"""
Main Entry Point — boot sequence and interactive CLI loop.
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pyfiglet
from rich.align import Align
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich import box

# Custom theme for a clean monochrome look
_theme = Theme({
    "markdown.h1": "bold white",
    "markdown.h2": "bold white",
    "markdown.h3": "bold white",
    "markdown.h4": "bold white",
    "markdown.code": "white on grey11",
    "markdown.item.bullet": "white",
    "markdown.link": "underline white",
    "markdown.link_url": "dim",
    "markdown.block_quote": "dim italic",
})

# Global rich console — all output goes through this
console = Console(theme=_theme)


def render_response(text: str) -> None:
    """Render a response string as formatted Markdown with left padding."""
    md = Markdown(text, code_theme="ansi_dark")
    console.print(Padding(md, (0, 0, 0, 2)))

# Add the project root to sys.path so 'src.*' imports work
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import httpx
import yaml
from dotenv import load_dotenv

from src.buddy.modes import BuddyMode
from src.embedding.embedder import Embedder
from src.ingestion import full_vault_sync, ingest_file, purge_file
from src.llm_client import LLMClient
from src.orchestrator.orchestrator import Orchestrator
from src.retrieval.lexical import LexicalIndex
from src.session.session_manager import SessionManager
from src.session.workspace_manager import WorkspaceManager
from src.storage.chroma_client import ChromaClient
from src.storage.schema import init_db
from src.storage.sqlite_client import SQLiteClient
from src.watcher import VaultWatcher

# Setup logging
user_mode = os.environ.get("USER_MODE") == "1"
logging.basicConfig(
    level=logging.ERROR if user_mode else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
for _name in list(logging.Logger.manager.loggerDict):
    if _name.startswith("google"):
        logging.getLogger(_name).setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)


def load_config() -> dict:
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        logger.error("Config not found at %s", config_path)
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def start_embedding_service(config: dict) -> subprocess.Popen:
    """Start the embedding microservice as a subprocess using the current Python interpreter."""
    emb_config = config.get("embedding_service", {})
    python_exe = sys.executable
    port = emb_config.get("port", 9500)

    logger.info("Starting embedding microservice on port %d...", port)
    
    # We suppress output so it doesn't clutter the main CLI unless it errors
    process = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "src.embedding_service:app", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process


async def wait_for_embedding_service(embedder: Embedder, timeout: int = 120) -> bool:
    """Wait for the embedding service /health endpoint to return OK."""
    start_time = time.time()
    user_mode = os.environ.get("USER_MODE") == "1"

    if not user_mode:
        logger.info("Waiting for embedding model to load (can take 1-2 mins on first run if downloading)...")

    if user_mode:
        with Progress(
            SpinnerColumn(spinner_name="dots", style="bold #EEDC72"),
            TextColumn("[bold #EEDC72]Waking up Uxie...[/bold #EEDC72] [dim][/dim]"),
            transient=True,
            console=console,
        ) as progress:
            task = progress.add_task("", status="loading memory model")
            while time.time() - start_time < timeout:
                if await embedder.health_check():
                    console.print("[bold #A1C6D9] [/bold #A1C6D9] [bold #EEDC72]Uxie is UP & READY.[/bold #EEDC72]")
                    return True
                await asyncio.sleep(0.2)
        console.print("[bold #DF4D47]  Service timeout — Uxie failed to wake up.[/bold #DF4D47]")
        return False
    else:
        while time.time() - start_time < timeout:
            if await embedder.health_check():
                logger.info("Embedding service is ready!")
                return True
            await asyncio.sleep(2)
        return False


async def async_main():
    # 0. Load env variables
    load_dotenv()

    # 1. Load config
    config = load_config()
    vault_path = Path(config.get("vault_path", "./vault"))
    vault_path.mkdir(exist_ok=True)
    logs_path = Path(config.get("logs_path", "./logs"))
    logs_path.mkdir(exist_ok=True)
    
    # Check LLM key
    llm_key_env = config.get("llm", {}).get("api_key_env", "GEMINI_API_KEY")
    if llm_key_env not in os.environ:
        logger.warning("%s environment variable is not set. LLM calls will fail.", llm_key_env)

    # 2. Start & wait for embedding service
    emb_process = start_embedding_service(config)
    
    emb_config = config.get("embedding_service", {})
    embedder = Embedder(host=emb_config.get("host", "localhost"), port=emb_config.get("port", 9500))
    await embedder.start()
    
    if not await wait_for_embedding_service(embedder, timeout=300):
        logger.error("Embedding service failed to start within timeout.")
        emb_process.terminate()
        await embedder.close()
        sys.exit(1)

    # 3. Init Databases
    db_path = config.get("db_path", "./db/metadata.sqlite")
    conn = await init_db(db_path)
    sqlite = SQLiteClient(conn)
    workspace_manager = WorkspaceManager(conn)
    
    chroma = ChromaClient(config.get("chroma_path", "./db/chroma"), config.get("chroma", {}).get("collection_name", "vault_chunks"))
    chroma.initialize()

    # 4. Vault Sync
    logger.info("Running full vault sync...")
    await full_vault_sync(
        vault_root=vault_path,
        sqlite=sqlite,
        chroma=chroma,
        embedder=embedder,
        max_words=config.get("chunking", {}).get("max_words", 800),
        min_words=config.get("chunking", {}).get("min_words", 50),
        overlap_words=config.get("chunking", {}).get("overlap_words", 50),
    )

    # 5. Build Lexical Index
    logger.info("Building BM25 index...")
    lexical_index = LexicalIndex()
    all_chunks = await sqlite.get_all_chunks()
    lexical_index.build_index(all_chunks)

    # 6. Init Orchestrator components
    llm = LLMClient(model=config.get("llm", {}).get("model", "gemini-3.5-flash"))
    session_config = config.get("session", {})
    session_manager = SessionManager(
        drafts_path=config.get("drafts_path", "./drafts"),
        auto_draft_interval=session_config.get("auto_draft_interval", 5),
        prune_threshold=session_config.get("history_prune_threshold", 20),
        prune_keep=session_config.get("history_prune_keep", 18),
    )
    
    orchestrator = Orchestrator(
        config=config,
        sqlite=sqlite,
        chroma=chroma,
        embedder=embedder,
        llm=llm,
        lexical_index=lexical_index,
        session_manager=session_manager,
        vault_path=str(vault_path),
        workspace_manager=workspace_manager,
    )

    # 7. Start file watcher
    async def _on_file_changed(path: Path):
        """Called by the watcher when a .md file is created or modified in the vault."""
        logger.info("Watcher triggered ingestion for: %s", path.name)
        try:
            stats = await ingest_file(
                file_path=path,
                vault_root=vault_path,
                sqlite=sqlite,
                chroma=chroma,
                embedder=embedder,
                max_words=config.get("chunking", {}).get("max_words", 800),
                min_words=config.get("chunking", {}).get("min_words", 50),
                overlap_words=config.get("chunking", {}).get("overlap_words", 50),
            )
            # Rebuild BM25 index to include the new chunks
            all_chunks = await sqlite.get_all_chunks()
            lexical_index.build_index(all_chunks)
            logger.info("Indexed '%s' into DB: %s", path.name, stats)
        except Exception as exc:
            logger.error("Failed to ingest '%s': %s", path.name, exc)

    async def _on_file_deleted(path: Path):
        """Called by the watcher when a .md file is deleted from the vault."""
        logger.info("Watcher triggered purge for: %s", path.name)
        try:
            removed = await purge_file(
                file_path=path,
                vault_root=vault_path,
                sqlite=sqlite,
                chroma=chroma,
            )
            # Rebuild BM25 index after removal
            all_chunks = await sqlite.get_all_chunks()
            lexical_index.build_index(all_chunks)
            logger.info("Purged %d chunks for '%s'", removed, path.name)
        except Exception as exc:
            logger.error("Failed to purge '%s': %s", path.name, exc)

    watcher = VaultWatcher(vault_path, _on_file_changed, _on_file_deleted)
    watcher.start()

    # 8. Interactive Loop
    uxie_art_raw = """
                                               ..:.                                                                 
                                       -===++-:...::::=:                                                            
                                    =-.::=-::.::::::::::--                                                          
                                   :..::-::::::::-=++++=--=                                                         
                                  -:::--::::::=-:.......::-+                                                        
                                 :::::+:::::--::::..::::::::--                                                      
                                :-:::=#:::::::------::::::::-:=                                                     
                               -::::==##:::::=------=**-:::::::=                                                    
                               -.:::*###-::::--=++=--+++*=-::::-*                                                   
                              =:=+#-####---:::+::::+=+++++*=:::-=                                                   
                              .-#+==-##=:::::=-::::=*+++++*+*---=           ..                                      
                               -#+::=::::::::=*+=-:-*++++#+++====          ++*:                                     
                               .++%=-=::::::::=:::-+++++#++++#==*   +*.   -+++*                                     
                                 ++*--::::::::::--#+++#*+++++%=+.  *++*  .#+++*-                                    
                                  .++*+::::::--==#++#++++++++**    *++*= +*+***+ +#**+                              
                                    .+++++###@%#@++%++++++++++    =#%%%@-******%****#-                              
                                       .=#+++++*%+#+%++++*+.      =*%******+##******=                               
                                             .#**+#=-==      ******#*@****####****%:                                
                                             -+*---=---*      ******%+%%#**@%***#.                                  
                                            :+*----*----=.     :%***##%**##**#*                                     
                                            %-::----+----+        :%%**@.:%**#=                                     
                                          =-:::::---+-----=        +***###*****+                                    
                                         =::+:::----=#-----+       ******=:*%.                                      
                                  +.=   +::-.:+----=++:-----*        -*   -*#                                       
                                  =::==*--:-::-+-==+++ .=----.       +*.  =*=                                       
                                   =--=*---=+--=#++++-   +---*       #*   #*                                        
                                   =*+++++++%+++*+++#=    :--::     :#%  :##                                        
                                    =*+++++++%+++++***.    = =*     +*+  =*#                                        
                                      +*-*++++*+++%+++*             #*   **                                         
                                            #@%%+:*++++%           **+  +*+                                         
                                              *++++.*+++#         -**  .**                                          
                                               ++++*--*++++      .*#.  *#=                                          
                                                 #++=*+=++++%.  =**- .#@+                                           
                                                   -*+++*%#++++*##==***-                                            
                                                      .+#+++*#%*+=+#*.                                              
"""
    lines = [line for line in uxie_art_raw.split('\n') if line.strip()]
    min_indent = min(len(line) - len(line.lstrip()) for line in lines)
    uxie_art_cleaned = '\n'.join(line[min_indent:].rstrip() for line in lines)
    uxie_art_text = Text(uxie_art_cleaned, style="bold #A1C6D9")

    logo_raw = pyfiglet.figlet_format("UXIE", font="block")
    logo = '\n'.join(line.rstrip() for line in logo_raw.split('\n'))

    body = Text(justify="left")
    body.append(logo, style="bold #EEDC72")
    body.append("\n  Your Personal Knowledge Companion\n", style="dim")
    body.append("\n")
    body.append("  MODES :-\n", style="bold white")
    body.append("    CASUAL     ", style="bold white")
    body.append("General conversation (default)\n", style="dim")
    body.append("    STUDY      ", style="bold white")
    body.append("Retrieval search · \n", style="dim")
    body.append("    BRAINSTORM ", style="bold white")
    body.append("Expanded search · \n", style="dim")
    body.append("\n")
    body.append("\n")
    body.append("  COMMANDS :-\n", style="bold white")
    body.append("   /mode <casual|study|brainstorm>  ", style="white")
    body.append("Switch mode\n", style="dim")
    body.append("   /stash save <name>               ", style="white")
    body.append("Archive session\n", style="dim")
    body.append("   /stash list                      ", style="white")
    body.append("Show saved workspaces\n", style="dim")
    body.append("   /load <name>                     ", style="white")
    body.append("Resume a workspace\n", style="dim")
    body.append("   /new  /clear                     ", style="white")
    body.append("Clean slate\n", style="dim")
    body.append("   /save                            ", style="white")
    body.append("Save session to vault\n", style="dim")
    body.append("   /help                            ", style="white")
    body.append("All slash commands\n", style="dim")
    body.append("   list topics / list notes         ", style="white")
    body.append("Query vault index\n", style="dim")
    body.append("\n")
    body.append("  Type ", style="dim")
    body.append("exit", style="bold white")
    body.append(" or ", style="dim")
    body.append("quit", style="bold white")
    body.append(" to close.", style="dim")
    body.append("\n")
    body.append("\n")

    grid = Table.grid(expand=True)
    grid.add_column(ratio=3, no_wrap=True)
    grid.add_column(ratio=2, no_wrap=True)
    grid.add_row(body, Align.center(uxie_art_text, vertical="middle"))

    console.print()
    console.print(Panel(grid, border_style="#A1C6D9", box=box.DOUBLE_EDGE, padding=(0, 2)))

    if getattr(orchestrator, "_continuity_loaded", False):
        console.print(Panel(
            "[bold #DF4D47][/bold #DF4D47] [bold white]Continuity memory loaded[/bold white] [dim]— I remember our last session.[/dim]",
            border_style="#A1C6D9",
            box=box.SIMPLE,
        ))
    console.print()

    try:
        while True:
            try:
                # Use to_thread so the blocking input() doesn't pause the async event loop
                user_input = await asyncio.to_thread(input, "\n>> ")
            except EOFError:
                break
                
            if user_input.strip().lower() in ("exit", "quit"):
                break
            if not user_input.strip():
                continue
                
            # Handle query
            response_data = await orchestrator.handle_query(user_input)

            # handle_query can return either (response_str, mode) or (async_generator, mode)
            if isinstance(response_data, tuple):
                response, mode = response_data
                console.print(f"\n[bold #EEDC72]Uxie · {mode.value.upper()}[/bold #EEDC72]")
                console.rule(style="dim")

                if hasattr(response, "__aiter__"):
                    # Collect the stream behind a spinner, then render as Markdown
                    full_response = ""
                    with console.status("[dim]thinking...[/dim]", spinner="dots"):
                        async for chunk in response:
                            full_response += chunk
                    render_response(full_response)
                    console.rule(style="dim")
                    # Add the complete message to the session manager after streaming
                    orchestrator.session_manager.add_message("assistant", full_response)
                else:
                    # It's a regular string — render as Markdown directly
                    render_response(response)
                    console.rule(style="dim")
                    # We assume orchestrator already added the string to session manager (like casual mode)
            else:
                console.print(Panel(str(response_data), title="[bold #EEDC72]Uxie[/bold #EEDC72]", border_style="#A1C6D9"))
            
    except KeyboardInterrupt:
        console.print("\n[dim]Shutting down...[/dim]")
    finally:
        # Generate continuity memory before tearing down
        console.print("[dim]Saving working memory...[/dim]")
        try:
            await orchestrator.generate_continuity_summary(config.get("drafts_path", "./drafts"))
        except Exception as e:
            logger.error("Continuity memory generation failed: %s", e)
        
        # Cleanup
        session_manager.end_session()
        watcher.stop()
        await sqlite.close()
        await embedder.close()
        emb_process.terminate()
        emb_process.wait()


def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

        # Single-instance guard — prevent duplicate launches
        import ctypes
        _mutex = ctypes.windll.kernel32.CreateMutexW(None, True, "Global\\UxieSecondBrainMutex")
        if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            console.print("[bold #DF4D47]Uxie is already running.[/bold #DF4D47] [dim]Only one instance allowed.[/dim]")
            sys.exit(0)

    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
