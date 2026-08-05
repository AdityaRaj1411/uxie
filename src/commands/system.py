import os
import re
from src.buddy.modes import find_closest_mode, levenshtein_distance
from src.commands.registry import registry, safe_command

@registry.register("/mode")
@safe_command
async def cmd_mode(context, arguments: str):
    if not arguments:
        return f"Current mode is: {context.session_manager.session.current_mode.value}", context.session_manager.session.current_mode
        
    fuzzy_match = find_closest_mode(arguments)
    if fuzzy_match is not None:
        matched_mode, matched_word = fuzzy_match
        split_args = arguments.split()
        if not split_args:
            return f"Did you mean to switch to '{matched_word} mode'?", context.session_manager.session.current_mode
        candidate_word = re.sub(r"[^\w]", "", split_args[0].lower())
        if levenshtein_distance(candidate_word, matched_word) == 0:
            context.session_manager.session.current_mode = matched_mode
            return f"Switched to {matched_mode.value} mode.", matched_mode
        else:
            context.session_manager.session.pending_mode_switch = matched_mode
            return f"Did you mean to switch to '{matched_word} mode'?", context.session_manager.session.current_mode
    return f"Unknown mode: {arguments}", context.session_manager.session.current_mode

@registry.register("/save")
@safe_command
async def cmd_save(context, arguments: str):
    response = await context._handle_save()
    return response, context.session_manager.session.current_mode

@registry.register("/help")
@safe_command
async def cmd_help(context, arguments: str):
    help_text = r"""### 🛠️ Functional Commands
| Command | Syntax | Description |
|---|---|---|
| `/mode` | `/mode [casual \| study \| brainstorm]` | Change behavior mode |
| `/save` | `/save` | Formats current session and saves to vault |
| `/help` | `/help` | Show this table |
| `/clear` | `/clear` | Discard active memory and clear console (alias: `/new`) |
| `/new` | `/new` | Discard active memory — clean slate |
| `/stash save` | `/stash save <name>` | Archive current session and clear workspace |
| `/stash list` | `/stash list` | Show all saved workspaces |
| `/stash delete` | `/stash delete <name>` | Delete a saved workspace |
| `/load` | `/load <name>` | Resume a stashed workspace |
| `/compact` | `/compact <note>` | LLM-rewrite a messy note |
| `/absorb` | `/absorb <file_path>` | Ingest foreign file |
| `/lint` | `/lint` | Evaluate structural health |
| `/reindex` | `/reindex` | Rebuild indexing layers |
| `/prune` | `/prune <note>` | Sever weak graph connections |"""
    return help_text, context.session_manager.session.current_mode

@registry.register("/clear")
@safe_command
async def cmd_clear(context, arguments: str):
    """Discard active memory and clear the console (alias for /new)."""
    context.session_manager.end_session()
    os.system('cls' if os.name == 'nt' else 'clear')
    return "🧹 Workspace cleared — clean slate.", context.session_manager.session.current_mode
