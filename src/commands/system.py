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
| `/prune` | `/prune <note>` | Sever weak graph connections |
| `/router` | `/router [threshold <val>]` | Inspect or tune C++ Semantic Router clusters |"""
    return help_text, context.session_manager.session.current_mode

@registry.register("/router")
@registry.register("/topics")
@safe_command
async def cmd_router(context, arguments: str):
    """Inspect or adjust the C++ Semantic Clustering Router state."""
    router = getattr(context, "semantic_router", None)
    if not router:
        return "Semantic router is not enabled.", context.session_manager.session.current_mode

    args = arguments.strip().split()
    if args and args[0].lower() == "threshold" and len(args) > 1:
        try:
            val = float(args[1])
            router.threshold = val
            return f"✅ Router similarity threshold updated to `{val:.2f}`.", context.session_manager.session.current_mode
        except ValueError:
            return f"❌ Invalid threshold `{args[1]}`. Please provide a float (e.g. `0.55`).", context.session_manager.session.current_mode

    import json
    raw_json = router.export_json()
    data = json.loads(raw_json)

    blocks = data.get("blocks", [])
    num_blocks = len(blocks)
    num_turns = data.get("num_turns", 0)
    thresh = data.get("similarity_threshold", router.threshold)
    dim = data.get("embedding_dim", router._dim)

    res = [
        f"### ⚡ Semantic Clustering Router Status (AVX2 SIMD)\n",
        f"- **Topic Blocks**: {num_blocks}",
        f"- **Total Turns**: {num_turns}",
        f"- **Similarity Threshold**: `{thresh:.2f}`",
        f"- **Vector Dimension**: {dim}\n",
    ]

    if not blocks:
        res.append("*No topic blocks formed yet. Send a query to begin clustering.*")
    else:
        res.append("#### 📂 Active Conversation Chapters:\n")
        for b in blocks:
            b_id = b.get("block_id")
            msgs = b.get("messages", [])
            first_msg = msgs[0]["content"][:60] if msgs else "(empty)"
            turns_str = f"Turns {msgs[0]['turn']}–{msgs[-1]['turn']}" if msgs else "empty"
            res.append(f"- **Block {b_id}** ({len(msgs)} turns, {turns_str}):\n  *\"{first_msg}...\"*")

    return "\n".join(res), context.session_manager.session.current_mode

@registry.register("/clear")
@safe_command
async def cmd_clear(context, arguments: str):
    """Discard active memory and clear the console (alias for /new)."""
    context.session_manager.end_session()
    os.system('cls' if os.name == 'nt' else 'clear')
    return "🧹 Workspace cleared — clean slate.", context.session_manager.session.current_mode
