"""
Workspace Commands — /stash, /load, /new for Git-style context switching.

These commands serialize and deserialize active chat sessions to/from
the SQLite workspaces table, enabling instant context pivots without
any context leakage between parallel threads of work.
"""

import os

from src.commands.registry import registry, safe_command


# ── /stash ────────────────────────────────────────────────────────────

@registry.register("/stash")
@safe_command
async def cmd_stash(context, arguments: str):
    """Route to subcommands: save, list, delete."""
    parts = arguments.strip().split(None, 1)
    subcmd = parts[0].lower() if parts else ""
    sub_args = parts[1].strip() if len(parts) > 1 else ""

    if subcmd == "save":
        return await _stash_save(context, sub_args)
    elif subcmd == "list":
        return await _stash_list(context)
    elif subcmd == "delete":
        return await _stash_delete(context, sub_args)
    else:
        help_text = (
            "**Stash commands:**\n"
            "  `/stash save <name>`   — Archive current session and clear workspace\n"
            "  `/stash list`          — Show all saved workspaces\n"
            "  `/stash delete <name>` — Delete a saved workspace"
        )
        return help_text, context.session_manager.session.current_mode


async def _stash_save(context, name: str):
    """Save the current session to the workspaces table and clear RAM."""
    mode = context.session_manager.session.current_mode

    if not name:
        return "Syntax: `/stash save <name>`", mode

    history = context.session_manager.session.history
    if not history:
        return "Nothing to stash — session is empty.", mode

    await context.workspace_manager.save(
        name=name,
        history=history,
        mode=mode,
    )

    # Clear active session (returns new_chunk_ids, which we discard here)
    context.session_manager.end_session()

    # Clear terminal for a fresh workspace
    os.system("cls" if os.name == "nt" else "clear")

    return f"  Session '{name}' archived. Workspace cleared.", context.session_manager.session.current_mode


async def _stash_list(context):
    """List all saved workspaces."""
    workspaces = await context.workspace_manager.list_all()
    mode = context.session_manager.session.current_mode

    if not workspaces:
        return "No saved workspaces found.", mode

    out = "###   Saved Workspaces\n\n"
    out += "| # | Name | Msgs | Mode | Last Updated | Preview |\n"
    out += "|---|------|------|------|--------------|---------|\n"

    for i, ws in enumerate(workspaces, 1):
        preview = ws["last_message"].replace("|", "\\|") if ws["last_message"] else "—"
        out += (
            f"| {i} | `{ws['name']}` | {ws['message_count']} "
            f"| {ws['mode']} | {ws['last_updated']} | {preview} |\n"
        )

    out += f"\n*Use `/load <name>` to resume a workspace.*"
    return out, mode


async def _stash_delete(context, name: str):
    """Delete a saved workspace."""
    mode = context.session_manager.session.current_mode

    if not name:
        return "Syntax: `/stash delete <name>`", mode

    deleted = await context.workspace_manager.delete(name)
    if deleted:
        return f"  Workspace '{name}' deleted.", mode
    else:
        return f"Workspace '{name}' not found.", mode


# ── /load ─────────────────────────────────────────────────────────────

@registry.register("/load")
@safe_command
async def cmd_load(context, arguments: str):
    """Load a stashed workspace, replacing the active session."""
    name = arguments.strip()
    mode = context.session_manager.session.current_mode

    if not name:
        return "Syntax: `/load <name>`", mode

    result = await context.workspace_manager.load(name)
    if result is None:
        return f"Workspace '{name}' not found. Use `/stash list` to see available workspaces.", mode

    history, loaded_mode = result

    # Replace active session
    context.session_manager.replace_session(history, loaded_mode)

    # Clear terminal
    os.system("cls" if os.name == "nt" else "clear")

    # Smart Screen Resume: show header + last 2-3 exchanges
    output = "=" * 55 + "\n"
    output += f"    WORKSPACE LOADED: {name}\n"
    output += "=" * 55 + "\n"

    recent = context.session_manager.get_last_n_exchanges(3)
    if recent:
        output += "  ... (older history hidden for clarity) ...\n\n"
        for msg in recent:
            role_label = "[User]" if msg["role"] == "user" else "[Buddy]"
            # Truncate very long messages for terminal display
            content = msg["content"]
            if len(content) > 300:
                content = content[:300] + "…"
            output += f"  {role_label}: {content}\n\n"

    output += "-" * 55 + "\n"
    output += "  Ready to resume! Type your prompt below.\n"

    return output, loaded_mode


# ── /new ──────────────────────────────────────────────────────────────

@registry.register("/new")
@safe_command
async def cmd_new(context, arguments: str):
    """Discard active memory without archiving (clean slate)."""
    context.session_manager.end_session()
    os.system("cls" if os.name == "nt" else "clear")
    return "   Workspace cleared — clean slate.", context.session_manager.session.current_mode
