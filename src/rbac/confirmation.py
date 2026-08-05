"""
RBAC Confirmation Flow — interactive approval for destructive actions.

Nothing writes, overwrites, or deletes without explicit command + confirmation.
This module intercepts Editor/Admin requests to render diffs/previews and
ask for user confirmation in the CLI.
"""

from __future__ import annotations

import difflib
import logging
from pathlib import Path

from src.rbac.permissions import Permission

logger = logging.getLogger(__name__)

# Optional color support if colorama is available
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    _GREEN = Fore.GREEN
    _RED = Fore.RED
    _YELLOW = Fore.YELLOW
    _RESET = Style.RESET_ALL
except ImportError:
    _GREEN = ""
    _RED = ""
    _YELLOW = ""
    _RESET = ""


def preview_save(content: str, target_path: str | Path) -> str:
    """Render a preview for saving a NEW file."""
    preview = f"\n{_YELLOW}=== PREVIEW: Save New File to '{target_path}' ==={_RESET}\n"
    preview += content
    preview += f"\n{_YELLOW}=============================================={_RESET}\n"
    return preview


async def preview_diff(original: str, modified: str, target_path: str | Path) -> str:
    """Render a red/green diff for modifying an EXISTING file."""
    preview = f"\n{_YELLOW}=== DIFF: Modify Existing File '{target_path}' ==={_RESET}\n"
    
    orig_lines = original.splitlines(keepends=True)
    mod_lines = modified.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        orig_lines, mod_lines,
        fromfile="Original", tofile="Modified",
        n=3  # Context lines
    )
    
    for line in diff:
        if line.startswith("+"):
            preview += f"{_GREEN}{line}{_RESET}"
        elif line.startswith("-"):
            preview += f"{_RED}{line}{_RESET}"
        elif line.startswith("@@"):
            preview += f"{_YELLOW}{line}{_RESET}"
        else:
            preview += line

    preview += f"\n{_YELLOW}================================================={_RESET}\n"
    return preview


def confirm_action(permission_level: Permission, prompt_text: str = "") -> bool:
    """
    Prompt the user for interactive confirmation.
    
    - WRITER/EDITOR: Yes/No
    - ADMIN: Must type 'DELETE'
    """
    if permission_level == Permission.READER:
        return True

    print("\n" + prompt_text)
    
    try:
        if permission_level in (Permission.WRITER, Permission.EDITOR):
            response = input(f"{_YELLOW}Approve this change? [y/N]: {_RESET}").strip().lower()
            return response in ("y", "yes")
            
        elif permission_level == Permission.ADMIN:
            response = input(f"{_RED}DANGER: Admin action. Type 'DELETE' to confirm: {_RESET}").strip()
            return response == "DELETE"
            
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled.")
        return False
        
    return False
