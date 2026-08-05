"""
RBAC Permissions — definitions and evaluation logic.

Defines the 4 permission levels (Reader, Writer, Editor, Admin) and
determines which level is required for specific system actions.
"""

from __future__ import annotations

import enum
import logging

logger = logging.getLogger(__name__)


class Permission(enum.IntEnum):
    """Hierarchical permission levels."""
    READER = 0  # Can search and read notes
    WRITER = 1  # Can save *new* notes
    EDITOR = 2  # Can modify existing notes
    ADMIN = 3   # Can delete notes or purge DB


class ActionType(enum.Enum):
    """System actions that require permissions."""
    SEARCH = "search"
    CHAT = "chat"
    SAVE_NEW = "save_new"
    MODIFY_EXISTING = "modify_existing"
    DELETE = "delete"


def evaluate_permission(action: ActionType) -> Permission:
    """Determine the required permission level for an action."""
    if action in (ActionType.SEARCH, ActionType.CHAT):
        return Permission.READER
    elif action == ActionType.SAVE_NEW:
        return Permission.WRITER
    elif action == ActionType.MODIFY_EXISTING:
        return Permission.EDITOR
    elif action == ActionType.DELETE:
        return Permission.ADMIN
    
    # Default safe fallback
    return Permission.ADMIN


def check_permission(required: Permission) -> bool:
    """
    Check if a permission is granted.
    
    Since this is a CLI app for a single user, READER is always granted.
    Anything higher requires an interactive confirmation flow (handled in confirmation.py).
    """
    return required <= Permission.READER
