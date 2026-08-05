"""
Buddy Config Loader — reads _buddy_config.md from the vault.

This config defines the persistent personality and preferences of the buddy.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_buddy_config(vault_path: Path | str) -> str:
    """
    Load the _buddy_config.md file from the vault.
    
    Returns the content as a string. If the file doesn't exist, returns
    an empty string and logs a warning.
    """
    config_file = Path(vault_path) / "_buddy_config.md"
    
    if not config_file.exists():
        logger.warning("Buddy config not found at %s. Using default empty config.", config_file)
        return ""
        
    try:
        content = config_file.read_text(encoding="utf-8").strip()
        return content
    except Exception as e:
        logger.error("Failed to read buddy config %s: %s", config_file, e)
        return ""
