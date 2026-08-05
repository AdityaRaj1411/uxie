"""
Session Save Template — enforces structured format when saving session notes.

The LLM is prompted to fill out this specific markdown template when saving,
preventing free-form conversational text from polluting the knowledge vault.
"""

from __future__ import annotations

import re
from datetime import datetime

_TEMPLATE_PROMPT = """
You are tasked with saving the current conversation into the user's knowledge vault.
You MUST format your response EXACTLY following this Markdown template. Do not add any conversational filler.

Template:

## Session Save — {date_str}

### Key Concepts Introduced
- **[Concept Name]**: [Concise definition or summary]
- ...

### Decisions Made
- **[Decision/Action]**: [Reasoning and context]
- ...

### Ideas to Explore Later
- **[Topic]**: [Research questions or exploration angles]
- ...

### Facts / References to Remember
- **[Reference]**: [Verbatim facts, code fragments, constraints, links]
- ...
"""


def get_save_prompt() -> str:
    """Get the system prompt instructing the LLM to use the structured template."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return _TEMPLATE_PROMPT.format(date_str=now)


def fill_template(llm_response: str) -> str:
    """
    Validate and clean the LLM's structured save response.
    
    Ensures the response has the required headings. If missing, it appends them empty.
    """
    content = llm_response.strip()
    
    required_sections = [
        "### Key Concepts Introduced",
        "### Decisions Made",
        "### Ideas to Explore Later",
        "### Facts / References to Remember"
    ]
    
    missing = [sec for sec in required_sections if sec not in content]
    
    if missing:
        content += "\n\n"
        for sec in missing:
            content += f"{sec}\n- None recorded.\n\n"
            
    # Strip out any conversational wrapper the LLM might have added (e.g. "Here is your saved note:\n")
    match = re.search(r"(## Session Save —.*)", content, flags=re.DOTALL)
    if match:
        content = match.group(1)
        
    return content
