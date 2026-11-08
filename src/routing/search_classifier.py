"""
Zero-latency Local Web Search Classifier.

Classifies whether a casual query requires live web search (substantive questions,
explanations, news, technical lookups) or is purely conversational fluff (greetings,
thanks, acknowledgements).
"""

import re

# Conversational expressions that never require web search
_FLUFF_PATTERN = re.compile(
    r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|"
    r"thanks|thank\s+you|thx|ty|"
    r"ok|okay|got\s+it|understood|cool|nice|great|awesome|perfect|sure|"
    r"bye|goodbye|see\s+ya|"
    r"yes|no|yep|nope|nah)\b|[?!.]*$",
    re.IGNORECASE
)

# Follow-up patterns referring directly to previous chat turns
_FOLLOWUP_PATTERN = re.compile(
    r"\b(above|previous|earlier|that\s+code|repeat\s+that|give\s+me\s+that|in\s+java|in\s+python|simpler\s+example)\b",
    re.IGNORECASE
)


def should_web_search(user_input: str) -> bool:
    """
    Determine if a user query warrants external web search.
    Returns True for substantive queries, False for fluff/simple followups.
    """
    cleaned = user_input.strip()
    if not cleaned:
        return False

    words = cleaned.split()

    # Very short conversational pleasantries
    if len(words) <= 5 and _FLUFF_PATTERN.match(cleaned):
        return False

    # Immediate follow-up requests modifying recent content
    if _FOLLOWUP_PATTERN.search(cleaned) and len(words) <= 7:
        return False

    # Default to substantive for queries
    return True
