"""
Zero-latency Local Web Search Classifier.

Classifies whether a casual query requires live web search (substantive questions,
explanations, news, technical lookups) or is purely conversational fluff (greetings,
thanks, acknowledgements).
"""

import re

# Single-token acknowledgements/affirmations/negations that never warrant a web search.
# Pattern is anchored at both ends and allows optional trailing punctuation.
_FLUFF_TOKENS = (
    "hi|hello|hey|greetings|"
    r"good\s+(morning|afternoon|evening)|"
    r"thanks|thank\s+you(\s+very\s+much)?(\s+so\s+much)?|thx|ty|"
    r"ok|okay|got\s+it|understood|sure|"
    r"cool|nice|great|awesome|perfect|"
    r"bye|goodbye|see\s+ya|"
    r"yes|yep|yup|no|nope|nah|"
    r"right|gotcha|makes\s+sense|agreed|exactly|absolutely|"
    r"go\s+on|tell\s+me\s+more|continue|proceed"
)
_FLUFF_PATTERN = re.compile(
    rf"^({_FLUFF_TOKENS})[?!.,]*$",
    re.IGNORECASE,
)

# Short follow-up patterns that refer to the previous chat turn's content
_FOLLOWUP_PATTERN = re.compile(
    r"\b(above|previous|earlier|that\s+code|repeat\s+that|give\s+me\s+that"
    r"|in\s+java|in\s+python|simpler\s+example|show\s+me\s+that)\b",
    re.IGNORECASE,
)

# Queries that contain at least one of these verbs/question-words are almost always
# substantive even if they are short (e.g. "how?", "why?", "explain that")
_QUESTION_PATTERN = re.compile(
    r"\b(what|who|where|when|why|how|explain|define|describe|compare|list|show|find|search|look\s+up)\b",
    re.IGNORECASE,
)


def should_web_search(user_input: str) -> bool:
    """
    Determine if a user query warrants external web search.
    Returns True for substantive queries, False for fluff/simple follow-ups.

    Decision order:
      1. Empty / blank                                        → False
      2. Pure fluff token (≤ 5 words, matches _FLUFF_PATTERN) → False
      3. Short in-context follow-up (≤ 7 words)              → False
      4. Contains explicit question/action word               → True
      5. Very short (≤ 3 words) with no question word        → False
      6. Everything else                                      → True
    """
    cleaned = user_input.strip()
    if not cleaned:
        return False

    words = cleaned.split()

    # Rule 1 — pure pleasantry / acknowledgement
    if len(words) <= 5 and _FLUFF_PATTERN.match(cleaned):
        return False

    # Rule 2 — explicit in-context follow-up
    if _FOLLOWUP_PATTERN.search(cleaned) and len(words) <= 7:
        return False

    # Rule 3 — contains a clear question/action word → always search
    if _QUESTION_PATTERN.search(cleaned):
        return True

    # Rule 4 — very short with no question word → no search
    if len(words) <= 3:
        return False

    # Default: treat as substantive
    return True
