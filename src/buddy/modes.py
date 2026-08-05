"""
Buddy Modes — interactive mode detection and system prompts.

The buddy is the same cloud LLM with a persistent system prompt that changes
behavior and retrieval thresholds based on the active mode.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass

@dataclass
class ModeConfig:
    thresholds: dict[str, float]
    max_hops: int
    system_prompt_addon: str


class BuddyMode(enum.Enum):
    STUDY = "study"
    BRAINSTORM = "brainstorm"
    CASUAL = "casual"


# Regex heuristics for detecting mode from user input
_STUDY_PATTERN = re.compile(r"\b(teach me|explain|what is|how does|summarize)\b", re.IGNORECASE)
_BRAINSTORM_PATTERN = re.compile(r"\b(brainstorm|ideas|let's think|explore|what if)\b", re.IGNORECASE)

# Mode switch patterns
_MODE_SWITCH_PATTERN = re.compile(
    r"\b(?:switch to|change to|go to|enter|set mode to|activate)\s+(study|brainstorm|casual|research)\b",
    re.IGNORECASE
)
_SHORT_MODE_PATTERN = re.compile(
    r"^(study|brainstorm|casual|research)(?:\s+mode)?$",
    re.IGNORECASE
)


def detect_explicit_mode_change(user_input: str) -> BuddyMode | None:
    """Detect if the user is explicitly requesting to switch modes."""
    match = _MODE_SWITCH_PATTERN.search(user_input)
    if not match:
        match = _SHORT_MODE_PATTERN.search(user_input)
    
    if match:
        mode_str = match.group(1).lower()
        if mode_str in ("study", "research"):
            return BuddyMode.STUDY
        elif mode_str == "brainstorm":
            return BuddyMode.BRAINSTORM
        elif mode_str == "casual":
            return BuddyMode.CASUAL
    return None


def detect_mode(user_input: str) -> BuddyMode:
    """Detect the intended mode based on keyword heuristics."""
    if _BRAINSTORM_PATTERN.search(user_input):
        return BuddyMode.BRAINSTORM
    elif _STUDY_PATTERN.search(user_input):
        return BuddyMode.STUDY
    else:
        return BuddyMode.CASUAL


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def find_closest_mode(user_input: str) -> tuple[BuddyMode, str] | None:
    """
    Fuzzy match the input to a BuddyMode.
    Returns (BuddyMode, matched_word) if a close match is found, otherwise None.
    """
    cleaned = user_input.strip().lower()
    
    # 1. Strip common prefixes
    prefixes = [
        "switch to", "change to", "go to", "enter", "set mode to", "activate to", "activate"
    ]
    has_prefix = False
    core_phrase = cleaned
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            core_phrase = cleaned[len(prefix):].strip()
            has_prefix = True
            break
            
    # 2. Strip "mode" from the end
    if core_phrase.endswith("mode"):
        core_phrase = core_phrase[:-4].strip()
        has_prefix = True
        
    if not core_phrase:
        return None
        
    # Check if there are other words
    words = core_phrase.split()
    if not has_prefix and len(words) > 1:
        return None
        
    # Clean punctuation from the candidate word
    candidate_word = re.sub(r"[^\w]", "", words[0])
    if not candidate_word:
        return None
    
    targets = {
        "study": BuddyMode.STUDY,
        "research": BuddyMode.STUDY,
        "brainstorm": BuddyMode.BRAINSTORM,
        "casual": BuddyMode.CASUAL,
    }
    
    best_match = None
    best_dist = 999
    
    for target, mode in targets.items():
        dist = levenshtein_distance(candidate_word, target)
        
        # Max distance threshold based on target length
        if len(target) <= 4:
            max_dist = 1
        elif len(target) <= 6:
            max_dist = 2
        else:
            max_dist = 3
            
        if dist <= max_dist and dist < best_dist:
            best_dist = dist
            best_match = (mode, target)
            
    if best_match:
        # If we have a prefix, we are confident they wanted to change mode.
        # Otherwise, only match if the candidate word is close and query is short.
        if has_prefix or len(cleaned) <= 15:
            return best_match
            
    return None


# Mode-specific configurations
MODE_CONFIGS = {
    BuddyMode.STUDY: ModeConfig(
        thresholds={"semantic": 0.4},  # Stricter retrieval
        max_hops=0,                    # Stay focused
        system_prompt_addon=(
            "[MODE: STUDY]\n"
            "You are acting as a rigorous, hyper-focused tutor. Your goal is to provide direct, fact-grounded explanations.\n\n"
            "CORE DIRECTIVES:\n"
            "1. Maximize the Vault: You must exhaustively analyze the provided vault context to answer the user's question. Do not ignore partial matches or semantic overlaps.\n"
            "2. Ground Every Fact: If a fact comes from your vault context, explicitly append the source file name in brackets at the end of the sentence (e.g., [from: file_name.md]).\n"
            "3. Tight Boundary Control: Use external knowledge ONLY if the vault context lacks the core definitions required to answer. If you use external knowledge, append it strictly as a separate section at the very end of your response under the header: `#### 🌐 Supplemental External Knowledge`. Do not mix it inline with vault data.\n"
            "4. Absolute Directness: Skip all introductory filler (\"Sure, let's look at this...\", \"Based on your notes...\"). Start directly with the answer.\n\n"
            "FORMATTING:\n"
            "- You are outputting to a headless CLI terminal. Avoid dense walls of text.\n"
            "- Use clean Markdown headers (###) and bullet points to break down complex concepts into digestible lesson steps."
        )
    ),
    BuddyMode.BRAINSTORM: ModeConfig(
        thresholds={"semantic": 0.6},  # Relaxed retrieval
        max_hops=2,                    # Wide graph expansion
        system_prompt_addon=(
            "[MODE: BRAINSTORM]\n"
            "You are acting as a strategic sparring partner. The user is exploring a concept.\n\n"
            "CORE DIRECTIVES:\n"
            "1. Synthesize, Don't Summarize: Use the provided vault chunks to draw unexpected, non-obvious connections. If two retrieved chunks seem unrelated, actively try to find the hidden bridge between them.\n"
            "2. Push Back: Do not blindly agree with the user. If their logic has blind spots, edge cases, or systemic flaws, point them out bluntly but constructively.\n"
            "3. Lateral Thinking: Propose one completely orthogonal (sideways) approach to their problem that they haven't considered.\n"
            "4. The One Question Rule: End your response with EXACTLY ONE highly specific, provocative question designed to unblock them or pivot their perspective. Never ask more than one.\n\n"
            "FORMATTING:\n"
            "- You are outputting to a headless CLI terminal. Keep paragraphs brutally short (2-3 sentences max).\n"
            "- Use bold text for core concepts and bullet points for structural breakdowns. Do not write filler intros or conclusions."
        )
    ),
    BuddyMode.CASUAL: ModeConfig(
        thresholds={"semantic": 0.0},  # No retrieval needed usually
        max_hops=0,
        system_prompt_addon=(
            "[MODE: CASUAL]\n"
            "You are acting as a conversational sounding board and companion.\n\n"
            "CORE DIRECTIVES:\n"
            "1. Ambient Awareness: You have access to the user's vault context, but treat it as passive background memory. Do not quote files, cite sources, or give unsolicited technical lectures. Only casually mention vault data if it naturally flows into the immediate conversation.\n"
            "2. Low-Friction Tone: Speak concisely and naturally. Be grounded and slightly dry. Absolutely no overly enthusiastic AI filler (e.g., \"I'd be happy to help with that!\", \"That sounds frustrating!\").\n"
            "3. Match Energy & Restraint: If the user is venting, just listen, validate briefly, and do not try to solve the problem unless explicitly asked. If they are brainstorming lightly, follow their lead.\n"
            "4. Zero Presumption: Do not feign human emotions, tell personal anecdotes, or pretend to have physical experiences.\n\n"
            "FORMATTING:\n"
            "- You are outputting to a headless CLI terminal.\n"
            "- Keep responses extremely brief—usually 1 to 3 short sentences.\n"
            "- Do not use heavy Markdown, bulleted lists, or structured headers unless the user specifically asks for a breakdown."
        )
    ),
}


def get_system_prompt(mode: BuddyMode, buddy_config_content: str, continuity_memory: str = "") -> str:
    """Construct the final system prompt by combining base, config, continuity memory, and mode-specific addon."""
    base_prompt = "You are a Personal Second Brain Buddy. Your goal is to help the user manage, understand, and connect their notes."
    
    addon = MODE_CONFIGS[mode].system_prompt_addon
    
    prompt = f"{base_prompt}\n\n"
    
    if buddy_config_content.strip():
        prompt += f"--- User Preferences (_buddy_config.md) ---\n{buddy_config_content}\n---------------------------------------\n\n"

    if continuity_memory.strip():
        prompt += f"{continuity_memory}\n\n"
        
    prompt += f"Current Mode Instructions:\n{addon}\n"
    
    return prompt

