"""
Groq Client — specifically for fast Context Pivot Detection using a small model.

Also decides whether a live web search is needed for casual mode queries.
"""

import asyncio
import os
import re
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Anaphoric / back-reference patterns that strongly signal a follow-up, not a pivot.
# If the user's message matches any of these, we skip the Groq call entirely.
_ANAPHORIC_PATTERNS = re.compile(
    r"\b(the above|above code|above example|that code|that example|the same"
    r"|give me that|give that|show me that|convert that|convert it|convert this"
    r"|translate that|translate it|translate this|rewrite that|rewrite it|rewrite this"
    r"|do that|do it|do this|repeat that|repeat it|repeat this"
    r"|in java|in python|in c\+\+|in javascript|in typescript|in rust|in go|in ruby"
    r"|in c#|in kotlin|in swift|in scala|in php|in perl"
    r"|more detail|more about|elaborate|expand on|can you explain"
    r"|what about|how about|and what|but what|but how|and how"
    r"|instead|as well|also show|also give|also explain)\b",
    re.IGNORECASE,
)


class PivotDetector:
    def __init__(self, config: dict):
        self.config = config.get("pivot_detector", {})
        api_key_env = self.config.get("api_key_env", "GROQ_API_KEY")
        self.model = self.config.get("model", "qwen/qwen3.8-27b")
        # Handle 32b naming variation -> active Groq model
        if "32b" in self.model.lower() and "qwen" in self.model.lower():
            logger.info("Mapping requested Qwen model '%s' to active Groq model 'qwen/qwen3.8-27b'", self.model)
            self.model = "qwen/qwen3.8-27b"

        api_key = os.environ.get(api_key_env)
        self.client = None
        if api_key:
            try:
                self.client = Groq(api_key=api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
        else:
            logger.warning(f"No {api_key_env} found. Pivot detection will be disabled.")

    async def detect_pivot(self, user_input: str, history: list[dict[str, str]]) -> tuple[bool, str, bool]:
        """
        Analyzes user_input against recent history. Runs the Groq API call in a
        thread so it never blocks the asyncio event loop.

        Returns:
            (is_pivot: bool, search_query: str, needs_web_search: bool)
            - is_pivot        → True means clear session history (topic changed).
            - search_query    → original or rewritten standalone query.
            - needs_web_search→ True means call Tavily in casual mode.
        """
        if not self.client:
            return False, user_input, False

        # --- Heuristic Guard: Anaphoric references force CONTINUE ---
        # Words like "above", "that code", "in Java", "elaborate" etc. are
        # dead giveaways that the user is referring to the previous context.
        if _ANAPHORIC_PATTERNS.search(user_input):
            logger.debug("Heuristic guard: anaphoric reference detected, forcing CONTINUE")
            return False, user_input, False

        recent_history = history[-6:]
        history_text = ""
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

        system_prompt = (
            "You are a strict conversational router. "
            "Analyze the user's latest query against recent conversation history.\n\n"
            
            "If the <chat_history> is empty, there is no context to pivot from. In this case, "
            "output CONTINUE, leave the query exactly as it is, and only determine SEARCH/NO_SEARCH.\n\n"

            "Task 1 — PIVOT check: Is the user COMPLETELY changing topic? "
            "Unrelated to history → PIVOT. Follow-up/clarification → CONTINUE.\n"
            "CRITICAL: Asking for the SAME content in a different language, format, or level of detail "
            "is ALWAYS a follow-up (CONTINUE), NOT a pivot.\n\n"

            "Task 2 — Rewrite: If PIVOT, rewrite query to stand alone. If CONTINUE, output as-is.\n\n"

            "Task 3 — Web search:\n"
            "  SEARCH   → any substantive query: explanations, concepts, how-tos, deep dives,\n"
            "              current events, news, prices, technical topics, comparisons.\n"
            "  NO_SEARCH → ONLY conversational fluff: greetings, thanks,\n"
            "               acknowledgements ('ok', 'got it'), or trivial arithmetic.\n\n"

            "### Examples\n"
            "Query: 'Explain how neural networks work'\n"
            "CONTINUE\nExplain how neural networks work\nSEARCH\n\n"
            "Query: 'How do I set up Docker?'\n"
            "CONTINUE\nHow do I set up Docker?\nSEARCH\n\n"
            "Query: 'Latest AI news'\n"
            "CONTINUE\nLatest AI news\nSEARCH\n\n"
            "Query: 'Thanks!'\n"
            "CONTINUE\nThanks!\nNO_SEARCH\n\n"
            "Query: 'Can you repeat that?'\n"
            "CONTINUE\nCan you repeat that?\nNO_SEARCH\n\n"
            "Query: 'Can you give the above code in Java?'\n"
            "CONTINUE\nCan you give the above code in Java?\nNO_SEARCH\n\n"
            "Query: 'Show me that in Python instead'\n"
            "CONTINUE\nShow me that in Python instead\nNO_SEARCH\n\n"
            "Query: 'Can you elaborate on the third point?'\n"
            "CONTINUE\nCan you elaborate on the third point?\nSEARCH\n\n"
            "Query: 'What about the disadvantages?'\n"
            "CONTINUE\nWhat about the disadvantages?\nSEARCH\n\n"
            "Query: 'Give me a simpler example'\n"
            "CONTINUE\nGive me a simpler example\nNO_SEARCH\n\n"

            "Output EXACTLY three lines:\n"
            "Line 1: PIVOT or CONTINUE\n"
            "Line 2: The query (rewritten or original)\n"
            "Line 3: SEARCH or NO_SEARCH"
        )

        user_prompt = f"<chat_history>\n{history_text}\n</chat_history>\n\n### Latest User Query:\n{user_input}"

        try:
            # Run synchronous Groq SDK in a thread — never blocks the event loop
            response = await asyncio.to_thread(
                lambda: self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=self.model,
                    temperature=0.0,
                    max_tokens=80,  # only 3 short lines needed
                )
            )

            output = response.choices[0].message.content.strip()
            lines = [line.strip() for line in output.split("\n") if line.strip()]

            if len(lines) >= 2:
                is_pivot = "PIVOT" in lines[0].upper()
                rewritten_query = lines[1]

                # Line 3 is optional — default NO_SEARCH if Groq truncates
                needs_web_search = False
                if len(lines) >= 3:
                    needs_web_search = (
                        "SEARCH" in lines[2].upper()
                        and "NO_SEARCH" not in lines[2].upper()
                    )

                logger.debug(
                    "Groq: pivot=%s, web_search=%s, query='%s'",
                    is_pivot, needs_web_search, rewritten_query[:60],
                )
                return is_pivot, rewritten_query, needs_web_search

        except Exception as e:
            err_msg = str(e)
            if ("model_not_found" in err_msg or "does not exist" in err_msg) and self.model == "llama-3.3-70b-versatile":
                fallback_model = "qwen/qwen3.8-27b"
                logger.warning(
                    "Groq model '%s' not available on this account. Falling back to '%s'.",
                    self.model, fallback_model
                )
                self.model = fallback_model
                try:
                    return await self.detect_pivot(user_input, history)
                except Exception as inner_e:
                    logger.error(f"Fallback Groq pivot detection failed: {inner_e}")
            else:
                logger.error(f"Groq pivot detection failed: {e}")

        return False, user_input, False
