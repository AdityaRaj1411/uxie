"""
LLM Client — wrapper around Google GenAI (Gemini).

Handles standard generation and structured classification tasks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TypeVar, Type

from google import genai
from google.genai import types
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Wrapper for the Gemini API."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-3.5-flash"):
        # The genai.Client automatically picks up GEMINI_API_KEY from the environment
        # if api_key is None.
        try:
            self._client = genai.Client(api_key=api_key)
            self._model = model
            logger.info("LLMClient initialized with model: %s", self._model)
        except Exception as e:
            logger.error("Failed to initialize Google GenAI client: %s", e)
            self._client = None
            self._model = model

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Generate a standard text response.
        
        Args:
            system_prompt: The system instructions.
            user_prompt: The current user message.
            history: Optional list of previous messages [{"role": "user"/"assistant", "content": "..."}, ...]
        """
        if not self._client:
            return "Error: LLM client is not configured. Missing GEMINI_API_KEY?"

        contents = []
        if history:
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
                
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]))

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
        )

        try:
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._model,
                contents=contents,
                config=config,
            )
            return response.text or ""
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return f"Error communicating with LLM: {e}"

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
    ):
        """Generate a streamed text response."""
        if not self._client:
            yield "Error: LLM client is not configured. Missing GEMINI_API_KEY?"
            return

        contents = []
        if history:
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
                
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]))

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # generate_content_stream currently runs synchronously in the google-genai SDK,
                # so we run it in a thread and iterate.
                response_stream = await asyncio.to_thread(
                    self._client.models.generate_content_stream,
                    model=self._model,
                    contents=contents,
                    config=config,
                )
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
                return  # Success, exit retry loop
                
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "resource_exhausted" in error_str or "quota" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s
                        logger.warning(f"Rate limited (429). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                        yield f"*(Hit Gemini rate limit. Retrying in {wait_time} seconds...)*\n"
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error("LLM streaming failed after retries: %s", e)
                        yield f"\n\n**Error:** I hit a rate limit (Quota Exceeded) and couldn't recover. You may need to wait a minute, or change your model in `settings.yaml`."
                else:
                    logger.error("LLM streaming failed: %s", e)
                    yield f"Error communicating with LLM: {e}"
                    return

    async def classify_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
    ) -> T:
        """
        Force the LLM to return a structured JSON response matching a Pydantic model.
        """
        if not self._client:
            raise RuntimeError("LLM client not configured.")

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,  # Low temp for classification
            response_mime_type="application/json",
            response_schema=response_model,
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self._model,
                    contents=user_prompt,
                    config=config,
                )
                
                # The SDK parses this back into the Pydantic model if structured outputs are supported
                if hasattr(response, "parsed") and response.parsed is not None:
                    if isinstance(response.parsed, response_model):
                        return response.parsed
                    elif isinstance(response.parsed, dict):
                        return response_model.model_validate(response.parsed)
                        
                # Fallback manual parse
                return response_model.model_validate_json(response.text)
                
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "resource_exhausted" in error_str or "quota" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2
                        logger.warning(f"Rate limited on structured classification. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error("Structured classification failed after retries: %s", e)
                        raise
                else:
                    logger.error("Structured LLM generation failed: %s", e)
                    raise
