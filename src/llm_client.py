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

    async def generate_with_tools_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[dict[str, str]] | None,
        tools: list,
        tool_executor,
        max_tool_rounds: int = 3,
    ):
        """
        Async generator that runs a tool-call loop then streams the final answer.

        Flow:
          1. Non-streaming call with tools so the model can emit function_call parts.
          2. For each function_call: execute via tool_executor, append result to contents.
          3. Repeat up to max_tool_rounds.
          4. When no more function_calls:
               - If a tool was used: stream the final answer (second LLM call).
               - If no tool used:    yield the response text directly (already in memory).

        Args:
            system_prompt:  System instructions.
            user_prompt:    Current user message.
            history:        Previous turns [{role, content}, ...].
            tools:          List of google.genai types.Tool objects.
            tool_executor:  async callable(name: str, args: dict) -> str
            max_tool_rounds: Safety cap on consecutive tool calls.

        Yields:
            str chunks (status messages + final answer text).
        """
        if not self._client:
            yield "Error: LLM client is not configured. Missing GEMINI_API_KEY?"
            return

        # Build initial contents list from history + current user message
        contents = []
        if history:
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(
                    types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
                )
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
        )

        config_with_tools = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
            tools=tools,
        )
        config_no_tools = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
        )

        tool_calls_made: list[str] = []
        response = None

        # --- Tool-call detection loop (non-streaming for each round) ---
        for round_num in range(max_tool_rounds):
            try:
                response = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self._model,
                    contents=contents,
                    config=config_with_tools,
                )
            except Exception as e:
                logger.error("LLM tool-call round %d failed: %s", round_num, e)
                yield f"Error communicating with LLM: {e}"
                return

            if not response.function_calls:
                break  # Model is done calling tools; response.text has the final answer

            # Execute each tool call in this round
            tool_response_parts = []
            for fc in response.function_calls:
                tool_calls_made.append(fc.name)
                logger.info(
                    "LLM requested tool '%s' with args: %s", fc.name, dict(fc.args)
                )
                result = await tool_executor(fc.name, dict(fc.args))
                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"result": result},
                    )
                )

            # Append model's function_call turn, then the tool results turn (Gemini requires role='user')
            contents.append(response.candidates[0].content)
            contents.append(types.Content(role="user", parts=tool_response_parts))

        # --- Yield status indicator if any tool was called ---
        if tool_calls_made:
            yield "*(Checking the web...)*\n\n"

        # --- Emit the final answer ---
        if not tool_calls_made:
            # No tools used: the first non-streaming response already has the answer.
            # Yield it directly — same latency as generate(), no second round-trip.
            yield response.text or ""
        else:
            # Tools were used: stream the final synthesis answer for a better UX.
            try:
                response_stream = await asyncio.to_thread(
                    self._client.models.generate_content_stream,
                    model=self._model,
                    contents=contents,
                    config=config_no_tools,
                )
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
            except Exception as e:
                logger.error("LLM final streaming after tool call failed: %s", e)
                yield f"Error communicating with LLM: {e}"


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
