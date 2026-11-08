"""
Web Search Tool — Tavily API integration for LLM-optimized web search.

Returns clean, structured content specifically designed for LLM consumption.
Also exposes a Gemini-native tool declaration so the LLM can decide to call
web search itself (function-calling / tool-use pattern).
"""

from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini Tool Declaration  (used by LLMClient.generate_with_tools_stream)
# ---------------------------------------------------------------------------
try:
    from google.genai import types as _genai_types

    SEARCH_TOOL = _genai_types.Tool(
        function_declarations=[
            _genai_types.FunctionDeclaration(
                name="web_search",
                description=(
                    "Search the web for current information, recent news, documentation, "
                    "tutorials, or any topic where real-time or detailed external data would "
                    "improve your answer. Use this when your internal knowledge is insufficient, "
                    "potentially outdated, or when the user is asking about something specific "
                    "and factual that benefits from live data."
                ),
                parameters=_genai_types.Schema(
                    type=_genai_types.Type.OBJECT,
                    properties={
                        "query": _genai_types.Schema(
                            type=_genai_types.Type.STRING,
                            description="The precise search query to look up on the web.",
                        )
                    },
                    required=["query"],
                ),
            )
        ]
    )
except ImportError:
    SEARCH_TOOL = None  # Graceful fallback if SDK not installed


async def execute_tool_call(name: str, args: dict, max_results: int = 5) -> str:
    """Dispatch an LLM-requested tool call by name and return the string result."""
    if name == "web_search":
        # Handle parameter variations LLMs might produce (query, queries, search_query, q)
        raw_query = (
            args.get("query")
            or args.get("queries")
            or args.get("search_query")
            or args.get("q")
            or ""
        )
        if isinstance(raw_query, (list, tuple)):
            query = " ".join(str(item) for item in raw_query if item)
        else:
            query = str(raw_query).strip()

        if not query:
            return "[web_search error: empty query]"
        return await tavily_search(query, max_results=max_results)
    return f"[Tool error: unknown tool '{name}']"


async def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily API and return clean, formatted results.
    
    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5).
        
    Returns:
        A formatted string of search results ready for LLM context injection.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set. Web search skipped.")
        return ""
    
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        
        # Run the synchronous Tavily client in a thread with a timeout
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.search,
                query=query,
                max_results=max_results,
                search_depth="basic",
                include_answer=True,
            ),
            timeout=15.0,  # 15 second hard timeout
        )
        
        # Format results for LLM consumption
        parts = ["--- WEB SEARCH RESULTS ---\n"]
        
        # Tavily provides a direct answer summary
        if response.get("answer"):
            parts.append(f"**Summary:** {response['answer']}\n")
        
        # Individual results
        for i, result in enumerate(response.get("results", []), 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", "")
            parts.append(f"[{i}] {title}\n    URL: {url}\n    {content}\n")
        
        formatted = "\n".join(parts)
        logger.info("Tavily search returned %d results for: %s", len(response.get("results", [])), query[:60])
        return formatted
        
    except ImportError:
        logger.error("tavily-python package not installed. Run: pip install tavily-python")
        return ""
    except Exception as e:
        logger.error("Tavily search failed: %s", e)
        return ""
