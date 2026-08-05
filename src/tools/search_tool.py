"""
Web Search Tool — Tavily API integration for LLM-optimized web search.

Returns clean, structured content specifically designed for LLM consumption.
"""

from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)


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
