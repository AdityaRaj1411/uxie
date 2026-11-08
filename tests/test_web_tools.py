"""
Unit tests for Web Search Tool Calling, /web, and /deep integrations.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.buddy.modes import BuddyMode
from src.tools.search_tool import SEARCH_TOOL, execute_tool_call


def test_search_tool_declaration():
    """Verify Gemini Tool and FunctionDeclaration structure."""
    assert SEARCH_TOOL is not None
    assert len(SEARCH_TOOL.function_declarations) == 1
    decl = SEARCH_TOOL.function_declarations[0]
    assert decl.name == "web_search"
    assert "query" in decl.parameters.properties
    assert "query" in decl.parameters.required


@pytest.mark.asyncio
async def test_execute_tool_call():
    """Verify tool execution dispatching."""
    with patch("src.tools.search_tool.tavily_search", new_callable=AsyncMock) as mock_tavily:
        mock_tavily.return_value = "Search Results For Python"

        # Valid tool call with 'query' string
        res = await execute_tool_call("web_search", {"query": "Python asyncio"})
        mock_tavily.assert_awaited_once_with("Python asyncio", max_results=5)
        assert res == "Search Results For Python"

        mock_tavily.reset_mock()
        # Valid tool call with 'queries' list (e.g. from Gemini tool call variations)
        res_list = await execute_tool_call("web_search", {"queries": ["weather in Ranchi", "current time"]})
        mock_tavily.assert_awaited_once_with("weather in Ranchi current time", max_results=5)
        assert res_list == "Search Results For Python"

        # Empty query
        res_empty = await execute_tool_call("web_search", {"query": ""})
        assert "empty query" in res_empty

        # Unknown tool
        res_unknown = await execute_tool_call("non_existent_tool", {})
        assert "unknown tool" in res_unknown


@pytest.mark.asyncio
async def test_web_and_deep_commands():
    """Verify /web and /deep commands in orchestrator."""
    from src.orchestrator.orchestrator import Orchestrator
    from src.session.session_manager import SessionManager

    mock_sqlite = AsyncMock()
    mock_chroma = MagicMock()
    mock_embedder = AsyncMock()
    mock_llm = AsyncMock()
    mock_lexical = MagicMock()
    session_manager = SessionManager(drafts_path="./drafts_test", auto_draft_interval=5)
    session_manager._draft_file = MagicMock()

    orchestrator = Orchestrator(
        config={"web_search": {"max_results": 3}},
        sqlite=mock_sqlite,
        chroma=mock_chroma,
        embedder=mock_embedder,
        llm=mock_llm,
        lexical_index=mock_lexical,
        session_manager=session_manager,
        vault_path="./vault_test",
    )

    # 1. /web without arguments -> returns usage string
    resp, mode = await orchestrator.handle_query("/web")
    assert "Usage: `/web <query>`" in resp

    # 2. /web with query -> returns generator and adds message
    with patch("src.tools.search_tool.tavily_search", new_callable=AsyncMock) as mock_tavily:
        mock_tavily.return_value = "Mocked Web Results"

        async def fake_stream(*args, **kwargs):
            yield "Web answer"

        mock_llm.generate_stream = fake_stream

        resp_gen, mode = await orchestrator.handle_query("/web latest rust features")
        assert mode == BuddyMode.CASUAL
        chunks = [c async for c in resp_gen]
        assert any("Web answer" in c for c in chunks)

    # 3. /deep without arguments -> returns usage string
    resp, mode = await orchestrator.handle_query("/deep")
    assert "Usage: `/deep <query>`" in resp

    # 4. /deep with query -> launches deep research
    with patch.object(orchestrator, "_handle_deep_research") as mock_deep:
        async def fake_deep(*args, **kwargs):
            yield "Deep research synthesis"

        mock_deep.return_value = fake_deep()

        resp_gen, mode = await orchestrator.handle_query("/deep quantum computing")
        assert mode == BuddyMode.CASUAL
        chunks = [c async for c in resp_gen]
        assert "Deep research synthesis" in chunks

