import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.buddy.modes import BuddyMode, detect_explicit_mode_change, levenshtein_distance, find_closest_mode, detect_mode
from src.orchestrator.orchestrator import Orchestrator
from src.session.session_manager import SessionManager, Session, Message

def test_detect_explicit_mode_change():
    # Explicit switch commands
    assert detect_explicit_mode_change("switch to study mode") == BuddyMode.STUDY
    assert detect_explicit_mode_change("switch to research mode") == BuddyMode.STUDY
    assert detect_explicit_mode_change("switch to brainstorm mode") == BuddyMode.BRAINSTORM
    assert detect_explicit_mode_change("switch to casual mode") == BuddyMode.CASUAL
    
    # Short mode commands
    assert detect_explicit_mode_change("study mode") == BuddyMode.STUDY
    assert detect_explicit_mode_change("research mode") == BuddyMode.STUDY
    assert detect_explicit_mode_change("brainstorm mode") == BuddyMode.BRAINSTORM
    assert detect_explicit_mode_change("casual mode") == BuddyMode.CASUAL
    
    # Non-mode queries
    assert detect_explicit_mode_change("tell me what is RAG") is None
    assert detect_explicit_mode_change("explain how to write notes") is None
    assert detect_explicit_mode_change("save this note") is None


def test_detect_mode():
    assert detect_mode("let's brainstorm ideas") == BuddyMode.BRAINSTORM
    assert detect_mode("teach me about python") == BuddyMode.STUDY
    assert detect_mode("how do I save money") == BuddyMode.CASUAL
    assert detect_mode("explain what a save point is in gaming") == BuddyMode.STUDY  # Contains explain -> STUDY


@pytest.mark.asyncio
async def test_orchestrator_mode_persistence_and_commands():
    # Setup mocks
    mock_sqlite = AsyncMock()
    # Mock SQLite get_all_chunks for the 'list topics' command
    mock_sqlite.get_all_chunks.return_value = [
        {"file_path": "cognitive_science.md", "heading": "Mental Models"},
        {"file_path": "cognitive_science.md", "heading": "Habit Formation (part 1)"},
        {"file_path": "tech_architecture.md", "heading": "API Design"},
        {"file_path": "tech_architecture.md", "heading": "untitled"},
    ]
    
    mock_chroma = MagicMock()
    mock_chroma.count.return_value = 0
    mock_embedder = AsyncMock()
    mock_llm = AsyncMock()
    # Mock LLM generate
    mock_llm.generate.return_value = "Mocked LLM Response"
    
    mock_lexical = MagicMock()
    
    # Create real SessionManager but with mock parameters
    session_manager = SessionManager(drafts_path="./drafts_test", auto_draft_interval=5)
    # Ensure test doesn't create file in actual drafts path if possible, or cleanup
    session_manager._draft_file = MagicMock() # Mock file writing for test
    
    orchestrator = Orchestrator(
        config={
            "retrieval": {"top_k": 5, "weights": {"semantic": 0.5, "bm25": 0.3, "importance": 0.1, "recency": 0.1}},
            "graph": {"min_strength": 0.6}
        },
        sqlite=mock_sqlite,
        chroma=mock_chroma,
        embedder=mock_embedder,
        llm=mock_llm,
        lexical_index=mock_lexical,
        session_manager=session_manager,
        vault_path="./vault_test"
    )
    
    # Initial state should be CASUAL
    assert session_manager.session.current_mode == BuddyMode.CASUAL
    
    # 1. Test database listing command (e.g. 'list topics')
    # Should return formatted string and NOT change mode
    response, mode = await orchestrator.handle_query("list topics")
    assert "Vault Library" in response
    assert "cognitive_science.md" in response
    assert "Mental Models" in response
    assert "Habit Formation" in response
    assert "tech_architecture.md" in response
    assert "API Design" in response
    assert mode == BuddyMode.CASUAL
    assert session_manager.session.current_mode == BuddyMode.CASUAL
    
    # 2. Test explicit mode change command
    response, mode = await orchestrator.handle_query("switch to research mode")
    assert mode == BuddyMode.STUDY
    assert session_manager.session.current_mode == BuddyMode.STUDY
    
    # 3. Test persistent mode - subsequent queries stay in STUDY mode
    # Mock retrieval search results to avoid issues in handle_retrieval
    with patch("src.orchestrator.orchestrator.semantic_search", new_callable=AsyncMock) as mock_sem, \
         patch("src.orchestrator.orchestrator.rerank", new_callable=AsyncMock) as mock_rerank, \
         patch("src.orchestrator.orchestrator.expand_neighbors", new_callable=AsyncMock) as mock_expand:
        
        mock_sem.return_value = []
        mock_rerank.return_value = []
        mock_expand.return_value = set()
        
        response, mode = await orchestrator.handle_query("tell me about mental models")
        assert mode == BuddyMode.STUDY
        assert session_manager.session.current_mode == BuddyMode.STUDY
        
    # 4. Switch back to casual mode
    response, mode = await orchestrator.handle_query("switch to casual mode")
    assert mode == BuddyMode.CASUAL
    assert session_manager.session.current_mode == BuddyMode.CASUAL


def test_levenshtein_distance():
    assert levenshtein_distance("study", "study") == 0
    assert levenshtein_distance("reaearch", "research") == 1
    assert levenshtein_distance("reserarch", "research") == 1
    assert levenshtein_distance("studdy", "study") == 1
    assert levenshtein_distance("casula", "casual") == 2
    assert levenshtein_distance("brainstorrm", "brainstorm") == 1


def test_find_closest_mode():
    # Matching misspelled modes with prefixes
    assert find_closest_mode("switch to reaearch")[0] == BuddyMode.STUDY
    assert find_closest_mode("switch to reaearch")[1] == "research"
    assert find_closest_mode("switch to studdy")[0] == BuddyMode.STUDY
    assert find_closest_mode("switch to studdy")[1] == "study"
    assert find_closest_mode("switch to casula mode")[0] == BuddyMode.CASUAL
    assert find_closest_mode("switch to casula mode")[1] == "casual"
    assert find_closest_mode("switch to brainstorrm")[0] == BuddyMode.BRAINSTORM
    assert find_closest_mode("switch to brainstorrm")[1] == "brainstorm"
    
    # Matching short misspelled inputs without prefix
    assert find_closest_mode("reserarch")[0] == BuddyMode.STUDY
    assert find_closest_mode("studdy mode")[0] == BuddyMode.STUDY
    
    # Non-matching cases (general query or long phrases without prefix)
    assert find_closest_mode("tell me about research in AI") is None
    assert find_closest_mode("yup") is None
    assert find_closest_mode("switch to yes") is None


@pytest.mark.asyncio
async def test_orchestrator_fuzzy_mode_confirmation_flow():
    # Setup mocks
    mock_sqlite = AsyncMock()
    mock_chroma = MagicMock()
    mock_embedder = AsyncMock()
    mock_llm = AsyncMock()
    mock_lexical = MagicMock()
    
    session_manager = SessionManager(drafts_path="./drafts_test", auto_draft_interval=5)
    session_manager._draft_file = MagicMock()
    
    orchestrator = Orchestrator(
        config={
            "retrieval": {"top_k": 5, "weights": {"semantic": 0.5, "bm25": 0.3, "importance": 0.1, "recency": 0.1}},
            "graph": {"min_strength": 0.6}
        },
        sqlite=mock_sqlite,
        chroma=mock_chroma,
        embedder=mock_embedder,
        llm=mock_llm,
        lexical_index=mock_lexical,
        session_manager=session_manager,
        vault_path="./vault_test"
    )
    
    # Verify starting mode
    assert session_manager.session.current_mode == BuddyMode.CASUAL
    assert session_manager.session.pending_mode_switch is None
    
    # 1. Trigger fuzzy match with misspelling
    response, mode = await orchestrator.handle_query("switch to reaearch")
    assert "Did you mean to switch to 'research mode'?" in response
    assert mode == BuddyMode.CASUAL  # Mode has not changed yet
    assert session_manager.session.current_mode == BuddyMode.CASUAL
    assert session_manager.session.pending_mode_switch == BuddyMode.STUDY
    
    # 2. Confirm the change with "yup"
    response, mode = await orchestrator.handle_query("yup")
    assert "Switched to study mode." in response
    assert mode == BuddyMode.STUDY  # Mode changed!
    assert session_manager.session.current_mode == BuddyMode.STUDY
    assert session_manager.session.pending_mode_switch is None
    
    # 3. Trigger another fuzzy match
    response, mode = await orchestrator.handle_query("casula")
    assert "Did you mean to switch to 'casual mode'?" in response
    assert mode == BuddyMode.STUDY  # Still study
    assert session_manager.session.pending_mode_switch == BuddyMode.CASUAL
    
    # 4. Reject the change with "no"
    response, mode = await orchestrator.handle_query("no")
    assert "Okay, remaining in study mode." in response
    assert mode == BuddyMode.STUDY  # Kept study
    assert session_manager.session.current_mode == BuddyMode.STUDY
    assert session_manager.session.pending_mode_switch is None

