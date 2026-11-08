"""
Unit tests for the C++ Semantic Clustering Router integration in Uxie.
"""

import pytest
from src.routing import SemanticRouter, should_web_search


def test_semantic_router_lifecycle():
    """Test router creation, turn insertion, routing, and clean teardown."""
    router = SemanticRouter(dim=384, threshold=0.25)
    assert router.num_blocks == 0
    assert router.num_turns == 0

    # Turn 0
    b0 = router.add_turn("user", "How do I configure git credentials?")
    assert b0 == 0
    assert router.num_blocks == 1
    assert router.num_turns == 1

    # Turn 1
    b1 = router.add_turn("assistant", "Use git config --global credential.helper store")
    assert b1 == 0  # Follow-up should cluster into same block
    assert router.num_blocks == 1
    assert router.num_turns == 2

    # Turn 2: Switch to distinct topic (Docker)
    b2 = router.add_turn("user", "How do I build a multi-stage Dockerfile?")
    assert b2 == 1  # Should create a new block
    assert router.num_blocks == 2
    assert router.num_turns == 3

    # Turn 3: Assistant answers Docker question
    b3 = router.add_turn("assistant", "Use a Docker multi-stage build with Dockerfile: FROM alpine AS builder.")
    assert b3 == 1
    assert router.num_turns == 4

    # Route: Return to Git topic
    payload = router.route("Where is the gitconfig file stored?")
    # Should match block 0 (Git), NOT block 1 (Docker)
    assert payload.matched_block_id == 0
    assert payload.best_similarity > router.threshold

    # Verify N-1 Safety Net
    # Preceding assistant turn was Turn 3 (Docker). Since it's from a different block,
    # it must be preserved as the safety net!
    assert payload.safety_net is not None
    assert payload.safety_net.turn == 3
    assert "builder" in payload.safety_net.content

    # Verify LLM payload construction
    llm_msgs = payload.to_llm_messages()
    # Should include: Turn 0, Turn 1, Safety Net (Turn 3), and the new query
    assert len(llm_msgs) == 4
    assert llm_msgs[0]["content"] == "How do I configure git credentials?"
    assert llm_msgs[-1]["content"] == "Where is the gitconfig file stored?"

    # Export JSON check
    json_str = router.export_json()
    assert '"num_blocks": 2' in json_str
    assert '"num_turns": 4' in json_str

    router.close()


def test_custom_precomputed_embeddings():
    """Verify routing with precomputed normalized vectors (as supplied by Uxie's Embedder)."""
    router = SemanticRouter(dim=4, threshold=0.50)

    # Orthogonal vectors for topic A and topic B
    vec_a = [1.0, 0.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0, 0.0]

    b0 = router.add_turn("user", "Topic A question", embedding=vec_a)
    b1 = router.add_turn("assistant", "Topic A answer", embedding=vec_a)
    assert b0 == 0
    assert b1 == 0

    b2 = router.add_turn("user", "Topic B question", embedding=vec_b)
    assert b2 == 1

    # Query matching Topic A
    payload_a = router.route("Query for A", embedding=vec_a)
    assert payload_a.matched_block_id == 0
    assert pytest.approx(payload_a.best_similarity, abs=1e-4) == 1.0
    assert not payload_a.new_topic_created

    # Query matching Topic B
    payload_b = router.route("Query for B", embedding=vec_b)
    assert payload_b.matched_block_id == 1
    assert pytest.approx(payload_b.best_similarity, abs=1e-4) == 1.0

    # Query matching neither (new topic)
    vec_c = [0.0, 0.0, 1.0, 0.0]
    payload_c = router.route("Query for C", embedding=vec_c)
    assert payload_c.matched_block_id is None
    assert payload_c.new_topic_created

    router.close()


def test_web_search_classifier():
    """Verify local zero-latency web search intent classification."""
    # Greetings & fluff -> No web search
    assert should_web_search("hello") is False
    assert should_web_search("hey there") is False
    assert should_web_search("thanks!") is False
    assert should_web_search("thank you very much") is False
    assert should_web_search("ok") is False
    assert should_web_search("got it") is False

    # Short follow-ups on previous chat -> No web search
    assert should_web_search("can you repeat that?") is False
    assert should_web_search("show me that in python") is False
    assert should_web_search("give me a simpler example") is False

    # Substantive queries -> Yes web search
    assert should_web_search("How does Raft consensus work?") is True
    assert should_web_search("Latest news on quantum computing") is True
    assert should_web_search("Compare PostgreSQL vs CockroachDB") is True
    assert should_web_search("What is the difference between TCP and UDP?") is True
