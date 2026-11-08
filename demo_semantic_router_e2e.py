"""
End-to-End Simulation: Testing Uxie's new C++ Semantic Clustering Router.

Simulates a user talking about Git, then switching to Docker, then returning
to Git, and demonstrates how Topic Blocks and the N-1 Safety Net work in practice.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.routing import SemanticRouter, should_web_search


def print_banner(text: str):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


async def main():
    print_banner("UXIE + C++ SEMANTIC CLUSTERING ROUTER E2E TEST")

    # Initialize router (384-dim matching Uxie's SentenceTransformer model)
    router = SemanticRouter(dim=384, threshold=0.25)
    print(f"✓ C++ AVX2 Router initialized (dim={router._dim}, threshold={router.threshold})")

    conversation = [
        # Chapter 1: Git setup
        ("user", "How do I setup GitHub Desktop on my Windows PC?"),
        ("assistant", "Download the setup installer from desktop.github.com and sign in with your GitHub account."),
        ("user", "Where do I configure my SSH key for GitHub?"),
        ("assistant", "Go to Settings > SSH and GPG keys > click 'New SSH key' and paste your public key."),

        # Chapter 2: Pivot to Networking
        ("user", "What is a subnet mask in computer networking?"),
        ("assistant", "A subnet mask determines which part of an IP address refers to the network and which refers to the host."),

        # Chapter 3: Return to Git
        ("user", "How do I configure GitHub Desktop again?"),
    ]

    print("\n[Simulating multi-turn conversation]")

    for idx, (role, text) in enumerate(conversation[:-1]):
        t0 = time.perf_counter()
        block_id = router.add_turn(role, text)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000

        icon = "👤" if role == "user" else "🤖"
        print(f"  Turn {idx} {icon} [{role.upper()}]: \"{text[:45]}...\"")
        print(f"       ↳ Assigned to Block {block_id} in {elapsed_us:.2f} µs")

    print(f"\n✓ Conversation populated: {router.num_blocks} Topic Blocks across {router.num_turns} turns.")

    # Now route the final query
    final_query = conversation[-1][1]
    print_banner("ROUTING QUERY: Topic Shift Back to GitHub")
    print(f"Query: \"{final_query}\"")

    t0 = time.perf_counter()
    payload = router.route(final_query)
    route_us = (time.perf_counter() - t0) * 1_000_000

    print(f"\n[Routing Results in {route_us:.2f} µs]")
    print(f"  • Best Similarity Score : {payload.best_similarity:.4f}")
    print(f"  • Matched Topic Block   : Block {payload.matched_block_id}")
    print(f"  • New Topic Created     : {payload.new_topic_created}")
    print(f"  • Local Web Search Flag : {should_web_search(final_query)}")

    print(f"\n[Context Payload Constructed for LLM ({len(payload.to_llm_messages())} messages)]:")
    for msg in payload.to_llm_messages():
        role = msg["role"]
        content = msg["content"]
        is_query = content == final_query
        tag = "CURRENT QUERY" if is_query else ("SAFETY NET (N-1)" if payload.safety_net and content == payload.safety_net.content else "TOPIC BLOCK MSG")
        print(f"  - [{tag}] {role}: {content}")

    # Assertions for verification
    assert payload.matched_block_id == 0, f"Expected match with Block 0, got {payload.matched_block_id}"
    assert payload.safety_net is not None, "Expected safety net from Turn 5 (Subnet mask)"
    assert payload.safety_net.turn == 5, f"Expected safety net turn 5, got {payload.safety_net.turn}"

    # Also test completely new topic
    print_banner("ROUTING QUERY: Brand New Topic (Baking)")
    new_query = "What ingredients do I need for sourdough bread?"
    t0 = time.perf_counter()
    new_payload = router.route(new_query)
    new_us = (time.perf_counter() - t0) * 1_000_000

    print(f"Query: \"{new_query}\"")
    print(f"  • Best Similarity Score : {new_payload.best_similarity:.4f}")
    print(f"  • Matched Topic Block   : {new_payload.matched_block_id}")
    print(f"  • New Topic Created     : {new_payload.new_topic_created} in {new_us:.2f} µs")
    print(f"  • Safety Net Preserved  : [Turn {new_payload.safety_net.turn}] {new_payload.safety_net.content[:50]}...")
    assert new_payload.new_topic_created is True
    assert new_payload.matched_block_id is None

    print_banner("TEST COMPLETE: ALL INVARIANTS VERIFIED!")


if __name__ == "__main__":
    asyncio.run(main())
