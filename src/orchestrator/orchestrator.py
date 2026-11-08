"""
Orchestrator — the central brain of the Second Brain backend.

Routes user queries through intent classification, retrieval arms, graph expansion,
reranking, context assembly, and finally the LLM. Also manages session state and RBAC.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.buddy.config_loader import load_buddy_config
from src.buddy.modes import (
    BuddyMode,
    get_system_prompt,
    detect_mode,
    detect_explicit_mode_change,
    find_closest_mode,
)
from src.embedding.embedder import Embedder
from src.graph.link_generator import generate_links
from src.graph.neighbor_expander import expand_neighbors
from src.ingestion import ingest_file
from src.llm_client import LLMClient
from src.routing import SemanticRouter, RoutePayload, should_web_search
from src.orchestrator.intent_classifier import get_intent_params
from src.rbac.confirmation import confirm_action, preview_save
from src.rbac.permissions import ActionType, evaluate_permission, Permission
from src.retrieval.lexical import LexicalIndex
from src.retrieval.reranker import rerank, RerankerWeights
from src.retrieval.semantic import semantic_search
from src.session.continuity import (
    CONTINUITY_EXTRACTION_PROMPT,
    format_memory_for_injection,
    load_continuity_memory,
    save_continuity_memory,
)
from src.session.save_template import fill_template, get_save_prompt
from src.session.session_manager import SessionManager
from src.session.workspace_manager import WorkspaceManager
from src.storage.chroma_client import ChromaClient
from src.storage.sqlite_client import SQLiteClient

logger = logging.getLogger(__name__)

_CONFIRM_YES_PATTERN = re.compile(r"\b(yes|yep|yup|y|sure|yeah|ok|okay|correct|indeed)\b", re.IGNORECASE)
_CONFIRM_NO_PATTERN = re.compile(r"\b(no|n|nope|nah|cancel|don't|dont|negative)\b", re.IGNORECASE)


class Orchestrator:
    def __init__(
        self,
        config: dict[str, Any],
        sqlite: SQLiteClient,
        chroma: ChromaClient,
        embedder: Embedder,
        llm: LLMClient,
        lexical_index: LexicalIndex,
        session_manager: SessionManager,
        vault_path: str,
        workspace_manager: WorkspaceManager | None = None,
    ):
        self.config = config
        self.sqlite = sqlite
        self.chroma = chroma
        self.embedder = embedder
        self.llm = llm
        self.lexical_index = lexical_index
        self.session_manager = session_manager
        self.vault_path = vault_path
        self.workspace_manager = workspace_manager
        self._chunking = config.get("chunking", {})
        self._drafts_path = config.get("drafts_path", "./drafts")

        # High-performance C++ Semantic Clustering Router (AVX2 SIMD)
        router_cfg = config.get("semantic_router", {})
        sim_threshold = router_cfg.get("threshold", 0.55)
        default_dim = 768 if "mpnet" in str(config.get("embedding_service", {}).get("model", "")) else 384
        emb_dim = router_cfg.get("dimension", default_dim)
        self.semantic_router = SemanticRouter(dim=emb_dim, threshold=sim_threshold)
        self.session_manager.router = self.semantic_router
        self._current_route_payload: RoutePayload | None = None
        
        self._buddy_config_content = load_buddy_config(vault_path)

        # Load continuity memory from previous session (if fresh enough)
        raw_memory = load_continuity_memory(self._drafts_path)
        if raw_memory:
            self._continuity_loaded = True
            
            # Gemini API requires history to start with a 'user' role.
            # We inject a synthetic user-model pair to satisfy the API.
            synthetic_user = f"--- SYSTEM: Continuity Memory Restored ---\n{raw_memory}"
            synthetic_model = "🧠 **Memory Restored:** Context acknowledged. Ready to continue."
            
            self.session_manager.add_message("user", synthetic_user)
            self.session_manager.add_message("assistant", synthetic_model)
        else:
            self._continuity_loaded = False

    async def handle_query(self, user_input: str) -> tuple[str, BuddyMode]:
        """Process a user message end-to-end and return (response, mode)."""
        
        # 0. Check for pending mode switch confirmation
        pending_mode = self.session_manager.session.pending_mode_switch
        if pending_mode is not None:
            self.session_manager.session.pending_mode_switch = None
            if _CONFIRM_YES_PATTERN.search(user_input):
                self.session_manager.session.current_mode = pending_mode
                self.session_manager.add_message("user", user_input)
                response = f"Switched to {pending_mode.value} mode."
                self.session_manager.add_message("assistant", response)
                return response, pending_mode
            elif _CONFIRM_NO_PATTERN.search(user_input):
                self.session_manager.add_message("user", user_input)
                response = f"Okay, remaining in {self.session_manager.session.current_mode.value} mode."
                self.session_manager.add_message("assistant", response)
                return response, self.session_manager.session.current_mode
            # If not confirming/rejecting, fall through to process as a regular query

        # Check for explicit mode change commands (e.g. "switch to research mode")
        explicit_mode = detect_explicit_mode_change(user_input)
        if explicit_mode is not None:
            self.session_manager.session.current_mode = explicit_mode
            self.session_manager.add_message("user", user_input)
            response = f"Switched to {explicit_mode.value} mode."
            self.session_manager.add_message("assistant", response)
            return response, explicit_mode

        # Check for fuzzy mode change (e.g. "switch to reaearch")
        fuzzy_match = find_closest_mode(user_input)
        if fuzzy_match is not None:
            candidate_mode, matched_word = fuzzy_match
            if candidate_mode != self.session_manager.session.current_mode:
                self.session_manager.session.pending_mode_switch = candidate_mode
                return f"Did you mean to switch to '{matched_word} mode'?", self.session_manager.session.current_mode
        
        # 1. /web <query> — force casual mode with web search regardless of decision
        cleaned_input = user_input.strip()
        if cleaned_input.lower().startswith("/web "):
            actual_query = cleaned_input[5:].strip()
            if actual_query:
                self.session_manager.add_message("user", actual_query)
                return self._handle_casual(actual_query, needs_web_search=True), self.session_manager.session.current_mode

        # 2. Check for other slash commands
        if cleaned_input.startswith("/"):
            parts = cleaned_input.split(" ", 1)
            command = parts[0].lower()
            arguments = parts[1] if len(parts) > 1 else ""
            return await self.execute_slash_command(command, arguments)

        # 2. Check for direct database listing command
        if any(cmd in cleaned_input.lower() for cmd in ("list topics", "list notes", "saved notes", "saved topics", "show vault", "check db")):
            is_just_notes = any(cmd in cleaned_input.lower() for cmd in ("list notes", "saved notes"))
            
            chunks = await self.sqlite.get_all_chunks()
            if not chunks:
                return "Your vault is currently empty. No notes or topics saved yet.", self.session_manager.session.current_mode
            
            # Group by file_path
            vault_content = {}
            for chunk in chunks:
                fp = chunk.get("file_path", "unknown")
                heading = chunk.get("heading")
                if fp not in vault_content:
                    vault_content[fp] = set()
                if heading and heading != "untitled":
                    # Clean up "part X" if sliding window was used
                    clean_heading = re.sub(r"\s+\(part \d+\)$", "", heading)
                    vault_content[fp].add(clean_heading)
            
            # Boilerplate headings from the session save template — never useful to show
            _TEMPLATE_HEADINGS = {
                "key concepts introduced", "decisions made",
                "ideas to explore later", "facts / references to remember",
            }

            # Format response
            response = "### 🧠 Vault Library\n\n"
            for fp, headings in sorted(vault_content.items()):
                if is_just_notes:
                    response += f"📄 `{fp}`\n"
                    continue
                
                file_path = Path(self.vault_path) / fp

                # Pull ALL H2/H3 headings from the file for richer topic display
                all_headings: list[str] = []
                if file_path.exists():
                    try:
                        # Only parse headers from markdown files
                        if file_path.suffix.lower() == '.md':
                            content = file_path.read_text(encoding="utf-8")
                            for m in re.finditer(r"^#{2,3}\s+(.+)$", content, re.MULTILINE):
                                h = m.group(1).strip()
                                # Skip session template boilerplate & datestamped session headers
                                if h.lower() not in _TEMPLATE_HEADINGS and not h.lower().startswith("session save"):
                                    all_headings.append(h)
                    except UnicodeDecodeError:
                        pass # Silently skip files that aren't valid UTF-8
                    except Exception as e:
                        logger.error("Failed to read file %s: %s", fp, e)

                # Also include chunk-level headings gathered from SQLite (may overlap)
                for h in headings:
                    if h.lower() not in _TEMPLATE_HEADINGS and not h.lower().startswith("session save") and h not in all_headings:
                        all_headings.append(h)

                if all_headings:
                    # Show topics prominently; file path as small annotation
                    response += f"📄 `{fp}`\n"
                    for h in all_headings:
                        response += f"  • **{h}**\n"
                else:
                    # Fallback: just show the file name
                    response += f"📄 `{fp}` *(no named topics)*\n"
                response += "\n"
            return response, self.session_manager.session.current_mode

        # Default to the persistent session mode
        mode = self.session_manager.session.current_mode
        
        # --- Semantic Clustering Routing + Zero-latency Web Search Classifier ---
        query_embedding = None
        if self.embedder:
            try:
                # Fast path: obtain normalized 384-dim embedding from local microservice
                raw_emb = await self.embedder.encode_query(user_input)
                if isinstance(raw_emb, (list, tuple)) and len(raw_emb) == self.semantic_router._dim:
                    query_embedding = list(raw_emb)
            except Exception as e:
                logger.debug("Embedder not available for semantic router: %s", e)

        # Route query against all existing Topic Block centroids (O(K*D) AVX2 SIMD)
        route_payload = self.semantic_router.route(user_input, query_embedding)
        self._current_route_payload = route_payload

        if route_payload.new_topic_created:
            logger.info(
                "Semantic topic shift detected (sim=%.4f < threshold=%.2f). Starting new topic block.",
                route_payload.best_similarity, self.semantic_router.threshold
            )
        elif route_payload.matched_block_id is not None:
            logger.info(
                "Semantic match with Topic Block %d (similarity: %.4f). Safety net: %s",
                route_payload.matched_block_id,
                route_payload.best_similarity,
                f"[Turn {route_payload.safety_net.turn}]" if route_payload.safety_net else "none"
            )

        # Local classifier determines web search intent instantly
        needs_web_search = should_web_search(user_input)

        # 3. Add to session history (updates both session_manager and semantic_router)
        self.session_manager.add_message("user", user_input, query_embedding)

        if mode == BuddyMode.CASUAL:
            response_gen = self._handle_casual(user_input, needs_web_search=needs_web_search)
            return response_gen, mode

        else:
            response_gen = self._handle_retrieval(user_input, mode)
            return response_gen, mode

    async def execute_slash_command(self, command: str, arguments: str) -> tuple[str, BuddyMode]:
        """Execute a slash command by delegating to the modular command registry."""
        from src.commands import registry
        handler = registry.get(command)
        if handler:
            return await handler(self, arguments)
        return f"Command '{command}' is not implemented yet or unknown.", self.session_manager.session.current_mode



    async def _handle_save(self) -> str:
        """Handle saving the current session to the vault.
        
        Writes two files:
          1. session_save_N.md   — LLM-structured summary (good for retrieval)
          2. session_raw_N.md    — verbatim transcript (preserves exact wording/code)
        Both are immediately ingested into SQLite + ChromaDB.
        """
        permission = evaluate_permission(ActionType.SAVE_NEW)
        
        history = self.session_manager.get_history_for_llm()
        msg_count = self.session_manager.session.message_count
        real_history = history  # The command itself is not in history

        # --- Build raw transcript ---
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        raw_lines = [f"## Raw Session Transcript — {now_str}\n\n"]
        for msg in real_history:
            role_label = "**USER**" if msg["role"] == "user" else "**BUDDY**"
            raw_lines.append(f"{role_label}:\n{msg['content']}\n\n---\n\n")
        raw_content = "".join(raw_lines)

        # --- LLM-structured summary ---
        print("\nDrafting structured save note...")
        system_prompt = get_save_prompt()
        structured_content = await self.llm.generate(
            system_prompt=system_prompt,
            user_prompt="Format our entire conversation history above into the Session Save Template.",
            history=real_history,
        )
        clean_content = fill_template(structured_content)

        # --- Preview the structured note before asking confirmation ---
        target_file     = f"session_save_{msg_count}.md"
        raw_file        = f"session_raw_{msg_count}.md"
        target_path     = Path(self.vault_path) / target_file
        raw_path        = Path(self.vault_path) / raw_file

        preview = preview_save(clean_content, target_file)
        print(preview)

        # Run blocking input() off the event loop so the watcher isn't frozen
        approved = await asyncio.to_thread(confirm_action, permission)
        if not approved:
            return "Save cancelled."

        # Prompt for optional rename
        def ask_filename():
            try:
                name = input(f"Enter filename to save as (leave blank for '{target_file}'): ").strip()
                return name
            except EOFError:
                return ""
                
        custom_name = await asyncio.to_thread(ask_filename)
        if custom_name:
            if not custom_name.endswith(".md"):
                custom_name += ".md"
            target_file = custom_name
            raw_file = custom_name.replace(".md", "_raw.md")
            target_path = Path(self.vault_path) / target_file
            raw_path = Path(self.vault_path) / raw_file

        # --- Write both files ---
        target_path.write_text(clean_content, encoding="utf-8")
        raw_path.write_text(raw_content, encoding="utf-8")

        # --- Ingest both immediately (don't rely on watcher) ---
        ingested = []
        for path in (target_path, raw_path):
            try:
                stats = await ingest_file(
                    file_path=path,
                    vault_root=Path(self.vault_path),
                    sqlite=self.sqlite,
                    chroma=self.chroma,
                    embedder=self.embedder,
                    max_words=self._chunking.get("max_words", 300),
                    min_words=self._chunking.get("min_words", 50),
                    overlap_words=self._chunking.get("overlap_words", 50),
                )
                ingested.append(path.name)
                logger.info("Ingested '%s': %s", path.name, stats)
            except Exception as exc:
                logger.error("Failed to index '%s': %s", path.name, exc)

        # Rebuild BM25 index once after both files are in SQLite
        all_chunks = await self.sqlite.get_all_chunks()
        self.lexical_index.build_index(all_chunks)

        return (
            f"✅ Saved and indexed:\n"
            f"  📄 {target_file}  ← structured summary\n"
            f"  📜 {raw_file}  ← verbatim transcript"
        )

    async def _handle_casual(self, user_input: str, needs_web_search: bool = False):
        """Handle a casual chat, optionally augmented with a Tavily web search.

        needs_web_search is decided by Groq's pivot detector. It is True for
        substantive queries (explanations, how-tos, current events, deep dives)
        and False for pure conversational fluff (thanks, follow-ups, greetings).
        /web <query> forces needs_web_search=True regardless of Groq's decision.
        """
        import os
        from src.tools.search_tool import tavily_search
        system_prompt = get_system_prompt(BuddyMode.CASUAL, self._buddy_config_content)
        history = self.session_manager.get_history_for_llm(self._current_route_payload)
        if history and history[-1]["role"] == "user" and history[-1]["content"] == user_input:
            history = history[:-1]

        web_context = ""
        if needs_web_search:
            if os.environ.get("USER_MODE") != "1":
                yield "*(Checking the web...)*\n\n"
            max_results = self.config.get("web_search", {}).get("max_results", 3)
            web_context = await tavily_search(user_input, max_results=max_results)

        if web_context:
            augmented_prompt = f"{web_context}\n\n--- USER QUERY ---\n{user_input}"
        else:
            augmented_prompt = user_input

        async for chunk in self.llm.generate_stream(system_prompt, augmented_prompt, history):
            yield chunk


    async def _handle_retrieval(self, user_input: str, mode: BuddyMode):
        """Handle Study or Brainstorm mode (requires retrieval pipeline)."""
        
        # 0. Query Evaluation (HyDE + Intent Classification)
        class QueryEvaluation(BaseModel):
            intent: str
            search_query: str

        hyde_prompt = (
            "You are an expert retrieval assistant. Evaluate the user's query.\n"
            "1. Output 'complex_synthesis' if the user wants to synthesize, compare, or deeply analyze multiple topics.\n"
            "   Output 'precise' for factual lookups (who, what, where).\n"
            "   Output 'brainstorm' if exploring new ideas.\n"
            "   Output 'exploratory' for general queries.\n"
            "2. Generate a hypothetical detailed answer or a list of highly relevant keywords for the query to improve vector search retrieval.\n\n"
            f"Query: {user_input}"
        )

        try:
            evaluation = await self.llm.classify_structured(
                system_prompt="You are a query expander and evaluator.",
                user_prompt=hyde_prompt,
                response_model=QueryEvaluation
            )
            intent_str = evaluation.intent.lower().strip()
            expanded_query = evaluation.search_query
        except Exception as e:
            logger.error(f"Structured classification failed: {e}")
            intent_str = "exploratory"
            expanded_query = user_input

        search_query = f"{user_input}\n{expanded_query}"
        
        if intent_str == "complex_synthesis":
            async for chunk in self._handle_deep_research(user_input, search_query, mode):
                yield chunk
            return

        from src.orchestrator.intent_classifier import get_intent_params
        intent = get_intent_params(intent_str, self.config)
        
        # 1. Semantic Search
        query_embedding = await self.embedder.encode_query(search_query)
        semantic_results = await semantic_search(
            query_embedding=query_embedding,
            chroma=self.chroma,
            n_results=self.config.get("retrieval", {}).get("top_k", 20),
            distance_threshold=intent.distance_threshold
        )
        
        # 2. Lexical Search
        lexical_results = self.lexical_index.search(
            query=search_query,
            top_k=self.config.get("retrieval", {}).get("top_k", 20)
        )
        
        # 3. Merge & Rerank (this also pulls metadata and recency)
        weights = RerankerWeights(
            semantic=intent.semantic_weight,
            bm25=intent.bm25_weight,
            importance=self.config.get("retrieval", {}).get("weights", {}).get("importance", 0.1),
            recency=self.config.get("retrieval", {}).get("weights", {}).get("recency", 0.1),
        )
        
        ranked_chunks = await rerank(
            semantic_results=semantic_results,
            bm25_results=lexical_results,
            sqlite=self.sqlite,
            top_k=self.config.get("retrieval", {}).get("top_k", 10),
            weights=weights,
            decay_lambda=self.config.get("retrieval", {}).get("recency_decay_lambda", 0.1)
        )
        
        # 4. Graph Expansion
        base_ids = [rc.chunk_id for rc in ranked_chunks]
        expanded_ids = await expand_neighbors(
            base_chunk_ids=base_ids,
            sqlite=self.sqlite,
            max_hops=intent.max_hops,
            min_strength=self.config.get("graph", {}).get("min_strength", 0.6)
        )
        
        # Fetch actual content for expanded neighbors not in original ranked list
        additional_ids = list(expanded_ids - set(base_ids))
        additional_content = []
        if additional_ids:
            # Need a method to get chunks by IDs directly, fast hack:
            for c_id in additional_ids:
                data = await self.sqlite.get_chunk(c_id)
                if data:
                    additional_content.append(data)
                    
        # 5. Assemble Context
        formatted_items = []
        for i, rc in enumerate(ranked_chunks):
            formatted_items.append(f"[{i+1}] {rc.file_path} > {rc.heading}\n{rc.content}\n\n")
        for i, data in enumerate(additional_content):
            formatted_items.append(f"[{len(ranked_chunks)+i+1} (Graph Link)] {data.get('file_path')} > {data.get('heading')}\n{data.get('content')}\n\n")
            
        # Apply outside-in reordering to mitigate 'Lost in the Middle'
        reordered = [None] * len(formatted_items)
        left, right = 0, len(formatted_items) - 1
        for i, item in enumerate(formatted_items):
            if i % 2 == 0:
                reordered[left] = item
                left += 1
            else:
                reordered[right] = item
                right -= 1

        context_str = "--- VAULT CONTEXT ---\n"
        for item in reordered:
            context_str += item
            
        if not ranked_chunks and not additional_content:
            context_str = "No relevant notes found in the vault."
            
            # OPTION B: Fallback Web Search if vault is empty (applies to ALL intents)
            from src.tools.search_tool import tavily_search
            yield "*(No vault results. Checking the web...)*\n\n"
            max_results = self.config.get("web_search", {}).get("max_results", 3)
            web_context = await tavily_search(user_input, max_results=max_results)
            if web_context:
                context_str += f"\n{web_context}\n"
        else:
            # 5b. Optional Web Search (exploratory / brainstorm intents only)
            if intent_str in ("exploratory", "brainstorm"):
                from src.tools.search_tool import tavily_search
                yield "*(Checking the web...)*\n\n"
                max_results = self.config.get("web_search", {}).get("max_results", 3)
                web_context = await tavily_search(user_input, max_results=max_results)
                if web_context:
                    context_str += f"\n{web_context}\n"

        # 6. LLM Call
        system_prompt = get_system_prompt(mode, self._buddy_config_content)
        history = self.session_manager.get_history_for_llm(self._current_route_payload)
        if history and history[-1]["role"] == "user" and history[-1]["content"] == user_input:
            history = history[:-1]
        
        augmented_prompt = f"{context_str}\n\n--- USER QUERY ---\n{user_input}"
        
        # Log telemetry (mocked to stdout for now)
        logger.info("RETRIEVAL LOG: mode=%s, intent=%s, semantic=%d, lexical=%d, expanded=%d", 
                    mode.value, intent.intent_type, len(semantic_results), len(lexical_results), len(additional_ids))
                    
        async for chunk in self.llm.generate_stream(system_prompt, augmented_prompt, history):
            yield chunk

    async def _handle_deep_research(self, user_input: str, search_query: str, mode: BuddyMode):
        """Perform a deep multi-step iterative research loop."""
        from src.orchestrator.intent_classifier import get_intent_params
        from src.tools.search_tool import tavily_search
        intent = get_intent_params("complex_synthesis", self.config)

        # Yield a status update so the user knows it's thinking
        yield "*(Entering Deep Research Mode... Searching internal vault and web...)*\n\n"

        # 1. Semantic Search with expanded query
        query_embedding = await self.embedder.encode_query(search_query)
        semantic_results = await semantic_search(
            query_embedding=query_embedding,
            chroma=self.chroma,
            n_results=self.config.get("retrieval", {}).get("top_k", 20),
            distance_threshold=intent.distance_threshold
        )
        
        # 2. Lexical Search
        lexical_results = self.lexical_index.search(
            query=search_query,
            top_k=self.config.get("retrieval", {}).get("top_k", 20)
        )
        
        # 3. Merge & Rerank
        weights = RerankerWeights(
            semantic=intent.semantic_weight,
            bm25=intent.bm25_weight,
            importance=self.config.get("retrieval", {}).get("weights", {}).get("importance", 0.1),
            recency=self.config.get("retrieval", {}).get("weights", {}).get("recency", 0.1),
        )
        
        ranked_chunks = await rerank(
            semantic_results=semantic_results,
            bm25_results=lexical_results,
            sqlite=self.sqlite,
            top_k=self.config.get("retrieval", {}).get("top_k", 15),
            weights=weights,
            decay_lambda=self.config.get("retrieval", {}).get("recency_decay_lambda", 0.1)
        )
        
        # 4. Web Search (Tavily)
        max_results = self.config.get("web_search", {}).get("max_results", 5)
        web_context = await tavily_search(user_input, max_results=max_results)
        
        # Assemble Context
        formatted_items = [f"[{i+1}] {rc.file_path} > {rc.heading}\n{rc.content}\n\n" for i, rc in enumerate(ranked_chunks)]
        
        # Apply outside-in reordering to mitigate 'Lost in the Middle'
        reordered = [None] * len(formatted_items)
        left, right = 0, len(formatted_items) - 1
        for i, item in enumerate(formatted_items):
            if i % 2 == 0:
                reordered[left] = item
                left += 1
            else:
                reordered[right] = item
                right -= 1

        context_str = "--- INTERNAL VAULT CONTEXT ---\n"
        for item in reordered:
            context_str += item
        
        if web_context:
            context_str += f"\n{web_context}\n"
            
        system_prompt = get_system_prompt(mode, self._buddy_config_content)
        history = self.session_manager.get_history_for_llm()[:-1]
        
        augmented_prompt = (
            f"You are in Deep Research Mode. OVERRIDE STRICT BREVITY RULES. "
            f"Synthesize the following internal vault context and external web search results "
            f"to provide a comprehensive, highly detailed, and exhaustive answer to the user's query. "
            f"If applicable, YOU MUST generate rich code examples to illustrate the concepts, just like a high-quality technical tutorial.\n\n"
            f"{context_str}\n\n"
            f"--- USER QUERY ---\n{user_input}"
        )
        
        logger.info("RETRIEVAL LOG: mode=DEEP_RESEARCH, semantic=%d, lexical=%d, web=%s", 
                    len(semantic_results), len(lexical_results), "yes" if web_context else "no")
        
        async for chunk in self.llm.generate_stream(system_prompt, augmented_prompt, history):
            yield chunk

    async def generate_continuity_summary(self, drafts_path: str | Path) -> None:
        """Distill the current session into a dense continuity memory and persist it.
        
        Called during the shutdown sequence (exit / Ctrl+C). Skips if the session
        had no substantive messages (< 2 user messages).
        """
        history = self.session_manager.get_history_for_llm()
        
        # Don't bother summarising trivially short sessions
        user_msg_count = sum(1 for m in history if m["role"] == "user")
        if user_msg_count < 2:
            logger.info("Session too short for continuity memory (%d user msgs). Skipping.", user_msg_count)
            return

        # Append current mode so the LLM can note it in the summary
        mode_label = self.session_manager.session.current_mode.value
        user_prompt = (
            f"The user's current mode is: {mode_label}\n\n"
            "Distill the conversation above into the continuity summary now."
        )

        try:
            summary = await self.llm.generate(
                system_prompt=CONTINUITY_EXTRACTION_PROMPT,
                user_prompt=user_prompt,
                history=history,
            )
            save_continuity_memory(drafts_path, summary)
        except Exception as e:
            logger.error("Failed to generate continuity summary: %s", e)
