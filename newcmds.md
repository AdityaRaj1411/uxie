# Personal Second Brain: CLI Slash Commands Comprehensive Specification

This document details the complete agentic slash-command subsystem specification for the Personal Second Brain CLI. These commands intercept input at the user-interface layer, bypassing regular conversational LLM processing to execute deterministic script routines, database mutations, or specialized multi-step agent actions.

---

## 🛠️ 1. Architecture & Global Router

All commands are intercepted immediately upon input collection within the primary orchestration loop (`src/orchestrator/orchestrator.py`). This guarantees that conversational state remains unpolluted by functional commands.

### Intercept Implementation
```python
async def handle_cli_input(user_input: str) -> bool:
    """
    Parses and routes user input. 
    Returns True if execution continues, False if exiting the application.
    """
    cleaned_input = user_input.strip()
    if not cleaned_input:
        return True

    # Intercept Slash Commands
    if cleaned_input.startswith("/"):
        parts = cleaned_input.split(" ", 1)
        command = parts[0].lower()
        arguments = parts[1] if len(parts) > 1 else ""
        
        await execute_slash_command(command, arguments)
        return True

    # Fallback to normal Conversational / RAG Pipeline
    return await process_conversation(cleaned_input)
📋 2. Command Index MatrixCommandCategoryPrimary Target Layers/modeSystem & SessionOrchestrator State/statusSystem & SessionSystem Environment & Paths/helpSystem & SessionStandard Output/clearSystem & SessionCLI Shell Context/linkGraph & Link ManipulationSQLite (Edges)/unlinkGraph & Link ManipulationSQLite (Edges)/traceGraph & Link ManipulationRetrieval Engine Debugger/orbitGraph & Link ManipulationGraph Traversal (Neighbor Expander)/stitchKnowledge SynthesisMulti-Armed Retrieval + Gen-AI Layer/compactKnowledge SynthesisMarkdown Chunker + LLM Refactor/absorbKnowledge SynthesisFile System + Chunk Ingestion Pipeline/lintVault MaintenanceFile System Integrity + SQLite/reindexVault Maintenancerank_bm25 Corpus + ChromaDB Sync/pruneVault MaintenanceSQLite Graph Cleanup + LLM Validation⚙️ 3. Detailed Command SpecificationsSystem & Session Management/modeSyntax: /mode [casual | study | brainstorm | save]Description: Forces an explicit state change in the orchestrator's behavioral mode. If no argument is passed, it displays the current active mode.Impact: Modifies retrieval score thresholds, graph traversal hop limits, and prompt templates dynamically./statusSyntax: /statusDescription: Outputs diagnostic metrics regarding system health.Output Metrics:Total unique Markdown files in ./vault.Total vector chunks stored in ChromaDB.Total conceptual semantic edges tracked in SQLite.Connectivity state of the local FastAPI embedding microservice./helpSyntax: /helpDescription: Renders a clean Markdown table mapping all available functional commands, their usage syntax, and their high-level intent directly to the console./clearSyntax: /clearDescription: Clears the active console window buffer and flushes the temporary in-memory chat session history (without modifying saved files).🕸️ Graph & Link Manipulation/linkSyntax: /link <source_note_id> <target_note_id> [relationship_type]Description: Programs a hard structural edge directly into the SQLite relationship tables between two target note chunks.Default Relationship Type: related_to (if not specified).Impact: Bypasses automated LLM classification to guarantee explicit retrieval paths during multi-hop expansions./unlinkSyntax: /unlink <source_note_id> <target_note_id>Description: Purges an existing edge definition linking two nodes inside the SQLite database.Impact: Instantly breaks topological connections without deleting raw text assets or vector data./traceSyntax: /trace <query_string>Description: Debugging command that dry-runs the multi-armed retrieval framework against a sample query. It ignores LLM synthesis and explicitly prints the numerical calculations behind score generation.Output Format Table:Plaintext  +----------+----------------+---------------+---------------+------------------+-------------+
  | Chunk ID | Semantic (50%) | Lexical (30%) | Recency (10%) | Importance (10%) | Total Score |
  +----------+----------------+---------------+---------------+------------------+-------------+
  | note_04a | 0.850          | 0.910         | 0.200         | 0.700            | 0.788       |
  | core_02c | 0.920          | 0.000         | 0.950         | 0.900            | 0.645       |
  +----------+----------------+---------------+---------------+------------------+-------------+
/orbitSyntax: /orbit <note_name> [--depth 1|2]Description: Discovers and displays a node's topological vicinity up to a specified hop limit.Impact: Generates a text-based hierarchy detailing every linked note, its connection metadata, and edge weights. Used to verify what context a "Brainstorm" mode sweep will pull into the prompt window.✍️ Knowledge Synthesis & Compaction/stitchSyntax: /stitch <"conceptual topic prompt">Description: Coordinates a wide-set multi-armed query. It harvests the top relevant content across separate documents, passes the assembled corpus to the generative model, and asks it to synthesize a new, single structured markdown reference page.Impact: Writes the newly consolidated output file directly to ./vault/stitched_[timestamp].md, triggering standard file ingestion automatically./compactSyntax: /compact <note_name>Description: Targets highly fractured notes or notes generated via informal text-dumps. The system pulls the target text chunks, invokes the LLM to rewrite and clean up structural layouts, fixes formatting anomalies, and enforces clean hierarchical markdown parsing.Impact: Replaces the existing file content on disk, calculating fresh SHA256 hashes for the newly unified chunks./absorbSyntax: /absorb <absolute_or_relative_file_path>Description: Command to ingest foreign files (e.g., Python scripts, text dumps, server output logs) outside the workspace folder.Impact: Copies the target file, strips unneeded syntax or boilerplate, pushes the content blocks through the system's markdown chunker, calls the embedding microservice, and maps it directly into ChromaDB and SQLite metadata storage.
🛡️ Vault Maintenance & Quality Control/lintSyntax: /lintDescription: Evaluates the structural health of your complete note directory.Scanned Anomalies:Broken relative Markdown wikilinks ([[Missing Note Link]]).Completely empty .md files.Orphaned chunks (notes tracking 0 semantic vector matches and 0 SQLite edges).Over-limit chunks (file blocks containing individual word counts crashing normal window bounds)./reindexSyntax: /reindex [--force]Description: Forces a complete rebuild of the systems indexing layers.Impact: Resolves runtime state desynchronization. It explicitly forces the rank_bm25 module to construct a new inverted document frequency index from the current database snapshot, and cleans stale reference pointers out of ChromaDB collections./pruneSyntax: /prune <note_name> [--threshold 0.3]Description: Evaluates edge decay. It analyzes the relationship paths attached to a specified note and severs graph connections whose strength scores fall below the target parameters.Impact: Keeps high-dimensional retrieval paths clean by deleting low-confidence automated associations from SQLite.