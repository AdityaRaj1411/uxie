from datetime import datetime
from pathlib import Path
from src.retrieval.semantic import semantic_search
from src.retrieval.reranker import rerank
from src.commands.registry import registry, safe_command

@registry.register("/trace")
@safe_command
async def cmd_trace(context, arguments: str):
    if not arguments:
        return "Syntax: /trace <query>", context.session_manager.session.current_mode
    query = arguments
    query_embedding = await context.embedder.encode_query(query)
        
    semantic_results = await semantic_search(
        query_embedding=query_embedding, chroma=context.chroma, n_results=10, distance_threshold=1.0
    )
    lexical_results = context.lexical_index.search(query=query, top_k=10)
    ranked_chunks = await rerank(
        semantic_results=semantic_results,
        bm25_results=lexical_results,
        sqlite=context.sqlite,
        top_k=10
    )
    out = "+------------------+----------------+---------------+---------------+------------------+-------------+\n"
    out += "| Chunk ID         | Semantic (50%) | Lexical (30%) | Recency (10%) | Importance (10%) | Total Score |\n"
    out += "+------------------+----------------+---------------+---------------+------------------+-------------+\n"
    for rc in ranked_chunks:
        cid = rc.chunk_id[:16].ljust(16)
        sem = f"{rc.semantic_score:.3f}".ljust(14)
        lex = f"{rc.bm25_score:.3f}".ljust(13)
        rec = f"{rc.recency_score:.3f}".ljust(13)
        imp = f"{rc.importance_score:.3f}".ljust(16)
        tot = f"{rc.final_score:.3f}".ljust(11)
        out += f"| {cid} | {sem} | {lex} | {rec} | {imp} | {tot} |\n"
    out += "+------------------+----------------+---------------+---------------+------------------+-------------+"
    return out, context.session_manager.session.current_mode

@registry.register("/stitch")
@safe_command
async def cmd_stitch(context, arguments: str):
    topic = arguments.strip()
    if not topic:
        return "Syntax: /stitch <topic>", context.session_manager.session.current_mode
    
    query_embedding = await context.embedder.encode_query(topic)
        
    semantic_results = await semantic_search(query_embedding=query_embedding, chroma=context.chroma, n_results=15, distance_threshold=0.7)
    lexical_results = context.lexical_index.search(query=topic, top_k=15)
    ranked_chunks = await rerank(semantic_results=semantic_results, bm25_results=lexical_results, sqlite=context.sqlite, top_k=10)
    
    context_str = "--- CONTEXT ---\n"
    for rc in ranked_chunks:
        context_str += f"File: {rc.file_path} (Heading: {rc.heading})\n{rc.content}\n\n"
        
    system_prompt = "You are a synthesizer. Synthesize the provided context into a single structured markdown reference page about the requested topic."
    user_prompt = f"Topic: {topic}\n\n{context_str}"
    
    stitched_content = await context.llm.generate(system_prompt, user_prompt, [])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"stitched_{timestamp}.md"
    target_path = Path(context.vault_path) / file_name
    target_path.write_text(stitched_content, encoding="utf-8")
    
    return f"Stitched note created: {file_name}", context.session_manager.session.current_mode

@registry.register("/compact")
@safe_command
async def cmd_compact(context, arguments: str):
    note_name = arguments.strip()
    if not note_name:
        return "Syntax: /compact <note_name>", context.session_manager.session.current_mode
    
    target_path = Path(context.vault_path) / note_name
    if not target_path.exists() or not target_path.is_file():
        return f"File '{note_name}' not found or is a directory.", context.session_manager.session.current_mode
        
    original_content = target_path.read_text(encoding="utf-8")
    system_prompt = "You are a markdown formatter. Clean up the provided text: fix formatting anomalies, enforce clean hierarchical markdown (H1-H3), and improve structure. Do NOT change the core meaning or drop information."
    user_prompt = f"Please compact and format this note:\n\n{original_content}"
    
    compacted = await context.llm.generate(system_prompt, user_prompt, [])
    target_path.write_text(compacted, encoding="utf-8")
    return f"Compacted {note_name}. File overwritten.", context.session_manager.session.current_mode

@registry.register("/absorb")
@safe_command
async def cmd_absorb(context, arguments: str):
    # Strip whitespace and any surrounding quotes (double or single)
    file_path = arguments.strip().strip('"').strip("'")
    if not file_path:
        return "Syntax: /absorb <file_path>", context.session_manager.session.current_mode
        
    source_path = Path(file_path)
    if not source_path.exists() or not source_path.is_file():
        return f"Cannot absorb: '{file_path}' not found or is a directory.", context.session_manager.session.current_mode
        
    import shutil

    if source_path.suffix.lower() == ".pdf":
        file_name = source_path.name
        target_path = Path(context.vault_path) / file_name
        counter = 1
        while target_path.exists():
            target_path = Path(context.vault_path) / f"{source_path.stem}_{counter}.pdf"
            counter += 1
        shutil.copy2(source_path, target_path)
        return f"Absorbed {source_path.name} into vault. It will be ingested shortly.", context.session_manager.session.current_mode

    try:
        content = source_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Cannot absorb: '{file_path}' is not a valid UTF-8 text file and is not a supported binary type.", context.session_manager.session.current_mode
    
    if not source_path.name.endswith(".md"):
        file_name = source_path.stem + ".md"
        content = f"# Absorbed: {source_path.name}\n\n```text\n{content}\n```"
    else:
        file_name = source_path.name
        
    target_path = Path(context.vault_path) / file_name
    
    counter = 1
    while target_path.exists():
        target_path = Path(context.vault_path) / f"{source_path.stem}_{counter}.md"
        counter += 1
        
    target_path.write_text(content, encoding="utf-8")
    return f"Absorbed {source_path.name} into {target_path.name}.", context.session_manager.session.current_mode
