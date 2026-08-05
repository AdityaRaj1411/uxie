from src.graph.neighbor_expander import expand_neighbors
from src.commands.registry import registry, safe_command

@registry.register("/link")
@safe_command
async def cmd_link(context, arguments: str):
    parts = arguments.split()
    if len(parts) < 2:
        return "Syntax: /link <source_note_id> <target_note_id> [relationship_type]", context.session_manager.session.current_mode
    src = parts[0]
    tgt = parts[1]
    rel_type = parts[2] if len(parts) > 2 else "related_to"
    await context.sqlite.upsert_link(src, tgt, rel_type, 1.0)
    return f"Linked {src} -> {tgt} ({rel_type})", context.session_manager.session.current_mode

@registry.register("/unlink")
@safe_command
async def cmd_unlink(context, arguments: str):
    parts = arguments.split()
    if len(parts) < 2:
        return "Syntax: /unlink <source_note_id> <target_note_id>", context.session_manager.session.current_mode
    src = parts[0]
    tgt = parts[1]
    await context.sqlite._conn.execute("DELETE FROM links WHERE source_chunk_id=? AND target_chunk_id=?", (src, tgt))
    await context.sqlite._conn.commit()
    return f"Unlinked {src} -> {tgt}", context.session_manager.session.current_mode

@registry.register("/orbit")
@safe_command
async def cmd_orbit(context, arguments: str):
    parts = arguments.split("--depth")
    note_name = parts[0].strip()
    if not note_name:
        return "Syntax: /orbit <note_name> [--depth 1|2]", context.session_manager.session.current_mode
    depth = 1
    if len(parts) > 1:
        try:
            depth = int(parts[1].strip())
        except ValueError:
            depth = 1
            
    chunks = await context.sqlite.get_chunks_by_file(note_name)
    if not chunks:
        return f"No chunks found for {note_name}.", context.session_manager.session.current_mode
        
    base_ids = [c["chunk_id"] for c in chunks]
    expanded_ids = await expand_neighbors(base_ids, context.sqlite, max_hops=depth, min_strength=0.1)
    
    out = f"Orbit for {note_name} (depth {depth}):\n"
    for eid in expanded_ids:
        if eid not in base_ids:
            chunk_data = await context.sqlite.get_chunk(eid)
            if chunk_data:
                out += f" - {chunk_data.get('file_path')} > {chunk_data.get('heading')} (ID: {eid[:8]})\n"
    if not expanded_ids or all(eid in base_ids for eid in expanded_ids):
        out += "No linked neighbors found."
    return out, context.session_manager.session.current_mode
