"""
Markdown Chunker — splits .md files into semantic chunks.

Strategy:
  1. Split on H1–H3 headings (the primary semantic boundary)
  2. If any section exceeds max_words, apply paragraph-fallback sliding window
  3. Maintain ~overlap_words overlap between adjacent chunks
  4. Each chunk is an independent idea unit with its own hash for change detection
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False
    logger.warning("PyMuPDF (fitz) is not installed. PDF ingestion will fail. Run 'pip install PyMuPDF'.")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    """A single semantic chunk extracted from a markdown file."""
    chunk_id: str           # SHA256-based unique identifier (16 hex chars)
    file_path: str          # Relative path from vault root
    heading: str            # Parent heading text (or "untitled")
    content: str            # Raw text content of the chunk
    content_hash: str       # SHA256 of content only (for change detection)
    tags: list[str] = field(default_factory=list)  # Extracted #tags


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------
def _sha256(text: str) -> str:
    """Return the full SHA256 hex digest of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_chunk_id(file_path: str, heading: str, content: str) -> str:
    """
    Generate a deterministic chunk ID from file path + heading + content.
    Truncated to 16 hex characters for readability.
    """
    raw = f"{file_path}::{heading}::{content}"
    return _sha256(raw)[:16]


# ---------------------------------------------------------------------------
# Tag extraction
# ---------------------------------------------------------------------------
_TAG_PATTERN = re.compile(r"(?<!\S)#([a-zA-Z_]\w{0,49})(?!\S)")


def _extract_tags(text: str) -> list[str]:
    """Extract inline #tags from text. Ignores markdown headings."""
    return list(set(_TAG_PATTERN.findall(text)))


# ---------------------------------------------------------------------------
# Heading-based splitting
# ---------------------------------------------------------------------------
_HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def _split_on_headings(text: str) -> list[tuple[str, str]]:
    """
    Split markdown text on H1–H3 headings.

    Returns list of (heading_text, section_body) tuples.
    Content before the first heading gets heading "untitled".
    """
    sections: list[tuple[str, str]] = []
    matches = list(_HEADING_PATTERN.finditer(text))

    if not matches:
        # No headings at all — treat entire content as one section
        stripped = text.strip()
        if stripped:
            sections.append(("untitled", stripped))
        return sections

    # Content before first heading
    pre_content = text[: matches[0].start()].strip()
    if pre_content:
        sections.append(("untitled", pre_content))

    # Each heading and its body
    for i, match in enumerate(matches):
        heading_text = match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body or heading_text:  # Keep even empty sections if heading exists
            sections.append((heading_text, body))

    return sections


# ---------------------------------------------------------------------------
# Paragraph-fallback sliding window
# ---------------------------------------------------------------------------
def _word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def _sliding_window_split(
    text: str,
    max_words: int = 800,
    overlap_words: int = 50,
) -> list[str]:
    """
    Split long text into overlapping chunks using paragraph boundaries.

    Prefers splitting at paragraph boundaries (double newline). Falls back
    to word-level splitting if a single paragraph exceeds max_words.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current_paragraphs: list[str] = []
    current_word_count = 0

    for para in paragraphs:
        para_wc = _word_count(para)

        # Single paragraph exceeds max — force split at word level
        if para_wc > max_words:
            # Flush current buffer first
            if current_paragraphs:
                chunks.append("\n\n".join(current_paragraphs))
                current_paragraphs = []
                current_word_count = 0

            # Word-level split of oversized paragraph
            words = para.split()
            start = 0
            while start < len(words):
                chunk_words = words[start : start + max_words]
                chunks.append(" ".join(chunk_words))
                if start + max_words >= len(words):
                    break
                start += (max_words - overlap_words)
            continue

        # Would this paragraph push us over the limit?
        if current_word_count + para_wc > max_words and current_paragraphs:
            chunks.append("\n\n".join(current_paragraphs))

            # Overlap: carry the last paragraph(s) forward
            overlap_paras: list[str] = []
            overlap_wc = 0
            for prev_para in reversed(current_paragraphs):
                prev_wc = _word_count(prev_para)
                if overlap_wc + prev_wc > overlap_words:
                    break
                overlap_paras.insert(0, prev_para)
                overlap_wc += prev_wc

            current_paragraphs = overlap_paras
            current_word_count = overlap_wc

        current_paragraphs.append(para)
        current_word_count += para_wc

    # Flush remaining
    if current_paragraphs:
        chunks.append("\n\n".join(current_paragraphs))

    return chunks


# ---------------------------------------------------------------------------
# Main entry point and Routers
# ---------------------------------------------------------------------------
def parse_file(
    file_path: Path | str,
    vault_root: Path | str,
    max_words: int = 800,
    min_words: int = 50,
    overlap_words: int = 50,
) -> list[Chunk]:
    """Route the file to the correct parser based on extension."""
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext == ".md":
        return parse_markdown_file(file_path, vault_root, max_words, min_words, overlap_words)
    elif ext == ".pdf":
        return parse_pdf_file(file_path, vault_root, max_words, min_words, overlap_words)
    elif ext in (".txt", ".json", ".csv"): # general text fallback
        return parse_text_file(file_path, vault_root, max_words, min_words, overlap_words)
    else:
        logger.warning(f"Unsupported file extension '{ext}' for file {file_path.name}")
        return []

def parse_markdown_file(
    file_path: Path | str,
    vault_root: Path | str,
    max_words: int = 800,
    min_words: int = 50,
    overlap_words: int = 50,
) -> list[Chunk]:
    """
    Parse a markdown file into semantic chunks.

    Steps:
      1. Read file content
      2. Split on H1–H3 headings
      3. Apply sliding-window fallback for oversized sections
      4. Generate chunk IDs and content hashes
      5. Extract inline #tags

    Args:
        file_path: Absolute or relative path to the .md file.
        vault_root: Root directory of the vault (for computing relative paths).
        max_words: Maximum words per chunk before sliding-window split.
        min_words: Minimum words — very short chunks get merged with neighbors.
        overlap_words: Word overlap between adjacent sliding-window chunks.

    Returns:
        List of Chunk objects, each representing one idea unit.
    """
    file_path = Path(file_path)
    vault_root = Path(vault_root)

    # Read file
    content = file_path.read_text(encoding="utf-8")
    if not content.strip():
        return []

    # Compute relative path for storage
    try:
        rel_path = file_path.relative_to(vault_root).as_posix()
    except ValueError:
        rel_path = file_path.name

    # Split on headings
    sections = _split_on_headings(content)

    # Process each section
    chunks: list[Chunk] = []
    for heading, body in sections:
        if not body.strip():
            continue

        wc = _word_count(body)

        if wc > max_words:
            # Oversized section → sliding window split
            sub_texts = _sliding_window_split(body, max_words, overlap_words)
            for idx, sub_text in enumerate(sub_texts):
                sub_heading = f"{heading} (part {idx + 1})" if len(sub_texts) > 1 else heading
                chunk = _make_chunk(rel_path, sub_heading, sub_text)
                chunks.append(chunk)
        else:
            chunk = _make_chunk(rel_path, heading, body)
            chunks.append(chunk)

    # Merge very short chunks with their neighbor
    chunks = _merge_short_chunks(chunks, min_words)

    return chunks


def parse_pdf_file(
    file_path: Path | str,
    vault_root: Path | str,
    max_words: int = 800,
    min_words: int = 50,
    overlap_words: int = 50,
) -> list[Chunk]:
    """Parse a PDF file using PyMuPDF, chunking page by page."""
    if not _HAS_FITZ:
        logger.error(f"Cannot parse PDF {file_path}. PyMuPDF is not installed.")
        return []

    file_path = Path(file_path)
    vault_root = Path(vault_root)
    
    try:
        rel_path = file_path.relative_to(vault_root).as_posix()
    except ValueError:
        rel_path = file_path.name

    chunks: list[Chunk] = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if not text.strip():
                continue
            
            heading = f"Page {page_num + 1}"
            wc = _word_count(text)
            
            if wc > max_words:
                sub_texts = _sliding_window_split(text, max_words, overlap_words)
                for idx, sub_text in enumerate(sub_texts):
                    sub_heading = f"{heading} (part {idx + 1})" if len(sub_texts) > 1 else heading
                    chunk = _make_chunk(rel_path, sub_heading, sub_text)
                    chunks.append(chunk)
            else:
                chunk = _make_chunk(rel_path, heading, text)
                chunks.append(chunk)
                
        doc.close()
    except Exception as e:
        logger.error(f"Failed to extract text from PDF {file_path}: {e}")
        return []

    return _merge_short_chunks(chunks, min_words)


def parse_text_file(
    file_path: Path | str,
    vault_root: Path | str,
    max_words: int = 800,
    min_words: int = 50,
    overlap_words: int = 50,
) -> list[Chunk]:
    """Parse a generic text file, splitting purely by sliding window."""
    file_path = Path(file_path)
    vault_root = Path(vault_root)

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.error(f"Failed to read {file_path} as UTF-8 text.")
        return []

    if not content.strip():
        return []

    try:
        rel_path = file_path.relative_to(vault_root).as_posix()
    except ValueError:
        rel_path = file_path.name

    chunks: list[Chunk] = []
    heading = "untitled"
    wc = _word_count(content)

    if wc > max_words:
        sub_texts = _sliding_window_split(content, max_words, overlap_words)
        for idx, sub_text in enumerate(sub_texts):
            sub_heading = f"{heading} (part {idx + 1})" if len(sub_texts) > 1 else heading
            chunk = _make_chunk(rel_path, sub_heading, sub_text)
            chunks.append(chunk)
    else:
        chunk = _make_chunk(rel_path, heading, content)
        chunks.append(chunk)

    return _merge_short_chunks(chunks, min_words)



def _make_chunk(rel_path: str, heading: str, content: str) -> Chunk:
    """Create a Chunk from raw components."""
    return Chunk(
        chunk_id=_make_chunk_id(rel_path, heading, content),
        file_path=rel_path,
        heading=heading,
        content=content,
        content_hash=_sha256(content),
        tags=_extract_tags(content),
    )


def _merge_short_chunks(chunks: list[Chunk], min_words: int) -> list[Chunk]:
    """
    Merge chunks that are too short (< min_words) with their next neighbor.
    This prevents tiny fragments from polluting the vector space.
    """
    if len(chunks) <= 1:
        return chunks

    merged: list[Chunk] = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        # If current chunk is very short and there's a next chunk to merge with
        if _word_count(current.content) < min_words and i + 1 < len(chunks):
            next_chunk = chunks[i + 1]
            combined_content = current.content + "\n\n" + next_chunk.content
            combined_heading = current.heading if current.heading != "untitled" else next_chunk.heading
            combined_tags = list(set(current.tags + next_chunk.tags))
            merged_chunk = Chunk(
                chunk_id=_make_chunk_id(current.file_path, combined_heading, combined_content),
                file_path=current.file_path,
                heading=combined_heading,
                content=combined_content,
                content_hash=_sha256(combined_content),
                tags=combined_tags,
            )
            merged.append(merged_chunk)
            i += 2  # Skip the next chunk since we merged it
        else:
            merged.append(current)
            i += 1

    return merged
