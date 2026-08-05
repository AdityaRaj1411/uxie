import pytest
from src.chunker.chunker import _split_on_headings, parse_markdown_file, _sliding_window_split

def test_heading_split():
    md = "# H1\ncontent 1\n## H2\ncontent 2"
    sections = _split_on_headings(md)
    assert len(sections) == 2
    assert sections[0][0] == "H1"
    assert sections[0][1] == "content 1"
    assert sections[1][0] == "H2"
    assert sections[1][1] == "content 2"

def test_sliding_window():
    # Generate 100 words
    words = "word " * 100
    chunks = _sliding_window_split(words, max_words=60, overlap_words=20)
    # Should split into two chunks because 100 > 60.
    assert len(chunks) == 2
    assert len(chunks[0].split()) == 60
    assert len(chunks[1].split()) == 60 # Due to 20 overlap (60 - 20) + 60 -> overlaps to cover 100. Actually: start at 0 (len 60), start at 40 (len 60). Covers up to 100.
