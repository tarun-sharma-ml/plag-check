import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plagiarism_checker import (
    Report,
    ChunkResult,
    chunk_document,
    clean_web_text,
    split_into_passages,
    split_into_sentences,
)


def test_split_into_sentences():
    text = "Hello world. This is a test! Is it working? Yes it is."
    assert split_into_sentences(text) == [
        "Hello world.",
        "This is a test!",
        "Is it working?",
        "Yes it is.",
    ]


def test_split_empty():
    assert split_into_sentences("") == []
    assert chunk_document("") == []


def test_chunk_document_overlap():
    text = "One is here. Two is here. Three is here. Four is here."
    chunks = chunk_document(text, sentences_per_chunk=2, overlap=1)
    assert chunks[0] == "One is here. Two is here."
    assert chunks[1] == "Two is here. Three is here."
    assert chunks[-1].endswith("Four is here.")


def test_chunk_single_sentence():
    assert chunk_document("Just one sentence.", sentences_per_chunk=3) == ["Just one sentence."]


def test_split_into_passages_skips_tiny():
    passages = split_into_passages("Hi. This is a reasonably long sentence for a passage.", 1)
    assert all(len(p) > 20 for p in passages)


def test_clean_web_text_strips_scripts():
    html = "<html><script>alert(1)</script><nav>menu</nav><p>Real content.</p></html>"
    cleaned = clean_web_text(html)
    assert "alert" not in cleaned
    assert "menu" not in cleaned
    assert "Real content." in cleaned


def test_report_overall_score():
    r = Report(total_chunks=4)
    assert r.overall_score == 0.0
    r.flagged_chunks.append(ChunkResult(0, "x", is_flagged=True))
    assert r.overall_score == 25.0
    assert Report().overall_score == 0.0
