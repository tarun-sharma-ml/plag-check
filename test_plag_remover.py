import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plag_remover import paraphrase_text, paraphrase_document, load_flagged_chunks, _match_case


def test_rate_zero_leaves_text_unchanged():
    text = "Photosynthesis is the process by which plants convert sunlight into energy."
    assert paraphrase_text(text, rate=0.0, seed=1) == text


def test_rate_one_can_change_text():
    text = "The enormous elephant walked slowly through the dense jungle today."
    out = paraphrase_text(text, rate=1.0, seed=7)
    assert isinstance(out, str) and len(out.split()) == len(text.split())


def test_stopwords_and_short_words_untouched():
    text = "the a it is of to in on"
    assert paraphrase_text(text, rate=1.0, seed=3) == text


def test_match_case_preserves_capitalization():
    assert _match_case("Happy", "glad") == "Glad"
    assert _match_case("HAPPY", "glad") == "GLAD"
    assert _match_case("happy", "glad") == "glad"


def test_paraphrase_document_same_sentence_count():
    text = "First sentence here. Second sentence follows. Third one too."
    out = paraphrase_document(text, rate=0.3, seed=5)
    assert out.count(".") == text.count(".")


def test_load_flagged_chunks(tmp_path):
    report = {"flagged_chunks": [{"chunk_text": "A flagged passage."}, {"chunk_text": "Another one."}]}
    f = tmp_path / "report.json"
    f.write_text(json.dumps(report))
    chunks = load_flagged_chunks(str(f))
    assert chunks == ["A flagged passage.", "Another one."]


def test_load_flagged_chunks_empty(tmp_path):
    f = tmp_path / "empty.json"
    f.write_text(json.dumps({"flagged_chunks": []}))
    assert load_flagged_chunks(str(f)) == []