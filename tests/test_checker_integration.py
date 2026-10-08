import plagiarism_checker


class FakeModel:
    def encode(self, value, convert_to_tensor):
        return value


class FakeScoreRow:
    def __init__(self, score):
        self.score = score

    def argmax(self):
        return 0

    def __getitem__(self, _index):
        return self.score


class FakeScores:
    def __init__(self, score):
        self.score = score

    def __getitem__(self, _index):
        return FakeScoreRow(self.score)


class FakeSimilarity:
    def __init__(self, score):
        self.score = score

    def cos_sim(self, _chunk_embedding, _passage_embeddings):
        return FakeScores(self.score)


def setup_checker_mocks(monkeypatch, score):
    monkeypatch.setattr(plagiarism_checker, "_lazy_imports", lambda: None)
    monkeypatch.setattr(
        plagiarism_checker,
        "SentenceTransformer",
        lambda _model_name: FakeModel(),
        raising=False,
    )
    monkeypatch.setattr(
        plagiarism_checker,
        "util",
        FakeSimilarity(score),
        raising=False,
    )
    monkeypatch.setattr(plagiarism_checker.time, "sleep", lambda _delay: None)


def test_check_document_flags_matching_web_page(monkeypatch):
    setup_checker_mocks(monkeypatch, 0.91)
    search_queries = []
    page_fetches = []

    def fake_search(query, max_results):
        search_queries.append((query, max_results))
        return [{"href": "https://example.com/source", "body": "Search snippet."}]

    def fake_fetch(url, cache):
        page_fetches.append((url, cache))
        return "This is a sufficiently long source passage with matching content."

    monkeypatch.setattr(plagiarism_checker, "search_web", fake_search)
    monkeypatch.setattr(plagiarism_checker, "fetch_page_text", fake_fetch)

    report = plagiarism_checker.check_document(
        "This original passage contains enough words for a complete comparison.",
        threshold=0.8,
        results_per_chunk=2,
        min_words=1,
        delay=0,
        verbose=False,
    )

    assert len(search_queries) == 1
    assert search_queries[0][0].startswith('"')
    assert search_queries[0][1] == 2
    assert page_fetches[0][0] == "https://example.com/source"
    assert report.total_chunks == 1
    assert report.coverage == 100.0
    assert report.overall_score == 100.0
    assert report.flagged_chunks[0].best_score == 0.91


def test_check_document_uses_snippet_when_page_is_unavailable(monkeypatch):
    setup_checker_mocks(monkeypatch, 0.84)
    monkeypatch.setattr(
        plagiarism_checker,
        "search_web",
        lambda *_args, **_kwargs: [
            {
                "href": "https://example.com/blocked",
                "body": "A long enough search snippet can still be compared.",
            }
        ],
    )
    monkeypatch.setattr(
        plagiarism_checker,
        "fetch_page_text",
        lambda *_args, **_kwargs: None,
    )

    report = plagiarism_checker.check_document(
        "This original passage contains enough words for a complete comparison.",
        threshold=0.8,
        min_words=1,
        delay=0,
        verbose=False,
    )

    assert report.flagged_chunks[0].best_source_snippet.startswith("A long enough")


def test_check_document_records_chunks_without_comparable_sources(monkeypatch):
    setup_checker_mocks(monkeypatch, 0.0)
    queries = []
    monkeypatch.setattr(
        plagiarism_checker,
        "search_web",
        lambda query, **_kwargs: queries.append(query) or [],
    )

    report = plagiarism_checker.check_document(
        "This original passage contains enough words for a complete comparison.",
        min_words=1,
        delay=0,
        verbose=False,
    )

    assert len(queries) == 2
    assert report.total_chunks == 1
    assert report.no_result_chunks == 1
    assert report.coverage == 0.0
    assert report.all_chunks[0].sources_checked == 0


def test_fetch_page_text_cleans_and_caches_response(monkeypatch):
    class FakeResponse:
        headers = {"Content-Type": "text/html; charset=utf-8"}
        text = (
            "<html><script>hidden()</script><p>Useful page content.</p></html>"
        )

        def raise_for_status(self):
            return None

    calls = []
    monkeypatch.setattr(
        plagiarism_checker.requests,
        "get",
        lambda url, **_kwargs: calls.append(url) or FakeResponse(),
    )
    cache = {}

    first = plagiarism_checker.fetch_page_text("https://example.com", cache=cache)
    second = plagiarism_checker.fetch_page_text("https://example.com", cache=cache)

    assert first == "Useful page content."
    assert second == first
    assert calls == ["https://example.com"]
