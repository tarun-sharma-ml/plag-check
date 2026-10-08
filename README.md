# Plagiarism Checker

A Python tool that checks a document (`.txt`, `.docx` or `.pdf`) against **web sources** using **semantic similarity** (AI sentence embeddings) instead of plain string matching, so it can catch paraphrased text as well as copy-paste.

## How it works

1. Splits the document into overlapping chunks of a few sentences.
2. Searches the web (DuckDuckGo via `ddgs`, no API key) for each chunk, using an exact-phrase query first.
3. Downloads the result pages and extracts clean text (if a site blocks the download, the search snippet is compared instead).
4. Embeds your chunk and the page passages with a local model (`sentence-transformers`, default `all-MiniLM-L6-v2`) and compares them using cosine similarity.
5. Flags chunks above the threshold and reports the best-matching source URL.

No paid API keys are required. The embedding model runs locally (a one-time ~90 MB download).

## Installation

```bash
git clone https://github.com/<your-username>/plagiarism-checker.git
cd plagiarism-checker
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python plagiarism_checker.py examples/sample.txt
python plagiarism_checker.py "C:\path\to\my essay.docx"
python plagiarism_checker.py paper.pdf
```

Supported inputs: `.txt`, `.docx`, `.pdf` (text-based PDFs; scanned images need OCR first). Wrap paths with spaces in quotes.

| Option | Default | Description |
| --- | --- | --- |
| `--threshold` | `0.80` | Similarity (0-1) above which a chunk is flagged |
| `--results-per-chunk` | `5` | Web results checked per chunk |
| `--sentences-per-chunk` | `3` | Sentences grouped into one chunk |
| `--min-words` | `8` | Skip chunks shorter than this |
| `--delay` | `1.0` | Seconds between chunks (raise to avoid rate limits) |
| `--keep-references` | off | Also check the References section (skipped by default) |
| `--model` | `all-MiniLM-L6-v2` | Any `sentence-transformers` model name |

Example:

```bash
python plagiarism_checker.py essay.txt --threshold 0.75 --results-per-chunk 3
```

## Running tests

```bash
pip install -r requirements-dev.txt   # note the -r
python -m ruff check --select E4,E7,E9,F plagiarism_checker.py plag_remover.py tests test_plag_remover.py
pytest
```

## Plag remover (paraphraser)

`plag_remover.py` rewrites text by swapping words for WordNet synonyms, to nudge flagged passages away from a near-exact match. It runs fully offline after a one-time WordNet data download (needs internet the first time only).

```bash
# Rewrite an entire document
python plag_remover.py my_document.txt -o rewritten.txt

# Rewrite only the passages the checker flagged
python plagiarism_checker.py my_document.txt --json report.json
python plag_remover.py my_document.txt --report report.json -o rewritten.txt

# Control how many words get swapped (0.0-1.0, default 0.4)
python plag_remover.py my_document.txt --rate 0.6 -o rewritten.txt
```

**This is a lightweight, dictionary-based paraphraser, not a generative rewrite.** It will sometimes pick an awkward synonym (e.g. "oxygen" → "O", "plants" → "flora"), because WordNet doesn't understand context. Always read and edit the output yourself — a good rule of thumb: you should be able to explain every sentence in your own words after reading it. It's meant to help you rephrase your own ideas, not to disguise copied work; if a passage is flagged because it states someone else's finding or data, the fix is a citation, not a rewording.

### Better quality: Gemini engine

For a real, context-aware rewrite (rather than word-swapping), use Google's Gemini API instead. It has a genuine free tier — no credit card needed:

1. Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. Copy `.env.example` to `.env` and paste your key in:
   ```bash
   cp .env.example .env
   # then edit .env and set GEMINI_API_KEY=your-actual-key
   ```
   (`.env` is already in `.gitignore`, so your key is never committed.)
3. Run with `--engine gemini`:
   ```bash
   python plag_remover.py my_document.txt --engine gemini -o rewritten.txt
   python plag_remover.py my_document.txt --engine gemini --report report.json -o rewritten.txt
   ```

Notes:
- This sends your document text to Google's servers. On the free tier, Google may use inputs to improve its models — don't use this on anything confidential or sensitive.
- Free-tier rate limits are modest; `--delay` (default 1 second between requests) helps avoid `429` errors on longer documents. If you still hit them, increase `--delay` or wait a bit.
- Model names change over time — if `gemini-3.5-flash` (the default) stops working, check [Google's current model list](https://ai.google.dev/gemini-api/docs/models) and pass e.g. `--gemini-model gemini-2.5-flash`.

## Reading the results

The report shows **web coverage**: the share of chunks that had at least one web source to compare. A low flagged rate only means something when coverage is high. If coverage is low, the search engine is probably rate-limiting; increase `--delay` or retry later.

## Limitations

- DuckDuckGo search is unofficial and may rate-limit heavy use.
- Some sites block scraping; those pages are skipped with a warning.
- Results are heuristic. Always review flagged passages manually; this is not an authority on academic integrity.
- To index more of the web, swap `search_web()` for the Google Custom Search or Bing Search API.

## Contributing

Issues and pull requests are welcome.

## License

MIT. See [LICENSE](LICENSE).