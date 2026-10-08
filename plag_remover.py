#!/usr/bin/env python3
"""
Plag Remover — paraphraser
----------------------------
Rewrites text to reduce similarity to a source. Two engines:

  wordnet (default) — swaps words for WordNet synonyms. Fully offline
      after a one-time data download. Free, no signup, but mechanical:
      it can produce awkward choices ("oxygen" -> "O") since it has no
      understanding of context.

  gemini — sends each passage to Google's Gemini API for a real,
      context-aware rewrite. Needs a free Gemini API key (Google AI
      Studio, no credit card): https://aistudio.google.com/apikey
      Put it in a .env file (see .env.example) or the GEMINI_API_KEY
      environment variable.
      Much better quality, but sends your text to Google's servers,
      and on the free tier Google may use inputs to improve its
      models — don't use it on anything sensitive/confidential.

Either way: always read the output yourself. A rewrite tool nudges
wording away from a near-exact match; it doesn't make copied ideas
your own. If a passage is flagged because it states someone else's
fact or finding, the fix is a citation, not a rewording.

Install:
    pip install nltk requests

    (WordNet data downloads automatically on first run for the
    wordnet engine; requires internet access once, then works
    offline. The gemini engine always needs internet.)

Usage:
    # Rewrite a whole document (default: wordnet engine)
    python plag_remover.py my_document.txt -o rewritten.txt

    # Rewrite only the passages a plagiarism_checker.py report flagged
    python plagiarism_checker.py my_document.txt --json report.json
    python plag_remover.py my_document.txt --report report.json -o rewritten.txt

    # Control how aggressively words get swapped (0.0-1.0, default 0.4)
    python plag_remover.py my_document.txt --rate 0.6 -o rewritten.txt

    # Use Gemini instead (needs a Gemini API key, see .env.example)
    python plag_remover.py my_document.txt --engine gemini -o rewritten.txt
"""

import argparse
import json
import os
import random
import re
import sys
import time
from typing import List, Optional, Set

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a .env file in the current directory, if present
except ImportError:
    pass  # .env support is optional; GEMINI_API_KEY can still be set another way

# Reuse the document loaders / sentence splitter from the checker so
# .docx / .pdf / .txt all work the same way here.
from plagiarism_checker import load_document, split_into_sentences


def _ensure_wordnet():
    import nltk
    for pkg, path in [("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


# Small, safe stopword list so we don't mangle grammar words. Deliberately
# NOT importing nltk's stopwords corpus to avoid a second download.
_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "because",
    "as", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "would", "could", "might",
    "must", "shall", "i", "you", "he", "she", "it", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "who", "whom", "which", "what",
}

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _synonym_for(word: str) -> Optional[str]:
    """Pick a single-word WordNet synonym for `word`, or None if none fit."""
    from nltk.corpus import wordnet

    lower = word.lower()
    synsets = wordnet.synsets(lower)
    if not synsets:
        return None

    candidates = []
    for syn in synsets[:3]:  # stick to the most common senses
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if (
                name.lower() != lower
                and " " not in name          # keep it a single word
                and "-" not in name
                and name.isalpha()
            ):
                candidates.append(name)

    if not candidates:
        return None

    choice = random.choice(candidates)
    return choice


def _match_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def paraphrase_text(text: str, rate: float = 0.4, seed: Optional[int] = None) -> str:
    """
    Replace a fraction of eligible words with WordNet synonyms.

    rate: rough probability (0-1) that an eligible word gets swapped.
    Proper nouns (capitalized mid-sentence), short words (<=3 letters),
    and common stopwords are left untouched to preserve names, facts,
    and grammar.
    """
    _ensure_wordnet()
    rng = random.Random(seed)

    def replace(match: re.Match) -> str:
        word = match.group(0)
        start = match.start()

        if len(word) <= 3 or word.lower() in _STOPWORDS:
            return word
        # Skip likely proper nouns: capitalized and not the first word of the text/sentence
        is_sentence_start = start == 0 or text[max(0, start - 2):start].strip()[-1:] in ("", ".", "!", "?")
        if word[0].isupper() and not is_sentence_start:
            return word
        if rng.random() > rate:
            return word

        syn = _synonym_for(word)
        if not syn:
            return word
        return _match_case(word, syn)

    return _WORD_RE.sub(replace, text)


def paraphrase_document(text: str, rate: float = 0.4, seed: Optional[int] = None) -> str:
    sentences = split_into_sentences(text)
    if not sentences:
        return paraphrase_text(text, rate=rate, seed=seed)
    return " ".join(paraphrase_text(s, rate=rate, seed=seed) for s in sentences)


GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

_GEMINI_PROMPT = """You are helping a writer rephrase a passage of their own text so it is \
no longer a close match to an existing source, while keeping the meaning exactly the same.

Rules:
- Keep all facts, numbers, names, and claims unchanged. Do not add or remove information.
- Use different sentence structure and different wording from the original.
- Keep roughly the same length and the same tone/register.
- Output ONLY the rewritten passage. No preamble, no quotes, no explanation.

Passage:
{passage}"""


def gemini_paraphrase(
    passage: str,
    api_key: str,
    model: str = "gemini-3.5-flash",
    timeout: int = 30,
    retries: int = 3,
) -> str:
    """Rewrite one passage using the Gemini API. Returns the original text on failure."""
    url = GEMINI_ENDPOINT.format(model=model)
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    body = {"contents": [{"parts": [{"text": _GEMINI_PROMPT.format(passage=passage)}]}]}

    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"  [warn] Gemini rate limit hit; waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                reason = data.get("promptFeedback", {}).get("blockReason", "no candidates returned")
                print(f"  [warn] Gemini returned nothing ({reason}); keeping original text", file=sys.stderr)
                return passage
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
            return text or passage
        except requests.exceptions.RequestException as e:
            print(f"  [warn] Gemini request failed ({e})", file=sys.stderr)
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    print("  [warn] giving up on this passage; keeping original text", file=sys.stderr)
    return passage


def gemini_paraphrase_document(
    text: str, api_key: str, model: str = "gemini-3.5-flash", delay: float = 1.0
) -> str:
    """Rewrite a whole document paragraph-by-paragraph (keeps blank-line breaks)."""
    paragraphs = text.split("\n\n")
    out = []
    for i, para in enumerate(paragraphs):
        if para.strip():
            out.append(gemini_paraphrase(para, api_key, model=model))
            if i < len(paragraphs) - 1:
                time.sleep(delay)
        else:
            out.append(para)
    return "\n\n".join(out)


def load_flagged_chunks(report_path: str) -> List[str]:
    """Load chunk texts from a plagiarism_checker.py --json report."""
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [c["chunk_text"] for c in data.get("flagged_chunks", [])]


def main():
    parser = argparse.ArgumentParser(description="Offline WordNet-based paraphraser")
    parser.add_argument("file", help="Path to the file to rewrite (.txt, .docx or .pdf)")
    parser.add_argument("-o", "--output", default=None,
                         help="Where to write the rewritten text (default: prints to stdout)")
    parser.add_argument("--rate", type=float, default=0.4,
                         help="Fraction (0-1) of eligible words to try to replace (default 0.4)")
    parser.add_argument("--report", default=None,
                         help="A plagiarism_checker.py --json report; if given, only the "
                              "flagged chunks are rewritten (each preceded by '--- Chunk N ---')")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, for reproducible output (wordnet engine only)")
    parser.add_argument("--engine", choices=["wordnet", "gemini"], default="wordnet",
                         help="Paraphrasing engine: 'wordnet' (offline, free, mechanical) or "
                              "'gemini' (needs an API key, much better quality) (default wordnet)")
    parser.add_argument("--api-key", default=None,
                         help="Gemini API key. If omitted, reads GEMINI_API_KEY from a .env file or the environment.")
    parser.add_argument("--gemini-model", default="gemini-3.5-flash",
                         help="Gemini model to use (default gemini-3.5-flash)")
    parser.add_argument("--delay", type=float, default=1.0,
                         help="Seconds between Gemini requests, to stay within free-tier rate limits (default 1.0)")
    args = parser.parse_args()

    if not 0.0 <= args.rate <= 1.0:
        sys.exit("--rate must be between 0.0 and 1.0")

    api_key = None
    if args.engine == "gemini":
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            sys.exit(
                "--engine gemini needs an API key. Get a free one (no credit card) at "
                "https://aistudio.google.com/apikey, then either:\n"
                "  1. Copy .env.example to .env and paste your key in, or\n"
                "  2. export GEMINI_API_KEY=\"your-key\"   (PowerShell: $env:GEMINI_API_KEY=\"your-key\")\n"
                "  or pass --api-key your-key"
            )

    try:
        text = load_document(args.file)
    except OSError as e:
        sys.exit(f"Could not read file: {e}")

    def rewrite(chunk_text: str) -> str:
        if args.engine == "gemini":
            return gemini_paraphrase(chunk_text, api_key, model=args.gemini_model)
        return paraphrase_document(chunk_text, rate=args.rate, seed=args.seed)

    if args.report:
        chunks = load_flagged_chunks(args.report)
        if not chunks:
            sys.exit("No flagged chunks found in that report.")
        out_parts = []
        for i, chunk in enumerate(chunks, 1):
            rewritten = rewrite(chunk)
            out_parts.append(f"--- Chunk {i} (original) ---\n{chunk}\n\n"
                              f"--- Chunk {i} (rewritten) ---\n{rewritten}\n")
            if args.engine == "gemini" and i < len(chunks):
                time.sleep(args.delay)
        output_text = "\n".join(out_parts)
    elif args.engine == "gemini":
        output_text = gemini_paraphrase_document(text, api_key, model=args.gemini_model, delay=args.delay)
    else:
        output_text = paraphrase_document(text, rate=args.rate, seed=args.seed)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Rewritten text written to {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()