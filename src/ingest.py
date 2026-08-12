"""
ingest.py — Build the FetchWise retrieval index.

Pipeline:
  1. Load each markdown doc in data/docs/
  2. Split into chunks (short docs mostly stay as a single chunk; longer
     ones get split at paragraph boundaries with a small word-count overlap)
  3. Embed each chunk with a sentence-transformers model
  4. Build a FAISS index (cosine similarity via L2-normalized inner product)
  5. Persist the index + chunk metadata to disk so agent.py / eval.py can
     load them without re-embedding on every run

Run directly to (re)build the index and sanity-check retrieval on a few
sample queries:

    python src/ingest.py
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Config ------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "data" / "docs"
INDEX_DIR = ROOT_DIR / "data" / "faiss_index"
INDEX_PATH = INDEX_DIR / "index.faiss"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking is word-count based and paragraph-aware. Our docs are short
# (~120-180 words) so most end up as a single chunk; the logic still handles
# longer docs sensibly by splitting on paragraph boundaries with overlap so
# no chunk straddles a paragraph mid-thought.
CHUNK_SIZE_WORDS = 120
CHUNK_OVERLAP_WORDS = 20

TOP_K_DEFAULT = 3


@dataclass
class Chunk:
    chunk_id: str      # e.g. "02-returns-refunds::0"
    doc_id: str         # filename stem, e.g. "02-returns-refunds"
    title: str           # doc title parsed from the "# Heading"
    category: str         # doc category parsed from "**Category:** X"
    chunk_index: int       # position of this chunk within its doc
    text: str                # chunk text actually embedded (title-prefixed)


# --- Parsing & chunking -------------------------------------------------------

def parse_doc(raw_text: str) -> tuple[str, str, str]:
    """Pull the title, category, and body out of one of our doc files.

    Docs follow a fixed convention:
        # Title
        **Category:** X
        <body paragraphs>
    """
    lines = raw_text.strip().splitlines()
    title = lines[0].lstrip("#").strip() if lines and lines[0].startswith("#") else ""

    category_match = re.search(r"\*\*Category:\*\*\s*(.+)", raw_text)
    category = category_match.group(1).strip() if category_match else "General"

    # Body = everything after the "**Category:** X" line.
    body_start = 0
    if category_match:
        body_start = raw_text.index(category_match.group(0)) + len(category_match.group(0))
    body = raw_text[body_start:].strip()

    return title, category, body


def chunk_text(
    body: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """Split a doc body into word-count-bounded chunks at paragraph boundaries.

    Paragraphs (split on blank lines) are greedily packed into a chunk until
    adding the next paragraph would exceed chunk_size words, then a new
    chunk starts, carrying the last `overlap` words forward so retrieval
    doesn't lose context right at a chunk boundary.
    """
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_words: list[str] = []

    for para in paragraphs:
        para_words = para.split()
        if current_words and len(current_words) + len(para_words) > chunk_size:
            chunks.append(" ".join(current_words))
            current_words = current_words[-overlap:] if overlap else []
        current_words.extend(para_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def build_chunks() -> list[Chunk]:
    """Load every doc in DOCS_DIR, chunk it, and return a flat list of Chunks."""
    doc_paths = sorted(DOCS_DIR.glob("*.md"))
    if not doc_paths:
        raise FileNotFoundError(f"No .md files found in {DOCS_DIR}")

    chunks: list[Chunk] = []
    for path in doc_paths:
        doc_id = path.stem
        raw_text = path.read_text(encoding="utf-8")
        title, category, body = parse_doc(raw_text)

        for i, chunk_body in enumerate(chunk_text(body)):
            # Prefix the title so short chunks still carry doc-level context
            # into the embedding (a bare paragraph like "Refunds are issued
            # once..." embeds much better with "Returns & Refunds Policy: "
            # in front of it).
            embedded_text = f"{title}: {chunk_body}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{i}",
                    doc_id=doc_id,
                    title=title,
                    category=category,
                    chunk_index=i,
                    text=embedded_text,
                )
            )

    return chunks


# --- Embedding & index ---------------------------------------------------------

def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embeddings)  # so inner product == cosine similarity
    return embeddings.astype("float32")


def build_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # brute-force cosine sim search — plenty fast at this corpus size
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, chunks: list[Chunk]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    CHUNKS_PATH.write_text(json.dumps([asdict(c) for c in chunks], indent=2), encoding="utf-8")


def load_index() -> tuple[faiss.Index, list[Chunk]]:
    """Load a previously built index + chunk metadata (used by agent.py / eval.py)."""
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"No index found at {INDEX_DIR}. Run `python src/ingest.py` first.")
    index = faiss.read_index(str(INDEX_PATH))
    raw_chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    chunks = [Chunk(**c) for c in raw_chunks]
    return index, chunks


# --- Retrieval -------------------------------------------------------------------

def search(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    chunks: list[Chunk],
    k: int = TOP_K_DEFAULT,
) -> list[tuple[Chunk, float]]:
    """Return the top-k (chunk, cosine_similarity) pairs for a query."""
    query_embedding = embed_texts(model, [query])
    scores, indices = index.search(query_embedding, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # FAISS pads with -1 if fewer than k results exist
            continue
        results.append((chunks[idx], float(score)))
    return results


# --- Script entry point ------------------------------------------------------------

SAMPLE_QUERIES = [
    "How long do I have to return an item?",
    "Can I pause my subscription instead of cancelling it?",
    "Do you ship to Canada?",
    "What's the weather like today?",  # off-topic — sanity check for low scores
]


def build_and_save_index(model: SentenceTransformer, verbose: bool = True) -> tuple[faiss.Index, list[Chunk]]:
    """End-to-end build: chunk docs, embed, index, persist to disk.

    Used both by `python src/ingest.py` directly and as an automatic
    fallback in agent.py's FetchWiseAgent when no prebuilt index is found on
    disk — e.g. on a fresh clone or a fresh deploy, since data/faiss_index/
    is a gitignored, rebuildable artifact rather than something committed.
    """
    if verbose:
        print(f"Chunking docs from {DOCS_DIR}...")
    chunks = build_chunks()
    if verbose:
        print(f"  -> {len(chunks)} chunks from {len(set(c.doc_id for c in chunks))} docs")
        print("Embedding chunks...")
    embeddings = embed_texts(model, [c.text for c in chunks])

    if verbose:
        print("Building FAISS index...")
    index = build_index(embeddings)

    save_index(index, chunks)
    if verbose:
        print(f"Saved index to {INDEX_PATH} and metadata to {CHUNKS_PATH}")

    return index, chunks


def main() -> None:
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    index, chunks = build_and_save_index(model)

    print("\n--- Sample retrieval checks ---")
    for query in SAMPLE_QUERIES:
        print(f"\nQuery: {query!r}")
        results = search(query, model, index, chunks, k=3)
        for rank, (chunk, score) in enumerate(results, start=1):
            preview = chunk.text[:90].replace("\n", " ")
            print(f"  {rank}. [{score:.3f}] {chunk.doc_id} ({chunk.category}) — {preview}...")


if __name__ == "__main__":
    main()
