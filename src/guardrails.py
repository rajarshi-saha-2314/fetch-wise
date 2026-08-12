"""
guardrails.py — Off-topic query guardrail for FetchWise.

Rule: before spending an LLM call, reject queries that aren't about
Fetchly's support domain at all (orders, shipping, returns, subscriptions,
billing, etc.) — as opposed to questions that ARE in-domain but just aren't
covered by any doc, which agent.py already handles by letting the LLM
honestly say "I don't know" once it sees no relevant context.

Detection reuses the same retrieval scores agent.py already computes: if the
single best-matching chunk anywhere in the knowledge base scores below
OFF_TOPIC_THRESHOLD, the query is judged unrelated to the support domain
entirely, and a canned refusal is returned immediately — no LLM call needed.

Calibration: from ingest.py's sample queries, on-topic queries scored
~0.55-0.71 while an off-topic ("what's the weather") query topped out at
~0.25. OFF_TOPIC_THRESHOLD=0.30 sits just above that off-topic ceiling, with
some margin below agent.py's own RETRIEVAL_THRESHOLD=0.35 (used to decide
whether a chunk is good enough to answer FROM). That gap is intentional:
this guardrail only screens out queries that are clearly outside the
domain; borderline-but-real support questions in between the two
thresholds still reach the LLM, which is instructed to admit honestly when
it lacks specific information rather than being hard-blocked by a rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import faiss
from sentence_transformers import SentenceTransformer

from ingest import Chunk, search

OFF_TOPIC_THRESHOLD = 0.30

REFUSAL_MESSAGE = (
    "I'm the Fetchly support assistant, so I can only help with questions about "
    "Fetchly orders, shipping, subscriptions, returns, billing, and related topics. "
    "I'm not able to help with that here — is there anything Fetchly-related I can help with instead?"
)


@dataclass
class GuardrailResult:
    blocked: bool
    top_score: float
    reason: Optional[str] = None


def is_off_topic(top_score: float) -> bool:
    """Pure threshold check given an already-computed top retrieval score.

    agent.py calls this directly with the top score from its own retrieval
    step, so the query never gets embedded twice.
    """
    return top_score < OFF_TOPIC_THRESHOLD


def check_off_topic(
    query: str,
    embedding_model: SentenceTransformer,
    index: faiss.Index,
    chunks: list[Chunk],
) -> GuardrailResult:
    """Standalone convenience wrapper that does its own retrieval.

    Useful for testing/demoing the guardrail in isolation (see __main__
    below and eval.py's guardrail-specific checks). agent.py does NOT call
    this — it reuses the top score from its own retrieval via is_off_topic()
    instead, to avoid a redundant embedding call per turn.
    """
    results = search(query, embedding_model, index, chunks, k=1)
    top_score = results[0][1] if results else 0.0
    blocked = is_off_topic(top_score)
    return GuardrailResult(blocked=blocked, top_score=top_score, reason="off_topic" if blocked else None)


if __name__ == "__main__":
    from ingest import EMBEDDING_MODEL_NAME, load_index

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    index, chunks = load_index()

    test_queries = [
        "What's your return policy?",
        "Can I get a refund on a recalled product?",
        "Where is my order FCH-10234?",
        "Can you write me a Python script to sort a list?",
        "What's the weather like today?",
        "Who won the last election?",
    ]
    print("\n--- Guardrail checks ---")
    for q in test_queries:
        result = check_off_topic(q, model, index, chunks)
        status = "BLOCKED (off-topic)" if result.blocked else "allowed"
        print(f"{status:20} [{result.top_score:.3f}]  {q}")
