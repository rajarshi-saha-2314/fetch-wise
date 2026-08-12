"""
agent.py — FetchWise core loop: retrieve -> decide -> act.

For each user query:
  1. Retrieve the top-k most relevant knowledge-base chunks (ingest.search).
     Only chunks scoring >= RETRIEVAL_THRESHOLD are shown to the model as
     context, so it isn't tempted to stretch an answer out of a barely
     related snippet.
  2. Decide + act, via a single LLM call with tool-calling enabled. The
     model itself decides whether to:
       (a) answer using only the retrieved excerpts,
       (b) call check_order_status (order-specific question with an ID), or
       (c) say honestly that it doesn't know, per its system-prompt
           instruction, when the excerpts don't cover the question.
  3. If the model called the tool, execute it against our mock data and
     make a second LLM call so it can phrase a natural answer grounded in
     the tool's actual output.
  4. Return a structured AgentResponse that tags which path was taken
     ("tool" / "retrieval" / "refusal"). That tag is derived deterministically
     from what actually happened (tool called? top retrieval score above
     threshold?) rather than by parsing the model's free-text wording — this
     keeps eval.py's scoring robust instead of relying on keyword-guessing.

Model: Groq-hosted openai/gpt-oss-120b, OpenAI-compatible tool calling.

Guardrail: before the LLM is called, guardrails.is_off_topic() checks the
same top retrieval score computed below against a slightly lower threshold
than RETRIEVAL_THRESHOLD. Queries clearly outside the support domain are
refused immediately with a canned message — no LLM call spent. See
guardrails.py for the threshold rationale.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from guardrails import REFUSAL_MESSAGE, is_off_topic
from ingest import EMBEDDING_MODEL_NAME, Chunk, load_index, search
from tools import TOOL_SCHEMAS, check_order_status

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 3

# Chunks scoring below this cosine-similarity are treated as "not actually
# relevant" and withheld from the model's context. Calibrated from the
# sample queries in ingest.py: on-topic queries scored ~0.55-0.71, while an
# off-topic query topped out at ~0.25. See step 3's output for those numbers.
RETRIEVAL_THRESHOLD = 0.35

SYSTEM_PROMPT = """You are FetchWise, the support assistant for Fetchly, an online pet food and supplies retailer with a subscription box service.

You have two ways to help a customer:
1. Answer using ONLY the "Retrieved policy excerpts" provided below. Do not use outside knowledge about Fetchly, and do not guess or invent policy details, prices, dates, or numbers that are not in the excerpts.
2. Call the check_order_status tool if the customer is asking about the status, tracking, or delivery of a SPECIFIC order and gives (or you can extract) an order ID in the format FCH-XXXXX. If they ask about an order but haven't given an ID, ask them for it instead of calling the tool.

If the retrieved excerpts do not contain the answer, and no tool applies, say so plainly and honestly instead of guessing — for example "I don't have information about that in our support docs." It's always better to admit you don't know than to fabricate an answer.

Keep answers short, friendly, and specific to what's actually in the excerpts or tool output.
"""


@dataclass
class RetrievedChunkInfo:
    doc_id: str
    title: str
    category: str
    score: float
    text: str  # the chunk text itself, so callers (eval.py, the UI) don't need to reload chunks.json


@dataclass
class ToolCallInfo:
    name: str
    arguments: dict
    result: dict


@dataclass
class AgentResponse:
    query: str
    answer: str
    path: str  # "retrieval" | "tool" | "refusal"
    retrieved_chunks: list[RetrievedChunkInfo]
    tool_call: Optional[ToolCallInfo] = None


class FetchWiseAgent:
    """Wraps the embedding model, FAISS index, and Groq client so callers
    (Streamlit app, eval harness) just construct once and call .answer(query)."""

    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
            )
        self.client = Groq(api_key=api_key)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index, self.chunks = load_index()

    def _retrieve(self, query: str, k: int = TOP_K) -> list[tuple[Chunk, float]]:
        return search(query, self.embedding_model, self.index, self.chunks, k=k)

    def _format_context(self, results: list[tuple[Chunk, float]]) -> str:
        """Render only the sufficiently-relevant chunks as context text.

        Chunks below RETRIEVAL_THRESHOLD are dropped here (not just used for
        path-labeling) so the model isn't shown weakly-related snippets it
        might otherwise be tempted to stretch into an answer.
        """
        relevant = [(c, s) for c, s in results if s >= RETRIEVAL_THRESHOLD]
        if not relevant:
            return "(No sufficiently relevant excerpts were found in the knowledge base for this question.)"
        blocks = [f"(Source: {c.doc_id})\n{c.text}" for c, _ in relevant]
        return "\n\n".join(blocks)

    def answer(self, query: str) -> AgentResponse:
        retrieved = self._retrieve(query)
        top_score = retrieved[0][1] if retrieved else 0.0
        retrieved_info = [
            RetrievedChunkInfo(doc_id=c.doc_id, title=c.title, category=c.category, score=s, text=c.text)
            for c, s in retrieved
        ]

        # Guardrail: reject queries clearly outside the support domain before
        # spending an LLM call. Reuses the top score from the retrieval above
        # rather than embedding the query a second time.
        if is_off_topic(top_score):
            return AgentResponse(
                query=query,
                answer=REFUSAL_MESSAGE,
                path="refusal",
                retrieved_chunks=retrieved_info,
                tool_call=None,
            )

        user_message = (
            f"Customer question: {query}\n\n"
            f"Retrieved policy excerpts:\n{self._format_context(retrieved)}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # First LLM call: the model decides to answer directly or call the tool.
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2,
        )
        message = response.choices[0].message

        tool_call_info: Optional[ToolCallInfo] = None

        if message.tool_calls:
            # The model chose to call check_order_status. We only act on the
            # first tool call — one order lookup per turn is all this agent
            # needs, and handling more would add complexity without a real
            # use case here.
            tool_call = message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            order_id = args.get("order_id", "")
            result = check_order_status(order_id)

            tool_call_info = ToolCallInfo(
                name=tool_call.function.name,
                arguments=args,
                result=result.to_dict(),
            )

            # Echo the assistant's tool-call turn, then the tool's result,
            # back into the conversation so the model can phrase a final
            # answer grounded in what the tool actually returned.
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result.to_dict()),
                }
            )

            follow_up = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.2,
            )
            final_text = follow_up.choices[0].message.content
            path = "tool"
        else:
            final_text = message.content
            # Deterministic path label: even though the LLM decided the
            # wording, we tag "refusal" whenever retrieval didn't surface
            # anything actually relevant (per the same threshold used to
            # build the model's context), rather than parsing free text.
            path = "retrieval" if top_score >= RETRIEVAL_THRESHOLD else "refusal"

        return AgentResponse(
            query=query,
            answer=final_text,
            path=path,
            retrieved_chunks=retrieved_info,
            tool_call=tool_call_info,
        )


if __name__ == "__main__":
    agent = FetchWiseAgent()
    sample_queries = [
        "What's your return window for opened dog food?",
        "Where is my order FCH-10234?",
        "What's the capital of France?",
    ]
    for q in sample_queries:
        resp = agent.answer(q)
        print(f"\nQ: {q}")
        print(f"Path: {resp.path}")
        if resp.tool_call:
            print(f"Tool call: {resp.tool_call.name}({resp.tool_call.arguments}) -> {resp.tool_call.result}")
        print(f"Retrieved: {[(c.doc_id, round(c.score, 3)) for c in resp.retrieved_chunks]}")
        print(f"Answer: {resp.answer}")
