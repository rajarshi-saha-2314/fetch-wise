"""
eval.py — Run the FetchWise agent against eval_set/test_queries.json and
score it on four axes, saving a report to results/eval_report.json.

Metrics:
  1. Retrieval correctness — for cases expecting a doc-grounded answer, did
     the top-k retrieval actually surface one of the expected doc(s)?
  2. Tool-call correctness — for cases expecting a tool call, did the agent
     call check_order_status with the correct order_id? (And, for the
     no-order-ID edge case classified under retrieval, did it correctly
     avoid calling the tool at all — caught via path_correct.)
  3. Hallucination check — a lightweight heuristic: every digit sequence
     (day counts, dollar amounts, dates, percentages, order IDs...) in the
     final answer must also appear somewhere in what the agent was actually
     shown (the doc excerpts passed to the LLM, plus any tool output). This
     won't catch every possible hallucination — a false claim with no
     numbers in it slips through — but it catches the highest-stakes
     failure mode for a support bot: inventing a policy number that isn't
     real. Documented limitation, not a claim of full coverage.
  4. Refusal correctness — for off-topic cases, did the agent actually
     refuse (path == "refusal")?

Every case also gets a soft "keyword hit rate" against expected_keywords,
reported for transparency but not treated as pass/fail (paraphrasing can
legitimately omit an expected substring).

Run:
    python src/eval.py
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from agent import AgentResponse, FetchWiseAgent, RETRIEVAL_THRESHOLD

ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_QUERIES_PATH = ROOT_DIR / "eval_set" / "test_queries.json"
REPORT_PATH = ROOT_DIR / "results" / "eval_report.json"

NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def load_test_cases() -> list[dict]:
    data = json.loads(TEST_QUERIES_PATH.read_text(encoding="utf-8"))
    return data["test_cases"]


LIST_MARKER_PATTERN = re.compile(r"(?m)^\s*\d+[.)]\s+")


def extract_numbers(text: str) -> set[str]:
    """Pull digit sequences out of model-generated text as a hallucination
    proxy — but first strip markdown ordered-list markers ("1. ", "2) ")
    from the start of lines, since the model formats answers as numbered
    steps and those digits aren't claims about anything, just list order.
    """
    cleaned = LIST_MARKER_PATTERN.sub("", text or "")
    return set(NUMBER_PATTERN.findall(cleaned))


def build_grounding_text(resp: AgentResponse) -> str:
    """Reconstruct what the agent was actually shown: the doc excerpts that
    passed RETRIEVAL_THRESHOLD (the same filter agent.py applies before
    calling the LLM) plus any tool output."""
    doc_context = "\n".join(c.text for c in resp.retrieved_chunks if c.score >= RETRIEVAL_THRESHOLD)
    tool_context = json.dumps(resp.tool_call.result) if resp.tool_call else ""
    return f"{doc_context}\n{tool_context}"


def check_hallucination(resp: AgentResponse) -> tuple[bool, list[str]]:
    grounding_text = build_grounding_text(resp)
    answer_numbers = extract_numbers(resp.answer)
    ungrounded = sorted(n for n in answer_numbers if n not in grounding_text)
    return (len(ungrounded) == 0, ungrounded)


@dataclass
class CaseResult:
    id: str
    query: str
    category: str
    expected_path: str
    actual_path: str
    path_correct: bool
    retrieval_correct: Optional[bool]
    retrieved_doc_ids: list[str]
    tool_call_correct: Optional[bool]
    actual_tool_call: Optional[dict]
    refusal_correct: Optional[bool]
    hallucination_free: bool
    ungrounded_numbers: list[str]
    expected_keywords: list[str]
    matched_keywords: list[str]
    answer: str


def evaluate_case(agent: FetchWiseAgent, case: dict) -> CaseResult:
    resp = agent.answer(case["query"])

    path_correct = resp.path == case["expected_path"]
    retrieved_doc_ids = [c.doc_id for c in resp.retrieved_chunks]

    retrieval_correct: Optional[bool] = None
    if case["expected_path"] == "retrieval":
        retrieval_correct = any(doc_id in case["expected_doc_ids"] for doc_id in retrieved_doc_ids)

    tool_call_correct: Optional[bool] = None
    actual_tool_call = None
    if resp.tool_call:
        actual_tool_call = {"order_id": resp.tool_call.arguments.get("order_id")}
    if case["expected_path"] == "tool":
        expected_order_id = case["expected_tool_call"]["order_id"].upper()
        tool_call_correct = (
            resp.path == "tool"
            and resp.tool_call is not None
            and str(resp.tool_call.arguments.get("order_id", "")).upper() == expected_order_id
        )

    refusal_correct: Optional[bool] = None
    if case["expected_path"] == "refusal":
        refusal_correct = resp.path == "refusal"

    hallucination_free, ungrounded_numbers = check_hallucination(resp)

    expected_keywords = case.get("expected_keywords", [])
    answer_lower = (resp.answer or "").lower()
    matched_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]

    return CaseResult(
        id=case["id"],
        query=case["query"],
        category=case["category"],
        expected_path=case["expected_path"],
        actual_path=resp.path,
        path_correct=path_correct,
        retrieval_correct=retrieval_correct,
        retrieved_doc_ids=retrieved_doc_ids,
        tool_call_correct=tool_call_correct,
        actual_tool_call=actual_tool_call,
        refusal_correct=refusal_correct,
        hallucination_free=hallucination_free,
        ungrounded_numbers=ungrounded_numbers,
        expected_keywords=expected_keywords,
        matched_keywords=matched_keywords,
        answer=resp.answer,
    )


def rate(numer: int, denom: int) -> Optional[float]:
    return round(numer / denom, 3) if denom else None


def summarize(results: list[CaseResult]) -> dict:
    n = len(results)
    path_ok = sum(r.path_correct for r in results)

    retrieval_cases = [r for r in results if r.retrieval_correct is not None]
    retrieval_ok = sum(r.retrieval_correct for r in retrieval_cases)

    tool_cases = [r for r in results if r.tool_call_correct is not None]
    tool_ok = sum(r.tool_call_correct for r in tool_cases)

    refusal_cases = [r for r in results if r.refusal_correct is not None]
    refusal_ok = sum(r.refusal_correct for r in refusal_cases)

    hallucination_free_ok = sum(r.hallucination_free for r in results)

    keyword_cases = [r for r in results if r.expected_keywords]
    keyword_hit_rate = (
        round(sum(len(r.matched_keywords) / len(r.expected_keywords) for r in keyword_cases) / len(keyword_cases), 3)
        if keyword_cases
        else None
    )

    return {
        "total_cases": n,
        "path_accuracy": rate(path_ok, n),
        "retrieval_accuracy": rate(retrieval_ok, len(retrieval_cases)),
        "retrieval_cases": len(retrieval_cases),
        "tool_call_accuracy": rate(tool_ok, len(tool_cases)),
        "tool_cases": len(tool_cases),
        "refusal_accuracy": rate(refusal_ok, len(refusal_cases)),
        "refusal_cases": len(refusal_cases),
        "hallucination_free_rate": rate(hallucination_free_ok, n),
        "keyword_hit_rate_advisory": keyword_hit_rate,
    }


def print_report(summary: dict, results: list[CaseResult], console: Console) -> None:
    console.print("\n[bold]FetchWise Eval Report[/bold]\n")

    summary_table = Table(show_header=True, header_style="bold")
    summary_table.add_column("Metric")
    summary_table.add_column("Score", justify="right")
    summary_table.add_row("Path accuracy (overall)", f"{summary['path_accuracy']:.0%} ({summary['total_cases']} cases)")
    summary_table.add_row(
        "Retrieval correctness", f"{summary['retrieval_accuracy']:.0%} ({summary['retrieval_cases']} cases)"
    )
    summary_table.add_row("Tool-call correctness", f"{summary['tool_call_accuracy']:.0%} ({summary['tool_cases']} cases)")
    summary_table.add_row("Refusal correctness", f"{summary['refusal_accuracy']:.0%} ({summary['refusal_cases']} cases)")
    summary_table.add_row("Hallucination-free rate", f"{summary['hallucination_free_rate']:.0%}")
    kw = summary["keyword_hit_rate_advisory"]
    summary_table.add_row("Keyword hit rate (advisory)", f"{kw:.0%}" if kw is not None else "n/a")
    console.print(summary_table)

    failures = [r for r in results if not r.path_correct or not r.hallucination_free]
    if failures:
        console.print(f"\n[bold red]{len(failures)} case(s) with issues:[/bold red]")
        fail_table = Table(show_header=True, header_style="bold")
        fail_table.add_column("ID")
        fail_table.add_column("Query")
        fail_table.add_column("Expected")
        fail_table.add_column("Actual")
        fail_table.add_column("Issue")
        for r in failures:
            issue = []
            if not r.path_correct:
                issue.append("wrong path")
            if not r.hallucination_free:
                issue.append(f"ungrounded numbers: {r.ungrounded_numbers}")
            fail_table.add_row(r.id, r.query, r.expected_path, r.actual_path, "; ".join(issue))
        console.print(fail_table)
    else:
        console.print("\n[bold green]No path or hallucination issues found.[/bold green]")


def run_eval() -> dict:
    console = Console()
    cases = load_test_cases()
    console.print(f"Loaded {len(cases)} test cases from {TEST_QUERIES_PATH}")

    console.print("Initializing agent (loading embedding model + FAISS index)...")
    agent = FetchWiseAgent()

    results = []
    for case in cases:
        console.print(f"  Running {case['id']}: {case['query'][:60]}...")
        results.append(evaluate_case(agent, case))

    summary = summarize(results)
    print_report(summary, results, console)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\nSaved full report to {REPORT_PATH}")

    return report


if __name__ == "__main__":
    run_eval()
