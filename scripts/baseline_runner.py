#!/usr/bin/env python
"""Baseline runner: score a local Ollama model on the Arabic evaluation set.

Usage (from the project root, with Ollama running)::

    python evaluation/baseline_runner.py --model qwen2.5:7b
    python evaluation/baseline_runner.py --model qwen2.5:7b --category grounded
    python evaluation/baseline_runner.py --model qwen2.5:7b --ids G01,G02,S01 --limit 3

Outputs go to ``evaluation/results/<timestamp>_<model>/``:

* ``report.md``          human-readable summary (read this first)
* ``results.jsonl``      every reply with every check result (machine-readable)
* ``manual_review.csv``  sheet for scoring Fusha quality, naturalness, etc.
* ``summary.json``       numbers for comparing runs later

WHY (architecture decisions)
----------------------------
1. **It runs the production prompt path.** Prompts are built with the same
   ``build_system_prompt`` / ``assemble_messages`` and the same
   ``[Source n: file, page p]`` context format the retriever produces. A baseline
   measured on a different prompt would say nothing about the real system.

2. **Retrieval is bypassed on purpose.** Each item carries its own context, so
   we measure the *model's* behaviour (language, grounding, citing, refusing)
   independent of retrieval quality. Retrieval is evaluated separately.

3. **Reproducible runs.** Fixed seed and low temperature, and the exact settings
   are written into the report, so a later run of the fine-tuned model is a fair
   comparison. Use ``--model`` to score any Ollama model with the same yardstick.

4. **Explicit context window.** Ollama's default context window is small and it
   silently drops the *start* of an over-long prompt, which here would be the
   system prompt itself. The runner sets ``num_ctx`` explicitly (default 8192) and
   warns when a prompt gets close to it. The production chat client must do the same.

5. **One failure never aborts the run.** A timeout on one item is recorded and
   the run continues; a pre-flight check reports a missing model or a stopped
   Ollama server up front with the command to fix it.
"""

import argparse
import csv
import json
import logging
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Allow `python evaluation/baseline_runner.py` (not only `python -m ...`).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402
from pydantic import BaseModel, Field, ValidationError, model_validator  # noqa: E402

from app.ai.prompts.system_prompts import (  # noqa: E402
    SYSTEM_PROMPT_CORE,
    assemble_messages,
    build_system_prompt,
)
from app.core.config import get_settings  # noqa: E402
from evaluation.checks import evaluate, item_passed, strip_reasoning  # noqa: E402

logger = logging.getLogger("baseline_runner")

DEFAULT_EVAL_SET = PROJECT_ROOT / "evaluation" / "evaluation_set.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "evaluation" / "results"
_CHARS_PER_TOKEN = 3  # same cautious estimate as app/ai/memory.py


# =============================================================================
# Evaluation-set schema
# =============================================================================
Category = Literal["grounded", "no_context", "general", "language", "memory", "limits"]


class ContextSource(BaseModel):
    source: str
    page: Optional[int] = None
    text: str


class MemoryInput(BaseModel):
    session_summary: str = ""
    past_conversations: str = ""


class Expect(BaseModel):
    citations: Literal["required", "forbidden", "optional"] = "optional"
    min_distinct_citations: int = Field(default=1, ge=1)
    full_structure: bool = False
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    no_context_admission: bool = False
    general_disclosure: bool = False
    notes: str = ""


class EvalItem(BaseModel):
    id: str
    category: Category
    question: str = Field(min_length=1)
    context_sources: list[ContextSource] = Field(default_factory=list)
    memory: Optional[MemoryInput] = None
    expect: Expect


class EvalSet(BaseModel):
    version: int
    description: str = ""
    items: list[EvalItem] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "EvalSet":
        ids = [item.id for item in self.items]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"Duplicate item ids in evaluation set: {duplicates}")
        return self


class RunRecord(BaseModel):
    """Everything recorded for one item."""

    id: str
    category: str
    question: str
    reply: str = ""
    error: Optional[str] = None
    passed: bool = False
    checks: list[dict[str, Any]] = Field(default_factory=list)
    latency_s: float = 0.0
    prompt_tokens_est: int = 0
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    tokens_per_sec: Optional[float] = None
    done_reason: Optional[str] = None
    notes: str = ""


# =============================================================================
# Loading and preparing
# =============================================================================
def load_eval_set(path: Path) -> EvalSet:
    """Load and validate the evaluation set, exiting with a clear message on errors."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return EvalSet.model_validate(raw)
    except FileNotFoundError:
        raise SystemExit(f"Evaluation set not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Evaluation set is not valid JSON ({path}): {exc}")
    except ValidationError as exc:
        raise SystemExit(f"Evaluation set failed validation:\n{exc}")


def format_context(sources: Sequence[ContextSource]) -> str:
    """Render context exactly like ``MultiQueryRetriever`` does (keep the two in sync)."""
    blocks = []
    for number, source in enumerate(sources, start=1):
        page = f", page {source.page}" if source.page is not None else ""
        blocks.append(f"[Source {number}: {source.source}{page}]\n{source.text.strip()}")
    return "\n\n".join(blocks)


def select_items(
    items: Sequence[EvalItem],
    *,
    ids: Optional[str],
    category: Optional[str],
    limit: Optional[int],
) -> list[EvalItem]:
    selected = list(items)
    if ids:
        wanted = {part.strip() for part in ids.split(",") if part.strip()}
        unknown = wanted - {item.id for item in selected}
        if unknown:
            raise SystemExit(f"Unknown item id(s): {sorted(unknown)}")
        selected = [item for item in selected if item.id in wanted]
    if category:
        selected = [item for item in selected if item.category == category]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise SystemExit("No evaluation items match the given filters.")
    return selected


# =============================================================================
# Talking to Ollama
# =============================================================================
def preflight(base_url: str, model: str) -> None:
    """Fail fast, with the fix, if Ollama is down or the model is not installed."""
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=5.0)
        response.raise_for_status()
        installed = [entry.get("name", "") for entry in response.json().get("models", [])]
    except (httpx.HTTPError, ValueError) as exc:
        raise SystemExit(
            f"Cannot reach Ollama at {base_url} ({exc}).\nStart it with `ollama serve`."
        )
    if model not in installed and f"{model}:latest" not in installed:
        shown = ", ".join(sorted(installed)) or "none"
        raise SystemExit(
            f"Model '{model}' is not installed in Ollama (installed: {shown}).\n"
            f"Install it with `ollama pull {model}`."
        )


def build_llm(
    model: str,
    base_url: str,
    *,
    temperature: float,
    seed: int,
    num_ctx: int,
    max_tokens: int,
    timeout: int,
) -> ChatOllama:
    kwargs: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "seed": seed,
        "num_ctx": num_ctx,
        "num_predict": max_tokens,
    }
    try:
        return ChatOllama(**kwargs, client_kwargs={"timeout": timeout})
    except (TypeError, ValueError):
        return ChatOllama(**kwargs)


def run_item(llm: ChatOllama, item: EvalItem) -> RunRecord:
    """Build the production prompt for one item, query the model, run the checks."""
    memory = item.memory or MemoryInput()
    system_prompt = build_system_prompt(
        rag_context=format_context(item.context_sources),
        session_summary=memory.session_summary,
        past_conversations=memory.past_conversations,
    )
    messages = assemble_messages(system_prompt, [], item.question)
    prompt_chars = sum(len(str(message.content)) for message in messages)
    record = RunRecord(
        id=item.id,
        category=item.category,
        question=item.question,
        prompt_tokens_est=math.ceil(prompt_chars / _CHARS_PER_TOKEN),
        notes=item.expect.notes,
    )

    started = time.perf_counter()
    try:
        response = llm.invoke(messages)
    except Exception as exc:  # timeout, connection reset, model crashed...
        record.error = f"{type(exc).__name__}: {exc}"
        record.latency_s = round(time.perf_counter() - started, 2)
        return record
    record.latency_s = round(time.perf_counter() - started, 2)

    content = response.content
    reply = content if isinstance(content, str) else "".join(str(part) for part in content)
    record.reply = strip_reasoning(reply)

    meta = getattr(response, "response_metadata", {}) or {}
    record.done_reason = meta.get("done_reason")
    record.prompt_eval_count = meta.get("prompt_eval_count")
    record.eval_count = meta.get("eval_count")
    eval_ns = meta.get("eval_duration")
    if record.eval_count and eval_ns:
        record.tokens_per_sec = round(record.eval_count / (eval_ns / 1e9), 1)

    results = evaluate(
        record.reply,
        item.expect.model_dump(),
        len(item.context_sources),
        done_reason=record.done_reason,
    )
    record.checks = [
        {"name": r.name, "passed": r.passed, "detail": r.detail, "weak": r.weak}
        for r in results
    ]
    record.passed = item_passed(results)
    return record


# =============================================================================
# Reporting
# =============================================================================
def _percent(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "-"


def build_summary(records: Sequence[RunRecord], meta: dict[str, Any]) -> dict[str, Any]:
    ok = [r for r in records if r.error is None]
    by_category: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        by_category[record.category].append(record)

    check_failures: Counter[str] = Counter()
    check_runs: Counter[str] = Counter()
    weak_flags: Counter[str] = Counter()
    for record in ok:
        for check in record.checks:
            if check["weak"]:
                if not check["passed"]:
                    weak_flags[check["name"]] += 1
                continue
            check_runs[check["name"]] += 1
            if not check["passed"]:
                check_failures[check["name"]] += 1

    latencies = [r.latency_s for r in ok]
    speeds = [r.tokens_per_sec for r in ok if r.tokens_per_sec]
    return {
        "meta": meta,
        "items": len(records),
        "errors": len(records) - len(ok),
        "passed": sum(1 for r in records if r.passed),
        "by_category": {
            cat: {"items": len(rs), "passed": sum(1 for r in rs if r.passed)}
            for cat, rs in sorted(by_category.items())
        },
        "check_failures": {
            name: {"failed": check_failures[name], "ran": check_runs[name]}
            for name in sorted(check_runs)
        },
        "weak_flags": dict(sorted(weak_flags.items())),
        "median_latency_s": round(statistics.median(latencies), 1) if latencies else None,
        "median_tokens_per_sec": round(statistics.median(speeds), 1) if speeds else None,
        "near_context_limit": [
            r.id for r in records if r.prompt_tokens_est > 0.8 * meta["num_ctx"]
        ],
    }


def build_report(records: Sequence[RunRecord], summary: dict[str, Any]) -> str:
    meta = summary["meta"]
    lines = [
        f"# Baseline report: {meta['model']}",
        "",
        f"- Date: {meta['timestamp']}",
        f"- Settings: temperature={meta['temperature']}, seed={meta['seed']}, "
        f"num_ctx={meta['num_ctx']}, max_tokens={meta['max_tokens']}",
        f"- System prompt size: ~{meta['system_prompt_tokens_est']} tokens (estimate)",
        f"- Items run: {summary['items']} (errors: {summary['errors']})",
        f"- **Overall automatic pass rate: {_percent(summary['passed'], summary['items'])}**",
        f"- Median latency: {summary['median_latency_s']} s, "
        f"median speed: {summary['median_tokens_per_sec']} tokens/s",
        "",
        "> The automatic checks cover language, structure, citations and facts only. "
        "Fusha quality and naturalness must be judged in `manual_review.csv`.",
        "",
        "## Pass rate by category",
        "",
        "| Category | Passed |",
        "|---|---|",
    ]
    for category, stats in summary["by_category"].items():
        lines.append(f"| {category} | {_percent(stats['passed'], stats['items'])} |")

    lines += ["", "## Hard-check failures (most frequent first)", ""]
    failures = sorted(
        summary["check_failures"].items(), key=lambda kv: kv[1]["failed"], reverse=True
    )
    shown = [(n, s) for n, s in failures if s["failed"]]
    if shown:
        lines += ["| Check | Failed |", "|---|---|"]
        lines += [f"| {name} | {_percent(s['failed'], s['ran'])} |" for name, s in shown]
    else:
        lines.append("No hard-check failures.")

    lines += ["", "## Advisory flags (heuristics, review manually)", ""]
    if summary["weak_flags"]:
        lines += [f"- {name}: {count} item(s)" for name, count in summary["weak_flags"].items()]
    else:
        lines.append("None.")

    if summary["near_context_limit"]:
        lines += [
            "",
            "## Context-window warning",
            "",
            f"Prompts of {', '.join(summary['near_context_limit'])} use more than 80% of "
            f"num_ctx={meta['num_ctx']}. The system prompt may be truncated; raise `--num-ctx`.",
        ]

    lines += ["", "## Failing items", ""]
    failing = [r for r in records if not r.passed]
    if not failing:
        lines.append("All items passed the automatic checks.")
    for record in failing:
        lines.append(f"### {record.id} ({record.category})")
        lines.append(f"**Question:** {record.question}")
        if record.error:
            lines.append(f"- ERROR: {record.error}")
        for check in record.checks:
            if not check["passed"] and not check["weak"]:
                lines.append(f"- FAILED `{check['name']}`: {check['detail']}")
        preview = " ".join(record.reply.split())[:300]
        if preview:
            lines.append(f"- Reply starts: {preview}…")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_outputs(
    out_dir: Path, records: Sequence[RunRecord], summary: dict[str, Any]
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(build_report(records, summary), encoding="utf-8")

    # utf-8-sig so Excel opens the Arabic text correctly.
    with (out_dir / "manual_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id", "category", "auto_passed", "question", "reply", "reviewer_hint",
                "fusha_1_5", "natural_not_translationese_1_5", "actionable_1_5",
                "faithful_1_5", "comments",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.id, record.category, record.passed, record.question,
                    record.reply or record.error, record.notes, "", "", "", "", "",
                ]
            )


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Score a local Ollama model on the Arabic eval set.")
    parser.add_argument("--model", default=settings.ollama_chat_model,
                        help=f"Ollama model tag (default from .env: {settings.ollama_chat_model})")
    parser.add_argument("--base-url", default=settings.ollama_base_url)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ids", help="Comma-separated item ids, e.g. G01,S02")
    parser.add_argument("--category", choices=[
        "grounded", "no_context", "general", "language", "memory", "limits"])
    parser.add_argument("--limit", type=int, help="Run only the first N selected items")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-ctx", type=int, default=8192,
                        help="Context window; must hold the Arabic system prompt + context + answer")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=settings.ollama_request_timeout * 2)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    eval_set = load_eval_set(args.eval_set)
    items = select_items(eval_set.items, ids=args.ids, category=args.category, limit=args.limit)
    preflight(args.base_url, args.model)

    llm = build_llm(
        args.model, args.base_url, temperature=args.temperature, seed=args.seed,
        num_ctx=args.num_ctx, max_tokens=args.max_tokens, timeout=args.timeout,
    )
    system_tokens = math.ceil(len(SYSTEM_PROMPT_CORE) / _CHARS_PER_TOKEN)
    print(f"Model: {args.model} | items: {len(items)} | num_ctx: {args.num_ctx}")
    print(f"System prompt is roughly {system_tokens} tokens, resent with every request.\n")

    records: list[RunRecord] = []
    for index, item in enumerate(items, start=1):
        record = run_item(llm, item)
        records.append(record)
        status = "ERROR" if record.error else ("PASS " if record.passed else "FAIL ")
        print(f"[{index:>2}/{len(items)}] {item.id:<4} {item.category:<10} {status} {record.latency_s:>6.1f}s")
        if record.error:
            print(f"        {record.error}")

    meta = {
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "temperature": args.temperature,
        "seed": args.seed,
        "num_ctx": args.num_ctx,
        "max_tokens": args.max_tokens,
        "system_prompt_tokens_est": system_tokens,
        "eval_set_version": eval_set.version,
    }
    summary = build_summary(records, meta)

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    out_dir = args.out_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{slug}"
    write_outputs(out_dir, records, summary)

    print(f"\nAutomatic pass rate: {_percent(summary['passed'], summary['items'])}")
    print(f"Results written to: {out_dir}")
    print("Next: open report.md, then score manual_review.csv (Fusha, naturalness, usefulness).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
