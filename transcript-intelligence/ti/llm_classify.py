"""OPTIONAL: classify calls with Claude using the same taxonomy as the rule-based classifier.

Why keep this optional?
  * The rule-based classifier already reaches 87% primary / 99% multi-label agreement on this dataset, runs in <1s,
    is free, deterministic and explainable -- the right default for 100 calls.
  * In production (thousands of calls/day, new products, transcripts *without* vendor summaries) keyword lists rot.
    The LLM path is the scale-out plan: same taxonomy, structured JSON output, a one-line rationale per call,
    and a cache so each transcript is only paid for once.
  * Running both lets us measure disagreement -- calls where rules and LLM disagree are the ones worth a human look.

Usage:  ANTHROPIC_API_KEY=... python run_pipeline.py --llm
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .taxonomy import THEMES

MODEL = "claude-opus-5-5"
MAX_TRANSCRIPT_CHARS = 60_000  # ~15k tokens; every transcript in this dataset fits, nothing is truncated

SCHEMA = {
    "type": "object",
    "properties": {
        "primary_theme": {"type": "string", "enum": list(THEMES)},
        "secondary_themes": {"type": "array", "items": {"type": "string", "enum": list(THEMES)}},
        "customer_sentiment": {"type": "integer", "description": "1 (very negative) to 5 (very positive), customer voice only"},
        "churn_risk": {"type": "string", "enum": ["none", "low", "medium", "high"]},
        "rationale": {"type": "string", "description": "One sentence citing what in the call drove the theme choice."},
    },
    "required": ["primary_theme", "secondary_themes", "customer_sentiment", "churn_risk", "rationale"],
    "additionalProperties": False,
}

SYSTEM = (
    "You categorise call transcripts for a B2B security SaaS company (Aegis Cloud; products Detect, Protect, "
    "Comply, Identity, LogVault). Pick the single theme that best describes why the call happened, plus any "
    "secondary themes that take up a meaningful part of the call.\n\nThemes:\n"
    + "\n".join(f"- {k}: {v['label']} - {v['desc']}" for k, v in THEMES.items())
)


def _prompt(row: pd.Series) -> str:
    text = row["transcript_text"]
    if len(text) > MAX_TRANSCRIPT_CHARS:
        raise ValueError(f"{row['meeting_id']}: transcript longer than {MAX_TRANSCRIPT_CHARS} chars; chunk it first")
    return f"Call type: {row['call_type']}\nTitle: {row['title']}\n\nTranscript:\n{text}"


def classify_with_llm(m: pd.DataFrame, cache: Path) -> pd.DataFrame:
    import anthropic

    client = anthropic.Anthropic()
    done = json.loads(cache.read_text()) if cache.exists() else {}
    for _, row in m.iterrows():
        mid = row["meeting_id"]
        if mid in done:
            continue
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=2048,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # if a safety classifier declines, re-run on a fallback model inside the same call
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            system=SYSTEM,
            messages=[{"role": "user", "content": _prompt(row)}],
        )
        if resp.stop_reason == "refusal":
            print(f"  refused: {mid}")
            continue
        text = next(b.text for b in resp.content if b.type == "text")
        done[mid] = json.loads(text)
        cache.write_text(json.dumps(done, indent=1))  # checkpoint after every call
    rows = [{"meeting_id": k, **{f"llm_{f}": v for f, v in d.items()}} for k, d in done.items()]
    return pd.DataFrame(rows)
