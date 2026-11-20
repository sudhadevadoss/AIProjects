"""End-to-end pipeline: load -> classify -> sentiment -> insights -> charts + CSV outputs.

    python run_pipeline.py                 # uses ../dataset
    python run_pipeline.py --dataset PATH  # any folder of meeting folders in the same format
    python run_pipeline.py --llm           # also run the Claude-based classifier (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import pandas as pd

from ti import charts, insights as I, sentiment as S
from ti.classify import classify, discover_topics, evaluate
from ti.load import DEFAULT_DATASET, load

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).parent


def run(dataset: Path, out: Path, use_llm: bool = False) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    step = lambda msg: print(f"[{time.time() - t0:5.1f}s] {msg}")

    step(f"Loading meetings from {dataset}")
    t = load(dataset)
    step(f"  {len(t['meetings'])} meetings, {len(t['sentences']):,} sentences, {len(t['key_moments'])} key moments")

    step("Classifying: call type -> products -> themes")
    m = classify(t)
    step("  call types: " + ", ".join(f"{k}={v}" for k, v in m.call_type.value_counts().items()))
    acc, ev = evaluate(m, ROOT / "data" / "gold_labels.csv")
    step(f"  primary-theme agreement with hand labels: {acc:.0%} | gold label within multi-label set: {ev.in_multilabel.mean():.0%}")
    m["cluster"], clusters = discover_topics(m)

    if use_llm:
        from ti.llm_classify import classify_with_llm
        step("Running LLM classifier (Claude)")
        llm = classify_with_llm(m, cache=out / "llm_cache.json")
        m = m.merge(llm, on="meeting_id", how="left")
        step(f"  LLM vs rules primary agreement: {(m.llm_primary_theme == m.primary_theme).mean():.0%}")

    step("Scoring sentiment (vendor labels + VADER cross-check, split by speaker side)")
    s = S.score_sentences(t["sentences"])
    m = S.meeting_sentiment(m, s)
    agree = S.label_agreement(s)
    tables = S.summary_tables(m)

    step("Generating stakeholder insights")
    km = t["key_moments"]
    health = I.account_health(m, km)
    ew = I.early_warnings(m, s)
    retro = I.retro_mentions_of_known_risk(m, s)
    fallout = I.remediation_fallout(m)
    versions = I.version_hotspots(m)
    fg_summary, fg_detail = I.feature_gap_backlog(m, km)
    comp = I.competitor_intel(m, s)
    billing = I.billing_friction(m)
    owners = I.owner_load(m, s, health)

    step("Writing CSVs and charts")
    cols = ["meeting_id", "start", "title", "account", "call_type", "primary_theme", "theme_label", "themes",
            "theme_margin", "primary_product", "products", "competitors", "vendor_sentiment_score", "nsi_customer",
            "nsi_internal", "cust_arc_delta", "phase", "cluster", "summary"]
    if use_llm:
        cols += [c for c in m.columns if c.startswith("llm_")]
    m[cols].to_csv(out / "meetings_enriched.csv", index=False)
    ev.to_csv(out / "theme_eval_vs_gold.csv", index=False)
    clusters.to_csv(out / "discovered_clusters.csv", index=False)
    for name, df in {**tables, "health": health, "early_warnings": ew, "known_risk_admissions": retro,
                     "remediation_fallout": fallout, "version_hotspots": versions, "feature_gaps_by_product": fg_summary,
                     "feature_gaps_detail": fg_detail, "competitors": comp, "billing_friction": billing,
                     "owner_load": owners}.items():
        df.to_csv(out / f"{name}.csv", index=name.startswith("by_"))
    charts.make_all(m, health, ew, fallout, comp, out / "charts")

    headline = {
        "meetings": len(m),
        "call_types": m.call_type.value_counts().to_dict(),
        "theme_accuracy_vs_gold": round(acc, 3),
        "vendor_vs_vader_direction_agreement": round(agree["direction_agreement"], 3),
        "sentiment_by_call_type": tables["by_type"]["vendor_score"].to_dict(),
        "at_risk_accounts": health[health.tier == "At risk"].account.tolist(),
        "early_warning_lead_days": int(ew.days_before.max()) if len(ew) else None,
        "remediation_fallout_calls": len(fallout),
        "preventable_known_bug_calls": int(versions.preventable.sum()) if len(versions) else 0,
    }
    (out / "headline.json").write_text(json.dumps(headline, indent=2, default=str))
    step(f"Done -> {out}")
    return {"m": m, "s": s, "t": t, "health": health, "headline": headline}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    ap.add_argument("--llm", action="store_true")
    a = ap.parse_args()
    r = run(a.dataset, a.out, a.llm)
    print(json.dumps(r["headline"], indent=2, default=str))
