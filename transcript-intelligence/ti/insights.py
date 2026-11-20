"""Beyond topic + sentiment: stakeholder-specific insights.

Each function answers one question for one stakeholder and returns a DataFrame with *evidence* (quotes / titles),
because an insight a leader can't verify is an insight they won't act on.

  account_health         CS / Sales leaders  : which accounts are at risk, and why?
  early_warnings         Eng leaders         : did we see it coming? (internal risk talk before customer pain)
  remediation_fallout    Eng / Support       : did our fixes create new customer problems?
  version_hotspots       Eng / QA            : which releases keep showing up in support calls?
  feature_gap_backlog    Product managers    : what are customers asking for, weighted by who's asking?
  competitor_intel       Sales / PMM         : who are we losing to, on which product, and why?
  billing_friction       Finance / Product   : are billing disputes a product problem?
  owner_load             Ops / CS leadership : who is carrying the at-risk relationships?
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .taxonomy import COMPETITORS, OUTAGE_START


# ------------------------------------------------------------------ account health
def account_health(m: pd.DataFrame, km: pd.DataFrame) -> pd.DataFrame:
    """Transparent, additive risk score per customer account (0-100). Not ML on purpose: 32 accounts and no
    churn labels -> a weighted rubric a CS leader can argue with beats an opaque model."""
    ext = m[m.account.notna()].sort_values("start")
    churn_q = (km[km.km_type == "churn_signal"].merge(m[["meeting_id", "account", "start"]], on="meeting_id")
               .sort_values("start").groupby("account")["km_text"].last())
    churn_n = km[km.km_type == "churn_signal"].merge(m[["meeting_id", "account"]], on="meeting_id").groupby("account").size()

    rows = []
    for acct, g in ext.groupby("account"):
        last = g.iloc[-1]
        text = " ".join(g.transcript_text)
        comp_counts = {c: len(re.findall(re.escape(c), text, re.I)) for c in COMPETITORS}
        comps = [c for c, n in sorted(comp_counts.items(), key=lambda kv: -kv[1]) if n]  # most-mentioned first
        recent = g[g.start >= g.start.max() - pd.Timedelta(days=45)]
        active_eval = bool(re.search(r"(actively|currently) evaluat|competitive (evaluation|quote)|exploratory conversations",
                                     " ".join(recent.summary), re.I))
        rows.append({
            "account": acct,
            "calls": len(g),
            "support_calls": int((g.call_type == "support").sum()),
            "first_score": g.vendor_sentiment_score.iloc[0],
            "last_score": last.vendor_sentiment_score,
            "trend": round(last.vendor_sentiment_score - g.vendor_sentiment_score.iloc[0], 1) if len(g) > 1 else 0.0,
            "last_call": last.start.date(),
            "last_title": last.title,
            "competitors_named": ", ".join(comps),
            "active_competitive_eval": active_eval,
            "churn_signals": int(churn_n.get(acct, 0)),
            "hit_by_outage": bool(g[g.start >= OUTAGE_START].transcript_text.str.contains(r"outage", case=False).any()),
            "evidence": churn_q.get(acct, ""),
        })
    df = pd.DataFrame(rows)
    # rubric: each component 0..1, weighted
    df["risk_score"] = (
        30 * ((5 - df.last_score) / 4).clip(0, 1)            # how unhappy were they last time we spoke
        + 25 * df.active_competitive_eval                     # explicit evaluation of an alternative
        + 15 * (df.churn_signals / 3).clip(0, 1)              # vendor-flagged churn moments
        + 10 * (df.support_calls / 3).clip(0, 1)              # support burden
        + 10 * (-df.trend / 2).clip(0, 1)                     # getting worse
        + 10 * df.hit_by_outage
    ).round(0)
    df["tier"] = pd.cut(df.risk_score, [-1, 35, 55, 101], labels=["Healthy", "Watch", "At risk"])
    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ early warning
RISK_PATTERNS = r"single point of failure|tech(?:nical)? debt|no redundancy|not prioriti[sz]ed|(?:bigger|real) risk|worried|concern(?:ed)? about|failover|we got lucky"


def early_warnings(m: pd.DataFrame, s: pd.DataFrame, incident_date: str = OUTAGE_START) -> pd.DataFrame:
    """Internal sentences that named a risk BEFORE the incident hit customers."""
    internal_before = m[(m.call_type == "internal") & (m.start < incident_date)][["meeting_id", "title", "start"]]
    hits = s.merge(internal_before, on="meeting_id")
    hits = hits[hits.text.str.contains(RISK_PATTERNS, case=False, regex=True)
                & hits.text.str.contains(r"pipeline|ingestion|node|detect|redundan|failover", case=False)]
    hits = hits.assign(days_before=(pd.Timestamp(incident_date) - hits.start).dt.days)
    return hits[["start", "title", "speaker", "text", "days_before"]].sort_values("start").reset_index(drop=True)


def retro_mentions_of_known_risk(m: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """After the incident: did people acknowledge it was a known/flagged issue?"""
    after = m[m.start >= OUTAGE_START][["meeting_id", "title", "start", "call_type"]]
    hits = s.merge(after, on="meeting_id")
    hits = hits[hits.text.str.contains(r"known (?:tech debt|risk)|known architectural|had been flagged|flagged it|was on the backlog|deprioriti[sz]ed\b(?! tuning)|we knew it", case=False)]
    return hits[["start", "title", "call_type", "speaker", "text"]].sort_values("start").reset_index(drop=True)


# ------------------------------------------------------------------ remediation fallout
def remediation_fallout(m: pd.DataFrame) -> pd.DataFrame:
    """Support calls *after* the outage whose root cause is attributed to the outage remediation work."""
    sup = m[(m.call_type == "support") & (m.start >= "2026-03-19")]
    pat = r"remediation|post-outage|circuit breaker|after the (?:march|outage)|mid-march|rebalanc|overloaded (?:processing )?(?:node|cluster)|shared cluster"
    hit = sup[sup.summary.str.contains(pat, case=False) | sup.transcript_text.str.contains(r"remediation (build|work)|circuit breaker|post-outage", case=False)]
    return hit[["start", "account", "title", "primary_product", "vendor_sentiment_score", "summary"]].reset_index(drop=True)


def version_hotspots(m: pd.DataFrame) -> pd.DataFrame:
    """Version strings cited in support calls -> recurring bad releases, and *preventable* calls
    (customer hit a known bug for which a fix had already shipped)."""
    sup = m[m.call_type == "support"]
    rows = []
    for _, r in sup.iterrows():
        text = r.summary + " " + r.transcript_text
        versions = sorted(set(re.findall(r"\b\d+\.\d+\.\d+\b", text)))
        if not versions:
            continue
        rows.append({
            "start": r.start.date(), "account": r.account, "title": r.title, "product": r.primary_product,
            "versions_cited": ", ".join(versions), "running_version": versions[0],
            "known_bug": bool(re.search(r"known (?:[\w-]+ ){0,3}(?:bug|issue)", r.summary, re.I)),
            # a newer version than the one the customer runs is cited as the fix
            "fix_already_shipped": len(versions) > 1 and "4.1.2" not in versions,
            "remediation_build": "4.1.2" in versions,
            "vendor_sentiment_score": r.vendor_sentiment_score,
        })
    df = pd.DataFrame(rows)
    df["preventable"] = df.known_bug & df.fix_already_shipped
    return df


# ------------------------------------------------------------------ product
def feature_gap_backlog(m: pd.DataFrame, km: pd.DataFrame) -> pd.DataFrame:
    fg = km[km.km_type == "feature_gap"].merge(
        m[["meeting_id", "account", "call_type", "primary_product", "start", "vendor_sentiment_score"]], on="meeting_id")
    fg = fg[fg.account.notna()]
    by_prod = fg.groupby("primary_product").agg(
        requests=("km_text", "count"), accounts=("account", "nunique"),
        avg_sentiment_of_asking_calls=("vendor_sentiment_score", "mean"),
        examples=("km_text", lambda x: " || ".join(list(x)[:3])),
    ).round(2).sort_values("accounts", ascending=False)
    return by_prod.reset_index(), fg[["start", "account", "primary_product", "km_text"]].sort_values("start")


def competitor_intel(m: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in COMPETITORS:
        hit = s[s.text.str.contains(re.escape(c), case=False)].merge(
            m[["meeting_id", "call_type", "account", "start", "primary_product", "title"]], on="meeting_id")
        if hit.empty:
            continue
        rows.append({
            "competitor": c,
            "mentions": len(hit),
            "calls": hit.meeting_id.nunique(),
            "customer_calls": hit[hit.call_type != "internal"].meeting_id.nunique(),
            "accounts": ", ".join(sorted(hit.account.dropna().unique())),
            "products_in_context": ", ".join(hit.primary_product.value_counts().index[:3]),
            "post_outage_share": round((hit.start >= OUTAGE_START).mean(), 2),
            "sample_quote": hit[hit.side == "customer"].text.iloc[0] if (hit.side == "customer").any() else hit.text.iloc[0],
        })
    return pd.DataFrame(rows).sort_values("mentions", ascending=False).reset_index(drop=True)


def billing_friction(m: pd.DataFrame) -> pd.DataFrame:
    """Billing calls + renewal calls that discuss overages: what caused them?"""
    b = m[(m.primary_theme == "BILLING") | m.summary.str.contains(r"overage charges", case=False)]
    cause = b.summary.str.extract(
        r"(test accounts|policy change|new (production )?facilities|no knowledge of plan limits|no in-app alerts|proration|pre-compression|seat|tier)",
        flags=re.I)[0]
    out = b[["start", "account", "title", "call_type", "vendor_sentiment_score", "cust_arc_delta"]].copy()
    out["root_cause_hint"] = cause.fillna("")
    out["customer_unaware_of_usage"] = b.summary.str.contains(r"no knowledge|no in-app|never deprovisioned|unaware|without (?:any )?(?:warning|alert)|traced", case=False)
    return out.reset_index(drop=True)


def owner_load(m: pd.DataFrame, s: pd.DataFrame, health: pd.DataFrame) -> pd.DataFrame:
    """Which Aegis people are in the room on calls with at-risk accounts? (key-person risk)"""
    at_risk = set(health[health.tier == "At risk"].account)
    ext = m[m.account.isin(at_risk)][["meeting_id", "account"]]
    ppl = s[s.side == "internal"].merge(ext, on="meeting_id").groupby("speaker").agg(
        at_risk_calls=("meeting_id", "nunique"), at_risk_accounts=("account", "nunique"))
    return ppl.sort_values("at_risk_calls", ascending=False).reset_index()


def talk_ratio(m: pd.DataFrame) -> pd.DataFrame:
    e = m[m.call_type == "external"].copy()
    e["customer_talk_share"] = e.talk_sec_customer / (e.talk_sec_customer + e.talk_sec_internal)
    e["talk_bucket"] = pd.cut(e.customer_talk_share, [0, 0.4, 0.55, 1], labels=["rep-dominated (<40%)", "balanced", "customer-led (>55%)"])
    return e
