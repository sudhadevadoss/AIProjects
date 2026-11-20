"""Sentiment analysis.

Three signals, deliberately kept separate:
  1. vendor meeting score   summary.json sentimentScore (1-5). Holistic, but a black box.
  2. vendor sentence labels transcript.json sentimentType (pos / neu / neg) per sentence.
  3. VADER                  an independent, open lexicon model re-scored on every sentence -- lets us
                            check whether the vendor labels are trustworthy before building on them.

Key design choice: split every sentence by *side* (customer vs Aegis employee). Aegis reps are paid to sound
positive ("Absolutely, happy to help!"), which dilutes call-level sentiment. The customer's voice is the signal.

Net sentiment index (NSI) = (#positive - #negative) / #sentences, in [-1, 1].
"""
from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .taxonomy import PHASES

_vader = SentimentIntensityAnalyzer()


def vader_label(text: str) -> tuple[float, str]:
    c = _vader.polarity_scores(text)["compound"]
    return c, ("positive" if c >= 0.35 else "negative" if c <= -0.2 else "neutral")


def score_sentences(sent: pd.DataFrame) -> pd.DataFrame:
    s = sent.copy()
    v = s["text"].apply(vader_label)
    s["vader_compound"] = v.str[0]
    s["vader_label"] = v.str[1]
    s["vendor_num"] = s["vendor_label"].map({"positive": 1, "neutral": 0, "negative": -1})
    s["third"] = pd.cut(s["pos"], [-0.01, 1 / 3, 2 / 3, 1.01], labels=["open", "middle", "close"])
    return s


def nsi(labels: pd.Series) -> float:
    n = len(labels)
    return ((labels == "positive").sum() - (labels == "negative").sum()) / n if n else float("nan")


def meeting_sentiment(m: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """Adds per-meeting sentiment features to the meetings table."""
    g = s.groupby("meeting_id")
    out = pd.DataFrame(
        {
            "nsi_all": g["vendor_label"].apply(nsi),
            "nsi_vader": g["vader_label"].apply(nsi),
            "neg_share": g["vendor_label"].apply(lambda x: (x == "negative").mean()),
        }
    )
    for side in ("customer", "internal"):
        out[f"nsi_{side}"] = s[s.side == side].groupby("meeting_id")["vendor_label"].apply(nsi)
    # within-call arc: customer sentiment at close minus at open ("did we land the call?")
    cust = s[s.side == "customer"]
    arc = cust.groupby(["meeting_id", "third"], observed=False)["vendor_label"].apply(nsi).unstack()
    out["cust_open"] = arc["open"]
    out["cust_close"] = arc["close"]
    out["cust_arc_delta"] = arc["close"] - arc["open"]
    m = m.merge(out, left_on="meeting_id", right_index=True, how="left")
    m["phase"] = m["date"].apply(phase_of)
    return m


def phase_of(d: pd.Timestamp) -> str:
    for name, lo, hi in PHASES:
        if (lo is None or d >= pd.Timestamp(lo)) and (hi is None or d <= pd.Timestamp(hi)):
            return name
    return "?"


def label_agreement(s: pd.DataFrame) -> dict:
    """How often does an independent model agree with the vendor sentence labels?"""
    agree = (s["vendor_label"] == s["vader_label"]).mean()
    # 'direction' agreement ignores neutral-vs-x disagreements, which are mostly threshold noise
    polar = s[(s.vendor_label != "neutral") & (s.vader_label != "neutral")]
    direction = (polar.vendor_label == polar.vader_label).mean()
    ct = pd.crosstab(s["vendor_label"], s["vader_label"], normalize="index").round(2)
    corr = s[["vendor_num", "vader_compound"]].corr().iloc[0, 1]
    return {"exact_agreement": agree, "direction_agreement": direction, "corr": corr, "crosstab": ct}


def summary_tables(m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_type = m.groupby("call_type").agg(
        calls=("meeting_id", "count"),
        vendor_score=("vendor_sentiment_score", "mean"),
        nsi_customer=("nsi_customer", "mean"),
        nsi_internal=("nsi_internal", "mean"),
        nsi_all=("nsi_all", "mean"),
        cust_arc_delta=("cust_arc_delta", "mean"),
    ).round(2)
    by_phase = m.pivot_table(index="phase", columns="call_type", values="vendor_sentiment_score", aggfunc="mean").round(2)
    by_phase_n = m.pivot_table(index="phase", columns="call_type", values="meeting_id", aggfunc="count")
    by_theme = m.groupby("theme_label").agg(
        calls=("meeting_id", "count"), vendor_score=("vendor_sentiment_score", "mean"),
        nsi_customer=("nsi_customer", "mean"), cust_arc_delta=("cust_arc_delta", "mean"),
    ).round(2).sort_values("vendor_score")
    by_product = m.groupby("primary_product").agg(
        calls=("meeting_id", "count"), vendor_score=("vendor_sentiment_score", "mean"),
        nsi_customer=("nsi_customer", "mean"),
    ).round(2).sort_values("vendor_score")
    return {"by_type": by_type, "by_phase": by_phase, "by_phase_n": by_phase_n, "by_theme": by_theme, "by_product": by_product}
