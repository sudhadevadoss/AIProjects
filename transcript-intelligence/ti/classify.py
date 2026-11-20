"""Categorisation pipeline: call type -> products -> themes (+ unsupervised discovery for validation).

Approach = HYBRID
  * call_type : deterministic rules on metadata (participant email domains + title). Cheap, explainable, ~100% accurate
                because the signal (who is on the call) is structural, not linguistic.
  * products  : lexicon counts over the transcript. Products are proper nouns -> regex is enough.
  * themes    : weighted keyword scoring over title, vendor summary, vendor topics, key-moment types and the raw
                transcript, multiplied by a call-type prior. Multi-label (a renewal call can also be a churn-risk call)
                with one primary theme for routing / dashboards.
  * discovery : TF-IDF + NMF clustering on transcripts, used to sanity-check that the taxonomy covers what is
                actually in the data (and to find themes we didn't think of).
  * LLM       : `ti.llm_classify` runs the same taxonomy through Claude for production-scale / no-summary data.

Every assignment keeps its evidence (which rules fired), so a reviewer can see *why* a call landed in a bucket.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from .taxonomy import CALL_TYPE_PRIOR, COMPETITORS, PRODUCTS, THEMES

SUPPORT_TITLE = re.compile(r"support case|^urgent:|^escalation:", re.I)
SUPPORT_SUMMARY = re.compile(r"(contacted|called|reached out to) (aegis( cloud security)? )?support", re.I)

KM_THEME_BOOST = {  # vendor key-moment type -> theme evidence
    "churn_signal": ("CHURN_COMPETITIVE", 1.5),
    "feature_gap": ("FEATURE_GAP", 1.5),
    "technical_issue": ("PRODUCT_DEFECT", 1.0),
    "pricing_offer": ("RENEWAL_EXPANSION", 1.5),
}


# ---------------------------------------------------------------- call type
def call_type(row: pd.Series) -> tuple[str, str]:
    if row["n_customer"] == 0:
        return "internal", "no external participants"
    if SUPPORT_TITLE.search(row["title"]):
        return "support", "title matches support/urgent/escalation pattern"
    if SUPPORT_SUMMARY.search(row["summary"]):
        return "support", "summary says customer contacted support"
    return "external", "external participants, account-management style title"


# ---------------------------------------------------------------- products
def products(text: str) -> dict[str, int]:
    t = text.lower()
    return {p: sum(len(re.findall(rx, t)) for rx in rxs) for p, rxs in PRODUCTS.items()}


def competitor_mentions(text: str) -> dict[str, int]:
    return {c: len(re.findall(re.escape(c), text, re.I)) for c in COMPETITORS}


# ---------------------------------------------------------------- themes
def _kw_score(text: str, kws: dict[str, float]) -> tuple[float, list[str]]:
    score, hits = 0.0, []
    t = text.lower()
    for rx, w in kws.items():
        n = len(re.findall(rx, t))
        if n:
            score += w * (1 + math.log(n))  # diminishing returns on repeats
            hits.append(rx)
    return score, hits


def theme_scores(row: pd.Series, km_types: list[str]) -> tuple[dict[str, float], dict[str, list[str]]]:
    topics_text = " | ".join(row["topics"])
    scores, evidence = {}, {}
    for key, spec in THEMES.items():
        ev = []
        s_title = sum(5 for rx in spec["title"] if re.search(rx, row["title"], re.I))
        if s_title:
            ev.append(f"title:{s_title}")
        s_sum, h1 = _kw_score(row["summary"], spec["keywords"])
        s_top, h2 = _kw_score(topics_text, spec["keywords"])
        s_tr, _ = _kw_score(row["transcript_text"], spec["keywords"])
        s_km = sum(w for t in km_types for th, w in [KM_THEME_BOOST.get(t, ("", 0))] if th == key)
        total = s_title + 1.0 * s_sum + 1.5 * s_top + 0.15 * s_tr + s_km
        total *= CALL_TYPE_PRIOR[row["call_type"]][key]
        scores[key] = round(total, 2)
        evidence[key] = ev + [f"summary:{h}" for h in h1[:3]] + [f"topic:{h}" for h in h2[:2]]
    return scores, evidence


def classify(tables: dict[str, pd.DataFrame], multi_label_ratio: float = 0.5) -> pd.DataFrame:
    m = tables["meetings"].copy()
    km = tables["key_moments"].groupby("meeting_id")["km_type"].apply(list).to_dict()

    ct = m.apply(call_type, axis=1, result_type="expand")
    m["call_type"], m["call_type_reason"] = ct[0], ct[1]

    prods = m["transcript_text"].apply(products)
    for p in PRODUCTS:
        m[f"mentions_{p}"] = prods.apply(lambda d: d[p])
    # primary product = most mentioned; secondary products need >= 25% of the top count
    m["primary_product"] = prods.apply(lambda d: max(d, key=d.get) if max(d.values()) > 0 else None)
    m["products"] = prods.apply(lambda d: [p for p, c in d.items() if c and c >= 0.25 * max(d.values())])

    comps = m["transcript_text"].apply(competitor_mentions)
    m["competitors"] = comps.apply(lambda d: [c for c, n in d.items() if n])

    prim, labels, conf, evid, all_scores = [], [], [], [], []
    for _, row in m.iterrows():
        sc, ev = theme_scores(row, km.get(row["meeting_id"], []))
        ranked = sorted(sc.items(), key=lambda kv: -kv[1])
        top, top_s = ranked[0]
        second_s = ranked[1][1] if len(ranked) > 1 else 0
        prim.append(top)
        labels.append([k for k, s in ranked if s >= multi_label_ratio * top_s and s > 0])
        conf.append(round((top_s - second_s) / top_s, 2) if top_s else 0.0)  # margin-based confidence
        evid.append(ev[top])
        all_scores.append(sc)
    m["primary_theme"] = prim
    m["theme_label"] = m["primary_theme"].map(lambda k: THEMES[k]["label"])
    m["themes"] = labels
    m["theme_margin"] = conf
    m["theme_evidence"] = evid
    m["theme_scores"] = all_scores
    return m


# ---------------------------------------------------------------- discovery (unsupervised)
def discover_topics(m: pd.DataFrame, k: int = 9, seed: int = 7) -> tuple[pd.Series, pd.DataFrame]:
    """TF-IDF + NMF over transcripts. Returns (cluster per meeting, top terms per cluster).

    NMF over TF-IDF is a good fit for ~100 short documents: deterministic with a seed, gives interpretable
    additive "parts", and needs no GPU / embeddings service. We use it to *validate* the taxonomy, not to drive it.
    """
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

    stop = list(ENGLISH_STOP_WORDS | {
        "yeah", "okay", "like", "just", "know", "think", "really", "right", "going", "gonna", "um", "uh",
        "aegis", "sure", "got", "thing", "things", "want", "make", "mean", "let", "actually", "need", "good",
        "great", "way", "lot", "kind", "sort", "said", "say", "come", "look", "time", "thanks", "yes",
    })
    vec = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=3, max_df=0.6, sublinear_tf=True)
    X = vec.fit_transform(m["transcript_text"])
    nmf = NMF(n_components=k, random_state=seed, init="nndsvda", max_iter=600)
    W = nmf.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    top = pd.DataFrame(
        {"cluster": range(k), "top_terms": [", ".join(terms[np.argsort(-c)[:8]]) for c in nmf.components_]}
    )
    return pd.Series(W.argmax(1), index=m.index, name="cluster"), top


def evaluate(m: pd.DataFrame, gold_path) -> tuple[float, pd.DataFrame]:
    gold = pd.read_csv(gold_path)
    key = m["start"].dt.strftime("%Y-%m-%d") + "|" + m["title"]
    gold_map = dict(zip(gold["date"] + "|" + gold["title"], gold["primary_theme"]))
    df = pd.DataFrame({"title": m["title"], "pred": m["primary_theme"], "gold": key.map(gold_map),
                       "themes": m["themes"]})
    df = df.dropna(subset=["gold"])
    df["primary_match"] = df["pred"] == df["gold"]
    df["in_multilabel"] = df.apply(lambda r: r["gold"] in r["themes"], axis=1)
    return df["primary_match"].mean(), df
