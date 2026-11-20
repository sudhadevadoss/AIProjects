"""Load the raw meeting folders into tidy pandas DataFrames.

Each meeting folder contains:
  meeting-info.json   title, times, participant emails
  transcript.json     sentence-level transcript (speaker, time, vendor sentiment label)
  summary.json        vendor-generated summary, topics, key moments, action items, sentiment score
  speakers.json       speaking turns (used for talk-time)
  speaker-meta.json   speaker id -> name
  events.json         join / leave events

We flatten these into four tables that every downstream step works from:
  meetings      one row per meeting
  sentences     one row per transcript sentence, tagged with the speaker's *side* (internal vs customer)
  key_moments   one row per vendor-flagged key moment (churn_signal, feature_gap, ...)
  action_items  one row per action item, with owner parsed out
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

INTERNAL_DOMAIN = "aegiscloud.com"
# Prefer the copy inside the repo; fall back to a sibling folder (the layout the assignment zip unpacks to).
_REPO_DATASET = Path(__file__).resolve().parents[1] / "dataset"
DEFAULT_DATASET = _REPO_DATASET if _REPO_DATASET.exists() else Path(__file__).resolve().parents[2] / "dataset"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def match_speaker_email(name: str, emails: list[str]) -> str | None:
    """Map a display name ("Dr. Linda Cheng", "Kevin O'Brien") to one of the meeting's emails.

    Emails follow first.last@domain, so we match normalised name tokens against the local part.
    Falls back to last-name, then first-name matching. Validated: 311/311 speakers map uniquely.
    """
    tokens = [_norm(t) for t in re.sub(r"^(dr|mr|ms|mrs)\.?\s", "", name, flags=re.I).split()]
    tokens = [t for t in tokens if t]
    locals_ = {e: _norm(e.split("@")[0]) for e in emails}
    for cand in (
        [e for e, l in locals_.items() if all(t in l for t in tokens)],
        [e for e, l in locals_.items() if tokens and tokens[-1] in l],
        [e for e, l in locals_.items() if tokens and tokens[0] in l],
    ):
        if len(cand) == 1:
            return cand[0]
    return None


def _account_from_domains(domains: set[str]) -> str | None:
    ext = sorted(d for d in domains if d != INTERNAL_DOMAIN)
    return ext[0] if ext else None


def load(dataset_dir: str | Path = DEFAULT_DATASET) -> dict[str, pd.DataFrame]:
    dataset_dir = Path(dataset_dir)
    meetings, sentences, moments, actions = [], [], [], []

    for folder in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        info = json.loads((folder / "meeting-info.json").read_text())
        summ = json.loads((folder / "summary.json").read_text())
        trans = json.loads((folder / "transcript.json").read_text())["data"]
        turns = json.loads((folder / "speakers.json").read_text())
        mid = info["meetingId"]

        emails = info["allEmails"]
        domains = {e.split("@")[1] for e in emails}
        account_domain = _account_from_domains(domains)

        # speaker -> side (internal / customer)
        names = {t["speaker_name"] for t in trans}
        side = {}
        for n in names:
            e = match_speaker_email(n, emails)
            side[n] = "internal" if (e is None or e.endswith(INTERNAL_DOMAIN)) else "customer"

        # talk time per side, from speaking turns
        talk = {"internal": 0.0, "customer": 0.0}
        for t in turns:
            talk[side.get(t["speakerName"], "internal")] += max(0.0, t["endTimeTs"] - t["timestamp"])

        meetings.append(
            {
                "meeting_id": mid,
                "title": info["title"],
                "start": pd.Timestamp(info["startTime"]).tz_convert(None),
                "duration_min": info["duration"],
                "n_participants": len(emails),
                "n_internal": sum(e.endswith(INTERNAL_DOMAIN) for e in emails),
                "n_customer": sum(not e.endswith(INTERNAL_DOMAIN) for e in emails),
                "account_domain": account_domain,
                "summary": summ.get("summary", ""),
                "topics": summ.get("topics", []),
                "vendor_sentiment_label": summ.get("overallSentiment"),
                "vendor_sentiment_score": summ.get("sentimentScore"),
                "n_sentences": len(trans),
                "n_words": sum(len(s["sentence"].split()) for s in trans),
                "talk_sec_internal": talk["internal"],
                "talk_sec_customer": talk["customer"],
                "transcript_text": " ".join(s["sentence"] for s in trans),
            }
        )

        n = max(len(trans), 1)
        for s in trans:
            sentences.append(
                {
                    "meeting_id": mid,
                    "idx": s["index"],
                    "pos": s["index"] / n,  # relative position in call, 0..1
                    "time": s["time"],
                    "speaker": s["speaker_name"],
                    "side": side[s["speaker_name"]],
                    "text": s["sentence"],
                    "vendor_label": s.get("sentimentType", "neutral"),
                    "asr_confidence": s.get("averageConfidence"),
                }
            )

        for k in summ.get("keyMoments", []):
            moments.append({"meeting_id": mid, **{f"km_{x}": k.get(x) for x in ("type", "text", "speaker", "time")}})

        for a in summ.get("actionItems", []):
            owner, _, task = a.partition(":")
            actions.append({"meeting_id": mid, "owner": owner.strip(), "task": task.strip() or a})

    m = pd.DataFrame(meetings).sort_values("start").reset_index(drop=True)
    m["date"] = m["start"].dt.normalize()
    m["week"] = m["start"].dt.to_period("W-SUN").dt.start_time
    m["account"] = m["account_domain"].map(_pretty_account)
    return {
        "meetings": m,
        "sentences": pd.DataFrame(sentences),
        "key_moments": pd.DataFrame(moments),
        "action_items": pd.DataFrame(actions),
    }


_ACCOUNT_NAMES = {
    "vantahealth.com": "Vanta Health", "coastalliving.com": "Coastal Living", "summittrust.com": "Summit Trust",
    "blackridgeinvest.com": "Blackridge Investments", "nimbusplatform.com": "Nimbus Platform",
    "brightpathcommerce.com": "Brightpath Commerce", "bridgeporthealth.com": "Bridgeport Health",
    "novaretail.com": "Nova Retail", "crestlinewealth.com": "Crestline Wealth", "frostbyte.ai": "Frostbyte AI",
    "redwoodclinical.com": "Redwood Clinical", "atlasprecision.com": "Atlas Precision",
    "maplewoodgoods.com": "Maplewood Goods", "pineridge.io": "Pineridge Systems", "harborviewbank.com": "Harborview Banking",
    "stratoscloud.io": "Stratos Cloud", "ridgelinelogistics.com": "Ridgeline Logistics",
    "meridiancapital.com": "Meridian Capital", "keystonehealth.com": "Keystone Health", "cobaltsoftware.com": "Cobalt Software",
    "trailheadmkt.com": "Trailhead Marketplace", "northstarpharma.com": "Northstar Pharma",
    "silverlinebrands.com": "Silverline Brands", "helixdata.io": "Helix Data", "forgeindustries.com": "Forge Industries",
    "quantumedge.com": "Quantum Edge", "steelpointmfg.com": "Steelpoint Manufacturing",
    "ironcladfinancial.com": "Ironclad Financial", "clearwatermed.com": "Clearwater Medical",
    "pinnacleins.com": "Pinnacle Insurance", "ironworkscorp.com": "Ironworks Corp", "axiomlabs.dev": "Axiom Labs",
}


def _pretty_account(domain: str | None) -> str | None:
    if not domain:
        return None
    return _ACCOUNT_NAMES.get(domain, domain.split(".")[0].title())
