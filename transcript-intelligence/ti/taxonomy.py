"""Business taxonomy used to categorise calls.

Why a curated taxonomy (and not just clusters)?
  * The vendor summaries already carry free-text "topics" -- 351 *unique* strings across 100 calls
    ("outage", "platform outage", "service outage", "outage post-mortem" ...). That is too fragmented
    to drive a dashboard or a routing rule.
  * Unsupervised clustering (see notebook section 2) is useful to *discover* themes, but its clusters
    are named by top words, drift every time new data arrives, and mix "what the call is about"
    with "which product it's about".
  * Stakeholders need stable, decision-oriented buckets: a support leader routes on "Billing" vs
    "Product defect", a sales leader watches "Churn risk & competitive".

So we separate three independent dimensions:
  1. call_type  (support / external / internal) -- who is on the call       -> metadata rules
  2. product    (Detect / Protect / Comply / Identity / LogVault)  -> lexicon counts
  3. theme      (10 business themes, multi-label + one primary)     -> weighted keyword scoring
"""

THEMES: dict[str, dict] = {
    "OUTAGE_INCIDENT": {
        "label": "Outage & Incident Response",
        "desc": "Live or recent service outage: war rooms, escalations, RCAs, post-incident reviews.",
        "keywords": {
            r"\boutage\b": 3, r"incident": 2, r"war room": 4, r"root cause": 2, r"post-?mortem": 3,
            r"post-incident": 3, r"cascading failure": 3, r"single point of failure": 2,
            r"zero (threat )?(monitoring )?visibility": 3, r"loss of (threat )?visibility": 3,
            r"\bp1\b": 2, r"status page": 2, r"restor(e|ation) window": 2, r"escalation bridge": 4,
            r"remediation": 1,
        },
        "title": [r"outage", r"incident", r"urgent", r"escalation", r"war room", r"data gaps", r"not firing", r"dashboard down"],
    },
    "PRODUCT_DEFECT": {
        "label": "Product Defect & Performance",
        "desc": "Bug, regression, failing connector or degraded performance (outside the major outage).",
        "keywords": {
            r"\bbug\b": 3, r"regression": 3, r"error": 1, r"failures?\b": 1, r"timeout": 2,
            r"workaround": 2, r"known (bug|issue)": 3, r"patch": 1, r"latency": 1, r"degraded": 2,
            r"throughput": 2, r"root cause": 1, r"version \d": 2, r"hotfix": 2, r"troubleshoot": 2,
            r"scalability": 2, r"queue": 1, r"false positive": 2,
        },
        "title": [r"failure", r"error", r"bug", r"issue", r"timeout", r"delay", r"slow", r"exceeded", r"false positives", r"scalability"],
    },
    "BILLING": {
        "label": "Billing & Invoicing",
        "desc": "Invoice disputes, overage charges, proration, seat counts.",
        "keywords": {
            r"invoice": 3, r"billing": 3, r"overage": 3, r"proration|prorat": 3, r"\bcredit\b": 2,
            r"charge[ds]?\b": 1, r"overbill": 3, r"discrepanc": 2, r"refund": 2,
        },
        "title": [r"billing", r"invoice", r"overage", r"license upgrade", r"charges"],
    },
    "RENEWAL_EXPANSION": {
        "label": "Renewal & Expansion",
        "desc": "Renewals, QBRs, account reviews, upsell / cross-sell on a healthy account.",
        "keywords": {
            r"renewal": 3, r"multi-year": 3, r"upsell|cross-sell": 2, r"expansion|expand": 2,
            r"business review|\bqbr\b": 3, r"pricing": 1, r"contract": 1, r"account review": 3,
            r"satisf": 1,
        },
        "title": [r"renewal", r"business review", r"account review", r"annual review", r"expansion", r"contract", r"q2 planning"],
    },
    "CHURN_COMPETITIVE": {
        "label": "Churn Risk & Competitive Pressure",
        "desc": "Customer evaluating alternatives, eroded trust, or internal competitive / win-loss strategy.",
        "keywords": {
            r"sentinelshield|vaultedge|cybernova|fortiguard": 3, r"competit": 2, r"evaluat\w+ (a |an )?(competitor|alternative)": 4,
            r"churn": 3, r"win[/-]loss|closed-lost": 4, r"eroded|erosion": 2, r"confidence": 1, r"at[- ]risk": 2,
            r"\bsla\b": 1, r"vendor comparison": 4,
        },
        "title": [r"competitive", r"concerns", r"win/loss", r"vendor comparison", r"recovery check-in", r"contract review"],
    },
    "COMPLIANCE_AUDIT": {
        "label": "Compliance & Audit Readiness",
        "desc": "SOC 2 / HIPAA / PCI DSS / ISO 27001 preparation and evidence collection.",
        "keywords": {
            r"soc ?2": 2, r"hipaa": 2, r"pci ?dss": 2, r"iso ?27001": 2, r"audit": 2, r"auditor": 2,
            r"evidence": 2, r"control(s)? mapping|controls?\b": 1, r"certification": 2, r"\bqsa\b": 2,
        },
        "title": [r"compliance", r"audit", r"soc 2", r"hipaa", r"pci", r"iso"],
    },
    "FEATURE_GAP": {
        "label": "Feature Request & Product Feedback",
        "desc": "Customer asks for capability that does not exist yet; structured product feedback.",
        "keywords": {
            r"feature (gap|request)": 4, r"not (currently )?supported|doesn'?t support|not yet (generally )?available": 3,
            r"on the roadmap|roadmap": 2, r"pain points?": 2, r"product feedback": 4, r"would love|wish": 1,
            r"design partner|early access": 1, r"escalated internally": 2,
        },
        "title": [r"request", r"feedback", r"roadmap (review|discussion)", r"template"],
    },
    "ONBOARDING": {
        "label": "Onboarding & Adoption",
        "desc": "Kickoffs, deployments, implementation planning, tuning after go-live.",
        "keywords": {
            r"onboard": 3, r"kickoff|kick-off": 3, r"deploy(ment)? (plan|sequenc|phase)": 2, r"rollout": 1,
            r"go-?live|went live": 2, r"implementation": 2, r"phased": 1, r"migrat": 1, r"tuning": 2,
            r"workshop": 2, r"discovery": 1,
        },
        "title": [r"onboarding", r"kickoff", r"deployment", r"setup", r"workshop", r"tuning"],
    },
    "LAUNCH": {
        "label": "Product Launch (Comply v2)",
        "desc": "Comply v2 design, launch readiness, GA, early-access demos and adoption tracking.",
        "keywords": {
            r"comply v2": 3, r"\bga\b|general availability": 2, r"launch": 2, r"early access": 2,
            r"adoption": 1, r"demo": 1, r"preview": 2,
        },
        "title": [r"comply v2", r"launch", r"preview", r"early access"],
    },
    "PLANNING": {
        "label": "Engineering Planning & Roadmap",
        "desc": "Internal sprint planning, retros, standups, roadmap and all-hands.",
        "keywords": {
            r"sprint": 3, r"retro": 3, r"standup|stand-up": 3, r"roadmap": 1, r"all[- ]hands": 4,
            r"carry-?over": 2, r"tickets?": 1, r"capacity": 1, r"quarterly planning|q[1-4] planning": 2,
        },
        "title": [r"sprint", r"standup", r"retro", r"roadmap", r"all hands", r"planning", r"sync"],
    },
}

# Simple priors: which themes are even plausible for a call type. Keeps "PLANNING" off customer calls etc.
CALL_TYPE_PRIOR = {
    "support": {"OUTAGE_INCIDENT": 1.0, "PRODUCT_DEFECT": 1.2, "BILLING": 1.2, "FEATURE_GAP": 1.0, "LAUNCH": 0.8,
                "RENEWAL_EXPANSION": 0.3, "CHURN_COMPETITIVE": 0.5, "COMPLIANCE_AUDIT": 0.5, "ONBOARDING": 0.5, "PLANNING": 0.0},
    "external": {"OUTAGE_INCIDENT": 1.0, "PRODUCT_DEFECT": 0.4, "BILLING": 0.6, "FEATURE_GAP": 1.0, "LAUNCH": 1.0,
                 "RENEWAL_EXPANSION": 1.2, "CHURN_COMPETITIVE": 1.2, "COMPLIANCE_AUDIT": 1.0, "ONBOARDING": 1.1, "PLANNING": 0.0},
    "internal": {"OUTAGE_INCIDENT": 1.0, "PRODUCT_DEFECT": 0.8, "BILLING": 0.3, "FEATURE_GAP": 0.3, "LAUNCH": 1.0,
                 "RENEWAL_EXPANSION": 0.3, "CHURN_COMPETITIVE": 1.0, "COMPLIANCE_AUDIT": 1.0, "ONBOARDING": 0.3, "PLANNING": 1.1},
}

PRODUCTS = {
    "Detect": [r"\bdetect\b", r"threat (monitoring|detection)", r"alerts?\b", r"event (processing|ingestion)"],
    "Protect": [r"\bprotect\b", r"backup", r"restore", r"\brto\b|\brpo\b", r"snapshot"],
    "Comply": [r"\bcomply\b", r"compliance report", r"evidence collection"],
    "Identity": [r"\bidentity\b", r"\bsso\b", r"\bmfa\b", r"\bsaml\b", r"\bscim\b", r"\bldap\b", r"provisioning"],
    "LogVault": [r"logvault", r"\bsiem\b"],
}

COMPETITORS = ["SentinelShield", "VaultEdge", "CyberNova", "Fortiguard Pro"]

# Phases of the quarter, anchored on the Detect outage that dominates the dataset.
OUTAGE_START = "2026-03-10"
PHASES = [
    ("1. Pre-outage", None, "2026-03-09"),
    ("2. Outage & fallout", "2026-03-10", "2026-03-31"),
    ("3. Recovery + Comply v2", "2026-04-01", None),
]
