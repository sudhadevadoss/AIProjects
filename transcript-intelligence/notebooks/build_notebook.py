"""Generates analysis.ipynb (so the notebook is reproducible from source). Run, then execute with nbconvert:
    python notebooks/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Transcript Intelligence: analysis notebook

**What this is:** the reference implementation behind the slide deck. Every number in the deck is produced here.
The heavy lifting lives in the `ti/` package (`load` → `classify` → `sentiment` → `insights` → `charts`); this
notebook calls it step by step and explains the decisions.

| Section | Question |
|---|---|
| 1 | What's in the data, and what did I change? |
| 2 | **Task 1:** how are calls categorised, and why this approach? |
| 3 | **Task 2:** what does sentiment look like across call types, and what does it mean? |
| 4 | **Task 3:** what else can stakeholders get from this data? |
| 5 | Limitations and what I'd do next |

Run everything end-to-end from the CLI with `python run_pipeline.py` (≈4s).
""")
code("""
import sys, warnings; sys.path.insert(0, '..'); warnings.filterwarnings('ignore')
import pandas as pd
from IPython.display import Image, display
pd.set_option('display.max_colwidth', 120); pd.set_option('display.width', 200)

from ti.load import load
from ti.classify import classify, discover_topics, evaluate
from ti import sentiment as S, insights as I, charts
from pathlib import Path
OUT = Path('../outputs/charts')

t = load()
m, s, km = t['meetings'], t['sentences'], t['key_moments']
print(f"{len(m)} meetings | {len(s):,} sentences | {m.n_words.sum():,} words | {m.start.min():%b %d} – {m.start.max():%b %d %Y}")
""")

md("""
## 1. The data, and what I changed

Each of the 100 folders is one recorded meeting: `meeting-info.json`, a sentence-level `transcript.json` (with a
vendor sentiment label per sentence), a vendor `summary.json` (summary, free-text topics, typed *key moments*,
action items, a 1–5 sentiment score), plus speaker turns and join/leave events.

**Call types aren't labelled.** I derived them from metadata rather than text, because *who is on the call*
is a structural fact:

* no external email domain → **internal**
* external participants **and** a support-style title (`Support Case #`, `URGENT:`, `ESCALATION:`) or a summary saying
  the customer contacted support → **support**
* otherwise → **external** (account management: renewals, QBRs, onboarding, feedback)

**Speaker side.** I mapped every speaker name to an email (311/311 matched) so every sentence is tagged
*customer* or *Aegis employee*. This turns out to be the single most useful enrichment (section 3).

**No synthetic data was added.** The three call types come out balanced (40/30/30), so there was no gap to fill.
I'd rather show real signal than pad it.
""")
code("""
m = classify(t)
display(m.call_type.value_counts().to_frame('calls'))
m.groupby('call_type').apply(lambda g: g.sample(3, random_state=1)[['title','call_type_reason']]).droplevel(0)
""")

md("""
## 2. Task 1: categorising calls by topic

### Step 1: Look at what's already there
The vendor summaries ship with free-text topics. They are **too fragmented to use directly**: 351 unique strings for
100 calls ("outage", "platform outage", "service outage", "outage post-mortem", ...).
""")
code("""
topics = pd.Series([x for ts in m.topics for x in ts])
print(topics.nunique(), 'unique topic strings; top 15:')
topics.value_counts().head(15)
""")

md("""
### Step 2: Unsupervised discovery (what does the data say?)
TF-IDF + NMF over the raw transcripts, with no labels involved. This shows which natural groupings exist before I impose any.
""")
code("""
m['cluster'], clusters = discover_topics(m)
clusters
""")
md("""
The clusters are coherent: billing, identity/SSO, competitive, connector errors, audit evidence, onboarding/backup,
incident pipeline, sprint work. They also show **why clustering alone isn't the answer**: they mix *product*
(cluster "identity, mfa, sso") with *business intent* (cluster "billing, invoice, overage"). A support leader routing
calls needs intent; a PM needs product. So I split those into separate dimensions.

### Step 3: A curated taxonomy, with three independent dimensions
| Dimension | Method | Why |
|---|---|---|
| **Call type** | metadata rules | structural, ~100% reliable |
| **Product** (Detect / Protect / Comply / Identity / LogVault) | lexicon counts | products are proper nouns |
| **Theme** (10 business themes, multi-label + one primary) | weighted keyword scoring over title, summary, vendor topics, key-moment types and transcript, × call-type prior | explainable, deterministic, free; every label carries its evidence |

The 10 themes were chosen from the discovery clusters plus the decisions stakeholders make
(e.g. *Billing* is separate from *Product defect* because they route to different teams).

**Why not LLM-only?** On 100 calls, rules are cheaper, faster and auditable, and they hit 87% (below). I built the
LLM path (`ti/llm_classify.py`, Claude with a JSON schema) as the **scale-out plan**: keyword lists rot as products
change, and in production you'd run both and send disagreements to a human.
""")
code("""
from ti.taxonomy import THEMES
pd.DataFrame([(k, v['label'], v['desc']) for k, v in THEMES.items()], columns=['key','theme','definition'])
""")

md("""
### Step 4: Validate
Before writing any rules I hand-labelled a primary theme for all 100 calls from their summaries
(`data/gold_labels.csv`). The classifier is scored against those labels.
""")
code("""
acc, ev = evaluate(m, '../data/gold_labels.csv')
print(f"Primary theme matches hand label: {acc:.0%}")
print(f"Hand label appears in the multi-label set: {ev.in_multilabel.mean():.0%}")
print(f"Calls with >1 theme: {(m.themes.str.len()>1).mean():.0%}")
ev[~ev.primary_match][['title','gold','pred']]
""")
md("""
**Reading the misses:** nearly all are genuinely multi-topic meetings: standups and all-hands that discuss *both*
the outage and the Comply v2 launch, or an account call that is both "post-incident" and "churn risk". In every
case but one, the hand label is still in the multi-label set. That's why the output is multi-label: forcing one
bucket loses information.

### Step 5: Results: categories with examples
""")
code("""
display(Image(OUT / '01_themes_by_call_type.png'))
ex = (m.sort_values('theme_margin', ascending=False).groupby('theme_label')
        .apply(lambda g: pd.Series({'calls': len(g), 'call_types': ', '.join(f"{k}:{v}" for k,v in g.call_type.value_counts().items()),
                                    'examples': ' | '.join(g.title.head(3))})))
ex.sort_values('calls', ascending=False)
""")
code("""
# The evidence trail for one call: why did it land where it did?
r = m[m.title.str.contains('Quantum Edge')].iloc[0]
print(r.title); print('primary:', r.primary_theme, '| all:', r.themes, '| margin:', r.theme_margin)
print('evidence:', r.theme_evidence)
print({k: v for k, v in sorted(r.theme_scores.items(), key=lambda kv: -kv[1])[:4]})
""")

md("""
## 3. Task 2: sentiment across call types

### First, can we trust the sentiment labels?
The vendor gives a 1–5 meeting score and a pos/neu/neg label per sentence. Before building on them I re-scored every
sentence with an independent model (VADER) and compared.
""")
code("""
s = S.score_sentences(s)
m = S.meeting_sentiment(m, s)
a = S.label_agreement(s)
print(f"Sentence-level exact agreement: {a['exact_agreement']:.0%} | direction agreement (ignoring neutrals): {a['direction_agreement']:.0%}")
display(a['crosstab'])
m[['vendor_sentiment_score','nsi_customer','nsi_internal','nsi_vader']].corr().round(2)
""")
md("""
**What this tells us:**
* At the **meeting level** the signals agree (vendor score vs. independent VADER: r = 0.83). The trend lines are trustworthy.
* At the **sentence level**, a generic lexicon misses this domain: VADER catches only **25%** of the sentences the
  vendor marks negative. Phrases like *"we were blind for six hours"* and *"zero visibility"* aren't negative words to a
  general-purpose model. **Implication for the product:** don't ship off-the-shelf sentiment; use a domain-tuned or LLM scorer.
* The vendor meeting score correlates **0.96 with the customer-only sentiment** and 0.84 with the rep-only. The meeting
  score is essentially *the customer's mood*, which is what we want.

### Customer voice vs. rep voice
""")
code("""
display(Image(OUT / '04_customer_vs_rep.png'))
s.groupby('side').vendor_label.value_counts(normalize=True).unstack().round(2)
""")
md("""
Customers say something negative in **29%** of their sentences; Aegis staff in **11%**. On support calls the gap is
0.38 NSI points. **Reps are professionally positive, so blended call sentiment understates customer pain.** Any
dashboard in this product should default to the *customer-side* score.

### By call type, over time
""")
code("""
tabs = S.summary_tables(m)
display(tabs['by_type'])
display(Image(OUT / '02_weekly_sentiment.png'))
display(Image(OUT / '03_sentiment_by_phase.png'))
display(tabs['by_phase']); tabs['by_phase_n']
""")
md("""
**What the trends mean:**

1. **Support is structurally the lowest (2.8)**, which is expected because people call support when something is broken. The
   useful signal is *change* in support sentiment, not its level.
2. **One incident moved every line at once.** In the outage window (Mar 10–31) external, internal *and* support
   sentiment all collapsed to ~2.3–2.5. Internal sentiment dropping as hard as customer sentiment matters:
   the outage cost the team morale and sprint capacity (two other teams' retros say so), not just customer trust.
3. **External sentiment "recovered" to 4.15 in April, and the average hides what's underneath.** April external calls are
   bimodal: Comply v2 demos and renewals at 4.7–4.8, competitive-evaluation calls at 2.6. The average says
   "we're fine"; the account view (section 4) says 9 accounts are at risk.
4. **Support did not recover the same way.** April support calls are back to pre-outage levels, but a new kind of
   ticket appears: problems *caused by the outage fix* (section 4).
""")
code("""
display(Image(OUT / '05_product_sentiment.png'))
display(tabs['by_product'])
display(tabs['by_theme'])
""")
md("""
**By product:** Detect (2.9) drags everything; Comply (4.3) is the bright spot thanks to the v2 launch. **By theme:**
outage < defect < churn < billing < feature requests < planning < renewal < launch < compliance < onboarding.
Onboarding and compliance-prep calls are the happiest. The *new-customer and audit-prep moments* are where Aegis is
winning.

### Do we land the call? (within-call sentiment arc)
""")
code("""
display(Image(OUT / '06_call_arc.png'))
""")
md("""
I compared the customer's sentiment in the first third of each call with the last third.
* Customers generally **leave calls happier than they arrived** (+0.36 external, +0.23 support). Reps are good at
  de-escalation in the moment.
* **Churn-risk calls show the biggest in-call recovery (+0.57)** yet those accounts are still evaluating competitors.
  In-call recovery ≠ account recovery. A rep's "great call" feeling is not a retention signal.
* **Billing calls barely move (+0.13).** The rep usually can't fix the invoice on the call and has to "submit a request
  to finance". **Recommendation:** give support agents credit authority up to a threshold.
""")

md("""
## 4. Task 3: what else can we see?

Each insight targets a specific stakeholder and comes with evidence (quotes and titles), because a leader won't act
on an insight they can't verify.

### 4.1 Account health and churn risk (CS and sales leaders)
A transparent, additive 0–100 rubric: last sentiment (30), active competitive evaluation (25), vendor-flagged
churn moments (15), support load (10), worsening trend (10), outage exposure (10). It's deliberately not ML: 32 accounts and
no churn labels. A rubric a CS leader can argue with beats a black box.
""")
code("""
h = I.account_health(m, km)
display(Image(OUT / '07_account_risk.png'))
h[['account','calls','support_calls','first_score','last_score','trend','competitors_named','active_competitive_eval','risk_score','tier','evidence']].head(12)
""")
md("""
**9 of 32 accounts are At risk, and every one of them is actively evaluating a competitor (6 name SentinelShield).** Summit Trust tops the list. It has had three calls
(billing → platform concerns → alert delays) and, in its April support call, found out about the March outage
**for the first time**, a month later. That's a communication failure that a transcript tool can catch automatically.

### 4.2 Early warning: did we see it coming? (Engineering leaders)
""")
code("""
ew = I.early_warnings(m, s)
display(ew)
I.retro_mentions_of_known_risk(m, s)[['start','title','speaker','text']].head(8)
""")
md("""
**The outage's exact root cause was named in internal meetings 32 and 7 days before it happened**, down to the fix
("add redundant processing nodes and a circuit breaker"). In the war room: *"It was flagged in the Q4 tech debt
review. I flagged it."* Then a customer is told *"a known architectural limitation that we hadn't yet prioritized."*

**And it's happening again:** a QA engineer flagged the Protect scheduler concurrency ceiling (QA-1847) in late Feb.
It was deprioritized because of the outage and became the April *"Protect Performance: Scalability Concerns"* meeting.

**Product idea:** a *risk register from transcripts*: extract "single point of failure / tech debt / I'm worried about"
statements from internal calls, link them to a product area, and resurface them when customer calls about that
area start to sour. This is the highest-leverage feature in the deck.
""")
code("""
display(Image(OUT / '08_incident_timeline.png'))
""")

md("""
### 4.3 Fixes that broke other things, and preventable tickets (Engineering, Support, QA)
""")
code("""
display(I.remediation_fallout(m)[['start','account','title','primary_product']])
I.version_hotspots(m)
""")
md("""
* **4 support calls in April trace back to the outage remediation itself**, across *three* products: Summit Trust
  (tenant moved to an overloaded cluster), Stratos (circuit-breaker payload threshold in build 4.1.2 dropping LogVault
  events), Silverline (backup throughput down after network routing changes), Crestline (Identity sync queue on a
  shared cluster). The fix to Detect had a blast radius into Protect and Identity.
* **Version 3.4.1 is a recurring bad release across three products** (LogVault connector, Protect backup agent,
  Identity). In each case **a fix already existed** (3.5.0 / 3.6.0 / 3.4.2) but the customer hadn't upgraded.
  These are preventable tickets → **proactive "you're on a version with a known bug" outreach.**

### 4.4 Feature-gap backlog (Product managers)
""")
code("""
fb, fg = I.feature_gap_backlog(m, km)
fb
""")
md("""
Counting *distinct accounts* asking (not mentions) makes this a defensible prioritisation input. Comply and Identity
gaps come from **happy** accounts (sentiment ~4): these are expansion asks. Detect gaps come from **unhappy** accounts
(2.4): these are retention asks. Same "feature request" label, very different urgency.

### 4.5 Competitive intelligence (Sales, PMM)
""")
code("""
display(Image(OUT / '09_competitors.png'))
I.competitor_intel(m, s)[['competitor','mentions','customer_calls','post_outage_share','products_in_context','sample_quote']]
""")
md("""
SentinelShield is mentioned 51 times across 14 customer calls, **80% after the outage**, pitched on uptime SLA
(99.9% vs. Aegis's 99.5%). The internal Win/Loss review attributes ≥5 lost deals to the outage. Sales needs a
reliability narrative (post-mortem + 30-day clean window); Product should consider a public SLA upgrade.

### 4.6 Billing friction is a product problem (Finance, Product)
""")
code("""
I.billing_friction(m)[['start','account','title','root_cause_hint','customer_unaware_of_usage','vendor_sentiment_score']]
""")
md("""
Every overage dispute has the same shape: **the customer didn't know they were over** (test accounts never
deprovisioned, a backup policy change, report generation during the v2 launch "with no in-app alerts"). One customer
literally asked for *"proactive seat-count threshold alerts."* **Usage alerts in-product** would remove a whole
support category.

### 4.7 Key-person load (CS and support ops)
""")
code("""
I.owner_load(m, s, h).head(8)
""")
md("""
Sarah Chen (support) and Maria Santos (CSM) are each on calls with 4 of the 9 at-risk accounts. If she leaves or burns out, that context goes with
her. A transcript tool can show relationship concentration that no CRM field captures.

### 4.8 A caution: talk-time ratio
""")
code("""
tr = I.talk_ratio(m)
print('corr(customer talk share, sentiment) =', round(tr[['customer_talk_share','vendor_sentiment_score']].corr().iloc[0,1], 2))
tr.groupby('talk_bucket', observed=True).agg(calls=('meeting_id','count'), sentiment=('vendor_sentiment_score','mean')).round(2)
""")
md("""
The naive reading ("calls where the rep talks more are happier, so reps should talk more") is **wrong**: the causation
runs the other way. Unhappy customers talk more because they're explaining what went wrong. I include this to show
the kind of conclusion the product should *not* surface without context.

## 5. Limitations and next steps
* **Small n.** 100 calls, 2–12 per week. Trends are directional; I report shapes and sizes, not significance.
* **Vendor-derived fields** (summary, key moments, topics) are themselves LLM outputs. I validated sentiment
  independently; key-moment types were spot-checked, not audited.
* **Gold labels** are mine (single annotator, LLM-assisted reading of summaries). A second annotator would provide
  inter-rater agreement.
* **Account risk** has no ground truth (we don't see renewals/churn outcomes). Next step: backtest against CRM outcomes.
* **Next:** run the LLM classifier in parallel for disagreement review, build the transcript risk register, link
  support tickets ↔ internal incident calls automatically, and add commitment tracking (did the action items from
  call N show up as done in call N+1?).
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
Path(__file__).with_name("analysis.ipynb").write_text(nbf.writes(nb))
print("wrote analysis.ipynb")
