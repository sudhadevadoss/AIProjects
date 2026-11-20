# Transcript Intelligence: take-home

Pipeline and analysis over 100 call transcripts (support, external account-management, internal) for a B2B security
SaaS company ("Aegis Cloud"). It categorises calls, analyses sentiment by call type, and generates stakeholder-specific insights.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_pipeline.py            # ~4s; writes outputs/
.venv/bin/jupyter lab notebooks/analysis.ipynb
```

**Dataset not included** (it belongs to the assignment provider). Put the 100 meeting folders in `transcript-intelligence/dataset/`, or pass `--dataset PATH`.
`--llm` also runs the optional Claude classifier (needs `ANTHROPIC_API_KEY`; results cached in `outputs/llm_cache.json`).

## Layout

```
ti/
  load.py          raw JSON folders -> meetings / sentences / key_moments / action_items tables;
                   maps every speaker to customer vs Aegis (311/311)
  taxonomy.py      10 business themes, product lexicon, competitors, outage phases
  classify.py      call type (metadata rules) -> products (lexicon) -> themes (weighted, multi-label)
                   + TF-IDF/NMF discovery + evaluation against hand labels
  sentiment.py     vendor labels + independent VADER cross-check, customer vs rep split, within-call arc
  insights.py      account health, early warnings, remediation fallout, version hotspots,
                   feature-gap backlog, competitor intel, billing friction, key-person load, talk ratio
  llm_classify.py  optional: same taxonomy through Claude (JSON schema output, cached)
  charts.py        static PNG charts used in the deck and notebook
run_pipeline.py    end-to-end CLI
notebooks/
  build_notebook.py  generates analysis.ipynb from source
  analysis.ipynb     executed walkthrough with commentary (also exported to outputs/analysis.html)
data/gold_labels.csv hand-labelled primary theme for all 100 calls (used only for evaluation)
outputs/           CSVs, charts/, headline.json
```

## Key decisions

| Decision | Why |
|---|---|
| Call type from **metadata**, not text | Who is on the call is a fact. Email domains + title patterns → 40 external / 30 support / 30 internal. |
| **No synthetic data** | The three call types arrived balanced; padding would dilute real signal. |
| **Hybrid** categorisation (rules + discovery + optional LLM) | Vendor topics are too fragmented (351 unique for 100 calls). Clusters mix product and intent. Rules are free, deterministic and explainable at this scale (87% primary / 99% multi-label agreement with hand labels); the LLM path is the scale-out plan. |
| **Multi-label** themes | 36% of calls are genuinely about more than one thing. |
| **Customer-side** sentiment | Reps are 2.6× less negative than customers; blended scores hide pain. The vendor call score tracks the customer side (r = 0.96). |
| Rubric, not ML, for account risk | 32 accounts and no churn outcomes; a transparent rubric can be argued with and tuned. |

## Headline findings

1. The root cause of the March 10 Detect outage was named in internal meetings **32 days** before it happened.
2. **9 of 32** accounts are at risk; every one is actively evaluating a competitor (6 name SentinelShield).
3. **4** April support calls were caused by the outage remediation itself, across three products. Version 3.4.1
   is a recurring bad release in three products, each with a fix already shipped.
4. Detect sentiment 2.9 vs Comply 4.3; April's "recovered" external average hides a bimodal split.

## Limitations

Small n (2–12 calls/week); vendor summaries/key moments are themselves model output; gold labels are a single
annotator; account risk has no ground-truth outcomes yet. The LLM classifier is implemented but was not run
(no API key in the build environment).
