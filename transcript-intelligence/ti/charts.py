"""Static charts (PNG) for the deck and notebook. One consistent visual system:
fixed categorical order for call types, status colours only for risk tiers, thin marks, direct labels."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
# categorical slots 1-3 (validated all-pairs) -> call types, fixed order
CT_COLORS = {"external": "#2a78d6", "support": "#eb6834", "internal": "#1baf7a"}
CT_LABEL = {"external": "External (account mgmt)", "support": "Support", "internal": "Internal"}
STATUS = {"Healthy": "#0ca30c", "Watch": "#fab219", "At risk": "#d03b3b"}
NEUTRAL = "#b9b8b2"
ACCENT = "#2a78d6"
from matplotlib.colors import LinearSegmentedColormap
DIVERGING = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#c9c8c3", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 14,
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 14, "legend.frameon": False,
})


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=180)
    plt.close(fig)


def theme_by_call_type(m: pd.DataFrame, out: Path):
    ct = pd.crosstab(m.theme_label, m.call_type).reindex(columns=["external", "support", "internal"], fill_value=0)
    ct = ct.loc[ct.sum(1).sort_values().index]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    left = np.zeros(len(ct))
    for c in ct.columns:
        ax.barh(ct.index, ct[c], left=left, color=CT_COLORS[c], label=CT_LABEL[c], height=0.62,
                edgecolor=SURFACE, linewidth=2)
        left += ct[c].values
    for i, tot in enumerate(left):
        ax.text(tot + 0.3, i, f"{int(tot)}", va="center", color=INK2, fontsize=10)
    ax.set_title("What the 100 calls are about: primary theme, by call type")
    ax.set_xlabel("Calls")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    _save(fig, out, "01_themes_by_call_type")


def weekly_trend(m: pd.DataFrame, out: Path):
    w = m.groupby(["week", "call_type"]).vendor_sentiment_score.mean().unstack()
    w.index = w.index + pd.Timedelta(days=3)  # plot at mid-week
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axvspan(pd.Timestamp("2026-03-10"), pd.Timestamp("2026-03-31"), color="#f0efec", zorder=0)
    ax.text(pd.Timestamp("2026-03-11"), 4.95, "Detect outage\n& fallout", color=INK2, fontsize=9, va="top")
    ax.axvline(pd.Timestamp("2026-04-07"), color=MUTED, lw=1, ls="--")
    ax.text(pd.Timestamp("2026-04-08"), 4.95, "Comply v2 GA", color=INK2, fontsize=9, va="top")
    for c in ["external", "support", "internal"]:
        s = w[c].dropna()
        ax.plot(s.index, s.values, color=CT_COLORS[c], lw=2, marker="o", ms=5, label=CT_LABEL[c])
        ax.text(s.index[-1] + pd.Timedelta(days=1.5), s.values[-1], CT_LABEL[c].split(" ")[0], color=INK2,
                fontsize=9, va="center")
    ax.set_ylim(1, 5.05)
    ax.set_ylabel("Avg sentiment score (1-5)")
    ax.set_title("Weekly sentiment by call type: one incident moved every line at once")
    ax.text(0, -0.13, "Points = mean score of that week's calls (2-12 calls/week); read the shape, not single points.", transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="lower left", ncol=3)
    _save(fig, out, "02_weekly_sentiment")


def phase_bars(m: pd.DataFrame, out: Path):
    p = m.pivot_table(index="phase", columns="call_type", values="vendor_sentiment_score").reindex(
        columns=["external", "support", "internal"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(p))
    wdt = 0.26
    for i, c in enumerate(p.columns):
        bars = ax.bar(x + (i - 1) * wdt, p[c], width=wdt - 0.02, color=CT_COLORS[c], label=CT_LABEL[c])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{b.get_height():.1f}", ha="center",
                    fontsize=9, color=INK2)
    ax.set_xticks(x, [s.split(". ")[1] for s in p.index])
    ax.set_ylim(1, 5)
    ax.set_ylabel("Avg sentiment score (1-5)")
    ax.set_title("Sentiment by phase: external calls bounced back, support never got above ~3")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    _save(fig, out, "03_sentiment_by_phase")


def customer_vs_rep(m: pd.DataFrame, out: Path):
    d = m[m.call_type != "internal"].groupby("call_type")[["nsi_customer", "nsi_internal"]].mean().reindex(["external", "support"])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, (ct, r) in enumerate(d.iterrows()):
        ax.plot([r.nsi_customer, r.nsi_internal], [i, i], color=NEUTRAL, lw=2, zorder=1)
        ax.scatter(r.nsi_customer, i, s=110, color=CT_COLORS[ct], zorder=2, edgecolor=SURFACE, linewidth=2)
        ax.scatter(r.nsi_internal, i, s=110, color=SURFACE, zorder=2, edgecolor=CT_COLORS[ct], linewidth=2)
        ax.text(r.nsi_customer, i + 0.22, f"Customer {r.nsi_customer:+.2f}", ha="center", fontsize=9, color=INK)
        ax.text(r.nsi_internal, i + 0.22, f"Aegis rep {r.nsi_internal:+.2f}", ha="center", fontsize=9, color=INK2)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(d)), [CT_LABEL[c] for c in d.index])
    ax.set_ylim(-0.6, len(d) - 0.3)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xlabel("Net sentiment index  (positive − negative sentences) / all sentences")
    ax.set_title("The customer's voice vs. the rep's voice on the same calls")
    ax.grid(axis="y", visible=False)
    _save(fig, out, "04_customer_vs_rep")


def product_sentiment(m: pd.DataFrame, out: Path):
    d = m[m.primary_product != "LogVault"].groupby("primary_product").agg(
        score=("vendor_sentiment_score", "mean"), n=("meeting_id", "count")).sort_values("score")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    colors = [STATUS["At risk"] if s < 3 else NEUTRAL for s in d.score]
    ax.barh(d.index, d.score, color=colors, height=0.55)
    for i, (p, r) in enumerate(d.iterrows()):
        ax.text(r.score + 0.05, i, f"{r.score:.2f}  ({r.n} calls)", va="center", fontsize=10, color=INK2)
    ax.set_xlim(1, 5.3)
    ax.set_xlabel("Avg sentiment score (1-5), calls grouped by most-discussed product")
    ax.set_title("Detect is the sentiment drag; Comply is the bright spot")
    ax.grid(axis="y", visible=False)
    _save(fig, out, "05_product_sentiment")


def call_arc(m: pd.DataFrame, out: Path):
    d = m[m.call_type != "internal"].groupby("theme_label")[["cust_open", "cust_close"]].mean().dropna()
    d = d.assign(delta=d.cust_close - d.cust_open).sort_values("delta")
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (t, r) in enumerate(d.iterrows()):
        ax.annotate("", xy=(r.cust_close, i), xytext=(r.cust_open, i),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT if r.delta > 0.2 else STATUS["At risk"], lw=2))
        ax.scatter(r.cust_open, i, s=40, color=NEUTRAL, zorder=3)
        ax.text(max(r.cust_open, r.cust_close) + 0.04, i, f"{r.delta:+.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(d)), d.index)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(-1.0, 1.1)
    ax.set_xlabel("Customer net sentiment: first third of call (dot) to last third (arrow head)")
    ax.set_title("Do we land the call? Customer sentiment from opening to close")
    ax.grid(axis="y", visible=False)
    _save(fig, out, "06_call_arc")


def account_risk(h: pd.DataFrame, out: Path, top: int = 15):
    d = h.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(d.account, d.risk_score, color=[STATUS[t] for t in d.tier], height=0.62)
    for i, (_, r) in enumerate(d.iterrows()):
        tag = r.competitors_named.split(",")[0] if r.active_competitive_eval and r.competitors_named else ""
        ax.text(r.risk_score + 1, i, f"{int(r.risk_score)}  {r.tier}" + (f"  · evaluating {tag}" if tag else ""),
                va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, 120)
    ax.set_xlabel("Risk score (0-100): last sentiment, competitive eval, churn signals, support load, trend, outage")
    ax.set_title(f"Account risk: top {top} of {len(h)} customer accounts")
    ax.grid(axis="y", visible=False)
    _save(fig, out, "07_account_risk")


def incident_timeline(m: pd.DataFrame, ew: pd.DataFrame, fallout: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(12, 4.6))
    det = m[m.mentions_Detect >= 3]
    for ct, y in (("internal", 2), ("external", 1), ("support", 0)):
        g = det[det.call_type == ct]
        sc = ax.scatter(g.start, [y] * len(g), c=g.vendor_sentiment_score, cmap=DIVERGING, vmin=1, vmax=5, s=70,
                        edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for _, r in ew.drop_duplicates("title").iterrows():
        ax.annotate("risk flagged\n(single point\nof failure)", xy=(r.start, 2), xytext=(r.start, 2.75), ha="center",
                    fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.axvline(pd.Timestamp("2026-03-10"), color=STATUS["At risk"], lw=1.5)
    ax.text(pd.Timestamp("2026-03-10 12:00"), 3.25, "Mar 10 outage", color=STATUS["At risk"], fontsize=9, fontweight="bold")
    for k, (_, r) in enumerate(fallout.iterrows()):
        ax.annotate(r.account, xy=(r.start, 0), xytext=(r.start, -0.55 - 0.3 * (k % 2)), ha="center", fontsize=8,
                    color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.text(fallout.start.min() - pd.Timedelta(days=1), -1.0, "side-effects of the fix:", ha="right", fontsize=8, color=INK2)
    ax.set_yticks([0, 1, 2], ["Support", "External", "Internal"])
    ax.set_ylim(-1.1, 3.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="y", visible=False)
    cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.025)
    cb.set_label("Sentiment (1-5)", color=INK2)
    ax.set_title("The Detect story: risk flagged 32 days early, then the outage, then fixes that broke other things")
    _save(fig, out, "08_incident_timeline")


def competitor_chart(ci: pd.DataFrame, out: Path):
    d = ci.sort_values("mentions")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    pre = d.mentions * (1 - d.post_outage_share)
    post = d.mentions * d.post_outage_share
    ax.barh(d.competitor, pre, color=NEUTRAL, label="Before Mar 10", height=0.55, edgecolor=SURFACE, linewidth=2)
    ax.barh(d.competitor, post, left=pre, color=STATUS["At risk"], label="After Mar 10", height=0.55,
            edgecolor=SURFACE, linewidth=2)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(r.mentions + 0.6, i, f"{r.mentions} mentions · {r.customer_calls} customer calls", va="center",
                fontsize=9, color=INK2)
    ax.set_xlim(0, d.mentions.max() * 1.6)
    ax.set_title("Competitor mentions across all transcripts")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    _save(fig, out, "09_competitors")


def make_all(m, h, ew, fallout, ci, out: Path):
    theme_by_call_type(m, out)
    weekly_trend(m, out)
    phase_bars(m, out)
    customer_vs_rep(m, out)
    product_sentiment(m, out)
    call_arc(m, out)
    account_risk(h, out)
    incident_timeline(m, ew, fallout, out)
    competitor_chart(ci, out)
