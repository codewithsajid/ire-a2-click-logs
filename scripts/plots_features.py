"""Figures for the user-representation and feature ablations.

Two questions that Q1 raised and Q2/Q3 never answered: how much of the user's
history to read, and whether the derived features the store carries are worth
anything to the ranking.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE = "#fcfcfb"
GOOD, BAD = "#127f5c", "#c9342f"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#d9d8d4", "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.grid": True, "grid.color": "#ececea", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
})
OUT = Path("reports/figures"); OUT.mkdir(parents=True, exist_ok=True)
WINDOWS = [5, 10, 20, 30, 50, 100, 0]
LABELS = ["5", "10", "20", "30", "50", "100", "all"]


def main():
    ur = [json.loads(p.read_text()) for p in sorted(Path("reports/q3").glob("userrep_*.json"))]
    fe = [json.loads(p.read_text()) for p in sorted(Path("reports/q4").glob("features_*.json"))]
    if not ur or not fe:
        raise SystemExit("run scripts/q3_userrep.py and scripts/q4_features.py first")

    fig, ax = plt.subplots(2, 3, figsize=(15.8, 9.2))

    # --- (a,b) how much history to read
    for j, d in enumerate(ur[:2]):
        a = ax[0][j]
        tag = f"{d['dataset']}/{d['variant']}"
        for i, kind in enumerate(("emb", "bm25")):
            rows = {r["n_recent"]: r for r in d["runs"]
                    if r["retriever"] == kind and r["halflife"] is None}
            y = [rows[w]["recall@100"] for w in WINDOWS]
            a.plot(range(len(WINDOWS)), y, color=C[i], marker="o", ms=6, mfc=SURFACE, mew=1.8,
                   label={"emb": "embeddings", "bm25": "BM25"}[kind])
            best = int(np.argmax(y))
            a.plot(best, y[best], marker="*", ms=15, color=C[i], zorder=5)
        a.axvline(WINDOWS.index(30), color=INK, lw=1.1, ls="--", alpha=0.65)
        a.set_xticks(range(len(WINDOWS))); a.set_xticklabels(LABELS)
        a.set_xlabel("clicks read from the history (n_recent)")
        a.set_ylabel("recall@100")
        a.set_title(f"({'ab'[j]}) {tag} — history median {d['history_median']}",
                    color=INK, weight="bold", loc="left", pad=20)
        a.legend(fontsize=8.5, frameon=False, ncol=2, loc="lower left",
                 bbox_to_anchor=(0, 1.005))
        a.text(WINDOWS.index(30), a.get_ylim()[0], "  shipped", fontsize=8, color=MUTED,
               va="bottom")

    # --- (c) does decaying old clicks help?
    a = ax[0][2]
    labels, vals, cols = [], [], []
    for d in ur[:2]:
        for kind in ("emb", "bm25"):
            rows = [r for r in d["runs"] if r["retriever"] == kind]
            flat = next(r for r in rows if r["n_recent"] == 0 and r["halflife"] is None)
            for hl in (5.0, 10.0, 20.0):
                r = next((x for x in rows if x["halflife"] == hl), None)
                if r is None:
                    continue
                pct = 100 * (r["recall@100"] / flat["recall@100"] - 1)
                labels.append(f"{d['dataset'][:2]}·{kind}·t½={hl:g}")
                vals.append(pct); cols.append(GOOD if pct > 0 else BAD)
    y = np.arange(len(labels))
    a.barh(y, vals, color=cols, edgecolor=SURFACE, linewidth=1.6)
    a.axvline(0, color=INK, lw=1.2)
    a.set_yticks(y); a.set_yticklabels(labels, fontsize=7.5); a.invert_yaxis()
    a.set_xlabel("recall@100 vs no decay (%)")
    a.set_title("(c) Decaying old clicks", color=INK, weight="bold", loc="left")
    a.grid(axis="y", visible=False)
    npos = sum(1 for v in vals if v > 0)
    msg = ("every setting hurts → old clicks still carry signal" if npos == 0
           else f"{len(vals)-npos} of {len(vals)} settings hurt; decay only helps "
                "where histories are short")
    a.text(0.97, 0.03, msg, transform=a.transAxes, ha="right", fontsize=8.5, color=MUTED)

    # --- (d) each feature on its own.
    # The two datasets do not carry the same features -- MIND ships no sentiment
    # or paywall flag -- so the axis is the union and a missing feature simply
    # has no bar, rather than being drawn as a zero it never earned.
    feats = list(dict.fromkeys([f for d in fe[:2] for f in d["features"]]))
    yv = np.arange(len(feats)); h = 0.38
    a = ax[1][0]
    for i, d in enumerate(fe[:2]):
        off = (i - (len(fe[:2]) - 1) / 2) * h
        present = [k for k, f in enumerate(feats) if f in d["alone"]]
        a.barh([yv[k] + off for k in present],
               [d["alone"][feats[k]]["auc"] for k in present], height=h * 0.9,
               color=C[i], edgecolor=SURFACE, linewidth=1.6,
               label=f"{d['dataset']}/{d['variant']}")
    a.axvline(0.5, color=INK, lw=1.2, ls="--")
    a.set_yticks(yv); a.set_yticklabels(feats, fontsize=8.5); a.invert_yaxis()
    a.set_xlim(0.40, max(v["auc"] for d in fe[:2] for v in d["alone"].values()) + 0.02)
    a.set_xlabel("AUC using this feature alone")
    a.set_title("(d) Each feature on its own", color=INK, weight="bold", loc="left", pad=20)
    a.legend(fontsize=8, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.005))
    a.grid(axis="y", visible=False)

    # --- (e) marginal contribution on top of embeddings
    a = ax[1][1]
    add = [f for f in feats if f != "emb"]
    yv = np.arange(len(add))
    for i, d in enumerate(fe[:2]):
        off = (i - (len(fe[:2]) - 1) / 2) * h
        present = [k for k, f in enumerate(add) if f in d["added_to_emb"]]
        delta = [d["added_to_emb"][add[k]]["auc"] - d["emb_baseline"]["auc"] for k in present]
        a.barh([yv[k] + off for k in present], delta, height=h * 0.9,
               color=[GOOD if v > 0 else BAD for v in delta],
               alpha=1.0 if i == 0 else 0.55,
               edgecolor=SURFACE, linewidth=1.6,
               label=f"{d['dataset']}/{d['variant']}")
    a.axvline(0, color=INK, lw=1.2)
    a.set_yticks(yv); a.set_yticklabels(add, fontsize=8.5); a.invert_yaxis()
    a.set_xlabel("Δ AUC when added to the embedding ranker")
    a.set_title("(e) Marginal contribution", color=INK, weight="bold", loc="left", pad=20)
    a.legend(fontsize=8, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.005))
    a.grid(axis="y", visible=False)

    # --- (f) naive fusion vs selected fusion
    a = ax[1][2]
    groups = ["embeddings\nalone", "all features\nsummed", "only the ones\nthat helped"]
    x = np.arange(len(groups)); w = 0.38
    for i, d in enumerate(fe[:2]):
        off = (i - (len(fe[:2]) - 1) / 2) * w
        vals = [d["emb_baseline"]["auc"], d["combined"]["all_features"]["auc"],
                d["combined"]["helpful_only"]["auc"]]
        a.bar(x + off, vals, w * 0.9, color=C[i], edgecolor=SURFACE, linewidth=1.8,
              label=f"{d['dataset']}/{d['variant']}")
        for xi, v in zip(x + off, vals):
            a.annotate(f"{v:.3f}", (xi, v), xytext=(0, 3), textcoords="offset points",
                       ha="center", fontsize=8, color=INK)
    a.axhline(0.5, color=INK, lw=1.1, ls="--", alpha=0.7)
    a.set_xticks(x); a.set_xticklabels(groups, fontsize=8.5)
    a.set_ylabel("AUC")
    a.set_ylim(0.44, max(d["combined"]["helpful_only"]["auc"] for d in fe[:2]) + 0.04)
    a.set_title("(f) Naive fusion loses", color=INK, weight="bold", loc="left")
    a.legend(fontsize=8, frameon=False, loc="upper left")
    a.grid(axis="x", visible=False)

    fig.suptitle("Q3/Q4 ablation — how the user is built, and which stored features earn their place",
                 color=INK, weight="bold", x=0.006, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(OUT / "fig14_features.png", dpi=160); plt.close(fig)
    print("wrote", OUT / "fig14_features.png")


if __name__ == "__main__":
    main()
