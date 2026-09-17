"""Figures for Q4: ranking quality with confidence intervals, the beyond-accuracy
trade-off, and the size of the future-leak illusion (Q9).

Conventions as in scripts/plots.py -- fixed categorical order, one y-axis per
panel, recessive grid, legend whenever two series share an axis.
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
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#d9d8d4", "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.grid": True, "grid.color": "#ececea", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
})
OUT = Path("reports/figures"); OUT.mkdir(parents=True, exist_ok=True)
Q4 = Path("reports/q4")
LEGAL = ["random", "pop_prior", "ctr_prior", "recency", "bm25", "emb", "hybrid_rrf"]


def load(mode: str = "shipped", variant: str | None = None) -> dict:
    """Result sets keyed by dataset/variant.

    Figures compare the two *datasets*, so they are generated once per variant
    rather than mixing scales onto one axis -- EB-NeRD small and large are the
    same corpus at 6x the size, and overlaying them would read as four systems.
    """
    out = {}
    for f in sorted(Q4.glob(f"q4_*_{mode}.json")):
        d = json.loads(f.read_text())
        if variant and d["variant"] != variant:
            continue
        out[f"{d['dataset']}/{d['variant']}"] = d
    return out


def _grouped(ax, ds: dict, metric: str, ci: bool, ref: float | None = None):
    names = [r for r in LEGAL if all(r in d["rankers"] for d in ds.values())]
    y = np.arange(len(names))
    h = 0.38
    for i, (tag, d) in enumerate(ds.items()):
        off = (i - (len(ds) - 1) / 2) * h
        vals = np.array([d["rankers"][n][metric] for n in names])
        ax.barh(y + off, vals, height=h * 0.9, color=C[i], label=tag,
                edgecolor=SURFACE, linewidth=2)
        if ci:
            lo = np.array([d["rankers"][n][f"{metric}_ci95"][0] for n in names])
            hi = np.array([d["rankers"][n][f"{metric}_ci95"][1] for n in names])
            ax.errorbar(vals, y + off, xerr=[vals - lo, hi - vals], fmt="none",
                        ecolor=INK2, elinewidth=1.2, capsize=2.5, alpha=0.8)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    if ref is not None:
        ax.axvline(ref, color=INK, lw=1.2, ls="--", alpha=0.75)
    ax.grid(axis="y", visible=False)
    return names


def fig_ranking(ds: dict, sfx: str = ""):
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.9))

    _grouped(ax[0], ds, "ndcg@10", ci=True)
    ax[0].set_xlabel("nDCG@10 (bars = 95% bootstrap CI)")
    ax[0].set_title("(a) Ranking quality", color=INK, weight="bold", loc="left", pad=22)
    # above the axes: horizontal bars start at x=0 and fill the panel, so there is
    # no in-plot corner a legend can occupy without covering a row
    ax[0].legend(fontsize=9, frameon=False, ncol=len(ds), loc="lower left",
                 bbox_to_anchor=(0, 1.005))

    _grouped(ax[1], ds, "auc", ci=True, ref=0.5)
    ax[1].set_xlabel("AUC")
    ax[1].set_title("(b) AUC — dashed line is chance", color=INK, weight="bold", loc="left")
    ax[1].set_xlim(0.40, max(0.62, max(d["rankers"][n]["auc"]
                                       for d in ds.values() for n in LEGAL) + 0.03))
    worst = min((d["rankers"]["pop_prior"]["auc"], t) for t, d in ds.items())
    note = ""
    if worst[0] < 0.5:
        note = (f"Prior popularity scores below chance on {worst[1]} ({worst[0]:.4f}): the "
                "week's clicks go to articles that did not exist when the feature window closed.")

    # (c) Q9 -- the leak
    tags = list(ds)
    metrics = ["auc", "ndcg@10"]
    x = np.arange(len(tags) * len(metrics))
    labels, safe, leak = [], [], []
    for t in tags:
        for m in metrics:
            labels.append(f"{t.split('/')[0]}\n{m.upper()}")
            safe.append(ds[t]["rankers"]["pop_prior"][m])
            leak.append(ds[t]["rankers"]["pop_oracle*"][m])
    w = 0.38
    ax[2].bar(x - w / 2, safe, w, color=C[0], label="serving-safe (prior window)",
              edgecolor=SURFACE, linewidth=2)
    ax[2].bar(x + w / 2, leak, w, color=C[7], label="counts clicks from the scored split",
              edgecolor=SURFACE, linewidth=2)
    for xi, (s, l) in enumerate(zip(safe, leak)):
        ax[2].annotate(f"{100*(l-s)/max(s,1e-9):+.0f}%", (xi, max(s, l)), xytext=(0, 4),
                       textcoords="offset points", ha="center", fontsize=9,
                       color=INK, weight="bold")
    ax[2].set_xticks(x); ax[2].set_xticklabels(labels, fontsize=9)
    ax[2].set_ylabel("metric value")
    ax[2].set_ylim(0, max(leak) * 1.22)
    ax[2].set_title("(c) Q9 — what one illegal feature buys", color=INK, weight="bold", loc="left")
    ax[2].legend(fontsize=8.5, frameon=False, loc="upper right")
    ax[2].grid(axis="x", visible=False)

    fig.suptitle("Q4 — ranking within the impression, the metric both leaderboards score",
                 color=INK, weight="bold", x=0.008, ha="left", fontsize=12)
    if note:
        fig.text(0.008, 0.008, note, fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.045, 1, 0.925))
    fig.savefig(OUT / f"fig8_q4_ranking{sfx}.png", dpi=160); plt.close(fig)
    print("wrote", OUT / f"fig8_q4_ranking{sfx}.png")


def fig_beyond(ds: dict, sfx: str = ""):
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.9))
    names = LEGAL

    for j, (key, xlab, title) in enumerate([
            ("ild@10", "intra-list diversity@10", "(a) Accuracy vs diversity"),
            ("novelty@10", "novelty@10  (mean −log₂ p)", "(b) Accuracy vs novelty")]):
        for i, (tag, d) in enumerate(ds.items()):
            xs = [d["rankers"][n][key] for n in names]
            ys = [d["rankers"][n]["ndcg@10"] for n in names]
            ax[j].scatter(xs, ys, s=52, color=C[i], alpha=0.85, edgecolor=SURFACE,
                          linewidth=1.5, label=tag, zorder=3)
            for n, xx, yy in zip(names, xs, ys):
                ax[j].annotate(n, (xx, yy), xytext=(4, 4), textcoords="offset points",
                               fontsize=7.5, color=INK2)
        ax[j].set_xlabel(xlab); ax[j].set_ylabel("nDCG@10")
        ax[j].set_title(title, color=INK, weight="bold", loc="left")
        ax[j].legend(fontsize=8.5, frameon=False, loc="best")

    # (c) slice deltas
    y = np.arange(len(names)); h = 0.38
    for i, (tag, d) in enumerate(ds.items()):
        off = (i - (len(ds) - 1) / 2) * h
        dv = [(d["rankers"][n]["head"]["ndcg@10"] or 0) - (d["rankers"][n]["tail"]["ndcg@10"] or 0)
              for n in names]
        ax[2].barh(y + off, dv, height=h * 0.9, color=C[i], label=tag,
                   edgecolor=SURFACE, linewidth=2)
    ax[2].axvline(0, color=INK, lw=1.2)
    ax[2].set_yticks(y); ax[2].set_yticklabels(names, fontsize=9); ax[2].invert_yaxis()
    ax[2].set_xlabel("nDCG@10  (head impressions − tail impressions)")
    ax[2].set_title("(c) Popularity bias by slice", color=INK, weight="bold", loc="left")
    ax[2].legend(fontsize=8.5, frameon=False, loc="lower right")
    ax[2].grid(axis="y", visible=False)
    ax[2].text(0.98, 0.04, "right = better on already-popular articles",
               transform=ax[2].transAxes, ha="right", fontsize=8.5, color=MUTED)

    fig.suptitle("Q4 — beyond-accuracy, and who each ranker is actually good for",
                 color=INK, weight="bold", x=0.008, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / f"fig9_q4_beyond{sfx}.png", dpi=160); plt.close(fig)
    print("wrote", OUT / f"fig9_q4_beyond{sfx}.png")


SEQ = ["#0e2847", "#1f5aa3", "#2a78d6", "#4a89da", "#7fabe4", "#b3cdf0", "#dbe8f8"]


def fig_rank_penalty(ds: dict, sfx: str = ""):
    """The discount every rank-weighted metric applies, and the rank distribution
    it is actually applied to."""
    fig, ax = plt.subplots(1, 1 + len(ds), figsize=(5.2 * (1 + len(ds)), 4.6))
    r = np.arange(1, 21)
    curves = [("nDCG gain  1/log₂(r+1)", 1 / np.log2(r + 1), C[0]),
              ("MRR / MAP  1/r", 1.0 / r, C[1]),
              ("RBP  0.8^(r−1)", 0.8 ** (r - 1), C[2]),
              ("hit@10  (no discount)", (r <= 10).astype(float), C[3])]
    for name, y, col in curves:
        ax[0].plot(r, y, color=col, marker="o", ms=4, mfc=SURFACE, mew=1.4, label=name)
    ax[0].set_xlabel("rank of the relevant item"); ax[0].set_ylabel("credit given")
    ax[0].set_xticks([1, 5, 10, 15, 20])
    ax[0].set_title("(a) How each metric penalises rank", color=INK, weight="bold", loc="left")
    ax[0].legend(fontsize=8.5, frameon=False)
    ax[0].text(0.97, 0.62, "MRR spends 50% of its\ncredit on rank 1–2;\nnDCG still pays 0.29\nat rank 10",
               transform=ax[0].transAxes, ha="right", va="top", fontsize=8.5, color=MUTED)

    for j, (tag, d) in enumerate(ds.items()):
        a = ax[1 + j]
        cap = d.get("rank_cap", 40)
        for i, n in enumerate(LEGAL):
            h = np.asarray(d["rankers"][n]["rank_profile"], dtype=float)
            surv = np.cumsum(h) / max(h.sum(), 1)
            a.plot(np.arange(1, len(surv) + 1), surv, color=C[i], label=n)
        a.set_xlim(1, 20); a.set_ylim(0, 1.02)
        a.set_xlabel("rank r"); a.set_ylabel("P(first relevant item at rank ≤ r)")
        a.set_xticks([1, 5, 10, 15, 20])
        a.set_title(f"({'bcde'[j]}) Where the first click lands — {tag}",
                    color=INK, weight="bold", loc="left")
        a.legend(fontsize=8, frameon=False, loc="lower right")

    fig.suptitle("Q4 — rank penalties: the weight a metric applies, and the distribution it applies it to",
                 color=INK, weight="bold", x=0.008, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / f"fig10_rank_penalty{sfx}.png", dpi=160); plt.close(fig)
    print("wrote", OUT / f"fig10_rank_penalty{sfx}.png")


def _label_ends(ax, ends, x):
    """Direct-label every series at its right end, pushed apart where curves converge.

    The rankers converge hard at K=20 -- without this the labels stack into an
    unreadable smear exactly where the interesting near-ties are.
    """
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.052
    placed = []
    for y, name, col in sorted(ends, reverse=True):
        if placed and y > placed[-1] - gap:
            y = placed[-1] - gap
        placed.append(y)
        ax.annotate(name, (x, y), xytext=(6, 0), textcoords="offset points",
                    fontsize=7.5, color=col, va="center", weight="bold",
                    annotation_clip=False)


def fig_cutoffs(ds: dict, sfx: str = ""):
    """Ablation over the cutoff K, for a discounted and an undiscounted metric."""
    fig, ax = plt.subplots(2, len(ds), figsize=(7.8 * len(ds), 8.4), squeeze=False)
    for j, (tag, d) in enumerate(ds.items()):
        ks = d["cutoffs"]
        for row, fam, lab in [(0, "ndcg", "nDCG@K  (rank-discounted)"),
                              (1, "hit", "hit@K  (no discount)")]:
            a = ax[row][j]
            ends = []
            for i, n in enumerate(LEGAL):
                y = [d["rankers"][n][f"{fam}@{k}"] for k in ks]
                a.plot(ks, y, color=C[i], marker="o", ms=5, mfc=SURFACE, mew=1.6, label=n)
                ends.append((y[-1], n, C[i]))
            _label_ends(a, ends, ks[-1])
            a.set_xscale("log"); a.set_xticks(ks)
            a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            a.set_xlim(ks[0] * 0.85, ks[-1] * 1.9)
            a.set_xlabel("cutoff K"); a.set_ylabel(lab)
            a.set_title(f"({'abcd'[row*len(ds)+j]}) {tag} — {fam}@K", color=INK,
                        weight="bold", loc="left")
    fig.suptitle("Q4 — cutoff ablation: how much of a metric's verdict is the cut, "
                 "and how much is the discount",
                 color=INK, weight="bold", x=0.006, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(OUT / f"fig11_metric_cutoffs{sfx}.png", dpi=160); plt.close(fig)
    print("wrote", OUT / f"fig11_metric_cutoffs{sfx}.png")


def fig_agreement(ds: dict, sfx: str = ""):
    """Does the choice of metric change which system you would ship?"""
    metrics = ["auc", "mrr", "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "ndcg@20",
               "hit@1", "hit@5", "hit@10"]
    fig, ax = plt.subplots(1, len(ds), figsize=(8.0 * len(ds), 4.6), squeeze=False)
    # rank 1 is the darkest step: a sequential ramp reads dark = most, and the
    # cell text is white, which needs the good cells dark to stay legible
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    for j, (tag, d) in enumerate(ds.items()):
        a = ax[0][j]
        M = np.zeros((len(LEGAL), len(metrics)))
        for c, m in enumerate(metrics):
            vals = np.array([d["rankers"][n][m] for n in LEGAL])
            M[:, c] = len(LEGAL) + 1 - vals.argsort().argsort() - 1     # 1 = best
        im = a.imshow(M, cmap=cmap, vmin=1, vmax=len(LEGAL), aspect="auto")
        for r in range(len(LEGAL)):
            for c in range(len(metrics)):
                a.text(c, r, f"{int(M[r, c])}", ha="center", va="center", fontsize=8.5,
                       color=SURFACE if M[r, c] <= 3 else INK, weight="bold")
        flips = {metrics[c] for c in range(len(metrics))
                 if LEGAL[int(M[:, c].argmin())] != LEGAL[int(M[:, 0].argmin())]}
        a.set_xlabel(("winner changes at: " + ", ".join(sorted(flips))) if flips
                     else "same winner under every metric", fontsize=8.5, color=MUTED)
        a.set_xticks(range(len(metrics)))
        a.set_xticklabels([m.upper().replace("NDCG", "nDCG") for m in metrics],
                          rotation=40, ha="right", fontsize=8)
        a.set_yticks(range(len(LEGAL))); a.set_yticklabels(LEGAL, fontsize=9)
        a.set_title(f"({'ab'[j]}) {tag} — rank of each ranker (1 = best)",
                    color=INK, weight="bold", loc="left")
        a.grid(False)
    fig.suptitle("Q4 — metric agreement: rank of each system under ten metric definitions "
                 "(1 = best)",
                 color=INK, weight="bold", x=0.006, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / f"fig12_metric_agreement{sfx}.png", dpi=160); plt.close(fig)
    print("wrote", OUT / f"fig12_metric_agreement{sfx}.png")


if __name__ == "__main__":
    variants = sorted({json.loads(f.read_text())["variant"]
                       for f in Q4.glob("q4_*_shipped.json")})
    if not variants:
        raise SystemExit("no reports/q4/*_shipped.json -- run `make q4` first")
    for v in variants:
        ds = load("shipped", variant=v)
        sfx = "" if v == "small" else f"_{v}"
        fig_ranking(ds, sfx)
        fig_beyond(ds, sfx)
        if "rank_profile" in next(iter(ds.values()))["rankers"]["random"]:
            fig_rank_penalty(ds, sfx)
            fig_cutoffs(ds, sfx)
            fig_agreement(ds, sfx)
