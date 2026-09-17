"""Figure for the similarity-threshold ablation.

The question is never "does a cutoff raise recall" -- it cannot, it only removes
candidates. It is whether a cutoff spends a *reduced* candidate budget better
than simply taking the top K'. So every frontier is drawn against that control.
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
Q3 = Path("reports/q3")
RANKS = [1, 10, 50, 100, 200]


def main():
    files = sorted(Q3.glob("threshold_*.json"))
    if not files:
        raise SystemExit("no reports/q3/threshold_*.json -- run scripts/q3_threshold.py")
    ds = [json.loads(f.read_text()) for f in files]
    n = len(ds)
    fig, ax = plt.subplots(2, max(n, 3), figsize=(5.2 * max(n, 3), 9.0), squeeze=False)

    # --- top row: the budget/recall frontier, one panel per corpus
    for j, d in enumerate(ds):
        a = ax[0][j]
        tag = f"{d['dataset']}/{d['variant']}"
        base = d["baseline"]["recall@200"]
        for key, col, lab, mk in [("global", C[0], "global cutoff τ", "o"),
                                  ("relative", C[1], "per-user τ = α·best", "s")]:
            rows = sorted(d[key], key=lambda r: r["mean_kept"])
            a.plot([r["mean_kept"] for r in rows], [r["recall@200"] for r in rows],
                   color=col, marker=mk, ms=5, mfc=SURFACE, mew=1.5, label=lab)
        rows = sorted(d["global"], key=lambda r: r["mean_kept"])
        a.plot([r["mean_kept"] for r in rows], [r["fixed_topk_control"] for r in rows],
               color=INK, ls="--", lw=1.6, marker="^", ms=5, mfc=SURFACE, mew=1.4,
               label="fixed top-K′ (control)")
        a.axhline(base, color=MUTED, lw=0.9, ls=":")
        a.set_xlabel("mean candidates kept per user")
        a.set_ylabel("recall@200")
        a.set_title(f"({'abc'[j]}) {tag} — universe {d['universe']:,}",
                    color=INK, weight="bold", loc="left")
        a.legend(fontsize=8, frameon=False, loc="lower right")
        a.text(0.03, 0.88, f"dotted: full top-200\ntop-200 = "
               f"{d['topk_share_of_universe']:.1%} of the universe",
               transform=a.transAxes, va="top", fontsize=8.5, color=MUTED)
    for j in range(n, ax.shape[1]):
        ax[0][j].axis("off")

    # --- bottom row: why the frontier looks the way it does
    a = ax[1][0]
    for i, d in enumerate(ds):
        p = d["score_percentiles"]
        med = [p[f"rank{r}"]["50"] if "50" in p[f"rank{r}"] else p[f"rank{r}"][50] for r in RANKS]
        lo = [p[f"rank{r}"].get("5", p[f"rank{r}"].get(5)) for r in RANKS]
        hi = [p[f"rank{r}"].get("95", p[f"rank{r}"].get(95)) for r in RANKS]
        a.plot(RANKS, med, color=C[i], marker="o", ms=5, mfc=SURFACE, mew=1.5,
               label=f"{d['dataset']}/{d['variant']}")
        a.fill_between(RANKS, lo, hi, color=C[i], alpha=0.13, linewidth=0)
    a.set_xscale("log"); a.set_xticks(RANKS)
    a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a.set_xlabel("rank"); a.set_ylabel("cosine similarity")
    a.set_title("(d) Why a cutoff struggles: score compression", color=INK, weight="bold", loc="left")
    a.legend(fontsize=8.5, frameon=False, loc="lower left")
    a.text(0.97, 0.95, "band = p5–p95 across users", transform=a.transAxes,
           ha="right", va="top", fontsize=8.5, color=MUTED)

    a = ax[1][1]
    for i, d in enumerate(ds):
        for key, ls, lab in [("global", "-", "global τ"), ("relative", "--", "per-user α")]:
            rows = sorted(d[key], key=lambda r: r["mean_kept"])
            a.plot([r["mean_kept"] for r in rows], [100 * r["empty_users"] for r in rows],
                   color=C[i], ls=ls, marker="o", ms=4, mfc=SURFACE, mew=1.2,
                   label=f"{d['dataset']}/{d['variant']} {lab}")
    a.set_xlabel("mean candidates kept per user")
    a.set_ylabel("users left with no candidates (%)")
    a.set_title("(e) The coverage a global cutoff costs", color=INK, weight="bold", loc="left")
    a.legend(fontsize=7.5, frameon=False, loc="upper right")

    a = ax[1][2]
    # below ~20 kept both sides are near zero, so the ratio explodes on noise;
    # the comparison only means anything at budgets anyone would actually serve
    FLOOR = 20
    for i, d in enumerate(ds):
        rows = [r for r in sorted(d["global"], key=lambda r: r["mean_kept"])
                if r["mean_kept"] >= FLOOR]
        a.plot([r["mean_kept"] for r in rows],
               [100 * r["vs_control"] / max(r["fixed_topk_control"], 1e-9) for r in rows],
               color=C[i], marker="o", ms=5, mfc=SURFACE, mew=1.5,
               label=f"{d['dataset']}/{d['variant']}")
    a.axhline(0, color=INK, lw=1.2)
    a.set_xlabel(f"mean candidates kept per user  (≥ {FLOOR})")
    a.set_ylabel("recall@200 vs fixed top-K′ (%)")
    a.set_title("(f) Does the cutoff beat its own budget?", color=INK, weight="bold", loc="left")
    a.legend(fontsize=8.5, frameon=False, loc="upper right")
    a.text(0.03, 0.06, "above 0 = adaptive allocation wins", transform=a.transAxes,
           fontsize=8.5, color=MUTED)
    for j in range(3, ax.shape[1]):
        ax[1][j].axis("off")

    fig.suptitle("Q3 ablation — similarity thresholds: what a cutoff costs, and what it buys",
                 color=INK, weight="bold", x=0.006, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(OUT / "fig13_threshold.png", dpi=160); plt.close(fig)
    print("wrote", OUT / "fig13_threshold.png")


if __name__ == "__main__":
    main()
