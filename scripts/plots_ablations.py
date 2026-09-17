"""Figures for the three ablations that test choices nobody had measured.

Each panel answers one question: does the query need saturating, what belongs in
the index, and is one vector enough to describe a user. The shipped setting is
marked on every panel, because the point of these is whether a judgement call
made early survives contact with evidence.
"""
from __future__ import annotations

import json
import textwrap
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


def _load(pattern: str, root: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(root).glob(pattern))]


def _tag(d: dict) -> str:
    return f"{d['dataset']}/{d['variant']}"


def query_panel(ax, d):
    """Recall against query-term saturation. Each series is one cutoff."""
    labels = [r["k3"].split()[0] for r in d["runs"]]
    x = np.arange(len(labels))
    for i, k in enumerate((50, 100, 200)):
        y = np.array([r[f"recall@{k}"] for r in d["runs"]])
        ax.plot(x, y / y[0], "o-", color=C[i], ms=5, label=f"recall@{k}")
    ax.axhline(1.0, color=MUTED, lw=1, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("query-term saturation  k₃  (left = none, right = binary)")
    ax.set_ylabel("recall relative to no saturation")
    ax.set_title(f"{_tag(d)} — saturating the query only loses recall", loc="left", fontsize=11)
    ax.annotate("shipped", xy=(0, 1.0), xytext=(0.35, 1.02), color=GOOD, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=GOOD, lw=1.2))


def index_panel(ax, d):
    """Recall against index size: the trade each tokeniser choice actually makes."""
    def name(r):
        if not r["stopwords"] and not r["stemming"]: return "no stop, no stem"
        if not r["stopwords"]: return "no stopwords"
        if not r["stemming"]: return "no stemming"
        return r["field"]
    shipped = next(r for r in d["runs"]
                   if r["field"] == "title+abstract" and r["stopwords"] and r["stemming"])
    for i, r in enumerate(d["runs"]):
        is_ship = r is shipped
        ax.scatter(r["postings"] / 1e6, r["recall@100"], s=140 if is_ship else 70,
                   color=C[i % len(C)], zorder=3, edgecolor=SURFACE, linewidth=1.5,
                   marker="*" if is_ship else "o")
        ax.annotate(name(r) + (" (shipped)" if is_ship else ""),
                    (r["postings"] / 1e6, r["recall@100"]), textcoords="offset points",
                    xytext=(7, 4), fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    lo = min(r["postings"] for r in d["runs"]) / 1e6
    hi = max(r["postings"] for r in d["runs"]) / 1e6
    ax.set_xlim(lo * 0.55, hi * 3.2)      # room for the labels, which sit to the right
    ax.set_xlabel("postings (millions, log scale)")
    ax.set_ylabel("recall@100")
    ax.set_title(f"{_tag(d)} — a bigger index is not a better one", loc="left", fontsize=11)


def multi_panel(ax, d):
    """Recall against number of user vectors, split by how they are formed."""
    for i, mode in enumerate(("chunks", "kmeans")):
        rows = [r for r in d["runs"] if r["mode"] == mode]
        ks = [r["k"] for r in rows]
        ax.plot(ks, [r["recall@100"] for r in rows], "o-", color=C[i], ms=5,
                label=f"{mode} — all users")
    warm = [r for r in d["runs"] if r["mode"] == "kmeans"]
    ax.plot([r["k"] for r in warm], [r["cold_recall@100"] for r in warm], "s--",
            color=C[3], ms=4, lw=1.5, label="kmeans — cold users only")
    base = d["single_vector"]
    ax.axhline(base, color=MUTED, lw=1, ls=":")
    ax.annotate("one mean-pooled vector", xy=(1, base), xytext=(6, -13),
                textcoords="offset points", ha="left", fontsize=8.5, color=MUTED)
    ax.set_xlabel("user vectors per user  (k)")
    ax.set_ylabel("recall@100")
    ax.set_title(f"{_tag(d)} — interests are concurrent, not sequential", loc="left", fontsize=11)


def main():
    q = _load("query_*.json", "reports/q2")
    ix = _load("index_*.json", "reports/q2")
    mi = _load("multiinterest_*.json", "reports/q3")
    if not (q and ix and mi):
        raise SystemExit("run scripts/q2_query.py, q2_index.py and q3_multiinterest.py first")

    n = max(len(q), len(ix), len(mi))
    fig, ax = plt.subplots(3, n, figsize=(7.9 * n, 13.2), squeeze=False)
    for j, d in enumerate(q):
        query_panel(ax[0][j], d)
    for j, d in enumerate(ix):
        index_panel(ax[1][j], d)
    for j, d in enumerate(mi):
        multi_panel(ax[2][j], d)
    for row in ax:
        for a in row:
            if a.get_legend_handles_labels()[0]:
                a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=3,
                         frameon=False, fontsize=9)

    best_mi = max((r for d in mi for r in d["runs"]), key=lambda r: r["recall@100"])
    title = ("Three choices made on judgement, measured afterwards: the query is right to "
             "stay unsaturated, the index is right to stop at the abstract, and clustering "
             f"the history into {best_mi['k']} interests beats mean-pooling it — for users "
             "who have enough history to split.")
    # wrap to the figure, not past it: one dataset is half the width of two
    fig.suptitle("\n".join(textwrap.wrap(title, width=int(52 * n))),
                 y=0.995, fontsize=12.5, color=INK, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.955 if n > 1 else 0.94))
    p = OUT / "fig15_ablations.png"
    fig.savefig(p, dpi=150)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
