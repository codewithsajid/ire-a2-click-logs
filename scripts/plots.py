"""Figures for Q2/Q3: BM25 mechanics on our corpora, the (k1,b) ablation, and
the lexical-vs-semantic comparison.

Palette and rules follow the project's dataviz conventions: fixed categorical
order (never cycled), a single-hue sequential ramp for the heatmap, one y-axis
per panel (distributions get their own panel rather than a second axis),
recessive grid, direct labels where there are few enough series to place them.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import PercentFormatter

import sys; sys.path.insert(0, str(Path(__file__).parent))
from q4_eval import tuned_bm25
from newsrec.lexical import BM25Index
from newsrec.store import FeatureStore

# categorical slots, fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURFACE = "#fcfcfb"
SEQ = ["#dbe8f8", "#b3cdf0", "#7fabe4", "#4a89da", "#2a78d6", "#1f5aa3", "#153c६e".replace("६",""), "#0e2847"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#d9d8d4", "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.grid": True, "grid.color": "#ececea", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
})
OUT = Path("reports/figures"); OUT.mkdir(parents=True, exist_ok=True)


def fig_mechanics():
    """The three BM25 knobs, drawn as functions."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) term-frequency saturation
    tf = np.linspace(0, 20, 400)
    for i, k1 in enumerate([0.5, 1.2, 2.0, 5.0]):
        y = tf * (k1 + 1) / (tf + k1)
        ax[0].plot(tf, y, color=C[i])
        ax[0].annotate(f"k₁={k1}", (tf[-1], y[-1]), xytext=(-2, 3), textcoords="offset points",
                       color=C[i], fontsize=9, ha="right", weight="bold")
        ax[0].axhline(k1 + 1, color=C[i], lw=0.8, ls=":", alpha=0.5)
    ax[0].set_title("Term-frequency saturation", color=INK, weight="bold", loc="left")
    ax[0].set_xlabel("term frequency f(t, D)"); ax[0].set_ylabel("saturation factor")
    ax[0].text(0.03, 0.95, "dotted = asymptote k₁+1", transform=ax[0].transAxes,
               fontsize=8.5, color=MUTED, va="top")

    # (b) IDF
    N = 100_000
    df = np.linspace(1, N, 2000)
    lucene = np.log(1 + (N - df + 0.5) / (df + 0.5))
    textbook = np.log((N - df + 0.5) / (df + 0.5))
    ax[1].plot(df / N, lucene, color=C[0], label="Lucene:  ln(1 + (N−df+0.5)/(df+0.5))")
    ax[1].plot(df / N, textbook, color=C[1], label="textbook:  ln((N−df+0.5)/(df+0.5))")
    ax[1].axhline(0, color=MUTED, lw=1)
    ax[1].axvline(0.5, color=MUTED, lw=0.8, ls="--")
    ax[1].annotate("textbook IDF crosses zero at df = 50%\n(a match on a stopword would\nsubtract from the score)",
                   (0.5, 0), xytext=(0.53, 4), fontsize=8.5, color=INK2,
                   arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax[1].set_title("Inverse document frequency", color=INK, weight="bold", loc="left")
    ax[1].set_xlabel("document frequency (share of corpus)"); ax[1].set_ylabel("IDF")
    ax[1].xaxis.set_major_formatter(PercentFormatter(1.0))
    ax[1].legend(frameon=False, fontsize=8.5, loc="upper right")

    # (c) length normalisation
    r = np.linspace(0, 3, 400)
    for i, b in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        y = 1 - b + b * r
        ax[2].plot(r, y, color=C[i])
        ax[2].annotate(f"b={b}", (r[-1], y[-1]), xytext=(-2, 2), textcoords="offset points",
                       color=C[i], fontsize=9, ha="right", weight="bold")
    ax[2].axvline(1, color=MUTED, lw=0.8, ls="--")
    ax[2].set_title("Length normalisation", color=INK, weight="bold", loc="left")
    ax[2].set_xlabel("|D| / avgdl"); ax[2].set_ylabel("denominator multiplier")
    ax[2].text(0.03, 0.95, "higher multiplier = lower score\n(dashed = average-length doc)",
               transform=ax[2].transAxes, fontsize=8.5, color=MUTED, va="top")
    fig.suptitle("BM25 has three knobs: how fast repetition stops helping, how much rarity is worth, "
                 "and how hard length is punished", color=INK, weight="bold", x=0.007, ha="left", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_bm25_mechanics.png", dpi=160, bbox_inches="tight")
    print("  fig1_bm25_mechanics.png")


def fig_corpus(idxs: dict[str, BM25Index]):
    """Why those knobs behave the way they do *here*: news text is short and
    almost never repeats a term."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (name, idx) in enumerate(idxs.items()):
        tf = idx.tf
        vals, counts = np.unique(np.clip(tf, 1, 6), return_counts=True)
        share = counts / counts.sum()
        ax[0].plot(vals, share, marker="o", ms=7, color=C[i], label=name)
        for x, y in zip(vals, share):
            if x <= 3:
                ax[0].annotate(f"{y:.0%}", (x, y), xytext=(0, 7), textcoords="offset points",
                               ha="center", fontsize=8.5, color=C[i], weight="bold")
    ax[0].set_yscale("log")
    ax[0].set_title("Term repeats within one article", color=INK, weight="bold", loc="left")
    ax[0].set_xlabel("term frequency in document (clipped at 6)")
    ax[0].set_ylabel("share of postings (log)")
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].text(0.35, 0.85, "almost every posting has f=1,\nso k₁ has little to bite on",
               transform=ax[0].transAxes, fontsize=8.5, color=INK2)

    for i, (name, idx) in enumerate(idxs.items()):
        dl = idx.doc_len[idx.doc_len > 0]
        ax[1].hist(dl / idx.avgdl, bins=60, range=(0, 3), histtype="step", lw=2,
                   color=C[i], label=f"{name} (avgdl {idx.avgdl:.0f})", density=True)
    ax[1].axvline(1, color=MUTED, lw=0.8, ls="--")
    ax[1].set_title("Document length, relative to corpus average", color=INK, weight="bold", loc="left")
    ax[1].set_xlabel("|D| / avgdl"); ax[1].set_ylabel("density")
    ax[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_corpus_shape.png", dpi=160, bbox_inches="tight")
    print("  fig2_corpus_shape.png")


def fig_grid(reports: dict[str, dict]):
    """(k1, b) ablation as a heatmap -- sequential single hue, one panel each."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#2a78d6", "#0e2847"])
    n = len(reports)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 4.4))
    axes = np.atleast_1d(axes)
    for ax, (name, rep) in zip(axes, reports.items()):
        g = pl.DataFrame(rep["grid"])
        k1s = sorted(g["k1"].unique().to_list()); bs = sorted(g["b"].unique().to_list())
        M = np.zeros((len(k1s), len(bs)))
        for r in g.iter_rows(named=True):
            M[k1s.index(r["k1"]), bs.index(r["b"])] = r["recall@100"]
        im = ax.imshow(M, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(bs)), [f"{b:g}" for b in bs])
        ax.set_yticks(range(len(k1s)), [f"{k:g}" for k in k1s])
        ax.set_xlabel("b  (length normalisation)"); ax.set_ylabel("k₁  (saturation)")
        best = M.max()
        for i in range(len(k1s)):
            for j in range(len(bs)):
                is_best = M[i, j] == best
                ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if M[i, j] > best * 0.97 else INK,
                        weight="bold" if is_best else "normal")
        ax.grid(False)
        spread = (M.max() - M.min()) / M.min() * 100
        ax.set_title(f"{name} — recall@100 across the grid  (spread {spread:.1f}%)",
                     color=INK, weight="bold", loc="left")
    fig.suptitle("b matters, k₁ barely does: length normalisation is the knob worth tuning on news text",
                 color=INK, weight="bold", x=0.007, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_k1_b_grid.png", dpi=160, bbox_inches="tight")
    print("  fig3_k1_b_grid.png")


def fig_recall(q3: dict[str, dict]):
    """Lexical vs semantic vs the reference points, per dataset."""
    ks = [50, 100, 200]
    order = ["random", "popularity", "recency", "bm25"]
    fig, axes = plt.subplots(1, len(q3), figsize=(6.6 * len(q3), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (name, rep) in zip(axes, q3.items()):
        methods = order + [m for m in rep["methods"] if m.startswith("emb:")]
        methods = [m for m in methods if m in rep["methods"]]
        ends = []
        for i, m in enumerate(methods):
            y = [rep["methods"][m][f"recall@{k}"] for k in ks]
            ls = "--" if m == "random" else "-"
            ax.plot(ks, y, marker="o", ms=7, color=C[i % len(C)], ls=ls)
            ends.append([y[-1], m, C[i % len(C)]])
        # nudge end-labels apart so close series stay readable
        span = max(e[0] for e in ends) - min(e[0] for e in ends)
        gap = span * 0.055
        ends.sort()
        for j in range(1, len(ends)):
            if ends[j][0] - ends[j - 1][0] < gap:
                ends[j][0] = ends[j - 1][0] + gap
        for ypos, m, col in ends:
            ax.annotate(m, (ks[-1], ypos), xytext=(8, 0), textcoords="offset points",
                        fontsize=9, color=col, va="center", weight="bold")
        ax.set_xticks(ks); ax.set_xlim(40, 345)
        ax.set_xlabel("K (candidates retrieved)"); ax.set_ylabel("recall@K")
        ax.set_title(f"{name} — universe {rep['universe']:,} articles, ceiling {rep['ceiling']:.2f}",
                     color=INK, weight="bold", loc="left")

    fig.suptitle("Which signal wins depends on the corpus: content on EB-NeRD, freshness on MIND",
                 color=INK, weight="bold", x=0.007, ha="left", y=1.06)
    fig.text(0.007, 0.99, "dashed = random baseline;  articles the user already read are excluded from every retriever",
             fontsize=9, color=MUTED, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_recall_comparison.png", dpi=160, bbox_inches="tight")
    print("  fig4_recall_comparison.png")


def fig_explain(fs: FeatureStore, idx: BM25Index, out_name="fig5_score_decomposition.png"):
    """One worked example: which terms actually carried a retrieved document."""
    import scipy.sparse as sp
    hist = fs.history("test").select("user_idx", "article_idx").collect()
    row = max(range(min(400, hist.height)), key=lambda i: len(hist["article_idx"][i]))
    docs = np.asarray(hist["article_idx"][row].to_list())[-30:]
    Q = idx.queries_from_history([docs])
    _, top = idx.search_sparse(Q, top_k=5)
    doc = int(top[0, 0])
    ex = idx.explain(Q[0], doc, top_n=10)
    titles = fs.articles().select("title").collect()["title"]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y = np.arange(ex.height)[::-1]
    ax.barh(y, ex["contribution"].to_numpy(), color=C[0], height=0.62)
    ax.set_yticks(y, ex["stem"].to_list())
    for yi, (c, d) in enumerate(zip(ex["contribution"].to_list(), ex["df"].to_list())):
        ax.annotate(f"{c:.1f}   (df={d:,})", (c, y[yi]), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.set_xlabel("contribution to BM25 score  (query weight × IDF × saturation)")
    ax.set_xlim(0, ex["contribution"].max() * 1.35)
    ax.set_title("Why this article was retrieved", color=INK, weight="bold", loc="left")
    ax.text(0, 1.02, f"retrieved: “{titles[doc][:80]}”", transform=ax.transAxes,
            fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / out_name, dpi=160, bbox_inches="tight")
    print(f"  {out_name}")
    return doc, ex


if __name__ == "__main__":
    idxs, q3, grids = {}, {}, {}
    for ds, var, label in [("ebnerd", "small", "EB-NeRD (da)"), ("mind", "small", "MIND (en)")]:
        fs = FeatureStore(ds, var)
        idxs[label] = BM25Index.build(fs.texts(), lang=fs.lang())
        cmp_path = Path("reports/q3/compare_dropseen.json")
        if cmp_path.exists():
            c = json.loads(cmp_path.read_text()).get(f"{ds}/{var}")
            if c:
                q3[label] = {
                    "universe": c["universe"], "ceiling": c["ceiling"],
                    "methods": {m: {f"recall@{k}": v["no-seen"][str(k)]
                                    for k in (50, 100, 200)}
                                for m, v in c["methods"].items() if "no-seen" in v},
                }
        g = Path(f"reports/q2/q2_bm25_{ds}_{var}.json")
        if g.exists():
            grids[label] = json.loads(g.read_text())
    fig_mechanics()
    fig_corpus(idxs)
    if grids:
        fig_grid(grids)
    if q3:
        fig_recall(q3)
    fs = FeatureStore("ebnerd", "small")
    fig_explain(fs, idxs["EB-NeRD (da)"].reweight(*tuned_bm25("ebnerd", "small", None, None)))
