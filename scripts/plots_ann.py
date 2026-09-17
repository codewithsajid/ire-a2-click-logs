"""Figures for the Q3 ANN ablation: the recall/throughput/memory trade-off at the
operating point, and the scale sweep that says where a graph index starts to pay.

Same conventions as scripts/plots.py -- fixed categorical order, one y-axis per
panel, recessive grid, direct labels where the series count allows.
"""
from __future__ import annotations

import json, re
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


def _num(s, key):
    m = re.search(rf"{key}=(\d+)", s)
    return int(m.group(1)) if m else None


def fig_pareto(tag: str):
    d = json.loads((Q3 / f"ann_operating_{tag}.json").read_text())
    rows = d["rows"]
    exact = next(r for r in rows if r["index"] == "faiss flat (exact)")
    dim = d["dim"]

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # --- (a) recall vs throughput, one line per index family/parameter setting
    def curve(pred, keyfn, color, name, marker="o"):
        pts = sorted([r for r in rows if pred(r)], key=keyfn)
        if not pts:
            return None
        x = [r["qps"] for r in pts]; y = [r["ann_recall@100"] for r in pts]
        ax[0].plot(x, y, color=color, marker=marker, ms=6, mfc=SURFACE, mew=2, label=name)
        return pts[-1], x[-1], y[-1]

    ends = []
    for i, M in enumerate((8, 16, 32, 64)):
        e = curve(lambda r, M=M: r["index"].startswith(f"hnsw M={M} efC=200") and "/L2" not in r["index"],
                  lambda r: _num(r["index"], "efS"), C[i], f"HNSW M={M}")
        if e: ends.append((f"M={M}", e[1], e[2], C[i]))
    for j, nl in enumerate((64, 256, 1024)):
        curve(lambda r, nl=nl: r["index"].startswith(f"ivf nlist={nl} "),
              lambda r: _num(r["index"], "nprobe"), C[4 + j], f"IVF nlist={nl}", marker="s")
    for r, col, mk, nm in [(exact, INK, "*", "flat (exact)"),
                           (next(r for r in rows if r["index"].startswith("numpy")), MUTED, "D", "numpy"),
                           (next(r for r in rows if r["index"].startswith("faiss SQ8")), C[7], "^", "SQ8")]:
        ax[0].plot(r["qps"], r["ann_recall@100"], marker=mk, ms=11 if mk == "*" else 7,
                   color=col, ls="none", label=nm)
    ax[0].axhline(1.0, color=INK, lw=0.8, ls=":", alpha=0.6)
    ax[0].axvline(exact["qps"], color=INK, lw=0.8, ls=":", alpha=0.6)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("throughput (queries / s, 1 thread)  →  better")
    ax[0].set_ylabel("ANN recall@100 vs exact")
    ax[0].set_title("(a) Fidelity vs speed", color=INK, weight="bold", loc="left")
    ax[0].legend(fontsize=8, frameon=False, ncol=2, loc="lower left")
    ax[0].text(0.98, 0.06, f"dotted: exact index\n({exact['qps']:,.0f} q/s, recall 1.0)",
               transform=ax[0].transAxes, ha="right", fontsize=8.5, color=MUTED)

    # --- (b) does the fidelity loss reach the task metric?
    xs = [r["ann_recall@100"] for r in rows]
    ys = [r["recall@100"] for r in rows]
    ax[1].axhline(exact["recall@100"], color=INK, lw=1.2, ls="--", alpha=0.7)
    ax[1].scatter(xs, ys, s=44, color=C[0], alpha=0.75, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    lo = min(ys + [exact["recall@100"]]); hi = max(ys + [exact["recall@100"]])
    pad = max((hi - lo) * 0.35, 1e-4)
    ax[1].set_ylim(lo - pad, hi + pad)
    ax[1].set_xlabel("ANN recall@100 (index fidelity)")
    ax[1].set_ylabel("task recall@100 (assignment metric)")
    ax[1].set_title("(b) What the approximation actually costs", color=INK, weight="bold", loc="left")
    ax[1].text(0.03, 0.95, f"dashed: exact = {exact['recall@100']:.4f}\n"
               f"worst config loses {100*(exact['recall@100']-min(ys))/exact['recall@100']:.1f}% of it",
               transform=ax[1].transAxes, va="top", fontsize=8.5, color=MUTED)

    # --- (c) memory
    fams, cols = [], []
    for r in rows:
        n = r["index"]
        fams.append("HNSW" if n.startswith("hnsw") else "IVF" if n.startswith("ivf nlist")
                    else "IVFPQ" if n.startswith("ivfpq") else "exact/SQ8")
    fam_col = {"HNSW": C[0], "IVF": C[1], "IVFPQ": C[2], "exact/SQ8": INK}
    for f in ["exact/SQ8", "HNSW", "IVF", "IVFPQ"]:
        sel = [r for r, g in zip(rows, fams) if g == f]
        if not sel:
            continue
        ax[2].scatter([r["bytes"] / d["universe"] for r in sel], [r["ann_recall@100"] for r in sel],
                      s=46, color=fam_col[f], alpha=0.8, edgecolor=SURFACE, linewidth=1.5, label=f)
    ax[2].axvline(4 * dim, color=MUTED, lw=0.9, ls=":")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("bytes per vector")
    ax[2].set_ylabel("ANN recall@100 vs exact")
    ax[2].set_title("(c) Memory vs fidelity", color=INK, weight="bold", loc="left")
    ax[2].legend(fontsize=8.5, frameon=False, loc="lower right")
    ax[2].text(4 * dim * 1.06, ax[2].get_ylim()[0] + 0.02, f"raw fp32\n({4*dim:,} B)",
               fontsize=8, color=MUTED)

    fig.suptitle(f"ANN ablation at the operating point — {tag}, {d['universe']:,} candidates × "
                 f"{dim}d, {d['n_queries']:,} user queries",
                 color=INK, weight="bold", x=0.008, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = OUT / f"fig6_ann_pareto_{tag}.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    print("wrote", p)


def fig_scale(tag: str):
    d = json.loads((Q3 / f"ann_scale_{tag}.json").read_text())
    rows = d["rows"]
    names = list(dict.fromkeys(r["index"] for r in rows))
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for i, nm in enumerate(names):
        sel = sorted([r for r in rows if r["index"] == nm], key=lambda r: r["n"])
        col = MUTED if nm.startswith("numpy") else C[i % len(C)]
        n = [r["n"] for r in sel]
        ax[0].plot(n, [r["qps"] for r in sel], color=col, marker="o", ms=5, mfc=SURFACE, mew=1.6, label=nm)
        ax[1].plot(n, [r["build_s"] for r in sel], color=col, marker="o", ms=5, mfc=SURFACE, mew=1.6)
        ax[2].plot(n, [r["ann_recall@100"] for r in sel], color=col, marker="o", ms=5, mfc=SURFACE, mew=1.6)
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("corpus size N (articles)")
    ax[0].set_yscale("log"); ax[0].set_ylabel("throughput (q/s, 1 thread)")
    ax[0].set_title("(a) Where the graph index starts to pay", color=INK, weight="bold", loc="left")
    ax[0].legend(fontsize=8, frameon=False, loc="lower left")
    ax[1].set_yscale("log"); ax[1].set_ylabel("index build time (s)")
    ax[1].set_title("(b) What it costs to build", color=INK, weight="bold", loc="left")
    ax[2].set_ylabel("ANN recall@100 vs exact")
    ax[2].set_title("(c) Fidelity as the corpus grows", color=INK, weight="bold", loc="left")
    fig.suptitle(f"ANN scale sweep — {tag}, {d['dim']}d, {d['n_queries']:,} queries, 1 thread",
                 color=INK, weight="bold", x=0.008, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = OUT / f"fig7_ann_scale_{tag}.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    import sys
    for f in sorted(Q3.glob("ann_operating_*.json")):
        fig_pareto(f.stem.replace("ann_operating_", ""))
    for f in sorted(Q3.glob("ann_scale_*.json")):
        fig_scale(f.stem.replace("ann_scale_", ""))
