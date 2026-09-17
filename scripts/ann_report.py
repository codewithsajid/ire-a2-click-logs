"""Turn the ANN ablation JSONs into a markdown table for the design note."""
from __future__ import annotations

import json
from pathlib import Path

Q3 = Path("reports/q3")


def op_table(p: Path) -> str:
    d = json.loads(p.read_text())
    ex = next(r for r in d["rows"] if r["index"] == "faiss flat (exact)")
    ci = ex.get("recall@100_ci95")
    note = ""
    if ci:
        half = 100 * (ci[1] - ci[0]) / 2 / max(ex["recall@100"], 1e-9)
        note = (f"\nExact `task r@100` is {ex['recall@100']:.4f}, 95% CI "
                f"[{ci[0]:.4f}, {ci[1]:.4f}]. An approximate index cannot beat exact "
                f"search on the objective it approximates, so every **Δ task** below "
                f"±{half:.1f}% is noise — including the positive ones.\n")
    L = [f"### {d['tag']} — {d['universe']:,} candidates × {d['dim']}d, "
         f"{d['n_queries']:,} queries, {d['threads']} thread\n" + note,
         "| index | build (s) | size (MB) | q/s | ANN r@100 | task r@100 | Δ task |",
         "|---|--:|--:|--:|--:|--:|--:|"]
    for r in d["rows"]:
        delta = (r["recall@100"] - ex["recall@100"]) / ex["recall@100"] * 100
        L.append(f"| {r['index']} | {r['build_s']:.2f} | {r['bytes']/2**20:.1f} | "
                 f"{r['qps']:,.0f} | {r['ann_recall@100']:.4f} | {r['recall@100']:.4f} | "
                 f"{delta:+.1f}% |")
    return "\n".join(L) + "\n"


def scale_table(p: Path) -> str:
    d = json.loads(p.read_text())
    names = list(dict.fromkeys(r["index"] for r in d["rows"]))
    sizes = sorted({r["n"] for r in d["rows"]})
    L = [f"### scale sweep — {d['tag']}, {d['dim']}d, {d['n_queries']:,} queries, "
         f"{d['threads']} thread\n",
         "| index | " + " | ".join(f"N={n:,}" for n in sizes) + " |",
         "|---|" + "--:|" * len(sizes)]
    for metric, fmt, title in [("qps", "{:,.0f}", "throughput (q/s)"),
                               ("build_s", "{:.2f}", "build time (s)"),
                               ("ann_recall@100", "{:.4f}", "ANN recall@100")]:
        L.append(f"| **{title}** |" + " |" * len(sizes))
        for nm in names:
            cells = []
            for n in sizes:
                m = [r for r in d["rows"] if r["index"] == nm and r["n"] == n]
                cells.append(fmt.format(m[0][metric]) if m else "—")
            L.append(f"| {nm} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    out = ["# Q3 ablation: ANN index and its parameters\n",
           "Every row searches the *same* vectors; the exact `IndexFlatIP` row is the\n"
           "reference, so any difference is attributable to the index alone. `ANN r@100`\n"
           "is overlap with the exact top-100 (the index's own fidelity); `task r@100` is\n"
           "the assignment's recall against the clicked articles. Timings are\n"
           "single-threaded on purpose — this box has 48 cores, and letting BLAS use all\n"
           "of them makes brute force look like an ANN index.\n"]
    for f in sorted(Q3.glob("ann_operating_*.json")):
        out.append(op_table(f))
    for f in sorted(Q3.glob("ann_scale_*.json")):
        out.append(scale_table(f))
    p = Q3 / "ann_ablation.md"
    p.write_text("\n".join(out))
    print("wrote", p)
