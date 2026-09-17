"""Q4 JSONs -> the markdown tables the design note quotes."""
from __future__ import annotations

import json
from pathlib import Path

Q4 = Path("reports/q4")
METRICS = ("auc", "mrr", "ndcg@5", "ndcg@10")
LEGAL = ("random", "pop_prior", "ctr_prior", "recency", "bm25", "emb", "hybrid_rrf")


def table(d: dict) -> str:
    L = [f"### {d['dataset']}/{d['variant']} — {d['split']} split, history={d['history_mode']}\n",
         f"{d['n_impressions']:,} labelled impressions · {d['n_pairs']:,} candidate rows · "
         f"{d['n_clicks']:,} clicks · catalogue {d['catalogue']:,} · "
         f"BM25 k1={d['bm25']['k1']} b={d['bm25']['b']} · {d['n_boot']} bootstrap resamples\n",
         "| ranker | AUC | MRR | nDCG@5 | nDCG@10 (95% CI) | ILD@10 | novelty@10 | coverage@10 |",
         "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for name, r in d["rankers"].items():
        ci = r["ndcg@10_ci95"]
        L.append(f"| {'**' + name + '**' if name not in LEGAL else name} | "
                 f"{r['auc']:.4f} | {r['mrr']:.4f} | {r['ndcg@5']:.4f} | "
                 f"{r['ndcg@10']:.4f} ({ci[0]:.4f}–{ci[1]:.4f}) | "
                 f"{r['ild@10']:.4f} | {r['novelty@10']:.3f} | {r['coverage@10']:.4f} |")
    L += ["", "**Slices** (nDCG@10)", "",
          "| ranker | cold | warm | head | tail |", "|---|--:|--:|--:|--:|"]
    for name, r in d["rankers"].items():
        L.append(f"| {name} | " + " | ".join(
            f"{(r[s]['ndcg@10'] if r[s]['ndcg@10'] is not None else float('nan')):.4f}"
            for s in ("cold", "warm", "head", "tail")) + " |")
    n = d["rankers"]
    L += ["", f"Slice sizes: cold {n['random']['cold']['n']:,} / warm {n['random']['warm']['n']:,} "
              f"(history < {d['cold_threshold']}), head {n['random']['head']['n']:,} / "
              f"tail {n['random']['tail']['n']:,} (clicked article in the top 20% by prior "
              f"decayed clicks).", ""]
    return "\n".join(L)


def leak_section(ds: list[dict]) -> str:
    L = ["## Q9 — what a serving-time-unavailable feature buys\n",
         "`pop_oracle*` is the same popularity ranker as `pop_prior`, with one change: it counts",
         "clicks from *inside* the scored split instead of strictly before it. Nothing else",
         "differs — same candidates, same impressions, same metric code. The gap is the size of",
         "the illusion that a future-leaking feature creates.\n",
         "| dataset | metric | serving-safe | with future clicks | inflation |",
         "|---|---|--:|--:|--:|"]
    for d in ds:
        if d["history_mode"] != "shipped":
            continue
        safe, leak = d["rankers"]["pop_prior"], d["rankers"]["pop_oracle*"]
        for m in ("auc", "ndcg@10"):
            L.append(f"| {d['dataset']}/{d['variant']} | {m.upper()} | {safe[m]:.4f} | "
                     f"{leak[m]:.4f} | {100*(leak[m]-safe[m])/max(safe[m],1e-9):+.1f}% |")
    return "\n".join(L) + "\n"


def history_ablation(ds: list[dict]) -> str:
    by = {}
    for d in ds:
        by.setdefault((d["dataset"], d["variant"], d["split"]), {})[d["history_mode"]] = d
    L = ["## History-window ablation (shipped vs augmented)\n",
         "`augmented` appends every click observed in earlier splits to the history the dataset",
         "ships, cut strictly at the target split's start. It is legal at serving time — a real",
         "system remembers what it logged — but it is not what the leaderboard hands you.\n",
         "| dataset | ranker | nDCG@10 shipped | nDCG@10 augmented | Δ |",
         "|---|---|--:|--:|--:|"]
    for (ds_, var, sp), modes in sorted(by.items()):
        if len(modes) < 2:
            continue
        for r in ("bm25", "emb", "hybrid_rrf"):
            a = modes["shipped"]["rankers"][r]["ndcg@10"]
            b = modes["augmented"]["rankers"][r]["ndcg@10"]
            L.append(f"| {ds_}/{var} | {r} | {a:.4f} | {b:.4f} | {100*(b-a)/max(a,1e-9):+.1f}% |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    files = sorted(Q4.glob("q4_*.json"))
    ds = [json.loads(f.read_text()) for f in files]
    out = ["# Q4 — offline evaluation harness\n",
           "Ranking quality *within* the impression, which is what both leaderboards score.",
           "Q2/Q3 measured retrieval from the whole candidate universe; a retriever that never",
           "surfaces an article can still rank it correctly once the impression puts it in front",
           "of the user, so the two tables answer different questions and disagree on purpose.\n",
           "`random` is the calibration check: a correct AUC implementation must put it at 0.5000.\n"]
    out += [table(d) for d in ds]
    out.append(leak_section(ds))
    out.append(history_ablation(ds))
    p = Q4 / "q4_results.md"
    p.write_text("\n".join(out))
    print("wrote", p)
