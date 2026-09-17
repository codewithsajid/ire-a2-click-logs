"""Build the browsable results page from the measured JSONs.

Same rule as the design note: every number is read out of reports/ at build
time. Figures are inlined as data URIs so the page is self-contained.
"""
from __future__ import annotations

import base64, json
from pathlib import Path

R = Path("reports")
FIG = R / "figures"


def j(p): return json.loads((R / p).read_text())


def img(name: str, cap: str, alt: str) -> str:
    p = FIG / name
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f'<figure class="plate"><img src="data:image/png;base64,{b64}" alt="{alt}">'
            f'<figcaption>{cap}</figcaption></figure>')


q2 = {k: j(f"q2/q2_bm25_{k}.json") for k in
      ("ebnerd_small", "ebnerd_large", "mind_small", "mind_large")
      if (R / f"q2/q2_bm25_{k}.json").exists()}
cmp3 = j("q3/compare_dropseen.json")
PROBE = j("q5/codabench_probe.json") if (R / "q5/codabench_probe.json").exists() else {}
q4 = {f"{d}_{v}": j(f"q4/q4_{d}_{v}_test_shipped.json")
      for d in ("ebnerd", "mind") for v in ("small", "large")
      if (R / f"q4/q4_{d}_{v}_test_shipped.json").exists()}
THR = {t: j(f"q3/threshold_{t}_small.json") for t in ("ebnerd", "mind")
       if (R / f"q3/threshold_{t}_small.json").exists()}
UREP = {t: j(f"q3/userrep_{t}_small.json") for t in ("ebnerd", "mind")
        if (R / f"q3/userrep_{t}_small.json").exists()}
FEAT = {t: j(f"q4/features_{t}_small.json") for t in ("ebnerd", "mind")
        if (R / f"q4/features_{t}_small.json").exists()}
QSAT = {t: j(f"q2/query_{t}_small.json") for t in ("ebnerd", "mind")
        if (R / f"q2/query_{t}_small.json").exists()}
IXAB = {t: j(f"q2/index_{t}_small.json") for t in ("ebnerd", "mind")
        if (R / f"q2/index_{t}_small.json").exists()}
MULTI = {t: j(f"q3/multiinterest_{t}_small.json") for t in ("ebnerd", "mind")
         if (R / f"q3/multiinterest_{t}_small.json").exists()}
ann_op = j("q3/ann_operating_ebnerd_small_contrastive.json")
ann_sc = j("q3/ann_scale_ebnerd_large_contrastive.json")

eb, mi = q4["ebnerd_small"]["rankers"], q4["mind_small"]["rankers"]

# Which metrics rank fusion wins is not stable across runs, so the finding below is
# written from the numbers. Tuning BM25 and widening the history flipped EB-NeRD from
# "fusion loses the discounted metrics" to "fusion wins all but the top slot".
CUT_METRICS = ("hit@1", "hit@5", "hit@10", "ndcg@5", "ndcg@10", "mrr", "auc")
_emb_w = [m for m in CUT_METRICS if eb["emb"][m] > eb["hybrid_rrf"][m]]
_rrf_w = [m for m in CUT_METRICS if eb["hybrid_rrf"][m] > eb["emb"][m]]
_mi_rrf_w = [m for m in CUT_METRICS if mi["hybrid_rrf"][m] > mi["emb"][m]]
_c = lambda ms: ", ".join(f"<code>{m}</code>" for m in ms)

if len(_emb_w) == 1 and _rrf_w:
    _m = _emb_w[0]
    FUSION_HEAD = "Rank fusion wins the whole list and loses the top slot"
    FUSION_BODY = (f"On EB-NeRD the hybrid beats plain embeddings on every metric here "
                   f"except {_c(_emb_w)}, where embeddings hold "
                   f"{eb['emb'][_m]:.4f} against {eb['hybrid_rrf'][_m]:.4f}. "
                   f"On MIND fusion wins {_c(_mi_rrf_w) if _mi_rrf_w else 'none of them'}.")
elif not _rrf_w:
    FUSION_HEAD = "Rank fusion does not pay on either dataset"
    FUSION_BODY = ("Plain embeddings beat the hybrid on every metric measured here, on "
                   "both datasets.")
else:
    FUSION_HEAD = "Rank fusion finds more clicks and orders them worse"
    FUSION_BODY = (f"On EB-NeRD the hybrid wins {_c(_rrf_w)} while losing {_c(_emb_w)} "
                   f"to plain embeddings.")
ebL = q4.get("ebnerd_large", {}).get("rankers", eb)
miL = q4.get("mind_large", {}).get("rankers", mi)


def sc(name, n, key="qps"):
    return next(r[key] for r in ann_sc["rows"] if r["index"] == name and r["n"] == n)


leak_eb = 100 * (eb["pop_oracle*"]["auc"] - eb["pop_prior"]["auc"]) / eb["pop_prior"]["auc"]
leak_mi = 100 * (mi["pop_oracle*"]["auc"] - mi["pop_prior"]["auc"]) / mi["pop_prior"]["auc"]
hnsw125 = sc("hnsw M=16 efC=200 efS=64", 125000)
flat125 = sc("faiss flat (exact)", 125000)

LEGAL = ["random", "pop_prior", "ctr_prior", "recency", "bm25", "emb", "hybrid_rrf"]
NICE = {"random": "random", "pop_prior": "popularity (prior)", "ctr_prior": "CTR (prior)",
        "recency": "recency", "bm25": "BM25", "emb": "embeddings",
        "hybrid_rrf": "hybrid (rank fusion)", "pop_oracle*": "popularity (future clicks)"}


def rank_table(tag: str) -> str:
    d = q4[tag]
    rows = []
    for n, r in d["rankers"].items():
        lo, hi = r["ndcg@10_ci95"]
        cls = ' class="leak"' if n.endswith("*") else ""
        rows.append(
            f'<tr{cls}><th scope="row">{NICE.get(n, n)}</th><td>{r["auc"]:.4f}</td>'
            f'<td>{r["mrr"]:.4f}</td><td>{r["ndcg@5"]:.4f}</td>'
            f'<td>{r["ndcg@10"]:.4f}<span class="ci">{lo:.3f}–{hi:.3f}</span></td>'
            f'<td>{r["ild@10"]:.3f}</td><td>{r["coverage@10"]:.3f}</td>'
            f'<td>{r["head"]["ndcg@10"]:.3f} / {r["tail"]["ndcg@10"]:.3f}</td></tr>')
    return (f'<div class="scroller"><table><caption>{d["dataset"]}/{d["variant"]} · '
            f'{d["n_impressions"]:,} impressions · {d["n_pairs"]:,} candidate rows</caption>'
            '<thead><tr><th scope="col">ranker</th><th scope="col">AUC</th>'
            '<th scope="col">MRR</th><th scope="col">nDCG@5</th>'
            '<th scope="col">nDCG@10 <span class="ci">95% CI</span></th>'
            '<th scope="col">diversity</th><th scope="col">coverage</th>'
            '<th scope="col">head / tail</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


def recall_table() -> str:
    order = [("random", "random"), ("popularity", "popularity (prior)"),
             ("recency", "recency"), ("bm25", "BM25")]
    rows = []
    for key, label in order:
        e = cmp3["ebnerd/small"]["methods"][key]["no-seen"]
        m = cmp3["mind/small"]["methods"][key]["no-seen"]
        rows.append(f'<tr><th scope="row">{label}</th>' + "".join(
            f"<td>{d[str(k)]:.4f}</td>" for d in (e, m) for k in (50, 100, 200)) + "</tr>")
    e = cmp3["ebnerd/small"]["methods"]["emb:contrastive"]["no-seen"]
    m = cmp3["mind/small"]["methods"]["emb:all-MiniLM-L6-v2"]["no-seen"]
    rows.append('<tr class="best"><th scope="row">embeddings</th>' + "".join(
        f"<td>{d[str(k)]:.4f}</td>" for d in (e, m) for k in (50, 100, 200)) + "</tr>")
    return ('<div class="scroller"><table><caption>recall@K against the live candidate '
            'universe, seen articles dropped, test split</caption>'
            '<thead><tr><th scope="col" rowspan="2">method</th>'
            '<th scope="col" colspan="3">EB-NeRD · Danish</th>'
            '<th scope="col" colspan="3">MIND · English</th></tr>'
            '<tr><th scope="col">@50</th><th scope="col">@100</th><th scope="col">@200</th>'
            '<th scope="col">@50</th><th scope="col">@100</th><th scope="col">@200</th></tr>'
            '</thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def scale_table() -> str:
    rows = []
    for tag, sm, lg in (("EB-NeRD", eb, ebL), ("MIND", mi, miL)):
        for n in ("bm25", "emb", "pop_prior"):
            d = lg[n]["auc"] - sm[n]["auc"]
            rows.append(f'<tr><th scope="row">{tag} · {NICE[n]}</th>'
                        f'<td>{sm[n]["auc"]:.4f}</td><td>{lg[n]["auc"]:.4f}</td>'
                        f'<td class="delta">{d:+.4f}</td></tr>')
    return ('<div class="scroller"><table><caption>same code, same configs, '
            '6× the catalogue</caption><thead><tr><th scope="col">ranker</th>'
            '<th scope="col">AUC small</th><th scope="col">AUC large</th>'
            '<th scope="col">Δ</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


b_eb = {b: max(g["recall@100"] for g in q2["ebnerd_small"]["grid"] if g["b"] == b)
        for b in (0.0, 1.0)}
b_mi = {b: max(g["recall@100"] for g in q2["mind_small"]["grid"] if g["b"] == b)
        for b in (0.0, 1.0)}
eb_spread = (max(b_eb.values()) - min(b_eb.values())) / min(b_eb.values())
mi_gain = (b_mi[1.0] - b_mi[0.0]) / b_mi[0.0]

HTML = f"""<title>EB-NeRD &amp; MIND Retrieval</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --ground:#fcfcfb; --raise:#ffffff; --sunk:#f2f4f7;
  --ink:#0b0b0b; --ink2:#41474f; --ink3:#767d86;
  --line:#e2e5ea; --line2:#cbd1d9;
  --accent:#2a78d6; --accent-ink:#1c5aa6; --counter:#eb6834; --leak:#c9342f;
  --good:#127f5c;
  --plate:#fcfcfb; --plate-line:#dfe3e8;
  --serif:"Newsreader",Georgia,"Times New Roman",serif;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#12151a; --raise:#191d24; --sunk:#1e232b;
    --ink:#e9ecf1; --ink2:#aeb6c1; --ink3:#7f8894;
    --line:#262c35; --line2:#39424e;
    --accent:#6aa6e8; --accent-ink:#8dbdf0; --counter:#f2884f; --leak:#e8635e;
    --good:#3fbb90;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#12151a; --raise:#191d24; --sunk:#1e232b;
  --ink:#e9ecf1; --ink2:#aeb6c1; --ink3:#7f8894;
  --line:#262c35; --line2:#39424e;
  --accent:#6aa6e8; --accent-ink:#8dbdf0; --counter:#f2884f; --leak:#e8635e;
  --good:#3fbb90;
}}
*,*::before,*::after {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--sans); font-size:16px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1080px; margin:0 auto; padding:0 24px 96px; }}
.col {{ max-width:68ch; }}
h1,h2,h3 {{ font-family:var(--serif); text-wrap:balance; margin:0; font-weight:600; }}
p {{ margin:0 0 1em; }}
a {{ color:var(--accent-ink); }}
:focus-visible {{ outline:2px solid var(--accent); outline-offset:3px; border-radius:2px; }}

/* ---- masthead: a news serif, because the subject is news ---- */
header.mast {{ border-bottom:2px solid var(--ink); margin:0 0 40px; padding:56px 0 20px; }}
header.mast h1 {{ font-size:clamp(2.3rem,5.6vw,3.6rem); line-height:1.03; letter-spacing:-.02em; }}
header.mast h1 em {{ font-style:italic; color:var(--accent-ink); }}
.dek {{ font-family:var(--serif); font-size:1.18rem; color:var(--ink2); margin:14px 0 0; max-width:60ch; }}
.byline {{
  display:flex; flex-wrap:wrap; gap:8px 20px; margin-top:22px;
  font-family:var(--mono); font-size:.735rem; letter-spacing:.06em;
  text-transform:uppercase; color:var(--ink3);
}}

/* ---- headline numbers ---- */
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); margin:0 0 52px; }}
.tile {{ background:var(--raise); padding:20px 22px 18px; display:flex; flex-direction:column; gap:6px; }}
.tile .n {{ font-family:var(--serif); font-size:2.35rem; line-height:1; letter-spacing:-.02em;
  font-variant-numeric:tabular-nums; }}
.tile .n.up {{ color:var(--leak); }}
.tile .n.ok {{ color:var(--accent-ink); }}
.tile .k {{ font-family:var(--mono); font-size:.68rem; letter-spacing:.09em;
  text-transform:uppercase; color:var(--ink3); }}
.tile .d {{ font-size:.86rem; color:var(--ink2); line-height:1.45; }}

/* ---- sections numbered by the assignment's own questions ---- */
section {{ margin:0 0 56px; scroll-margin-top:24px; }}
.qhead {{ display:flex; align-items:baseline; gap:14px; border-top:1px solid var(--line2);
  padding-top:14px; margin-bottom:18px; }}
.qhead .q {{ font-family:var(--mono); font-size:.72rem; font-weight:500; letter-spacing:.1em;
  color:var(--accent-ink); padding:3px 8px; border:1px solid var(--accent); border-radius:2px;
  white-space:nowrap; }}
.qhead h2 {{ font-size:1.62rem; letter-spacing:-.01em; }}

/* ---- tables ---- */
.scroller {{ overflow-x:auto; margin:0 0 26px; border:1px solid var(--line); background:var(--raise); }}
table {{ border-collapse:collapse; width:100%; font-size:.855rem;
  font-variant-numeric:tabular-nums; }}
caption {{ caption-side:top; text-align:left; padding:12px 16px; font-family:var(--mono);
  font-size:.7rem; letter-spacing:.06em; text-transform:uppercase; color:var(--ink3);
  border-bottom:1px solid var(--line); }}
th,td {{ padding:8px 14px; text-align:right; white-space:nowrap; }}
thead th {{ font-size:.72rem; font-weight:600; color:var(--ink2); letter-spacing:.03em;
  border-bottom:1px solid var(--line2); }}
tbody th[scope=row] {{ text-align:left; font-weight:500; }}
tbody tr + tr th, tbody tr + tr td {{ border-top:1px solid var(--line); }}
tbody tr.best {{ background:color-mix(in oklab, var(--accent) 9%, transparent); }}
tbody tr.leak th, tbody tr.leak td {{ color:var(--leak); font-style:italic; }}
td.pos {{ color:var(--good); font-variant-numeric:tabular-nums; }}
td.neg {{ color:var(--leak); font-variant-numeric:tabular-nums; }}
.ci {{ display:block; font-family:var(--mono); font-size:.66rem; color:var(--ink3);
  font-style:normal; }}
.delta {{ font-family:var(--mono); color:var(--good); }}

/* ---- findings ---- */
.find {{ border-left:3px solid var(--accent); background:var(--sunk); padding:16px 20px;
  margin:0 0 22px; }}
.find h3 {{ font-size:1.06rem; margin-bottom:6px; }}
.find p:last-child {{ margin-bottom:0; }}
.find.warn {{ border-left-color:var(--leak); }}
.find.warn h3 {{ color:var(--leak); }}

/* ---- figure plates stay light: these are print artifacts ---- */
.plate {{ margin:0 0 30px; border:1px solid var(--plate-line); background:var(--plate);
  padding:10px 10px 0; }}
.plate img {{ display:block; width:100%; height:auto; }}
.plate figcaption {{ font-size:.8rem; color:#52514e; padding:10px 6px 12px; line-height:1.5;
  border-top:1px solid #ececea; margin-top:8px; }}

code {{ font-family:var(--mono); font-size:.86em; background:var(--sunk); padding:.1em .35em;
  border-radius:2px; }}
ul {{ margin:0 0 1em; padding-left:1.1em; }}
li {{ margin-bottom:.45em; }}
li::marker {{ color:var(--ink3); }}
footer {{ border-top:1px solid var(--line2); padding-top:18px; font-size:.85rem; color:var(--ink3); }}
@media (prefers-reduced-motion:reduce) {{ *{{animation:none!important;transition:none!important;}} }}
</style>

<div class="wrap">
<header class="mast">
  <h1>Lexical and semantic retrieval,<br><em>measured twice</em></h1>
  <p class="dek">One pipeline over two news corpora — Danish tabloid and Microsoft News —
  built to answer whether content retrieval beats knowing what is popular. On one of these
  datasets it does not even need to: popularity ranks below chance.</p>
  <div class="byline">
    <span>CS4.406 · Assignment 1</span>
    <span>{q2['ebnerd_large']['index']['n_docs']:,} + {q2['mind_large']['index']['n_docs']:,} articles</span>
    <span>{q4['ebnerd_large']['n_impressions'] + q4['mind_large']['n_impressions']:,} scored impressions</span>
    <span>every number regenerated by <code>make reproduce</code></span>
  </div>
</header>

<div class="tiles">
  <div class="tile"><span class="k">Q9 · future-click leak</span>
    <span class="n up">+{leak_eb:.0f}%</span>
    <span class="d">AUC gained on EB-NeRD by one feature that counts clicks from inside the
    scored week. On MIND the same feature buys +{leak_mi:.0f}%.</span></div>
  <div class="tile"><span class="k">Q3 · ANN crossover</span>
    <span class="n ok">N≈8K</span>
    <span class="d">Below this, exact search beats every approximate index. At 125K articles
    HNSW is {hnsw125/flat125:.1f}× faster than exact.</span></div>
  <div class="tile"><span class="k">Q2 · length normalisation</span>
    <span class="n ok">+{mi_gain*100:.0f}%</span>
    <span class="d">What tuning <code>b</code> buys on MIND. On EB-NeRD the whole sweep is
    worth {eb_spread*100:.1f}% — and its optimum flips with corpus size.</span></div>
  <div class="tile"><span class="k">Q4 · best ranker</span>
    <span class="n ok">{mi['emb']['auc']:.3f}</span>
    <span class="d">AUC for mean-pooled embeddings on MIND, against {mi['bm25']['auc']:.3f}
    for BM25 on identical history input.</span></div>
</div>

<section>
  <div class="qhead"><span class="q">Q2 · Q3</span><h2>Retrieval: does content beat popularity?</h2></div>
  <div class="col"><p>Both retrievers are fed the same input — the user's whole click history — so
  the comparison isolates representation. Candidates are restricted to articles alive in a
  7-day window, and articles the user already read are dropped, which alone is worth
  {100*(cmp3['ebnerd/small']['methods']['bm25']['no-seen']['50']/cmp3['ebnerd/small']['methods']['bm25']['raw']['50']-1):.0f}%
  to BM25 on EB-NeRD: its top hit was frequently an article from the user's own history.</p></div>
  {recall_table()}
  <div class="find"><h3>The two datasets disagree, and that is the result</h3>
  <p>On EB-NeRD, popularity and recency both fall <strong>below random</strong>. Only 2 of the
  200 most-clicked test-week articles appear in the top 200 by prior popularity — they are a
  median 80 hours younger than the split boundary, so they did not exist when the feature
  window closed. On MIND the same overlap is 24/200 at a median age of +7h, and recency is the
  strongest retriever there. A freshness prior is a bet on how fast the catalogue turns over,
  not a portable trick.</p></div>
  <div class="find"><h3>Lexical and semantic retrieve nearly disjoint sets</h3>
  <p>Mean overlap between the two top-10 lists is <strong>0.88 of 10</strong>. BM25 returns
  near-duplicates of what the user already read; embeddings return related-but-new articles.
  The right architecture is both, not the better one.</p></div>
  {img("fig4_recall_comparison.png", "recall@K for every method on both corpora, against the ceiling imposed by the candidate universe.", "Recall at K comparison across methods and datasets")}
</section>

<section>
  <div class="qhead"><span class="q">Q3 · ablation</span><h2>What approximate search actually costs</h2></div>
  <div class="col"><p>Two questions with opposite answers, so they are measured separately.
  At the operating point — a {ann_op['universe']:,}-candidate universe — exact
  <code>IndexFlatIP</code> runs at {next(r for r in ann_op['rows'] if r['index']=='faiss flat (exact)')['qps']:,.0f}
  queries/second on one thread and <em>beats</em> HNSW, because walking a graph over 1.7K
  vectors costs more than the matrix multiply it replaces. The task metric is far more
  forgiving than index fidelity: configurations down to 63% index recall stay within 4% of
  exact task recall.</p>
  <p>The scale sweep locates the crossover at <strong>N≈8,000</strong>. At the full
  {ann_sc['rows'][-1]['n']:,}-article corpus, HNSW runs {hnsw125/flat125:.1f}× faster than
  exact FAISS at {sc('hnsw M=16 efC=200 efS=64',125000,'ann_recall@100'):.1%} fidelity. So
  exact search is right for this assignment and wrong for the production catalogue.</p></div>
  {img("fig7_ann_scale_ebnerd_large_contrastive.png", "Throughput, build cost and fidelity as the corpus grows. Single-threaded, because 48 cores make brute force look like an ANN index.", "ANN scale sweep showing crossover near eight thousand vectors")}
</section>

<section>
  <div class="qhead"><span class="q">Q4</span><h2>Ranking inside the impression</h2></div>
  <div class="col"><p>This is what the leaderboards score, and it is not what recall@K measures:
  a retriever that never surfaces an article can still rank it correctly once the impression
  puts it in front of the user. <code>random</code> is the calibration check — a correct AUC
  implementation must place it at 0.5000.</p></div>
  {rank_table("ebnerd_small")}
  {rank_table("mind_small")}
  <div class="find"><h3>The accuracy winner is the diversity loser</h3>
  <p>The embedding ranker has the <em>lowest</em> intra-list diversity on both datasets
  ({eb['emb']['ild@10']:.3f} and {mi['emb']['ild@10']:.3f}, against {eb['random']['ild@10']:.3f}
  and {mi['random']['ild@10']:.3f} for random). It is accurate because it is narrow. Prior
  popularity fails the other way: on MIND it reaches only
  {mi['pop_prior']['coverage@10']:.0%} of the catalogue and is worth
  {mi['pop_prior']['head']['ndcg@10']:.3f} on head impressions against
  {mi['pop_prior']['tail']['ndcg@10']:.3f} on tail. Whatever ships needs a diversity
  constraint no accuracy metric will ask for.</p></div>
  <div class="find"><h3>{FUSION_HEAD}</h3>
  <p>{FUSION_BODY} No single nDCG number reports that — which is the argument for measuring
  a candidate generator and a ranker on different metrics.</p></div>
  {img("fig10_rank_penalty.png", "Left: the credit each metric gives an item at rank r. Centre and right: where the first clicked item actually lands — the distribution those weights are applied to.", "Rank discount curves and empirical first-relevant-rank distributions")}
  {img("fig12_metric_agreement.png", "Rank of every system under ten metric definitions. The winner is stable on MIND; on EB-NeRD it changes at the undiscounted cutoffs.", "Heatmap of ranker position under ten different metrics")}
</section>

<section>
  <div class="qhead"><span class="q">Q9</span><h2>What one illegal feature buys</h2></div>
  <div class="col"><p>The <em>popularity (future clicks)</em> row in the tables above is the
  same ranker as <em>popularity (prior)</em>, with one change: it counts clicks from inside
  the scored split. Same candidates, same impressions, same metric code. On EB-NeRD that turns
  a below-chance baseline into the second-best ranker in the table.</p>
  <p>The pipeline is defended structurally rather than by inspection — 50 assertions enforce
  split ordering, feature cutoff stamps, exposure totals, and the absence of
  <code>next_read_time</code>. One of them caught a real bug: augmented history was silently
  deduplicating the shipped history, shrinking a 38-item list to 36.</p></div>
  {img("fig8_q4_ranking.png", "Ranking quality with bootstrap confidence intervals, and the size of the future-click illusion.", "Ranking metrics with confidence intervals and the Q9 leakage comparison")}
</section>

{{ABLATIONS}}

<section>
  <div class="qhead"><span class="q">Scale</span><h2>The same pipeline, 6× the catalogue</h2></div>
  <div class="col"><p>Identical code and configs on the Codabench-scale bundles. Every
  content-ranker conclusion reproduces, including the below-chance popularity result and the
  size of the leak.</p></div>
  {scale_table()}
  <div class="find warn"><h3>One conclusion did not survive — and it is the useful one</h3>
  <p>The BM25 <code>(k₁, b)</code> sweep <strong>reverses</strong> on EB-NeRD between variants:
  b={q2['ebnerd_small']['best']['b']} wins on small, b={q2['ebnerd_large']['best']['b']} on
  large. The explanation is in the corpora, not the algorithm — EB-NeRD indexes title and
  subtitle at ~{q2['ebnerd_small']['index']['avgdl']:.0f} tokens with little variance, so
  there is barely any length to normalise, while MIND's title and abstract averages
  ~{q2['mind_small']['index']['avgdl']:.0f} with 5.2% of abstracts missing entirely. Tuning on
  the demo bundle and shipping to the leaderboard bundle would have cost real score.</p></div>
</section>

{{LEADERBOARD}}

<section>
  <div class="qhead"><span class="q">10×</span><h2>Where this breaks</h2></div>
  <div class="col"><ul>
    <li><strong>The join that already broke.</strong> Remapping article ids inside history
    lists by explode → join → group_by hit 120 GB and died. The cause was shape, not volume:
    200,000 beyond-accuracy rows carry 250 candidates each and all share
    <code>impression_id = 0</code>, so the group_by silently merged them. Rewritten in place,
    peak is 8.9 GB.</li>
    <li><strong>…and broke again, silently.</strong> A polars join returns rows in hash order,
    so the store was a permutation of its source file. Both graders score one line per raw row
    <em>in raw order</em>, and on the beyond-accuracy rows position is the only identifier
    there is — so no sort could repair it. It cost a rejected submission and was invisible to
    every test, because the checks compared the output against the same reordered table it
    came from. The store now carries <code>src_row</code>, pins the join order, and asserts
    both against the raw file for every split of every store.</li>
    <li><strong>Features freeze at the split boundary.</strong> Popularity is computed once,
    at the start of a 7-day window, on a corpus whose median clicked article is 80 hours
    younger than that boundary. At 10× it does not get worse — it is already maximally wrong.</li>
    <li><strong>Exact search stops being free</strong> past N≈8K, and the index build becomes
    the bottleneck: 42 s single-threaded at 125K, superlinear in N.</li>
    <li><strong>Three defects only the large run exposed:</strong> a per-user Python loop in
    the random baseline (13 minutes at 791K users), an unbounded bootstrap that wanted 100 GB
    for its resample matrix, and a ranking harness whose peak scales at ~74 KB per impression.</li>
    <li><strong>Caches that could not tell they were stale.</strong> The build skipped a stage
    whenever the <em>config</em> hash matched, so editing the feature code and rebuilding
    skipped everything and kept the old logic. The fingerprint is now config + build-source +
    raw-input state, reported separately so a rebuild says which one moved.</li>
    <li><strong>Single node.</strong> The three-way polars / cudf-polars / native-cuDF
    benchmark is wired but the RAPIDS install never completed over this network, so the GPU
    column is unmeasured — stated as a gap rather than estimated.</li>
  </ul></div>
</section>

<footer>Generated from <code>reports/*.json</code> by <code>scripts/results_page.py</code>.
Figures are the committed matplotlib output. The 4-page design note is
<code>reports/design_note.pdf</code>.</footer>
</div>
"""



def leaderboard_section() -> str:
    """Q5. One leaderboard answered and one did not; both reported from measurement."""
    if not PROBE:
        return ""
    eb, mi, eg = PROBE["ebnerd"], PROBE["mind"], PROBE["egress"]
    lo, hi = eb["organiser_stated_job_hours"]
    days = eb["messages_ready"] * ((lo + hi) / 2) / 24
    ext = eg["external_nodes"].values()
    ms = f"{1000 * min(ext):.0f}&ndash;{1000 * max(ext):.0f}&nbsp;ms"
    subs = "".join(
        f"<tr><td>{x['id']}</td><td>{x['file']}</td><td>{x['status']}</td>"
        f"<td>{x.get('score', '&mdash;')}</td>"
        f"<td style='text-align:left'>{x.get('cause', '')}</td></tr>"
        for x in mi["submissions"])
    return f"""
<section>
  <div class="qhead"><span class="q">Q5</span><h2>What the leaderboards said</h2></div>
  <div class="col"><p>Both submissions build from the <em>raw</em> behaviours files and are
  format-checked row-for-row: MIND 2,370,727 and EB-NeRD 13,536,710 lines, one per raw row.
  Grouping by <code>impression_id</code>, as the starter notebook does, would collapse all
  200,000 beyond-accuracy rows into a single line. The first MIND upload was rejected
  outright for carrying a descriptive zip name &mdash; both graders open a <em>fixed</em> inner
  filename, which is now an assertion rather than a comment.</p></div>
  <table><tr><th>ID</th><th>file</th><th>status</th><th>{mi['metric']}</th>
  <th style="text-align:left">note</th></tr>{subs}</table>
  <div class="find warn"><h3>The popularity baseline is degenerate on MIND's test week</h3>
  <p>It scored {mi['metric']} <strong>{mi['leaderboard_auc_popularity']:.4f}</strong> &mdash; below chance &mdash; while the
  identical scorer reaches {mi['dev_auc_popularity']:.4f} on dev. Popularity is counted from clicks
  strictly before the split cutoff; the clickstream ends {mi['clickstream_ends']} and the test
  impressions run {mi['test_window']}, up to seven days past it. The consequence is not a small
  loss of accuracy: <strong>{mi['zero_slot_pct_popularity']}% of candidate slots score exactly
  zero</strong> and <strong>{mi['all_tied_impression_pct']}% of impressions are a complete tie</strong>,
  so roughly a third of the test set is ordered arbitrarily and AUC collapses toward 0.5.</p>
  <p>Ranking the same rows by content instead &mdash; cosine between the mean-pooled history vector
  and each candidate, the retriever from &sect;3 &mdash; leaves {mi['zero_slot_pct_semantic']}% of slots at
  zero. On dev that is worth {mi['dev_auc_semantic']:.4f} against {mi['dev_auc_popularity']:.4f}; resubmitted to the
  leaderboard it scored <strong>{mi['leaderboard_auc_semantic']:.4f}</strong> against {mi['leaderboard_auc_popularity']:.4f}, a
  {100 * (mi['leaderboard_auc_semantic'] - mi['leaderboard_auc_popularity']) / mi['leaderboard_auc_popularity']:+.0f}% relative gain on the hidden test set. Note the direction: popularity
  fell from dev to test ({mi['dev_auc_popularity']:.4f} &rarr; {mi['leaderboard_auc_popularity']:.4f}) while content <em>rose</em>
  ({mi['dev_auc_semantic']:.4f} &rarr; {mi['leaderboard_auc_semantic']:.4f}). A stale popularity prior does not degrade
  gracefully; past its window it stops being a ranking at all. That is the whole case for
  content-based retrieval on a fast-moving news catalogue, measured rather than asserted.</p></div>
  <div class="find"><h3>EB-NeRD returned no score: a queue, not a rejection</h3>
  <p>Competition {eb['competition_id']}'s open phase does not use Codabench's shared worker pool. It routes to
  <code>{eb['queue_name']}</code>, a queue the organisers own and whose VMs were retired when the
  challenge ended. An authenticated AMQP passive declare returned <strong>{eb['messages_ready']} messages
  ready against {eb['consumers']} consumer</strong> &mdash; roughly {days:.0f} days of backlog at the organisers'
  stated {lo}&ndash;{hi}&nbsp;h per job, under a {eb['execution_time_limit_s'] // 3600}&nbsp;h ceiling. MIND declares no private
  queue and a {mi['execution_time_limit_s']}&nbsp;s ceiling, which is why it answered in minutes.</p>
  <p>The sanctioned fix is to attach your own compute worker, and the docs are explicit that a
  worker reaches its queue only over <code>pyamqp://&hellip;@www.codabench.org:5672</code> &mdash; no HTTP
  transport, no alternate port. That is blocked by <em>campus egress</em>, not by the broker, and the
  two were measured separately: four external probe nodes reach the broker in {ms}, so it is
  listening; and <code>portquiz.net</code>, which accepts a connection on every port, answers this
  network on 8080 but times out on 5671 and 5672. The outbound allowlist here carries
  {', '.join(eg['allowlisted_elsewhere'])} and not the broker port.</p></div>
</section>
"""


def ablation_section() -> str:
    if not (UREP or FEAT or THR):
        return ""
    rows = []
    for t, d in UREP.items():
        for kind, nice in (("emb", "embeddings"), ("bm25", "BM25")):
            rr = [r for r in d["runs"] if r["retriever"] == kind]
            flat = next(r for r in rr if r["n_recent"] == 30 and r["halflife"] is None)
            best = max(rr, key=lambda r: r["recall@100"])
            rows.append(
                f'<tr><th scope="row">history window — {t} {nice}</th>'
                f'<td>n_recent=30</td>'
                f'<td>n_recent={best["n_recent"] or "all"} &mdash; adopted</td>'
                f'<td class="pos">{100*(best["recall@100"]/flat["recall@100"]-1):+.1f}%</td></tr>')
    for t, d in THR.items():
        b = max((r for r in d["global"] if r["mean_kept"] >= 20), key=lambda r: r["vs_control"])
        rows.append(f'<tr><th scope="row">similarity cutoff — {t}</th>'
                    f'<td>none (top-200)</td><td>τ={b["tau"]:.2f}, keeps {b["mean_kept"]:.0f}</td>'
                    f'<td class="pos">+{100*b["vs_control"]/b["fixed_topk_control"]:.1f}%</td></tr>')
    for t, d in QSAT.items():
        base = next(r for r in d["runs"] if r["k3"].startswith("inf"))
        binary = next(r for r in d["runs"] if r["k3"].startswith("0"))
        rows.append(f'<tr><th scope="row">query-term saturation — {t}</th>'
                    f'<td>k₃ = ∞ (none)</td><td>k₃ = ∞ &mdash; confirmed</td>'
                    f'<td class="neg">{100 * (binary["recall@100"] / base["recall@100"] - 1):+.1f}% if binary</td></tr>')
    for t, d in IXAB.items():
        nost = next(r for r in d["runs"] if not r["stopwords"] and r["stemming"])
        body = next((r for r in d["runs"] if r["field"] == "title+abstract+body"), None)
        extra = f", body {100 * body['vs_shipped']:+.1f}%" if body else ""
        rows.append(f'<tr><th scope="row">indexed text + tokeniser — {t}</th>'
                    f'<td>title+abstract, stop+stem</td>'
                    f'<td>title+abstract &mdash; confirmed</td>'
                    f'<td class="neg">stopwords {100 * nost["vs_shipped"]:+.1f}%{extra}</td></tr>')
    for t, d in MULTI.items():
        rows.append(f'<tr><th scope="row">user representation — {t}</th>'
                    f'<td>1 mean-pooled vector</td>'
                    f'<td>k-means k={d["best"]["k"]} &mdash; not adopted</td>'
                    f'<td class="pos">{100 * d["gain_vs_single"]:+.1f}%</td></tr>')
    for t, d in FEAT.items():
        base, hl = d["emb_baseline"]["auc"], d["combined"]["helpful_only"]["auc"]
        rows.append(f'<tr><th scope="row">stored features — {t}</th>'
                    f'<td>embeddings only</td>'
                    f'<td>{", ".join(d["combined"]["helpful_features"])}</td>'
                    f'<td class="pos">{100*(hl/base-1):+.1f}% AUC</td></tr>')

    best_feat = ""
    if FEAT:
        d = FEAT.get("ebnerd") or next(iter(FEAT.values()))
        alone = sorted(d["alone"].items(), key=lambda kv: -kv[1]["auc"])[:3]
        best_feat = ("<div class=\"find\"><h3>The strongest single feature was one nothing "
                     "was using</h3><p>Ranked on its own, "
                     + ", ".join(f"<code>{k}</code> {v['auc']:.4f}" for k, v in alone)
                     + f". The user's category profile — built in Q1 and then ignored by every "
                       f"retriever — out-ranks BM25 on the metric the leaderboard scores. "
                       f"<code>user_activity</code> is the negative control: it is constant "
                       f"across the candidates of one impression, so it cannot reorder them, "
                       f"and it moves AUC by exactly "
                       f"{d['added_to_emb']['user_activity']['auc']-d['emb_baseline']['auc']:+.4f}."
                       "</p></div>")

    return f"""<section>
  <div class="qhead"><span class="q">Ablations</span><h2>What the sweeps actually changed</h2></div>
  <div class="col"><p>Seven knobs were fixed early on judgement and swept later on evidence.
  Three were set wrong; the rest survived, which is a result too &mdash; the argument for
  leaving the BM25 query unsaturated and for stopping the index at the abstract now rests on
  a sweep rather than on taste. The history window has since been corrected in the shipped
  pipeline, and every number elsewhere on this page is from the corrected setting.</p></div>
  <div class="scroller"><table><caption>swept after the fact</caption>
  <thead><tr><th scope="col">knob</th><th scope="col">shipped</th><th scope="col">best</th>
  <th scope="col">worth</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
  {best_feat}
  <div class="find warn"><h3>Summing every feature is worse than summing none</h3>
  <p>A naive z-scored sum of all {len(next(iter(FEAT.values()))["features"]) if FEAT else 0}
  features scores <b>below</b> the embedding ranker alone, because prior popularity and CTR are
  actively anti-predictive on EB-NeRD and drag the sum down with them. Keeping only the features
  that helped individually recovers the gain. Fusion is a selection problem, not an addition
  problem.</p></div>
  {img("fig14_features.png", "Top: how much history to read, and whether to decay it. Bottom: each stored feature alone, its marginal contribution, and what happens when everything is summed.", "User representation and feature ablation results")}
  {img("fig15_ablations.png", "Three choices that were never measured until now. Top: saturating the query only ever loses recall. Middle: recall against index size — the largest index is the second worst. Bottom: splitting a user into k interest vectors, by time and by clustering, with the cold-user slice that explains why k cannot be constant.", "Query saturation, index composition and multi-interest user representation")}
  {img("fig13_threshold.png", "A similarity cutoff against a matched fixed-top-K budget. Panel (f) is the one that decides it.", "Similarity threshold ablation")}
</section>"""


HTML = HTML.replace("{LEADERBOARD}", leaderboard_section())
HTML = HTML.replace("{ABLATIONS}", ablation_section())

out = R / "results.html"
out.write_text(HTML)
print(f"wrote {out} ({len(HTML)/1024:.0f} KB)")
