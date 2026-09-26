"""Codabench submission files.

Format (identical for both leaderboards):

    <impression_id> [<rank_1>,<rank_2>,...,<rank_N>]

one line **per behaviours row**, N = number of candidates in that row, rank 1 =
most likely click. Per row, not per impression_id: EB-NeRD's test split reuses
`impression_id = 0` for all 200,000 beyond-accuracy rows, so grouping by id
would collapse them into a single line and silently drop that whole track.

Scoring stays inside the list column (`replace_strict` then `list.eval`), so a
13.5M-row / 206M-candidate file is produced without ever exploding.

Rows are emitted **in the order of the raw behaviours file**, and the candidate
lists come from that file rather than from the feature store. Both graders walk
their ground-truth file line by line and assert that line *i* of the submission
carries the same impression id, so a submission built from the internal store --
which sorts rows during the build -- is rejected on line 1 ("Inconsistent
Impression Id"). The store's dense article indices are for analysis; a
submission has to speak the raw dataset's own ids and ordering.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from .store import FeatureStore


def _rank_expr(scores: str = "_scores") -> pl.Expr:
    """Dense ranks within each candidate list, 1 = highest score."""
    return (
        pl.col(scores)
        .list.eval(pl.element().rank(method="ordinal", descending=True))
        .cast(pl.List(pl.UInt32))
    )


def score_by_lookup(lf: pl.LazyFrame, idx, score, default: float = 0.0) -> pl.LazyFrame:
    """Attach a per-candidate score from an article-level lookup table.

    `idx` may be dense article indices or the dataset's own ids (int for EB-NeRD,
    str for MIND); it only has to match the dtype of the candidate list.
    """
    keys = idx.tolist() if isinstance(idx, np.ndarray) else list(idx)
    vals = score.tolist() if isinstance(score, np.ndarray) else list(score)
    return lf.with_columns(
        pl.col("candidates")
        .list.eval(pl.element().replace_strict(
            keys, vals, default=default, return_dtype=pl.Float64))
        .alias("_scores")
    )


def raw_popularity(fs: FeatureStore, split: str, decayed: bool = True):
    """Article popularity keyed by the dataset's *own* article id.

    The scores still come from `article_features`, which is computed on a window
    strictly before `split` -- the unlabelled test set contributes nothing. Only
    the key changes, from dense index to raw id.
    """
    col = "clicks_decayed" if decayed else "clicks"
    pop = (
        fs.article_features(split).select("article_idx", col)
        .join(fs.articles().with_row_index("article_idx").select("article_idx", "src_id"),
              on="article_idx", how="inner")
        .select("src_id", col)
        .collect()
    )
    return pop["src_id"], pop[col]


def raw_behaviours(fs: FeatureStore, raw_root: Path) -> tuple[pl.LazyFrame, pl.DataType]:
    """The unlabelled test behaviours, in file order, with raw candidate ids.

    Reads the raw file rather than the store because the store used to reorder
    rows. It no longer does -- impressions carry `src_row` and are written in
    source order, asserted by tests/test_row_order.py -- so the store route is
    now viable. This path stays the shipped one because it is the route the
    graders actually accepted, and it needs no store at all.
    """
    if fs.dataset == "ebnerd":
        lf = (pl.scan_parquet(Path(raw_root) / "ebnerd_testset/test/behaviors.parquet")
              .select(pl.col("impression_id").cast(pl.Int64),
                      pl.col("article_ids_inview").cast(pl.List(pl.Utf8)).alias("candidates")))
    else:
        lf = (pl.scan_csv(Path(raw_root) / "MINDlarge_test/behaviors.tsv", separator="\t",
                          has_header=False, quote_char=None,
                          new_columns=["impression_id", "user_id", "time", "history",
                                       "impressions"])
              .select(pl.col("impression_id").cast(pl.Int64),
                      pl.col("impressions").str.split(" ").alias("candidates")))
    return lf, pl.Utf8


def write_submission(lf: pl.LazyFrame, path: Path, chunk_rows: int = 2_000_000,
                     order_by: str | None = None) -> int:
    """Write ranked predictions, streaming in row chunks.

    The frame is materialised to a temporary parquet before slicing. `lf` is
    the product of a join, and a *lazy* slice re-runs the entire plan for every
    chunk -- a hash join does not promise the same row order across two
    executions, so rows near a boundary were emitted by both neighbouring
    slices while others were emitted by neither. That cost 193 of MIND's
    2,370,727 lines and 29,003 of EB-NeRD's, and it was invisible in the line
    count because each slice still returned exactly `chunk_rows` rows.

    `order_by` is a unique column (the raw row index) that fixes the output
    order, so the file lands in raw file order rather than join order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".ordered.parquet")
    if order_by:
        lf = lf.sort(order_by)
    lf.sink_parquet(tmp)                     # one execution; order now fixed on disk
    try:
        scan = pl.scan_parquet(tmp)
        total = scan.select(pl.len()).collect().item()
        written = 0
        with open(path, "w") as fh:
            for start in range(0, total, chunk_rows):
                part = (
                    scan.slice(start, chunk_rows)
                    .with_columns(_rank_expr().alias("_ranks"))
                    .select(
                        pl.format("{} [{}]",
                                  pl.col("impression_id"),
                                  pl.col("_ranks").cast(pl.List(pl.Utf8)).list.join(","))
                        .alias("line")
                    )
                    .collect(engine="streaming")
                )
                fh.write("\n".join(part["line"].to_list()))
                fh.write("\n")
                written += part.height
        return written
    finally:
        tmp.unlink(missing_ok=True)


# What each grader expects to find *inside* the zip. This is not cosmetic: both
# scorers open a fixed filename, so a descriptive local name like
# `ebnerd_popularity_predictions.txt` is a rejected submission.
ARCNAME = {"ebnerd": "predictions.txt", "mind": "prediction.txt"}


def zip_submission(path: Path, zip_path: Path | None = None,
                   arcname: str | None = None) -> Path:
    path = Path(path)
    zip_path = Path(zip_path or path.with_suffix(".zip"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(path, arcname=arcname or path.name)
    return zip_path


def popularity_submission(dataset: str, variant: str, split: str, out: Path,
                          decayed: bool = True) -> dict:
    """The baseline every later system has to beat: rank candidates by how often
    they were clicked in the window before the scored split."""
    fs = FeatureStore(dataset, variant)
    col = "clicks_decayed" if decayed else "clicks"
    pop = fs.article_features(split).select("article_idx", col).collect()
    lf = score_by_lookup(fs.impressions(split).select("impression_id", "candidates"),
                         pop["article_idx"].to_numpy(), pop[col].to_numpy())
    n = write_submission(lf, out)
    arc = ARCNAME[dataset]
    z = zip_submission(out, arcname=arc)
    return {"rows": n, "txt": str(out), "zip": str(z), "inner_name": arc,
            "zip_mb": round(z.stat().st_size / 1e6, 1),
            "scored_articles": pop.height}



def _semantic_scores(df: pl.DataFrame, embn: np.ndarray, amap: pl.DataFrame,
                     pop: pl.DataFrame) -> np.ndarray:
    """Per-candidate scores for one chunk, flat and aligned with `candidates` exploded.

    Ranking happens inside a row, so different rows may use different scoring
    functions without any scale to reconcile. Rows with a history are scored by
    cosine against the mean-pooled history vector; the 1.2% of MIND test rows that
    carry no history are scored by decayed popularity instead, which is the only
    signal left for them. Candidates whose article is not in the store sort last.
    """
    from scipy.sparse import csr_matrix

    n, dim = df.height, embn.shape[1]
    a2i = dict(zip(amap["src_id"].to_list(), amap["idx"].to_list()))

    # --- user vectors: sum of history embeddings via one sparse matmul, then mean
    h = (df.select(pl.col("_hist"))
           .with_row_index("r").explode("_hist").drop_nulls("_hist"))
    hr = h["r"].to_numpy()
    hi = np.fromiter((a2i.get(x, -1) for x in h["_hist"].to_list()), dtype=np.int64,
                     count=h.height)
    keep = hi >= 0
    hr, hi = hr[keep], hi[keep]
    S = csr_matrix((np.ones(len(hi), dtype=np.float32), (hr, hi)),
                   shape=(n, embn.shape[0]))
    uv = np.asarray(S @ embn, dtype=np.float32)
    cnt = np.asarray(S.sum(axis=1)).ravel()
    uv /= np.maximum(np.linalg.norm(uv, axis=1, keepdims=True), 1e-12)
    warm = cnt > 0

    # --- candidates
    c = df.select("candidates").with_row_index("r").explode("candidates")
    cr = c["r"].to_numpy()
    ci = np.fromiter((a2i.get(x, -1) for x in c["candidates"].to_list()), dtype=np.int64,
                     count=c.height)
    known = ci >= 0

    out = np.full(c.height, -2.0, dtype=np.float32)          # unknown article: last
    m = known & warm[cr]
    out[m] = np.einsum("ij,ij->i", uv[cr[m]], embn[ci[m]]).astype(np.float32)

    # --- cold rows fall back to popularity, on their own rows only
    cold = known & ~warm[cr]
    if cold.any():
        p2s = dict(zip(pop["src_id"].to_list(), pop["s"].to_list()))
        out[cold] = np.fromiter((p2s.get(x, 0.0) for x in
                                 c["candidates"].to_numpy()[cold].tolist()),
                                dtype=np.float32, count=int(cold.sum()))
    return out


def semantic_submission(dataset: str, variant: str, raw_root: Path, out: Path,
                        emb_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                        split: str = "submit", chunk: int = 250_000) -> dict:
    """Leaderboard file ranked by content, not by popularity.

    The popularity baseline is degenerate on MIND's test week: it is computed from
    clicks strictly before the split cutoff, and the test impressions run for seven
    days past it, so 85.8% of candidate slots score exactly 0 and 31.4% of rows are
    a complete tie. This path scores every candidate against the user's own history
    instead, which is what the retrieval systems in this repo are actually for.
    """
    from .embeddings import article_embeddings

    fs = FeatureStore(dataset, variant)
    emb = np.asarray(article_embeddings(fs, emb_name))
    embn = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    amap = fs.articles().with_row_index("idx").select("src_id", "idx").collect()
    ids, scores = raw_popularity(fs, split, decayed=True)
    pop = pl.DataFrame({"src_id": ids.cast(pl.Utf8), "s": scores.cast(pl.Float32)})

    lf = raw_behaviours_with_history(fs, raw_root)
    total = lf.select(pl.len()).collect().item()
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out, "w") as fh:
        for start in range(0, total, chunk):
            df = lf.slice(start, chunk).collect()
            s = _semantic_scores(df, embn, amap, pop)
            back = (df.select("candidates").with_row_index("r").explode("candidates")
                      .with_columns(pl.Series("_s", s))
                      .group_by("r", maintain_order=True).agg(pl.col("_s").alias("_scores")))
            part = (df.select("impression_id").with_row_index("r")
                      .join(back, on="r", how="left")
                      .with_columns(_rank_expr().alias("_ranks"))
                      .select(pl.format("{} [{}]", pl.col("impression_id"),
                                        pl.col("_ranks").cast(pl.List(pl.Utf8)).list.join(","))
                              .alias("line")))
            fh.write("\n".join(part["line"].to_list()))
            fh.write("\n")
            written += part.height
            print(f"  {written:,}/{total:,}", flush=True)
    arc = ARCNAME[dataset]
    z = zip_submission(out, arcname=arc)
    return {"rows": written, "txt": str(out), "zip": str(z), "inner_name": arc,
            "zip_mb": round(z.stat().st_size / 1e6, 1), "embedding": emb_name}



def raw_behaviours_with_history(fs: FeatureStore, raw_root: Path) -> pl.LazyFrame:
    """Raw test behaviours in file order, with the history column kept."""
    if fs.dataset != "mind":
        raise NotImplementedError(
            "history-carrying raw read is wired for MIND only; EB-NeRD keeps test "
            "history in a separate history.parquet that this path does not join yet")
    return (pl.scan_csv(Path(raw_root) / "MINDlarge_test/behaviors.tsv", separator="\t",
                        has_header=False, quote_char=None,
                        new_columns=["impression_id", "user_id", "time", "history",
                                     "impressions"])
            .select(pl.col("impression_id").cast(pl.Int64),
                    pl.col("history").fill_null("").str.split(" ").alias("_hist"),
                    pl.col("impressions").str.split(" ").alias("candidates")))




def semantic_submission_ebnerd(raw_root: Path, out: Path,
                               emb_name: str = "contrastive",
                               split: str = "submit", max_slots: int = 6_000_000) -> dict:
    """EB-NeRD content submission.

    Two shape problems this path has to respect, both of which killed an earlier
    version. Chunks are bounded by *candidate slots*, not rows: 200,000
    beyond-accuracy rows carry 250 candidates each and sit at the end of the file,
    so a fixed row chunk that is harmless for the first 13M rows becomes a 55M-slot
    chunk for the last one. And ids are remapped inside polars with `replace_strict`
    rather than through a Python dict, so 200M candidate ids never become 200M
    Python strings.

    The user vector is built once per *user*, not once per row -- joining history
    lists onto 13.5M behaviour rows would materialise the same vectors a dozen
    times over.
    """
    from scipy.sparse import csr_matrix
    from .embeddings import article_embeddings

    fs = FeatureStore("ebnerd", "large")
    emb = np.asarray(article_embeddings(fs, emb_name))
    embn = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    amap = fs.articles().with_row_index("idx").select("src_id", "idx").collect()
    a_src, a_idx = amap["src_id"].to_list(), amap["idx"].to_list()
    print(f"  {emb.shape[0]:,} articles x {emb.shape[1]} dims", flush=True)

    root = Path(raw_root) / "ebnerd_testset/test"
    h = (pl.scan_parquet(root / "history.parquet")
         .select("user_id", pl.col("article_id_fixed").cast(pl.List(pl.Utf8)).alias("h"))
         .collect())
    users = h["user_id"].to_numpy()
    u2r = {int(u): i for i, u in enumerate(users)}
    ex = (h.select("h").with_row_index("r").explode("h").drop_nulls("h")
           .with_columns(pl.col("h").replace_strict(a_src, a_idx, default=-1).alias("i")))
    hr, hi = ex["r"].to_numpy(), ex["i"].to_numpy()
    k = hi >= 0
    S = csr_matrix((np.ones(int(k.sum()), np.float32), (hr[k], hi[k])),
                   shape=(len(users), emb.shape[0]))
    uv = np.asarray(S @ embn, dtype=np.float32)
    uv /= np.maximum(np.linalg.norm(uv, axis=1, keepdims=True), 1e-12)
    warm = np.asarray(S.sum(axis=1)).ravel() > 0
    print(f"  {len(users):,} users, {warm.sum():,} usable ({100 * warm.mean():.1f}%)", flush=True)
    del ex, h, S

    ids, sc = raw_popularity(fs, split, decayed=True)
    p_src, p_val = ids.cast(pl.Utf8).to_list(), [float(x) for x in sc.to_list()]

    lf = (pl.scan_parquet(root / "behaviors.parquet")
          .select(pl.col("impression_id").cast(pl.Int64), "user_id",
                  pl.col("article_ids_inview").cast(pl.List(pl.Utf8)).alias("candidates")))
    n_cand = (pl.scan_parquet(root / "behaviors.parquet")
              .select(pl.col("article_ids_inview").list.len().alias("n"))
              .collect()["n"].to_numpy().astype(np.int64))
    total = len(n_cand)

    # slot-bounded slices
    bounds, i = [], 0
    while i < total:
        j, acc = i, 0
        while j < total and (acc + n_cand[j] <= max_slots or j == i):
            acc += n_cand[j]; j += 1
        bounds.append((i, j - i)); i = j
    print(f"  {total:,} rows, {n_cand.sum():,} slots, {len(bounds)} slices", flush=True)

    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out, "w") as fh:
        for start_row, take in bounds:
            df = lf.slice(start_row, take).collect()
            ur = np.fromiter((u2r.get(int(x), -1) for x in df["user_id"].to_list()),
                             np.int64, df.height)
            c = (df.select("candidates").with_row_index("r").explode("candidates")
                   .with_columns(
                       pl.col("candidates").replace_strict(a_src, a_idx, default=-1).alias("i"),
                       pl.col("candidates").replace_strict(p_src, p_val, default=0.0,
                                                           return_dtype=pl.Float32).alias("p")))
            cr, ci = c["r"].to_numpy(), c["i"].to_numpy()
            urow = ur[cr]
            ok = (ci >= 0) & (urow >= 0)
            hot = ok & warm[np.maximum(urow, 0)]
            s = np.full(c.height, -2.0, dtype=np.float32)
            s[hot] = np.einsum("ij,ij->i", uv[urow[hot]], embn[ci[hot]]).astype(np.float32)
            cold = ok & ~hot
            if cold.any():
                s[cold] = c["p"].to_numpy()[cold]
            back = (c.select("r").with_columns(pl.Series("_s", s))
                     .group_by("r", maintain_order=True).agg(pl.col("_s").alias("_scores")))
            part = (df.select("impression_id").with_row_index("r")
                      .join(back, on="r", how="left")
                      .with_columns(_rank_expr().alias("_ranks"))
                      .select(pl.format("{} [{}]", pl.col("impression_id"),
                                        pl.col("_ranks").cast(pl.List(pl.Utf8)).list.join(","))
                              .alias("line")))
            fh.write("\n".join(part["line"].to_list()))
            fh.write("\n")
            written += part.height
            del df, c, back, part, s
            print(f"  {written:,}/{total:,}", flush=True)
    z = zip_submission(out, arcname=ARCNAME["ebnerd"])
    return {"rows": written, "txt": str(out), "zip": str(z),
            "inner_name": ARCNAME["ebnerd"],
            "zip_mb": round(z.stat().st_size / 1e6, 1), "embedding": emb_name}


def raw_submission(dataset: str, variant: str, raw_root: Path, out: Path,
                   split: str = "submit", decayed: bool = True) -> dict:
    """Leaderboard file built against the raw behaviours file, in its own order."""
    fs = FeatureStore(dataset, variant)
    ids, scores = raw_popularity(fs, split, decayed=decayed)
    lf, _ = raw_behaviours(fs, raw_root)
    lf = score_by_lookup(lf, ids.cast(pl.Utf8), scores)
    n = write_submission(lf, out)
    arc = ARCNAME[dataset]
    z = zip_submission(out, arcname=arc)
    return {"rows": n, "txt": str(out), "zip": str(z), "inner_name": arc,
            "zip_mb": round(z.stat().st_size / 1e6, 1), "scored_articles": len(ids)}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--split", default="submit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-root", default=None,
                    help="build against the raw behaviours file, in its row order "
                         "(required by both graders)")
    ap.add_argument("--raw-clicks", action="store_true", help="use raw instead of decayed clicks")
    a = ap.parse_args()
    if a.raw_root:
        r = raw_submission(a.dataset, a.variant, Path(a.raw_root), Path(a.out),
                           a.split, decayed=not a.raw_clicks)
    else:
        r = popularity_submission(a.dataset, a.variant, a.split, Path(a.out),
                                  decayed=not a.raw_clicks)
    print(json.dumps(r, indent=2))
