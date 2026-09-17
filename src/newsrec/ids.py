"""Dense id assignment and list-column remapping.

BM25 and FAISS both want contiguous row ids, and both datasets ship sparse or
string ids, so every raw id is mapped to a dense `idx` once and the mapping is
persisted.

Remapping the ids *inside* a list column (candidates, history) is done with
`replace_strict` under `list.eval` rather than explode -> join -> group_by. The
explode route cost 120 GB and an OOM on EB-NeRD's test split, because its 200,000
beyond-accuracy rows carry 250 candidates each (50M elements landing in a single
hour). It was also subtly wrong: that split reuses `impression_id = 0` for all
200,000 of those rows, so regrouping on impression_id collapsed them into one.
The in-place form touches neither issue -- it is row-preserving and keeps element
order by construction -- and does the same hour in 2.8 s at 3 GB.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def remap_list(
    lf: pl.LazyFrame,
    list_col: str,
    id_map: pl.DataFrame | pl.LazyFrame,
    key: str = "src_id",
    out: str | None = None,
    drop_unknown: bool = True,
) -> pl.LazyFrame:
    """Map ids inside `list_col` to dense idx, in place, preserving order.

    Elements missing from `id_map` become null and are dropped by default --
    MIND's test news set contains ids absent from train, and a candidate the
    article table has never seen cannot be scored anyway.
    """
    out = out or list_col
    m = id_map.collect() if isinstance(id_map, pl.LazyFrame) else id_map
    old = m[key].cast(pl.Utf8).to_list()
    new = m["idx"].cast(pl.UInt32).to_list()

    expr = (
        pl.col(list_col).cast(pl.List(pl.Utf8))
        .list.eval(pl.element().replace_strict(old, new, default=None, return_dtype=pl.UInt32))
    )
    if drop_unknown:
        expr = expr.list.drop_nulls()
    return lf.with_columns(expr.alias(out))


def build_id_map(values: pl.LazyFrame, key: str) -> pl.LazyFrame:
    """Assign 0..n-1 to the distinct values of `key`, in sorted order.

    Sorted rather than first-seen so the mapping is reproducible regardless of
    row order or which engine built it.
    """
    return (
        values.select(pl.col(key).cast(pl.Utf8))
        .unique()
        .sort(key)
        .with_row_index("idx")
        .with_columns(pl.col("idx").cast(pl.UInt32))
    )


def stable_uniform(a: np.ndarray, b: np.ndarray, seed: int = 0) -> np.ndarray:
    """Uniform [0,1) values keyed on the *pair* (a, b), not on row position.

    splitmix64's finaliser over (a, b, seed): a counter-based hash, so the value
    a row gets depends on what the row is and not on where it sits. Anywhere a
    sequential RNG is indexed by position instead, re-ordering the table silently
    re-rolls every draw -- which happened twice here. The random baseline drew
    per row, so rebuilding the store moved the denominator of every
    lift-over-random figure; the metric harness broke score ties per row, so the
    same rebuild re-resolved every tie and shifted the constant-scoring baselines
    in the fourth decimal. Both now key on (user/impression, article).
    """
    x = (np.asarray(a, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)) \
        ^ (np.asarray(b, dtype=np.uint64) * np.uint64(0xBF58476D1CE4E5B9)) \
        ^ np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(30); x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27); x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
