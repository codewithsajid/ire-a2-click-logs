"""The submission writer must reproduce the raw id sequence exactly.

`write_submission` streams in row chunks. It used to slice a *lazy* frame, and
because that frame is the output of a join, every slice re-ran the whole plan --
a hash join gives no promise that two executions emit rows in the same order, so
rows near a chunk boundary were written by both neighbouring slices and others
by neither. The line count stayed exactly right, which is why it survived:
each slice still returned `chunk_rows` rows. It cost 193 of MIND's 2,370,727
lines and 29,003 of EB-NeRD's.
"""
import numpy as np
import polars as pl
import pytest

from newsrec.submit import write_submission


def _joined(n: int) -> pl.LazyFrame:
    """A frame shaped like the real one: raw rows left-joined to scored lists."""
    rng = np.random.default_rng(0)
    raw = pl.DataFrame({
        "src_row": np.arange(n, dtype=np.uint32),
        "impression_id": np.arange(1, n + 1, dtype=np.int64),
    })
    # shuffled, as a join result is -- the writer must not depend on this order
    order = rng.permutation(n)
    scores = pl.DataFrame({
        "src_row": order.astype(np.uint32),
        "_scores": [list(rng.random(1 + (i % 4))) for i in order],
    })
    return raw.lazy().join(scores.lazy(), on="src_row", how="left")


def _ids(path) -> np.ndarray:
    return np.array([int(l.split(" ", 1)[0]) for l in open(path) if l.strip()],
                    dtype=np.int64)


@pytest.mark.parametrize("n,chunk", [(1000, 250), (1000, 999), (1000, 1000), (777, 100)])
def test_every_raw_row_is_written_once_and_in_order(tmp_path, n, chunk):
    out = tmp_path / "sub.txt"
    written = write_submission(_joined(n).select("src_row", "impression_id", "_scores"),
                               out, chunk_rows=chunk, order_by="src_row")
    got = _ids(out)
    assert written == n
    # exact sequence, not merely the right multiset or the right count
    assert np.array_equal(got, np.arange(1, n + 1))


def test_no_duplicates_across_a_chunk_boundary(tmp_path):
    """The specific failure: a boundary that emits one row twice and drops another."""
    out = tmp_path / "sub.txt"
    write_submission(_joined(500).select("src_row", "impression_id", "_scores"),
                     out, chunk_rows=100, order_by="src_row")
    got = _ids(out)
    assert len(got) == len(set(got.tolist())), "a row was written twice"
    assert set(got.tolist()) == set(range(1, 501)), "a row was dropped"


def test_ranks_are_a_permutation_of_the_candidate_positions(tmp_path):
    out = tmp_path / "sub.txt"
    write_submission(_joined(200).select("src_row", "impression_id", "_scores"),
                     out, chunk_rows=64, order_by="src_row")
    for line in open(out):
        body = line.strip().split(" ", 1)[1]
        ranks = [int(x) for x in body.strip("[]").split(",")]
        assert sorted(ranks) == list(range(1, len(ranks) + 1))


def test_the_temporary_parquet_does_not_survive(tmp_path):
    out = tmp_path / "sub.txt"
    write_submission(_joined(50).select("src_row", "impression_id", "_scores"),
                     out, chunk_rows=16, order_by="src_row")
    assert not list(tmp_path.glob("*.parquet")), "scratch file left behind"
