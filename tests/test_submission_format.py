"""Both graders reject a submission before scoring it. These are the rejections.

Each competition walks its ground-truth file line by line and asserts that line
*i* of the submission carries the same impression id -- MIND's evaluate.py raises
"Inconsistent Impression Id" and stops. A file built from the feature store is
rejected on line 1, because the build sorts rows. So the contract under test is
the raw behaviours file's own ordering, not anything the store knows about.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from newsrec.config import DATA_ROOT
from newsrec.submit import ARCNAME

SUB = Path("reports/sub")
RAW = DATA_ROOT / "raw"

CASES = [
    ("mind", SUB / "prediction.txt", SUB / "prediction.zip"),
    ("ebnerd", SUB / "predictions.txt", SUB / "predictions.zip"),
]


def raw_reference(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """(impression_id, n_candidates) per row, in raw file order."""
    if dataset == "ebnerd":
        d = pl.scan_parquet(RAW / "ebnerd/ebnerd_testset/test/behaviors.parquet").select(
            pl.col("impression_id").cast(pl.Int64),
            pl.col("article_ids_inview").list.len().cast(pl.Int64).alias("n")).collect()
    else:
        d = pl.read_csv(RAW / "mind/MINDlarge_test/behaviors.tsv", separator="\t",
                        has_header=False, quote_char=None,
                        new_columns=["impression_id", "user", "time", "hist", "imp"]).select(
            pl.col("impression_id").cast(pl.Int64),
            pl.col("imp").str.split(" ").list.len().cast(pl.Int64).alias("n"))
    return d["impression_id"].to_numpy(), d["n"].to_numpy()


@pytest.mark.parametrize("dataset,txt,zp", CASES)
def test_submission_matches_raw_row_for_row(dataset, txt, zp):
    if not txt.exists():
        pytest.skip(f"{txt} not built")
    ids, counts = raw_reference(dataset)
    n = 0
    with open(txt) as fh:
        for i, line in enumerate(fh):
            n += 1
            assert i < len(ids), f"{dataset}: submission longer than the test set"
            head, ranks = line.rstrip("\n").split(" ", 1)
            assert int(head) == ids[i], (
                f"{dataset} line {i+1}: impression id {head} but the raw file has {ids[i]} "
                "-- this is the grader's 'Inconsistent Impression Id' rejection")
            assert ranks.startswith("[") and ranks.endswith("]")
            r = [int(x) for x in ranks[1:-1].split(",")]
            assert len(r) == counts[i], f"{dataset} line {i+1}: {len(r)} ranks for {counts[i]} candidates"
            assert sorted(r) == list(range(1, len(r) + 1)), (
                f"{dataset} line {i+1}: ranks are not a permutation of 1..N")
    assert n == len(ids), f"{dataset}: {n:,} lines for {len(ids):,} test rows"


@pytest.mark.parametrize("dataset,txt,zp", CASES)
def test_zip_holds_exactly_the_expected_bare_filename(dataset, txt, zp):
    if not zp.exists():
        pytest.skip(f"{zp} not built")
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
    assert names == [ARCNAME[dataset]], (
        f"{dataset}: zip contains {names}, but the grader opens '{ARCNAME[dataset]}'")
    assert not any("/" in n or n.startswith("__MACOSX") for n in names)
