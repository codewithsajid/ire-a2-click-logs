"""The store must reproduce its source files' row order, exactly.

Both graders read one prediction per row, in the raw file's order, and check
each line's impression id against the raw file at that position. EB-NeRD's
beyond-accuracy rows all carry impression_id = 0, so row position is the only
identifier there is -- an out-of-order store cannot produce a valid submission
and cannot be repaired by sorting. A plain polars join returns rows in hash
order; this asserts the fix (`src_row` + `maintain_order`) actually holds.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from newsrec.config import DATA_ROOT, Config

CONFIGS = sorted(Path("configs").glob("*.yaml"))


def _cfg_with_store(path: Path):
    cfg = Config.load(path)
    return cfg if (cfg.out / "manifest.json").exists() else None


CASES = [(p, s) for p in CONFIGS
         if (c := _cfg_with_store(p)) is not None
         for s in c.splits]
pytestmark = pytest.mark.skipif(not CASES, reason="no feature store built yet")


def _raw_ids(cfg: Config, spec) -> list[int]:
    if cfg.dataset == "ebnerd":
        part = f"{spec.part}/" if spec.part else ""
        lf = pl.scan_parquet(cfg.raw / spec.source / part / "behaviors.parquet")
        return lf.select("impression_id").collect()["impression_id"].to_list()
    lf = pl.scan_csv(cfg.raw / spec.source / "behaviors.tsv", separator="\t",
                     has_header=False, quote_char=None,
                     new_columns=["impression_id", "user_id", "time", "history", "impressions"],
                     schema_overrides={"impression_id": pl.Int64})
    return lf.select("impression_id").collect()["impression_id"].to_list()


@pytest.mark.parametrize("cfg_path,split", CASES, ids=[f"{p.stem}:{s}" for p, s in CASES])
def test_src_row_indexes_the_raw_file(cfg_path: Path, split: str):
    """src_row must be the row's position in its source behaviours file."""
    cfg = Config.load(cfg_path)
    store = pl.read_parquet(cfg.out / "splits" / split / "impressions.parquet")
    assert "src_row" in store.columns, "src_row is what makes submissions reconstructable"

    raw_ids = _raw_ids(cfg, cfg.splits[split])
    sr = store["src_row"].to_list()
    assert max(sr) < len(raw_ids), "src_row points past the end of the source file"
    # every stored row's id equals the raw file's id at that position
    mism = [i for i, (r, got) in enumerate(zip(sr, store["impression_id"].to_list()))
            if raw_ids[r] != got]
    assert not mism, f"{len(mism)} rows disagree with the raw file at their src_row"


@pytest.mark.parametrize("cfg_path,split", CASES, ids=[f"{p.stem}:{s}" for p, s in CASES])
def test_store_rows_are_in_source_order(cfg_path: Path, split: str):
    """And the rows are stored in that order, so no sort is needed to submit."""
    cfg = Config.load(cfg_path)
    sr = pl.read_parquet(cfg.out / "splits" / split / "impressions.parquet")["src_row"].to_list()
    assert sr == sorted(sr), "impressions are not in source-file order"
