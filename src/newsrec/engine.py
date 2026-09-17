"""Execution-engine shim.

The pipeline is written once in polars' lazy API. `collect()` decides how it
actually runs:

  polars : CPU, streaming executor -- bounded memory over the 12M-row bundles
  gpu    : cudf-polars, i.e. the same LazyFrame executed on the GPU

Native-cuDF rewrites of the hot operators live in `newsrec.bench`, which is
where the three-way comparison for the design note is measured. Keeping the GPU
option behind this one function means no dataframe dialect leaks into the
pipeline code.
"""
from __future__ import annotations

import functools
import os
import warnings

import polars as pl

ENGINE = os.environ.get("NEWSREC_ENGINE", "polars")


@functools.lru_cache(maxsize=1)
def gpu_available() -> bool:
    try:
        import cudf_polars  # noqa: F401
    except Exception:
        return False
    return hasattr(pl, "GPUEngine")


def collect(lf: pl.LazyFrame, engine: str | None = None) -> pl.DataFrame:
    """Materialise a LazyFrame under the requested engine."""
    engine = engine or ENGINE
    if engine == "gpu":
        if not gpu_available():
            warnings.warn("GPU engine requested but cudf-polars is unavailable; using CPU")
        else:
            # raise_on_fail=False: unsupported nodes silently fall back to CPU,
            # which is what we want in the pipeline (bench.py sets it True so
            # fallbacks show up as failures rather than as fake GPU timings).
            return lf.collect(engine=pl.GPUEngine(raise_on_fail=False))
    return lf.collect(engine="streaming")


def sink(lf: pl.LazyFrame, path, engine: str | None = None) -> None:
    """Write a LazyFrame straight to parquet, streaming where possible."""
    engine = engine or ENGINE
    if engine == "polars":
        lf.sink_parquet(path, compression="zstd")
    else:
        collect(lf, engine).write_parquet(path, compression="zstd")
