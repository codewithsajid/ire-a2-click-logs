"""Build configuration: one YAML per (dataset, variant)."""
from __future__ import annotations

import functools
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import yaml

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))

# How many of a user's most recent clicks build their query. 0 reads the whole
# history. The sweep in scripts/q3_userrep.py (reports/q3/userrep_*.json) picked
# this: reading everything beat every truncation on both datasets, by +22.6%
# BM25 recall@100 on EB-NeRD (median history 81) and +1.6-2.7% on MIND (median
# 15) -- the gain tracks how much history a window would have discarded.
N_RECENT_DEFAULT = 0

# Modules whose source can change what the build writes. A change to any of them
# invalidates every cached stage; retrieval/eval code is deliberately excluded, so
# editing a plot script does not trigger a multi-hour rebuild.
BUILD_MODULES = ("config.py", "schema.py", "ids.py", "build.py", "features.py",
                 "ingest_ebnerd.py", "ingest_mind.py", "embeddings.py", "engine.py")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@functools.lru_cache(maxsize=1)
def _code_fingerprint() -> str:
    here = Path(__file__).resolve().parent
    parts = []
    for name in BUILD_MODULES:
        f = here / name
        parts.append(f"{name}:{hashlib.sha256(f.read_bytes()).hexdigest()}" if f.exists()
                     else f"{name}:MISSING")
    return _sha("\n".join(parts))



@dataclass
class SplitSpec:
    """Where one canonical split comes from in the raw tree.

    `source` names a raw bundle directory; `part` the split inside it (EB-NeRD
    nests train/validation/test, MIND does not). `time_from`/`time_to` carve a
    time window out of that source -- this is how train/val are split out of the
    official train week without touching the official val/test.
    """
    source: str
    part: str | None = None
    time_from: datetime | None = None
    time_to: datetime | None = None
    labelled: bool = True


@dataclass
class Config:
    dataset: str                      # "ebnerd" | "mind"
    variant: str                      # "demo" | "small" | "large"
    splits: dict[str, SplitSpec]
    article_sources: list[str]        # raw bundles whose article tables are unioned
    engine: str = "polars"            # "polars" | "gpu"
    history_mode: str = "shipped"     # "shipped" | "augmented"
    popularity_halflife_hours: float = 24.0
    candidate_universe_days: int = 7   # articles considered "live" at retrieval time
    extra: dict = field(default_factory=dict)

    @property
    def raw(self) -> Path:
        return DATA_ROOT / "raw" / ("ebnerd" if self.dataset == "ebnerd" else "mind")

    @property
    def out(self) -> Path:
        return DATA_ROOT / "processed" / self.dataset / self.variant

    def fingerprint(self) -> dict[str, str]:
        """Everything that can change a stage's output, in three parts.

        Hashing the config alone was not enough: editing `features.py` and
        re-running the build skipped every stage, so the store silently kept the
        old logic while the reports claimed the new one. Code and raw-input state
        are hashed too, and kept separate so a mismatch says *what* moved.

        Raw inputs are fingerprinted by (path, size) rather than content -- the
        bundles are gigabytes and a size change is what a truncated or swapped
        download actually looks like. mtime is deliberately excluded: re-extracting
        an unchanged zip would otherwise force a full rebuild.
        """
        return {
            "config": _sha(json.dumps(asdict(self), sort_keys=True, default=str)),
            "code": _code_fingerprint(),
            "raw": self._raw_fingerprint(),
        }

    def _raw_fingerprint(self) -> str:
        roots = {self.raw / src for src in self.article_sources}
        roots |= {self.raw / spec.source for spec in self.splits.values()}
        items = []
        for root in sorted(roots):
            if not root.exists():
                items.append(f"{root.name}:MISSING")
                continue
            for f in sorted(root.rglob("*")):
                if f.is_file() and f.suffix in (".parquet", ".tsv"):
                    items.append(f"{f.relative_to(self.raw)}:{f.stat().st_size}")
        return _sha("\n".join(items))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text())
        splits = {k: SplitSpec(**v) for k, v in raw.pop("splits").items()}
        return cls(splits=splits, **raw)
