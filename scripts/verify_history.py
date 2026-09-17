"""Verify the assumptions the MIND recency features rest on.

Three things are asserted elsewhere in this codebase and none of them is
self-evident from the data:

1. **MIND ships no history timestamps.** `behaviors.tsv` has five fields and the
   fourth is a bare list of news ids. If that is right, MIND's recency features
   can only decay over *rank*, which is what `history_recency` does.

2. **MIND's history is ordered oldest-first.** This is the load-bearing one. The
   rank decay weights element *i* by `0.5 ** ((n - 1 - i) / halflife)`, i.e. it
   treats the *last* element as the most recent. If the list is newest-first the
   weighting is exactly inverted, and nothing downstream would complain -- the
   feature would simply be wrong in a way that looks like noise.

   Tested by pairing each user's train history against their dev history. Dev is
   strictly later in time, so a user present in both should have accumulated
   clicks. Where one history is a prefix of the other, the new clicks were
   appended and the list is oldest-first; where it is a suffix, they were
   prepended and it is newest-first.

3. **EB-NeRD's history timestamps are ascending.** Same question for the dataset
   that does carry times, where it can be checked directly.

Writes reports/q1/verify_history.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from newsrec.config import DATA_ROOT
from newsrec.store import FeatureStore

BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


def mind_raw(bundle: str) -> pl.LazyFrame:
    return pl.scan_csv(
        DATA_ROOT / "raw" / "mind" / bundle / "behaviors.tsv",
        separator="\t", has_header=False, quote_char=None,
        new_columns=BEHAVIOR_COLS,
        schema_overrides={c: pl.Utf8 for c in BEHAVIOR_COLS},
    )


def check_field_count() -> dict:
    p = DATA_ROOT / "raw" / "mind" / "MINDsmall_train" / "behaviors.tsv"
    with open(p) as f:
        first = f.readline().rstrip("\n")
    n = len(first.split("\t"))
    hist = first.split("\t")[3] if n >= 4 else ""
    return {
        "n_fields": n,
        "history_sample": hist[:80],
        # a timestamp would need a digit-slash-digit or a colon in it
        "history_contains_time_chars": any(c in hist for c in (":", "/")),
    }


def check_mind_order() -> dict:
    """Prefix vs suffix: which end did the newer clicks arrive at."""
    def hist_map(bundle: str) -> pl.DataFrame:
        return (
            mind_raw(bundle)
            .select("user_id", "history")
            .drop_nulls("history")
            .unique(subset=["user_id"])
            .with_columns(pl.col("history").str.split(" ").alias("h"))
            .select("user_id", "h", pl.col("h").list.len().alias("n"))
            .collect()
        )

    tr, dv = hist_map("MINDsmall_train"), hist_map("MINDsmall_dev")
    j = tr.join(dv, on="user_id", how="inner", suffix="_dev")
    grew = j.filter(pl.col("n_dev") > pl.col("n"))
    if grew.is_empty():
        return {"n_users_both": j.height, "n_grew": 0,
                "verdict": "histories never grow between train and dev"}

    prefix = suffix = neither = 0
    for row in grew.head(20000).iter_rows(named=True):
        old, new, k = row["h"], row["h_dev"], row["n"]
        if new[:k] == old:
            prefix += 1          # appended at the end -> oldest-first
        elif new[-k:] == old:
            suffix += 1          # prepended at the front -> newest-first
        else:
            neither += 1
    total = prefix + suffix + neither
    verdict = ("oldest-first (new clicks appended)" if prefix > suffix
               else "newest-first (new clicks prepended)" if suffix > prefix
               else "indeterminate")
    return {
        "n_users_both": j.height, "n_grew": grew.height, "n_checked": total,
        "old_is_prefix_of_new": prefix, "old_is_suffix_of_new": suffix,
        "neither": neither, "verdict": verdict,
    }


def check_ebnerd_order() -> dict:
    """EB-NeRD carries per-click times, so ascendingness is directly checkable."""
    try:
        fs = FeatureStore("ebnerd", "small")
    except FileNotFoundError:
        return {"skipped": "no ebnerd/small store"}
    h = (
        fs.history("train").select("ts")
        .filter(pl.col("ts").list.len() > 1)
        .head(50_000)
        .collect()
    )
    asc = desc = 0
    for (ts,) in h.iter_rows():
        if ts is None or len(ts) < 2:
            continue
        s = list(ts)
        if all(a <= b for a, b in zip(s, s[1:])):
            asc += 1
        elif all(a >= b for a, b in zip(s, s[1:])):
            desc += 1
    return {"n_checked": h.height, "ascending": asc, "descending": desc,
            "verdict": "ascending (oldest-first)" if asc > desc else "descending"}


def check_store_ts_is_null() -> dict:
    out = {}
    for ds in ("mind", "ebnerd"):
        try:
            fs = FeatureStore(ds, "small")
        except FileNotFoundError:
            continue
        n = (
            fs.history("train")
            .select(pl.col("ts").list.len().fill_null(0).sum().alias("n_ts"),
                    pl.col("article_idx").list.len().sum().alias("n_art"))
            .collect().to_dicts()[0]
        )
        out[ds] = n
    return out


def main() -> None:
    rep = {
        "mind_raw_fields": check_field_count(),
        "mind_history_order": check_mind_order(),
        "ebnerd_history_order": check_ebnerd_order(),
        "store_history_timestamps": check_store_ts_is_null(),
    }
    print(json.dumps(rep, indent=2))
    out = Path("reports/q1/verify_history.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
