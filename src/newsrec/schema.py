"""The canonical schema both datasets are mapped onto.

Design intent: one set of tables, with columns a dataset genuinely lacks left
null rather than faked. The asymmetries that matter downstream:

  * MIND has no article `published_time` -- we derive `first_seen_time` from the
    earliest impression an article appears in, which is the only honest proxy.
  * MIND history carries no timestamps, so `history.ts` is null there and
    recency features degrade to rank position ("k-th most recent click").
  * MIND has no body text; EB-NeRD has no Wikidata entity ids.
"""
from __future__ import annotations

import polars as pl

ARTICLES = {
    "idx": pl.UInt32,             # dense row index -- BM25/FAISS row id
    "src_id": pl.Utf8,            # "N55528" (MIND) or "9771627" (EB-NeRD)
    "title": pl.Utf8,
    "abstract": pl.Utf8,          # MIND abstract / EB-NeRD subtitle
    "body": pl.Utf8,              # EB-NeRD only
    "category": pl.Utf8,
    "subcategory": pl.List(pl.Utf8),
    "entities": pl.List(pl.Utf8),
    "published_time": pl.Datetime("us"),   # EB-NeRD only
    "first_seen_time": pl.Datetime("us"),  # derived; the usable proxy for MIND
    "text": pl.Utf8,              # title + abstract, pre-joined for lexical retrieval
    "lang": pl.Utf8,              # "da" | "en" -- picks the BM25 stemmer in Q2
    "has_abstract": pl.Boolean,   # 5.2% of MIND and 1,709 EB-NeRD articles lack one.
                                  # Flagged, never dropped: Codabench needs a
                                  # prediction for every impression, so shrinking
                                  # the catalogue is not an option.
}

IMPRESSIONS = {
    # 0-based position of this row in its source behaviours file. Both graders
    # score one line per raw row in the raw file's order, and EB-NeRD's
    # beyond-accuracy rows all carry impression_id = 0, so position is the only
    # identifier there is. Ingest joins reorder rows, so the order has to be
    # carried explicitly rather than assumed.
    "src_row": pl.UInt32,
    "impression_id": pl.Int64,
    "user_idx": pl.UInt32,
    "time": pl.Datetime("us"),
    "candidates": pl.List(pl.UInt32),
    "clicked": pl.List(pl.UInt32),   # empty list on unlabelled splits
    "n_candidates": pl.UInt16,
}

# Context available at serving time. Deliberately excludes EB-NeRD's
# next_read_time / next_scroll_percentage: those describe the *following*
# impression and are pure future information (assignment Q9).
CONTEXT = {
    "device_type": pl.Int8,
    "session_id": pl.UInt32,
    "is_subscriber": pl.Boolean,
    "is_sso_user": pl.Boolean,
    "read_time": pl.Float32,
    "scroll_percentage": pl.Float32,
}

HISTORY = {
    "user_idx": pl.UInt32,
    "article_idx": pl.List(pl.UInt32),
    "ts": pl.List(pl.Datetime("us")),   # null for MIND
    "n_hist": pl.UInt32,
}

# (user, article, time) triples for every observed click on a labelled split.
# This is what `history_mode: augmented` draws on, always with a strict
# time cutoff at the target split's start.
CLICKSTREAM = {
    "user_idx": pl.UInt32,
    "article_idx": pl.UInt32,
    "ts": pl.Datetime("us"),
}

LEAK_COLUMNS = ("next_read_time", "next_scroll_percentage")
