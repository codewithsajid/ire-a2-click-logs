"""MIND (headerless TSV) -> canonical tables."""
from __future__ import annotations

import polars as pl

from .config import Config, SplitSpec
from .ids import remap_list

BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]
NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract", "url",
             "title_entities", "abstract_entities"]
# MIND stores US-format 12-hour timestamps; the raw strings sort wrong
# lexicographically, so everything downstream must use the parsed column.
TIME_FMT = "%m/%d/%Y %I:%M:%S %p"

_ENTITY_DTYPE = pl.List(pl.Struct({"Label": pl.Utf8, "WikidataId": pl.Utf8}))


def _news(cfg: Config, source: str) -> pl.LazyFrame:
    return pl.scan_csv(
        cfg.raw / source / "news.tsv",
        separator="\t", has_header=False, quote_char=None,
        new_columns=NEWS_COLS,
        schema_overrides={c: pl.Utf8 for c in NEWS_COLS},
    )


def articles(cfg: Config) -> pl.LazyFrame:
    """Union news.tsv across splits: MIND ships a different (overlapping) news
    table per split -- 51K train / 42K dev / 121K large-test -- and an article
    seen only at test time still has to be retrievable."""
    frames = [_news(cfg, src) for src in cfg.article_sources]
    lf = pl.concat(frames, how="vertical_relaxed").unique(subset=["news_id"], keep="first").sort("news_id")
    return lf.select(
        pl.col("news_id").alias("src_id"),
        pl.col("title"),
        pl.col("abstract"),
        pl.lit(None, dtype=pl.Utf8).alias("body"),          # MIND ships no body text
        pl.col("category"),
        pl.col("subcategory").cast(pl.Utf8).str.split(" ").alias("subcategory"),
        # Wikidata-linked entities from both fields, deduplicated by label.
        pl.concat_list(
            pl.col("title_entities").str.json_decode(_ENTITY_DTYPE, infer_schema_length=None)
              .list.eval(pl.element().struct.field("Label")),
            pl.col("abstract_entities").str.json_decode(_ENTITY_DTYPE, infer_schema_length=None)
              .list.eval(pl.element().struct.field("Label")),
        ).list.unique().alias("entities"),
        pl.lit(None, dtype=pl.Datetime("us")).alias("published_time"),  # not provided
    ).with_columns(
        pl.concat_str([pl.col("title"), pl.col("abstract")], separator=" ", ignore_nulls=True).alias("text"),
        pl.lit("en").alias("lang"),
        (pl.col("abstract").fill_null("").str.strip_chars().str.len_chars() > 0).alias("has_abstract"),
    )


def _behaviors(cfg: Config, spec: SplitSpec) -> pl.LazyFrame:
    lf = pl.scan_csv(
        cfg.raw / spec.source / "behaviors.tsv",
        separator="\t", has_header=False, quote_char=None,
        new_columns=BEHAVIOR_COLS,
        schema_overrides={"impression_id": pl.Int64, "user_id": pl.Utf8,
                          "time": pl.Utf8, "history": pl.Utf8, "impressions": pl.Utf8},
    ).with_row_index("src_row").with_columns(pl.col("time").str.to_datetime(TIME_FMT).alias("time"))
    if spec.time_from is not None:
        lf = lf.filter(pl.col("time") >= pl.lit(spec.time_from))
    if spec.time_to is not None:
        lf = lf.filter(pl.col("time") < pl.lit(spec.time_to))
    return lf


def impressions(cfg: Config, spec: SplitSpec, art_map: pl.LazyFrame, user_map: pl.LazyFrame) -> pl.LazyFrame:
    """`impressions` is "N123-1 N456-0" on labelled splits and bare "N123 N456"
    on the leaderboard split.

    `src_row` and the pinned join order matter: the MIND grader reads one line
    per row and checks each line's impression id against the raw file at that
    position, so a hash-ordered store cannot produce a valid submission.
    """
    lf = _behaviors(cfg, spec).with_columns(pl.col("impressions").str.split(" ").alias("_items"))
    if spec.labelled:
        lf = lf.with_columns(
            pl.col("_items").list.eval(pl.element().str.head(-2)).alias("_cand"),
            pl.col("_items").list.eval(
                pl.element().filter(pl.element().str.ends_with("-1")).str.head(-2)
            ).alias("_click"),
        )
    else:
        lf = lf.with_columns(
            pl.col("_items").alias("_cand"),
            pl.lit([], dtype=pl.List(pl.Utf8)).alias("_click"),
        )
    lf = remap_list(lf, "_cand", art_map, out="candidates")
    lf = remap_list(lf, "_click", art_map, out="clicked")
    return (
        lf.select("src_row", "impression_id", "user_id", "time", "candidates", "clicked")
        .join(user_map.rename({"src_id": "user_id"}), on="user_id", how="left",
              maintain_order="left")
        .rename({"idx": "user_idx"})
        .drop("user_id")
        .with_columns(pl.col("candidates").list.len().cast(pl.UInt16).alias("n_candidates"))
    )


def history(cfg: Config, spec: SplitSpec, art_map: pl.LazyFrame, user_map: pl.LazyFrame) -> pl.LazyFrame:
    """MIND carries history inline per impression, with no timestamps. Rows for
    the same user repeat it, so we keep the longest observed history per user
    and leave `ts` null -- recency downstream is rank position only."""
    lf = (
        _behaviors(cfg, spec)
        .select("user_id", pl.col("history").fill_null("").str.split(" ").alias("_hist"))
        .with_columns(pl.col("_hist").list.eval(pl.element().filter(pl.element().str.len_chars() > 0)))
        .with_columns(pl.col("_hist").list.len().alias("_n"))
        .sort("_n", descending=True)
        .unique(subset=["user_id"], keep="first")
    )
    lf = remap_list(lf, "_hist", art_map, out="article_idx")
    return (
        lf.select("user_id", "article_idx")
        .join(user_map.rename({"src_id": "user_id"}), on="user_id", how="left",
              maintain_order="left")
        .rename({"idx": "user_idx"})
        .drop("user_id")
        .with_columns(pl.col("article_idx").fill_null(pl.lit([], dtype=pl.List(pl.UInt32))))
        .with_columns(
            pl.lit(None, dtype=pl.List(pl.Datetime("us"))).alias("ts"),
            pl.col("article_idx").list.len().cast(pl.UInt32).alias("n_hist"),
        )
    )


def user_ids(cfg: Config) -> pl.LazyFrame:
    return pl.concat(
        [
            pl.scan_csv(cfg.raw / spec.source / "behaviors.tsv", separator="\t", has_header=False,
                        quote_char=None, new_columns=BEHAVIOR_COLS,
                        schema_overrides={c: pl.Utf8 for c in BEHAVIOR_COLS})
            .select(pl.col("user_id").alias("src_id"))
            for spec in cfg.splits.values()
        ],
        how="vertical_relaxed",
    )
