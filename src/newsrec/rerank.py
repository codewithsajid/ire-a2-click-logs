"""Q2: the second stage -- a LambdaMART re-ranker over behavioural features.

Stage one (A1) answers "which articles could plausibly be shown". Stage two
answers "in what order", and it is allowed to be expensive because it sees a
hundred candidates rather than a hundred thousand. That asymmetry is the whole
argument for a cascade: feature cost is what forces the funnel, so a cheap model
runs on many documents and an expensive one on few.

Why a GBDT. The feature matrix assembled here is heterogeneous and full of
structural nulls -- a MIND user has no `session_rank` because MIND ships no
sessions, a cold user has no `hours_since_last_click` because they have no prior
clicks, and BM25 scores live on a different scale from a cosine. Axis-aligned
splits handle all three without imputation or scaling, and tuned GBDTs remain the
benchmark to beat on tabular learning-to-rank.

Why `lambdarank` rather than logistic regression on the same features. The
ranking metrics are flat almost everywhere in the scores -- moving a candidate
from rank 40 to rank 39 changes nDCG@10 by exactly nothing -- so they cannot be
differentiated directly. LambdaRank's move is to take the pairwise gradient and
scale it by the metric change a swap would cause, which concentrates capacity at
the top of the list where the metric actually moves. Whether that is worth
anything here is `scripts/q3_ablation.py`'s objective sweep, not an assumption.

Features are declared in families rather than as one flat list, because the
ablation that matters is "what does this *kind* of signal contribute", and
because the two datasets legitimately support different families.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp

from .config import N_RECENT_DEFAULT
from .embeddings import article_embeddings
from .lexical import BM25Index
from .retrieval import user_histories
from .semantic import l2_normalise, user_vectors
from .store import FeatureStore

# ---------------------------------------------------------------- families
#
# The taxonomy is the standard learning-to-rank one: signals about the user
# alone, about the article alone, about the match between them, and about the
# serving context. `grey` is this assignment's Q9 column -- features that exist
# in the logs but that a system choosing what to show has not yet observed.

FAMILIES: dict[str, tuple[str, ...]] = {
    "user": (
        "n_hist", "clicks_24h", "clicks_7d", "hours_since_last_click",
        "top_category_share", "n_categories", "category_entropy",
        # dwell, aggregated over impressions that closed before this one -- the
        # serving-safe half of EB-NeRD's read_time (null on MIND)
        "prior_read_time", "prior_scroll", "prior_impressions",
    ),
    "article": (
        "ctr_smoothed", "clicks_decayed", "prior_clicks", "prior_inview",
        "age_hours",
        # rolling exposure count and in-split age. Both are computable on an
        # unlabelled split -- candidate lists are published -- so both may ship
        # to the leaderboard.
        "roll_inview", "roll_age_hours",
    ),
    # Counters that need in-split *labels*. A production feature store has these
    # (a click an hour ago is in the log); the Codabench test set withholds them
    # by construction. So they are built, measured, and reported as "what the
    # system can do with a live log" -- never folded into a submission.
    "rolling": (
        "roll_clicks", "roll_ctr",
        # the dwell an article earned before this impression: the one signal that
        # separates "attracted a click" from "held attention"
        "art_read_time", "art_dwell_n",
    ),
    "match": (
        "bm25", "emb_cos", "emb_max", "emb_recent",
        "cat_share", "cat_clicks", "cat_recency", "cat_recency_share",
    ),
    "context": (
        "session_rank", "session_seconds", "secs_since_prev",
        "device_type", "is_subscriber", "is_sso_user", "n_candidates",
    ),
    # Never in the shipped model. Built so the with/without comparison Q9 asks
    # for is a measurement rather than a claim.
    "grey": ("read_time", "scroll_percentage", "position", "position_frac"),
}

# What goes to Codabench: every family whose features can be computed on a split
# with no labels in it.
SHIPPED = ("user", "article", "match", "context")

# What a system with a live click log can use. The gap between this and SHIPPED
# is the honest measure of what the withheld labels cost us, and is reported as
# such rather than being presented as a leaderboard result.
PRODUCTION = SHIPPED + ("rolling",)


def feature_names(df: pl.DataFrame, families=SHIPPED) -> list[str]:
    """The columns of `df` belonging to the requested families, in a fixed order.

    Intersected with what the frame actually has, because MIND has no session or
    device columns at all. A missing family shrinks the model rather than
    crashing it, and the design note reports the two feature counts separately
    instead of implying one model was trained twice.
    """
    have = set(df.columns)
    return [c for fam in families for c in FAMILIES[fam] if c in have]


# ------------------------------------------------------- matching features

def add_matching_features(fs: FeatureStore, split: str, pairs: pl.DataFrame,
                          emb_name: str, k1: float, b: float,
                          history_mode: str = "shipped",
                          n_recent: int = N_RECENT_DEFAULT,
                          halflife_rank: float = 10.0,
                          newest_last: bool = True) -> pl.DataFrame:
    """Score every (user, candidate) pair with A1's two retrievers.

    These are the `match` family, and they are what carries the tail: a user with
    no click history has no behavioural features worth the name, and only the
    content signals have anything to say about them.

    Four columns:

      * `bm25`       -- BM25 of the candidate against the user's whole history as
                        a bag of terms, at the (k1, b) A1 tuned for this corpus.
      * `emb_cos`    -- cosine to the mean-pooled history vector, i.e. exactly
                        A1's semantic ranker.
      * `emb_max`    -- the *largest* cosine to any single clicked article. Mean
                        pooling answers "is this like their average interest";
                        the max answers "is this like anything they read", which
                        is a different question for a multi-interest reader and is
                        the cheap stand-in for the multi-vector user
                        representation A1 measured and declined to ship.
      * `emb_recent` -- cosine to a recency-weighted history vector. On EB-NeRD
                        the weights come from timestamps; on MIND, which ships no
                        history times, they decay over rank position instead.

    Computed with `BM25Index.score_pairs` rather than by retrieving and joining:
    scoring the pairs directly is O(pairs x terms-per-document) with no dependence
    on corpus size, where a dense user-by-corpus product would be O(users x corpus)
    to read out a few million cells.
    """
    out = pairs
    user_ids, hists = user_histories(fs, split, history_mode)
    # dense map from user_idx -> row in the history arrays, so a pair can find
    # its user's query vector without a join
    urow = np.full(int(user_ids.max()) + 1 if len(user_ids) else 1, -1, dtype=np.int64)
    urow[user_ids] = np.arange(len(user_ids))

    pu = pairs["user_idx"].to_numpy()
    pa = pairs["article_idx"].to_numpy()
    # Bounds-check rather than clip. Clipping an out-of-range user_idx lands it on
    # the last slot of `urow`, which holds a real and entirely different user --
    # so an unknown user would silently be scored with someone else's query vector
    # instead of being marked unknown. No split has such a user at dev scale
    # (measured: 0 rows on every split of both datasets), but the large variants
    # and the unlabelled submit split carry users the history table need not cover.
    rows = np.full(len(pu), -1, dtype=np.int64)
    inb = pu < len(urow)
    rows[inb] = urow[pu[inb]]
    known = rows >= 0

    # ---- lexical
    idx = BM25Index.build(fs.texts(), lang=fs.lang(), k1=k1, b=b)
    Q = idx.queries_from_history(hists, n_recent=n_recent)
    bm25 = np.zeros(len(pairs), dtype=np.float32)
    if known.any():
        bm25[known] = idx.score_pairs(Q, rows[known], pa[known])
    out = out.with_columns(pl.Series("bm25", bm25))

    # ---- semantic
    emb = np.asarray(article_embeddings(fs, emb_name), dtype=np.float32)
    embn = l2_normalise(emb)

    uv = user_vectors(hists, embn, n_recent=n_recent)
    # `user_vectors(recency_weighted=True)` decays over list position and treats
    # the last element as the most recent. On MIND that end is an assumption the
    # data cannot confirm (the history snapshot is identical in every bundle), so
    # the lists are reversed here when the assumption is being tested the other
    # way round -- rather than duplicating the decay logic.
    rh = hists if newest_last else [h[::-1] for h in hists]
    uvr = user_vectors(rh, embn, n_recent=n_recent, recency_weighted=True,
                       halflife=halflife_rank)

    cos = np.zeros(len(pairs), dtype=np.float32)
    cosr = np.zeros(len(pairs), dtype=np.float32)
    if known.any():
        cos[known] = np.einsum("ij,ij->i", uv[rows[known]], embn[pa[known]])
        cosr[known] = np.einsum("ij,ij->i", uvr[rows[known]], embn[pa[known]])

    # ---- max-over-history cosine, as one sparse product per user block
    #
    # The obvious loop over pairs gathers |history| vectors per pair and is the
    # slowest thing in the assembly. Instead the candidates of each user are
    # scored against that user's history in one small dense product; histories
    # average 160 clicks on EB-NeRD and 19 on MIND, so each product is tiny and
    # there is one per user rather than one per pair.
    emax = np.zeros(len(pairs), dtype=np.float32)
    order = np.argsort(rows, kind="stable")
    ru, starts = np.unique(rows[order], return_index=True)
    bounds = np.append(starts, len(order))
    for j, r in enumerate(ru):
        if r < 0:
            continue
        sel = order[bounds[j]:bounds[j + 1]]
        h = hists[r]
        h = h[h < embn.shape[0]]
        if h.size == 0:
            continue
        emax[sel] = (embn[pa[sel]] @ embn[h].T).max(axis=1)

    return out.with_columns(
        pl.Series("emb_cos", cos),
        pl.Series("emb_recent", cosr),
        pl.Series("emb_max", emax),
    )


# ------------------------------------------------------------------ model

def _groups(df: pl.DataFrame) -> np.ndarray:
    """Candidate counts per impression, in the order the rows appear.

    LightGBM's ranking objectives take group sizes rather than group ids, so the
    frame must already be sorted by `imp` -- a group boundary computed from an
    unsorted frame silently trains on interleaved impressions and still returns a
    plausible-looking model.
    """
    imp = df["imp"].to_numpy()
    if np.any(np.diff(imp) < 0):
        raise ValueError("rows are not grouped by `imp`; sort before training")
    _, counts = np.unique(imp, return_counts=True)
    return counts


def train(train_df: pl.DataFrame, val_df: pl.DataFrame, features: list[str],
          objective: str = "lambdarank", num_boost_round: int = 600,
          learning_rate: float = 0.05, num_leaves: int = 63,
          min_data_in_leaf: int = 50, early_stopping: int = 50,
          weights: np.ndarray | None = None, seed: int = 0,
          ndcg_eval_at: tuple[int, ...] = (5, 10), verbose_eval: int = 50,
          truncation: int = 30):
    """Fit a LightGBM ranker, early-stopped on the validation split.

    `objective` spans the pointwise -> pairwise -> listwise arc on one flag:

        binary          pointwise -- regress the click, ignore that ranking is relative
        rank_xendcg     pairwise-ish -- a listwise cross-entropy surrogate
        lambdarank      listwise -- pair gradients weighted by their nDCG impact

    The sweep over these is the point of `scripts/q3_ablation.py`; this function
    just takes whichever one it is told.

    Early stopping is on the validation split's nDCG@10 rather than on a fixed
    round count, because the two datasets do not want the same number of trees --
    MIND has twice the candidates per impression and half the history depth.
    """
    import lightgbm as lgb

    params = {
        "objective": objective,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "seed": seed,
        "num_threads": 0,
        "verbosity": -1,
    }
    # Early stopping always watches nDCG, whatever the objective optimises.
    # Letting the stopping rule change with the objective confounds the
    # comparison between them: a pointwise run stopped on AUC and a listwise run
    # stopped on nDCG differ in two things at once, and on MIND that showed up as
    # 148 trees against 13. Groups are set for every objective so the ranking
    # metric is computable even when the loss is pointwise.
    params |= {"metric": "ndcg", "ndcg_eval_at": list(ndcg_eval_at)}
    if objective == "lambdarank":
        # How deep into each list LambdaRank forms pairs. Beyond it, a swap
        # contributes no gradient at all -- so a truncation shorter than the list
        # leaves part of every impression untrained. EB-NeRD shows ~11 candidates
        # and fits inside the default; MIND shows ~36 and up to 299 and does not.
        params |= {"lambdarank_truncation_level": truncation}

    X = train_df.select(features).to_numpy()
    y = train_df["label"].to_numpy().astype(np.int8)
    dtrain = lgb.Dataset(X, label=y, feature_name=features, weight=weights,
                         free_raw_data=False)
    dtrain.set_group(_groups(train_df))

    Xv = val_df.select(features).to_numpy()
    dval = lgb.Dataset(Xv, label=val_df["label"].to_numpy().astype(np.int8),
                       feature_name=features, reference=dtrain, free_raw_data=False)
    dval.set_group(_groups(val_df))

    return lgb.train(
        params, dtrain, num_boost_round=num_boost_round,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False),
                   lgb.log_evaluation(verbose_eval)],
    )


def predict(model, df: pl.DataFrame, features: list[str],
            batch: int = 2_000_000) -> np.ndarray:
    """Score in blocks -- the large splits do not fit as one dense matrix."""
    out = np.empty(df.height, dtype=np.float32)
    for s in range(0, df.height, batch):
        chunk = df.slice(s, batch).select(features).to_numpy()
        out[s:s + len(chunk)] = model.predict(chunk, num_iteration=model.best_iteration)
    return out


def importances(model, features: list[str]) -> pl.DataFrame:
    """Gain-based importance, normalised to shares.

    Gain rather than split count: a feature can be split on constantly and move
    the objective very little, which is how a high-cardinality id column looks
    important without being useful.
    """
    gain = np.asarray(model.feature_importance("gain"), dtype=np.float64)
    tot = gain.sum() or 1.0
    fam = {c: f for f, cs in FAMILIES.items() for c in cs}
    return (
        pl.DataFrame({
            "feature": features,
            "family": [fam.get(c, "?") for c in features],
            "gain": gain,
            "share": gain / tot,
        })
        .sort("gain", descending=True)
    )
