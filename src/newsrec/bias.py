"""Position bias: estimating it from logs, and correcting for it when training.

The click log is not a relevance judgement. It records what happened under the
ranking we chose to show, so a logged click mixes two things the data never
separates:

    P(click at rank k) = P(examined at rank k) * P(attractive | examined)

That is the examination hypothesis (Craswell et al., 2008). The first factor is
attention and belongs to the slot; the second is relevance and belongs to the
article. Training on raw clicks fits their product, which teaches the model that
whatever the platform put at the top is good -- and since the model then puts
those articles at the top, the loop closes on itself.

Two estimators are implemented, and they answer different questions:

  * `ctr_by_position` is the *observed* curve. It is quality and attention mixed,
    and on its own it proves nothing: a platform that ranks well will show a
    decaying curve even with no attention effect at all, because it put the good
    articles first on purpose.

  * `propensity_by_position` holds the *article* fixed and asks how its click
    rate moves as the platform happens to place it at different ranks. Because
    the article is the same, its attractiveness is the same, and what remains is
    examination. This is the observational stand-in for the randomisation
    experiment the lecture's dashed curve comes from -- we cannot randomise a
    historical log, but the platform's own churn has already scattered most
    articles across many positions.

The correction is inverse propensity weighting (Joachims et al., 2017): a click
observed at a rank that is rarely examined is worth more evidence than one at the
top, so it is weighted by 1/P(examined at that rank). Weights are clipped,
because the variance of an IPW estimator is set by its smallest propensity and an
unclipped 1/0.02 term will dominate any finite sample.

Whether any of this *matters* is a measurement, not an assumption, and it is not
the same measurement on the two datasets: it depends on whether the candidate
list in the log preserves the order the user saw. `scripts/q1_position.py` tests
that rather than taking it on faith.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def ctr_by_position(pairs: pl.DataFrame, max_pos: int = 40) -> pl.DataFrame:
    """Observed click-through rate at each rendered position.

    The lecture's solid curve. Reported alongside the propensity curve so the
    design note can show the gap between "CTR falls with position" (which any
    competent ranker produces) and "attention falls with position" (which is the
    confound).
    """
    return (
        pairs.lazy()
        .filter(pl.col("position") < max_pos)
        .group_by("position")
        .agg(
            pl.len().alias("n_shown"),
            pl.col("label").sum().alias("n_clicked"),
        )
        .with_columns((pl.col("n_clicked") / pl.col("n_shown")).alias("ctr"))
        .sort("position")
        .collect()
    )


def ctr_by_position_stratified(pairs: pl.DataFrame, max_pos: int = 40,
                               min_impressions: int = 500) -> pl.DataFrame:
    """CTR by position *within impressions that showed the same number of items*.

    This is the estimator that matters, and the naive one above is a trap. Both
    datasets are near-single-click -- EB-NeRD averages 1.01 clicks per impression
    -- so the per-slot click rate of an impression showing L candidates is about
    1/L before any behaviour is involved. Position k is reachable only by lists of
    length > k, so the deep end of an unstratified curve is populated entirely by
    long lists, and it falls for arithmetic reasons that have nothing to do with
    attention. Measured on EB-NeRD, that artifact alone produced a 6.5x "decay".

    Conditioning on L removes it: inside a fixed L, every position has the same
    denominator and the same 1/L baseline, so a difference across positions is a
    difference in behaviour. `ratio` normalises each length's curve by its own
    mean rate, which makes curves from different L commensurable and puts the
    no-bias null at exactly 1.0 for every position.
    """
    per_len = (
        pairs.lazy()
        .filter(pl.col("position") < max_pos)
        .group_by("n_candidates", "position")
        .agg(pl.len().alias("n_shown"), pl.col("label").sum().alias("n_clicked"))
    )
    lens = (
        per_len.group_by("n_candidates")
        .agg(pl.col("n_shown").max().alias("_imps"),
             pl.col("n_clicked").sum().alias("_clicks"),
             pl.col("n_shown").sum().alias("_slots"))
        .filter((pl.col("_imps") >= min_impressions) & (pl.col("_clicks") > 0))
    )
    return (
        per_len.join(lens, on="n_candidates", how="inner")
        .with_columns(
            (pl.col("n_clicked") / pl.col("n_shown")).alias("_ctr"),
            (pl.col("_clicks") / pl.col("_slots")).alias("_base"),
        )
        .with_columns((pl.col("_ctr") / pl.col("_base")).alias("_ratio"))
        .group_by("position")
        .agg(
            # weight each length's contribution by how many slots it supplied, so
            # a length seen 500 times does not outvote one seen 200,000 times
            ((pl.col("_ratio") * pl.col("n_shown")).sum()
             / pl.col("n_shown").sum()).alias("ratio"),
            pl.col("n_shown").sum().alias("n_shown"),
            pl.col("n_clicked").sum().alias("n_clicked"),
            pl.col("n_candidates").n_unique().alias("n_lengths"),
        )
        .sort("position")
        .collect()
    )


def propensity_by_position(pairs: pl.DataFrame, max_pos: int = 40,
                           min_shown: int = 50, min_positions: int = 2,
                           ) -> pl.DataFrame:
    """Examination propensity per position, with the article held fixed.

    Method. Keep articles that the platform placed at at least `min_positions`
    distinct positions and showed at least `min_shown` times. Within each such
    article, compute its CTR at every position it reached, and divide by that
    article's own mean CTR. The ratio is unitless and the article's
    attractiveness cancels, so averaging it across articles leaves the positional
    component. Normalised to 1.0 at the top position, which is the convention
    IPW assumes.

    What it can still get wrong. The platform did not place articles at random,
    so an article can reach position 8 disproportionately on days when it was
    stale, and staleness lowers its CTR for reasons that are not attention. That
    residual confound biases the curve *towards* a steeper decay, i.e. towards
    over-correcting. This is why the shipped run reports the correction as an
    ablation against the uncorrected model rather than asserting it is right --
    and why the weights are clipped.

    Returns one row per position: `propensity` in (0, 1], plus the support it
    was estimated from.
    """
    p = (
        pairs.lazy()
        .filter(pl.col("position") < max_pos)
        .group_by("article_idx", "n_candidates", "position")
        .agg(pl.len().alias("n_shown"), pl.col("label").sum().alias("n_clicked"))
    )
    # Condition on list length as well as on the article. Holding the article
    # fixed removes differences in attractiveness; holding L fixed removes the
    # 1/L denominator effect that `ctr_by_position_stratified` documents. Both
    # are needed -- an article drifting to deeper positions is also drifting into
    # longer lists, and the naive version charges that arithmetic to attention.
    keep = (
        p.group_by("article_idx", "n_candidates")
        .agg(
            pl.col("n_shown").sum().alias("_tot_shown"),
            pl.col("n_clicked").sum().alias("_tot_clicked"),
            pl.col("position").n_unique().alias("_n_pos"),
        )
        .filter((pl.col("_tot_shown") >= min_shown)
                & (pl.col("_n_pos") >= min_positions)
                # an article nobody ever clicked carries no positional signal and
                # would contribute a 0/0 ratio
                & (pl.col("_tot_clicked") > 0))
    )
    joined = (
        p.join(keep, on=["article_idx", "n_candidates"], how="inner")
        .with_columns(
            (pl.col("n_clicked") / pl.col("n_shown")).alias("_ctr_at_pos"),
            (pl.col("_tot_clicked") / pl.col("_tot_shown")).alias("_ctr_overall"),
        )
        .with_columns((pl.col("_ctr_at_pos") / pl.col("_ctr_overall")).alias("_ratio"))
    )
    curve = (
        joined.group_by("position")
        .agg(
            # weight each article's ratio by how often it was shown there: a
            # ratio from 3 impressions is not evidence on a par with one from 3000
            ((pl.col("_ratio") * pl.col("n_shown")).sum()
             / pl.col("n_shown").sum()).alias("_raw"),
            pl.col("article_idx").n_unique().alias("n_articles"),
            pl.col("n_shown").sum().alias("n_shown"),
        )
        .sort("position")
        .collect()
    )
    if curve.is_empty():
        return curve.with_columns(pl.lit(1.0).alias("propensity"))
    top = float(curve["_raw"][0])
    return curve.with_columns(
        (pl.col("_raw") / (top if top > 0 else 1.0)).alias("propensity")
    ).drop("_raw")


def ipw_weights(position: np.ndarray, propensity: np.ndarray,
                clip: float = 0.1) -> np.ndarray:
    """1 / P(examined at this position), clipped.

    `clip` is a floor on the propensity, not a ceiling on the weight, so the
    maximum weight is 1/clip and is stated rather than emergent. The default of
    0.1 caps the weight at 10x, which the variance of a 2M-row training set can
    carry; the sweep in `scripts/q3_ablation.py` is what chose it.
    """
    pos = np.asarray(position, dtype=np.int64)
    pr = np.asarray(propensity, dtype=np.float64)
    safe = np.maximum(pr, clip)
    idx = np.clip(pos, 0, len(safe) - 1)
    return (1.0 / safe[idx]).astype(np.float32)


def curve_to_array(curve: pl.DataFrame, max_pos: int) -> np.ndarray:
    """Dense propensity array indexed by position, forward-filled past support.

    Positions deep in a long candidate list are seen a handful of times and their
    estimate is noise. Rather than trust it, the last well-supported value is
    carried forward -- a flat tail is a weaker claim than a jagged one, and IPW
    is sensitive to exactly the small propensities that noise produces.
    """
    out = np.ones(max_pos, dtype=np.float64)
    if curve.is_empty():
        return out
    have = dict(zip(curve["position"].to_list(), curve["propensity"].to_list()))
    last = 1.0
    for k in range(max_pos):
        if k in have and np.isfinite(have[k]) and have[k] > 0:
            last = float(have[k])
        out[k] = last
    return out


def position_bias_report(pairs: pl.DataFrame, max_pos: int = 40,
                         min_shown: int = 50) -> dict:
    """Both curves plus the summary numbers the design note quotes.

    `decay` is the headline: how much of the top position's click rate survives
    at the deepest well-supported rank, once the article is held fixed. A value
    near 1.0 means the log carries no usable positional signal -- which is the
    expected answer wherever the stored candidate order is not the order the user
    saw, and is a fact about the dataset rather than about attention.
    """
    obs = ctr_by_position(pairs, max_pos)
    strat = ctr_by_position_stratified(pairs, max_pos)
    prop = propensity_by_position(pairs, max_pos, min_shown)
    out = {
        "max_pos": int(max_pos),
        "observed_ctr": obs.to_dicts(),
        "stratified": strat.to_dicts(),
        "propensity": prop.to_dicts(),
    }
    if not obs.is_empty():
        c = obs["ctr"].to_numpy()
        out["observed_decay"] = float(c[-1] / c[0]) if c[0] > 0 else None
    if not strat.is_empty():
        r = strat["ratio"].to_numpy()
        out["stratified_decay"] = float(r[-1] / r[0]) if r[0] > 0 else None
        out["stratified_ratio_min"] = float(np.nanmin(r))
        out["stratified_ratio_max"] = float(np.nanmax(r))
        # The null hypothesis "the stored list order is not the order the user
        # saw" predicts ratio == 1 at every position. Deviation from 1, in units
        # of the binomial standard error, is what decides it -- 2.2M rows make
        # tiny effects significant, so the *size* is reported next to the test.
        n, k = strat["n_shown"].to_numpy(), strat["n_clicked"].to_numpy()
        base = k.sum() / max(n.sum(), 1)
        se = np.sqrt(np.maximum(base * (1 - base) / np.maximum(n, 1), 1e-18)) / max(base, 1e-12)
        out["stratified_max_z"] = float(np.nanmax(np.abs(r - 1.0) / np.maximum(se, 1e-12)))
    if not prop.is_empty():
        pr = prop["propensity"].to_numpy()
        out["propensity_decay"] = float(pr[-1])
        out["propensity_min"] = float(np.nanmin(pr))
        out["n_articles_used"] = int(prop["n_articles"].max())
    return out
