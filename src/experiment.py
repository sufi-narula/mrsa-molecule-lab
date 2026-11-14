"""
The heart of the app: would the model have been a better shopping list?

The question the 2023 MIT and Broad paper really answers is not "what is your
ROC-AUC". It is: if a lab can only afford to buy and test a handful of
molecules, does ranking them with a model find more real antibiotics than
picking at random?

So we take the test molecules the model has never seen, rank them by the
model's score, pretend to buy the top N, and count how many of those really
were active. Then we compare that hit rate against the hit rate of picking N
molecules blindly.

Every number here is computed from the data that is passed in. Nothing is
assumed, stored or hardcoded.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import hypergeom


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple:
    """
    A 95% confidence interval for a hit rate, the Wilson way.

    Why not the simple formula: if you test 10 molecules and 10 are active, the
    simple formula says "100%, give or take 0%", which is obviously nonsense
    from 10 molecules. The Wilson interval stays sensible with small counts and
    never runs below 0 or above 1.

    Returns (low, high) as fractions. With 0 trials it returns (0.0, 0.0).
    """
    successes = int(successes)
    trials = int(trials)
    if trials <= 0:
        return 0.0, 0.0
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = (p + z2 / (2.0 * trials)) / denominator
    half_width = (
        z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))
    ) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def rank_order(scores) -> np.ndarray:
    """
    The row positions sorted from the highest score to the lowest.

    Ties keep their original order, which makes every number in the app
    reproducible: numpy's "stable" sort does not shuffle equal values.
    """
    scores = np.asarray(scores, dtype=float)
    # Sort ascending on the negated scores: stable, so ties keep input order.
    return np.argsort(-scores, kind="stable")


def hypergeometric_p_value(hits: int, population: int, successes: int, draws: int) -> float:
    """
    The chance of doing this well by pure luck.

    Imagine a bag holding `population` molecules of which `successes` are
    active. You pull out `draws` of them blindly. This is the probability of
    getting at least `hits` actives that way. A small number (say under 0.05)
    means luck alone rarely does this well.

    This is the exact test (scipy's hypergeom.sf), not an approximation.
    """
    hits = int(hits)
    population = int(population)
    successes = int(successes)
    draws = int(draws)
    if draws <= 0 or population <= 0 or successes <= 0:
        return 1.0
    if hits <= 0:
        return 1.0
    return float(hypergeom.sf(hits - 1, population, successes, draws))


def model_vs_random(scores, y_true, n_pick: int) -> dict:
    """
    Compare "buy the model's top N" against "buy N at random".

    scores are the model's predicted probabilities for the test molecules and
    y_true are their real labels (1 active, 0 inactive). n_pick is how many
    molecules we pretend to buy.

    Everything returned is derived from those three inputs:
      random_hit_rate   actives in the whole test set, divided by its size
      model_hit_rate    actives among the top N, divided by N
      model_ci          95% Wilson interval around the model hit rate
      enrichment        how many times better the model is (None if the random
                        rate is 0, which the app shows as "n/a")
      actives_found     actives in the top N
      expected_random   how many actives random picking would find on average
      p_value           the chance of finding this many actives by luck
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if scores.shape[0] != y_true.shape[0]:
        raise ValueError("scores and y_true must have the same length.")

    n_test = int(y_true.shape[0])
    if n_test == 0:
        raise ValueError("There are no test molecules to compare.")

    n_pick = int(max(1, min(int(n_pick), n_test)))
    actives_total = int(y_true.sum())

    order = rank_order(scores)
    top = order[:n_pick]
    actives_found = int(y_true[top].sum())

    random_hit_rate = actives_total / n_test
    model_hit_rate = actives_found / n_pick
    ci_low, ci_high = wilson_interval(actives_found, n_pick)

    enrichment = (model_hit_rate / random_hit_rate) if random_hit_rate > 0 else None

    return {
        "n_test": n_test,
        "n_pick": n_pick,
        "actives_total": actives_total,
        "actives_found": actives_found,
        "random_hit_rate": random_hit_rate,
        "model_hit_rate": model_hit_rate,
        "model_ci_low": ci_low,
        "model_ci_high": ci_high,
        "enrichment": enrichment,
        "expected_random": n_pick * random_hit_rate,
        "p_value": hypergeometric_p_value(actives_found, n_test, actives_total, n_pick),
        "ranked_order": order,
    }


def cumulative_gain(scores, y_true) -> tuple:
    """
    Data for the cumulative gain curve.

    Walk down the ranked list one molecule at a time and ask: having tested
    this share of the molecules, what share of all the actives have I found?
    A model that ranks well climbs steeply at the start. Random picking just
    follows the diagonal.

    Returns (percent_tested, percent_actives_found), both starting at 0.
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    n = y_true.shape[0]
    actives_total = int(y_true.sum())
    if n == 0 or actives_total == 0:
        return np.zeros(1), np.zeros(1)

    order = rank_order(scores)
    found = np.cumsum(y_true[order])
    x = np.arange(1, n + 1) / n * 100.0
    y = found / actives_total * 100.0
    # Start the curve at the origin so it reads correctly.
    return np.concatenate([[0.0], x]), np.concatenate([[0.0], y])


def enrichment_text(result: dict) -> str:
    """
    One plain sentence describing the comparison, built from the real numbers.

    Used as the caption under the hit-rate chart and inside the copy-ready
    summary, so the wording and the figures can never drift apart.
    """
    model_pct = result["model_hit_rate"] * 100.0
    random_pct = result["random_hit_rate"] * 100.0
    base = (
        f"In the top {result['n_pick']} picks, {result['actives_found']} "
        f"{'was' if result['actives_found'] == 1 else 'were'} truly active "
        f"({model_pct:.0f}%), compared with {random_pct:.0f}% if we picked at random"
    )
    if result["enrichment"] is None:
        return base + " (there are no actives in this test set, so there is nothing to compare)."
    return base + f", about {result['enrichment']:.1f} times better."


def verdict_text(result: dict) -> str:
    """
    An honest one-line verdict, including when the model is barely any good.

    The app shows this instead of quietly letting a weak result look impressive.
    """
    enrichment = result["enrichment"]
    if enrichment is None:
        return (
            "This test set contains no active molecules, so a hit rate cannot be "
            "compared. Try a different threshold or a larger dataset."
        )
    if result["actives_found"] == 0:
        return (
            "The model found no actives in its top picks, so on this data it is no "
            "better than guessing."
        )
    if enrichment < 1.0:
        return (
            "The model did worse than random picking on this data. That is a real "
            "result, not a bug: this dataset may be too small or too hard for it."
        )
    if enrichment < 1.5:
        return (
            "The model is only slightly better than random picking here, so it would "
            "not save a lab much money."
        )
    if result["p_value"] > 0.05:
        return (
            f"The model looks about {enrichment:.1f} times better than random, but with "
            f"a p-value of {result['p_value']:.2f} this could still be luck. A bigger "
            "test set would tell you more."
        )
    return (
        f"The model's picks were about {enrichment:.1f} times richer in real actives "
        f"than random picking, and luck alone would do this well only about "
        f"{result['p_value'] * 100:.1f}% of the time."
    )


__all__ = [
    "cumulative_gain",
    "enrichment_text",
    "hypergeometric_p_value",
    "model_vs_random",
    "rank_order",
    "verdict_text",
    "wilson_interval",
]
