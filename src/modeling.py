"""
Training the model and scoring it honestly.

Two models are offered, both deliberately small and fast:

  * Random Forest: hundreds of decision trees, each one looking at a random
    part of the fingerprint, voting together. Good at "this combination of
    chemical pieces tends to work".
  * Logistic Regression: one straight-line rule over all the bits. Simpler,
    and a useful sanity check: if the forest cannot beat it, the extra
    complexity is not buying anything.

Both are told class_weight="balanced", because actives are rare. Without that,
a model can score 98% accuracy by answering "inactive" to everything, which is
useless and is exactly why this app plays down plain accuracy.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .features import fingerprint_array

MODEL_RANDOM_FOREST = "Random Forest"
MODEL_LOGISTIC = "Logistic Regression"
MODEL_CHOICES = (MODEL_RANDOM_FOREST, MODEL_LOGISTIC)

DEFAULT_TREES = 300
CHUNK_TREES = 25  # how many trees we add between progress-bar updates

# What the memory safeguard falls back to.
SAFEGUARD_BITS = 1024
SAFEGUARD_TREES = 150
SAFEGUARD_NOTICE = "Large dataset: used 1024 bits and 150 trees to fit in memory."


def train_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = DEFAULT_TREES,
    seed: int = 42,
    progress_cb=None,
) -> RandomForestClassifier:
    """
    Grow a random forest a few trees at a time so we can show progress.

    warm_start=True lets us keep the trees we already grew and add more, which
    is how the progress bar can move during what is otherwise one long call.
    """
    model = RandomForestClassifier(
        n_estimators=0,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
        warm_start=True,
    )
    grown = 0
    while grown < n_estimators:
        grown = min(grown + CHUNK_TREES, n_estimators)
        model.n_estimators = grown
        model.fit(x_train, y_train)
        if progress_cb is not None:
            progress_cb(grown, n_estimators)
    return model


def train_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    progress_cb=None,
) -> LogisticRegression:
    """Fit the simple straight-line model."""
    model = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    if progress_cb is not None:
        progress_cb(1, 1)
    return model


def fit_with_safeguard(
    train_smiles,
    y_train,
    model_kind: str = MODEL_RANDOM_FOREST,
    radius: int = 2,
    bits: int = 2048,
    n_estimators: int = DEFAULT_TREES,
    seed: int = 42,
    feature_progress_cb=None,
    train_progress_cb=None,
) -> dict:
    """
    Build the fingerprints and train the model, surviving a memory squeeze.

    If the machine runs out of memory we try exactly once more with smaller
    fingerprints (1024 bits) and fewer trees (150), and report a notice so the
    user knows the settings changed under them. No molecules are ever silently
    dropped: a big dataset is still trained on in full.

    Returns a dictionary with the model, the settings actually used, the
    training fingerprints, and how long it took.
    """
    y_train = np.asarray(y_train, dtype=int)

    def build_and_fit(use_bits: int, use_trees: int):
        x_train = fingerprint_array(
            train_smiles, radius=radius, bits=use_bits, progress_cb=feature_progress_cb
        )
        if model_kind == MODEL_LOGISTIC:
            model = train_logistic(x_train, y_train, seed=seed, progress_cb=train_progress_cb)
        else:
            model = train_random_forest(
                x_train, y_train, n_estimators=use_trees, seed=seed,
                progress_cb=train_progress_cb,
            )
        return model, x_train

    started = time.perf_counter()
    notice = None
    used_bits = int(bits)
    used_trees = int(n_estimators)
    try:
        model, x_train = build_and_fit(used_bits, used_trees)
    except MemoryError:
        # One retry, smaller. If this fails too, the app shows a friendly error.
        used_bits = SAFEGUARD_BITS
        used_trees = SAFEGUARD_TREES
        notice = SAFEGUARD_NOTICE
        model, x_train = build_and_fit(used_bits, used_trees)
    elapsed = time.perf_counter() - started

    return {
        "model": model,
        "x_train": x_train,
        "radius": int(radius),
        "bits": used_bits,
        "n_estimators": used_trees if model_kind == MODEL_RANDOM_FOREST else None,
        "model_kind": model_kind,
        "seed": int(seed),
        "seconds": elapsed,
        "safeguard_notice": notice,
    }


def predict_scores(model, x) -> np.ndarray:
    """
    The model's probability that each molecule is active.

    Both model types expose predict_proba; column 1 is the "active" class.
    """
    probabilities = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    # A model trained on a single class cannot really score anything; return
    # zeros rather than crash, and the app explains the situation.
    return np.zeros(x.shape[0], dtype=float)


def evaluate(y_true, scores, threshold: float = 0.5) -> dict:
    """
    Score the model on the test set only.

    Returns ROC-AUC, PR-AUC (average precision) with its "no skill" baseline,
    and the confusion-matrix numbers at the chosen probability cut-off.

    When the test set happens to contain only one class, ROC-AUC and PR-AUC are
    mathematically undefined. We return None for them and set single_class, so
    the app can say so plainly instead of printing a meaningless figure.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n = int(y_true.shape[0])
    n_active = int(y_true.sum())
    single_class = n_active == 0 or n_active == n

    result = {
        "n_test": n,
        "n_active": n_active,
        "n_inactive": n - n_active,
        "single_class": single_class,
        "threshold": float(threshold),
        "baseline_pr_auc": (n_active / n) if n else 0.0,
    }

    if single_class:
        result.update(
            {
                "roc_auc": None,
                "pr_auc": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "accuracy": None,
                "confusion": None,
                "roc_curve": None,
            }
        )
        return result

    predicted = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_true, scores)

    result.update(
        {
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "pr_auc": float(average_precision_score(y_true, scores)),
            "precision": float(precision_score(y_true, predicted, zero_division=0)),
            "recall": float(recall_score(y_true, predicted, zero_division=0)),
            "f1": float(f1_score(y_true, predicted, zero_division=0)),
            "accuracy": float((predicted == y_true).mean()),
            "confusion": {
                "true_negative": int(matrix[0, 0]),
                "false_positive": int(matrix[0, 1]),
                "false_negative": int(matrix[1, 0]),
                "true_positive": int(matrix[1, 1]),
            },
            "roc_curve": (fpr, tpr),
        }
    )
    return result


def score_band(score: float, reference_scores) -> dict:
    """
    Put one prediction in context: higher than what share of the test molecules?

    A raw probability such as 0.42 means little on its own. Comparing it with
    the scores the model gave the molecules it was tested on is far more
    informative, and it is computed from those real scores every time.
    """
    reference = np.asarray(reference_scores, dtype=float)
    if reference.size == 0:
        return {"percentile": None, "text": "No test scores are available for comparison yet."}
    percentile = float((reference < float(score)).mean() * 100.0)
    if percentile >= 95:
        word = "This is one of the highest scores the model gave."
    elif percentile >= 75:
        word = "This is a relatively high score for this model."
    elif percentile >= 25:
        word = "This is a middling score for this model."
    else:
        word = "This is a relatively low score for this model."
    return {
        "percentile": percentile,
        "text": f"Higher than {percentile:.0f}% of the test molecules. {word}",
    }


__all__ = [
    "CHUNK_TREES",
    "DEFAULT_TREES",
    "MODEL_CHOICES",
    "MODEL_LOGISTIC",
    "MODEL_RANDOM_FOREST",
    "SAFEGUARD_BITS",
    "SAFEGUARD_NOTICE",
    "SAFEGUARD_TREES",
    "evaluate",
    "fit_with_safeguard",
    "predict_scores",
    "score_band",
    "train_logistic",
    "train_random_forest",
]
