"""
Charts and molecule drawings.

Every chart here is built from numbers that were passed in, has a title and
axis labels, and comes with a caption written from those same numbers so the
words can never disagree with the picture.

Molecules are drawn as SVG. That keeps them sharp at any size and avoids
depending on RDKit having been built with bitmap image support.
"""

from __future__ import annotations

import matplotlib

# Agg is the non-interactive backend: it draws into memory, which is what a web
# app needs. This must be set before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .features import mol_from_smiles  # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402

TEAL = "#0F766E"
SLATE = "#64748B"
AMBER = "#B45309"
INK = "#0F172A"

FIG_SIZE = (6.0, 3.8)


def _style(ax) -> None:
    """One calm, consistent look for every chart."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors=INK, labelsize=9)
    ax.title.set_color(INK)
    ax.title.set_fontsize(11)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)


def title_with_demo(title: str, is_demo: bool) -> str:
    """Add "(Demo data)" to a chart title when demo data is loaded."""
    return f"{title} (Demo data)" if is_demo else title


# ---------------------------------------------------------------------------
# Molecule drawings
# ---------------------------------------------------------------------------

def mol_svg(smiles_or_mol, size: int = 230, caption: str = "") -> "str | None":
    """
    Draw one molecule as SVG text, or return None if it cannot be drawn.

    caption, if given, is written under the structure by RDKit itself.
    """
    mol = (
        smiles_or_mol
        if hasattr(smiles_or_mol, "GetNumAtoms")
        else mol_from_smiles(smiles_or_mol)
    )
    if mol is None:
        return None
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
        drawer.drawOptions().addStereoAnnotation = False
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, legend=str(caption))
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data tab charts
# ---------------------------------------------------------------------------

def label_bar_chart(n_active: int, n_inactive: int, is_demo: bool = False) -> tuple:
    """How many molecules ended up on each side of the threshold."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.bar(
        ["Active", "Inactive"],
        [n_active, n_inactive],
        color=[TEAL, SLATE],
        width=0.55,
    )
    for i, value in enumerate([n_active, n_inactive]):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_title(title_with_demo("How your molecules are labelled", is_demo))
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of molecules")
    total = n_active + n_inactive
    ax.set_ylim(0, max(1, max(n_active, n_inactive) * 1.18))
    _style(ax)
    fig.tight_layout()
    percent = (100.0 * n_active / total) if total else 0.0
    caption = (
        f"{n_active:,} of {total:,} molecules ({percent:.1f}%) count as active with your "
        f"current setting, and {n_inactive:,} count as inactive."
    )
    return fig, caption


def measurement_histogram(values, threshold: float, direction: str, is_demo: bool = False) -> tuple:
    """
    The spread of the lab measurements, with the threshold drawn on top.

    A log x-axis is used when the numbers span more than a hundredfold, which
    is normal for potency measurements such as MIC.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    if values.size == 0:
        ax.text(0.5, 0.5, "No numbers to show", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return fig, "This column holds no numbers I can plot."

    positive = values[values > 0]
    use_log = (
        positive.size > 0
        and positive.max() / positive.min() > 100
        and positive.size == values.size
    )
    if use_log:
        bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 30)
        ax.set_xscale("log")
    else:
        bins = 30

    ax.hist(values, bins=bins, color=TEAL, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(threshold, color=AMBER, linestyle="--", linewidth=2)
    ax.annotate(
        f"threshold = {threshold:g}",
        xy=(threshold, ax.get_ylim()[1] * 0.92),
        xytext=(6, 0),
        textcoords="offset points",
        color=AMBER,
        fontsize=9,
    )
    ax.set_title(title_with_demo("The spread of your measurements", is_demo))
    ax.set_xlabel("Measurement value" + (" (log scale)" if use_log else ""))
    ax.set_ylabel("Number of molecules")
    _style(ax)
    fig.tight_layout()

    n_active = int((values <= threshold).sum() if direction == "at or below" else (values >= threshold).sum())
    caption = (
        f"Measurements run from {values.min():g} to {values.max():g}. "
        f"With “active = {direction} {threshold:g}”, {n_active:,} of {values.size:,} "
        f"molecules count as active."
        + (" The x-axis uses a log scale because the values span a very wide range." if use_log else "")
    )
    return fig, caption


# ---------------------------------------------------------------------------
# Results tab charts
# ---------------------------------------------------------------------------

def hit_rate_chart(result: dict, is_demo: bool = False) -> tuple:
    """
    Random picking against the model's top N, with an error bar on the model.

    The error bar is the 95% Wilson interval: with only a handful of picks the
    hit rate is a shaky number, and the bar shows how shaky.
    """
    random_pct = result["random_hit_rate"] * 100.0
    model_pct = result["model_hit_rate"] * 100.0
    low = max(0.0, model_pct - result["model_ci_low"] * 100.0)
    high = max(0.0, result["model_ci_high"] * 100.0 - model_pct)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.bar(["Random picking"], [random_pct], color=SLATE, width=0.5)
    ax.bar(
        [f"Model’s top {result['n_pick']}"],
        [model_pct],
        color=TEAL,
        width=0.5,
        yerr=[[low], [high]],
        capsize=8,
        ecolor=INK,
    )
    for i, value in enumerate([random_pct, model_pct]):
        ax.text(i, value, f"  {value:.1f}%", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_title(title_with_demo("Hit rate: random picking vs the model", is_demo))
    ax.set_xlabel("How the molecules were chosen")
    ax.set_ylabel("Share that were truly active (%)")
    ax.set_ylim(0, max(1.0, max(random_pct, result["model_ci_high"] * 100.0) * 1.25))
    _style(ax)
    fig.tight_layout()

    from .experiment import enrichment_text

    return fig, enrichment_text(result)


def gain_curve_chart(x_percent, y_percent, is_demo: bool = False) -> tuple:
    """
    The cumulative gain curve: how fast you find the actives working down the
    model's ranking, against the straight line you would get picking blindly.
    """
    x_percent = np.asarray(x_percent, dtype=float)
    y_percent = np.asarray(y_percent, dtype=float)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot([0, 100], [0, 100], color=SLATE, linestyle="--", linewidth=1.6,
            label="Random picking")
    ax.plot(x_percent, y_percent, color=TEAL, linewidth=2.2, label="Model’s ranking")
    ax.set_title(title_with_demo("How quickly the actives are found", is_demo))
    ax.set_xlabel("Share of the test molecules tested (%)")
    ax.set_ylabel("Share of all test actives found (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()

    # How much of the actives you have after testing the first 10%.
    if x_percent.size > 1:
        found_at_10 = float(np.interp(10.0, x_percent, y_percent))
        caption = (
            f"Testing only the model’s top 10% of molecules would have found "
            f"{found_at_10:.0f}% of all the actives in the test set; random picking "
            "would find about 10%."
        )
    else:
        caption = "There are not enough test molecules to draw a useful curve."
    return fig, caption


def roc_chart(fpr, tpr, roc_auc: float, is_demo: bool = False) -> tuple:
    """The ROC curve, with the no-skill diagonal for comparison."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot([0, 1], [0, 1], color=SLATE, linestyle="--", linewidth=1.6, label="No skill")
    ax.plot(fpr, tpr, color=TEAL, linewidth=2.2, label=f"Model (AUC = {roc_auc:.2f})")
    ax.set_title(title_with_demo("ROC curve on the unseen test molecules", is_demo))
    ax.set_xlabel("False positive rate (inactives wrongly flagged)")
    ax.set_ylabel("True positive rate (actives correctly found)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    fig.tight_layout()
    caption = (
        f"ROC-AUC is {roc_auc:.3f}. Read it as: pick one active and one inactive "
        f"molecule at random, and the model gives the active one the higher score "
        f"about {roc_auc * 100:.0f}% of the time. 0.5 would be pure guessing."
    )
    return fig, caption


# ---------------------------------------------------------------------------
# Batch Screen tab chart
# ---------------------------------------------------------------------------

def funnel_chart(stages, is_demo: bool = False) -> tuple:
    """
    The screening funnel as horizontal bars.

    stages is a list of (label, count) pairs from widest to narrowest. This
    mirrors the real study, which went from millions of scored compounds down
    to a couple of hundred actually bought and tested.
    """
    labels = [s[0] for s in stages]
    counts = [int(s[1]) for s in stages]

    fig, ax = plt.subplots(figsize=(6.0, 0.62 * len(stages) + 1.6))
    positions = np.arange(len(stages))[::-1]
    ax.barh(positions, counts, color=TEAL, height=0.6)
    for pos, value in zip(positions, counts):
        ax.text(value, pos, f"  {value:,}", va="center", fontsize=10, color=INK)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title_with_demo("Your screening funnel", is_demo))
    ax.set_xlabel("Number of molecules")
    ax.set_ylabel("")
    ax.set_xlim(0, max(1, max(counts) * 1.22))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors=INK, labelsize=9)
    ax.title.set_color(INK)
    ax.title.set_fontsize(11)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()

    first, last = counts[0], counts[-1]
    kept = (100.0 * last / first) if first else 0.0
    caption = (
        f"Out of {first:,} uploaded rows, {last:,} molecules ({kept:.1f}%) make it "
        "through every filter to the shortlist."
    )
    return fig, caption


def close(fig) -> None:
    """
    Release a figure's memory.

    Streamlit re-runs the script on every interaction. Without this the open
    figures pile up until matplotlib starts warning about it.
    """
    try:
        plt.close(fig)
    except Exception:
        pass


__all__ = [
    "AMBER",
    "FIG_SIZE",
    "INK",
    "SLATE",
    "TEAL",
    "close",
    "funnel_chart",
    "gain_curve_chart",
    "hit_rate_chart",
    "label_bar_chart",
    "measurement_histogram",
    "mol_svg",
    "roc_chart",
    "title_with_demo",
]
