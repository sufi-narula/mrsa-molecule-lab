"""
The built-in demo dataset.

These 79 molecules were hand-labelled from general knowledge: well known
antibiotics were given small "MIC" numbers and everyday chemicals were given
large ones. They are NOT real laboratory measurements. They exist so that the
app can be clicked through end to end without downloading anything, and so the
tests have something fixed to run against.

Any result produced from them is a demonstration of the machinery, not a
scientific finding, and the app says so in a banner wherever they are used.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# data/demo/ sits next to the src/ folder, one level up from this file.
DEMO_DIR = Path(__file__).resolve().parent.parent / "data" / "demo"
TRAINING_PATH = DEMO_DIR / "demo_training.csv"
SCREENING_PATH = DEMO_DIR / "demo_screening.csv"

# The column choices and threshold the app selects automatically for the demo.
DEMO_SMILES_COLUMN = "smiles"
DEMO_ACTIVITY_COLUMN = "measurement_demo_mic_ug_per_ml"
DEMO_NAME_COLUMN = "name"
DEMO_DIRECTION = "at or below"
DEMO_THRESHOLD = 50.0

DEMO_BANNER = (
    "DEMO DATA: hand-labeled from general knowledge, not real lab measurements. "
    "Do not use these results in an article."
)


def load_demo_training() -> pd.DataFrame:
    """The demo training table: a name, a SMILES string and a fake MIC value."""
    if not TRAINING_PATH.exists():
        raise FileNotFoundError(
            f"The demo training file is missing. I expected it at {TRAINING_PATH}."
        )
    return pd.read_csv(TRAINING_PATH)


def load_demo_screening() -> pd.DataFrame:
    """
    The demo screening list: molecules with no labels, to be scored.

    It deliberately contains two broken rows and one duplicate so that the
    Batch Screen tab's cleaning and funnel can be seen working.
    """
    if not SCREENING_PATH.exists():
        raise FileNotFoundError(
            f"The demo screening file is missing. I expected it at {SCREENING_PATH}."
        )
    return pd.read_csv(SCREENING_PATH)


def demo_screening_bytes() -> bytes:
    """The demo screening file as raw bytes, as if it had just been uploaded."""
    return SCREENING_PATH.read_bytes()


__all__ = [
    "DEMO_ACTIVITY_COLUMN",
    "DEMO_BANNER",
    "DEMO_DIRECTION",
    "DEMO_NAME_COLUMN",
    "DEMO_SMILES_COLUMN",
    "DEMO_THRESHOLD",
    "SCREENING_PATH",
    "TRAINING_PATH",
    "demo_screening_bytes",
    "load_demo_screening",
    "load_demo_training",
]
