"""
Getting a messy file into a clean, labelled table of molecules.

Four jobs live here:
  1. read a file (csv, tsv, txt, xlsx) into a pandas table;
  2. work out which column holds the molecules and which holds the lab
     measurement;
  3. turn that measurement into a yes/no label ("active" or "inactive");
  4. clean the molecules: remove salts, drop unreadable ones, and drop
     duplicates.
"""

from __future__ import annotations

import io
import re
from typing import Sequence

import numpy as np
import pandas as pd

from .features import canonical_smiles, largest_fragment, mol_from_smiles

# Headers that strongly suggest a column of molecules.
SMILES_HEADER_HINTS = ("smiles", "canonical_smiles", "smile", "structure")

# Headers that strongly suggest a column of lab measurements.
ACTIVITY_HEADER_HINTS = (
    "activity",
    "label",
    "active",
    "class",
    "mic",
    "ic50",
    "inhibition",
    "growth",
    "measurement",
    "response",
    "potency",
)

# Text values we accept as a yes/no label, and what they mean.
TRUE_WORDS = {"1", "1.0", "true", "yes", "y", "active", "t", "pos", "positive"}
FALSE_WORDS = {"0", "0.0", "false", "no", "n", "inactive", "f", "neg", "negative"}

# How many actives and inactives we insist on before training is allowed.
MIN_PER_CLASS = 5


# ---------------------------------------------------------------------------
# 1. Reading files
# ---------------------------------------------------------------------------

def excel_sheet_names(file_like) -> list:
    """The sheet names inside an .xlsx file, so the user can pick one."""
    data = _as_bytes(file_like)
    with pd.ExcelFile(io.BytesIO(data), engine="openpyxl") as book:
        return list(book.sheet_names)


def _as_bytes(file_like) -> bytes:
    """Read an uploaded file or a path into raw bytes."""
    if isinstance(file_like, (bytes, bytearray)):
        return bytes(file_like)
    if isinstance(file_like, str):
        with open(file_like, "rb") as handle:
            return handle.read()
    # A Streamlit UploadedFile, or any file object.
    try:
        file_like.seek(0)
    except Exception:
        pass
    return file_like.read()


def detect_separator(data: bytes) -> str:
    """
    Work out whether a text file is separated by commas, tabs or semicolons.

    We look at the header line and count each candidate, then pick the most
    common one. If the header contains none of them the file has a single
    column, and we return a comma so pandas reads it as exactly that.

    Why not let pandas sniff it: pandas' sniffer raises or guesses wildly on a
    one-column file, which used to turn a tidy list of molecules into three
    columns called "Unnamed: 0", "Unnamed: 1" and "Unnamed: 2".
    """
    try:
        text = data[:65536].decode("utf-8", errors="replace")
    except Exception:
        return ","
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return ","
    header = lines[0]
    counts = {
        ",": header.count(","),
        "\t": header.count("\t"),
        ";": header.count(";"),
        "|": header.count("|"),
    }
    best = max(counts, key=lambda key: counts[key])
    return best if counts[best] > 0 else ","


def read_table(file_like, filename: str, sheet_name=None) -> pd.DataFrame:
    """
    Read a data file into a pandas table.

    Supports .csv, .tsv, .txt (tab or comma separated, detected automatically)
    and .xlsx. Raises ValueError with a friendly message if the file cannot be
    read or turns out to be empty, so the app can show that message directly.
    """
    name = (filename or "").lower()
    data = _as_bytes(file_like)
    if not data.strip():
        raise ValueError("That file is empty. Please choose a file with some rows in it.")

    if name.endswith((".xlsx", ".xlsm", ".xltx")):
        try:
            frame = pd.read_excel(
                io.BytesIO(data),
                sheet_name=sheet_name if sheet_name is not None else 0,
                engine="openpyxl",
            )
        except Exception as exc:
            raise ValueError(f"I could not read that Excel file: {exc}") from exc
    else:
        separator = detect_separator(data)
        try:
            frame = pd.read_csv(
                io.BytesIO(data),
                sep=separator,
                skip_blank_lines=True,
            )
        except Exception as exc:
            raise ValueError(f"I could not read that file as a table: {exc}") from exc

    if isinstance(frame, dict):  # read_excel with sheet_name=None
        frame = next(iter(frame.values()))
    if frame is None or frame.shape[0] == 0:
        raise ValueError("That file has a header but no rows of data.")
    if frame.shape[1] == 0:
        raise ValueError("That file has no columns I can read.")

    # Tidy the column names so later matching is predictable.
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


# ---------------------------------------------------------------------------
# 2. Finding the right columns
# ---------------------------------------------------------------------------

def _header_rank(column: str, hints: Sequence[str]) -> int:
    """
    How strongly a column name matches a list of hint words.

    2 means the name is exactly a hint, 1 means it contains one, 0 means no
    match. Bigger is better.
    """
    low = str(column).strip().lower()
    if low in hints:
        return 2
    if any(h in low for h in hints):
        return 1
    return 0


def smiles_valid_fraction(values: Sequence, sample: int = 200) -> tuple:
    """
    What share of the first `sample` non-empty values are real molecules?

    Returns (fraction, number_checked). This is the test we use to find the
    SMILES column: a column of names such as "aspirin" scores near 0, while a
    column of real SMILES scores near 1.
    """
    checked = 0
    valid = 0
    for value in values:
        if checked >= sample:
            break
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() in ("nan", "none"):
            continue
        checked += 1
        if mol_from_smiles(text) is not None:
            valid += 1
    if checked == 0:
        return 0.0, 0
    return valid / checked, checked


def detect_smiles_column(
    frame: pd.DataFrame, sample: int = 200, threshold: float = 0.8
) -> tuple:
    """
    Guess which column holds the molecules.

    A column qualifies when at least `threshold` (80% by default) of its first
    `sample` non-empty values parse as molecules in RDKit. Among the columns
    that qualify we prefer the ones whose header looks like a SMILES header,
    then the one with the highest valid fraction.

    Returns (best_column_or_None, scores) where scores maps every column to
    (fraction, number_checked). A column of chemical *names* can never win,
    because names do not parse as molecules.
    """
    scores = {}
    for column in frame.columns:
        scores[column] = smiles_valid_fraction(frame[column].tolist(), sample=sample)

    qualifying = [c for c, (frac, n) in scores.items() if n > 0 and frac >= threshold]
    if not qualifying:
        return None, scores

    def sort_key(column):
        frac, _n = scores[column]
        return (_header_rank(column, SMILES_HEADER_HINTS), frac)

    best = sorted(qualifying, key=sort_key, reverse=True)[0]
    return best, scores


def numeric_or_binary_fraction(values: Sequence) -> tuple:
    """
    What share of the non-empty values are numbers or yes/no words?

    Returns (fraction, number_checked). Used to decide whether a column could
    be the lab measurement.
    """
    checked = 0
    usable = 0
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        text = str(value).strip()
        if not text or text.lower() in ("nan", "none"):
            continue
        checked += 1
        low = text.lower()
        if low in TRUE_WORDS or low in FALSE_WORDS:
            usable += 1
            continue
        try:
            float(text)
            usable += 1
        except ValueError:
            pass
    if checked == 0:
        return 0.0, 0
    return usable / checked, checked


def detect_activity_column(
    frame: pd.DataFrame, smiles_column, threshold: float = 0.9
) -> tuple:
    """
    Guess which column holds the lab measurement.

    Candidates are every column except the SMILES column in which at least 90%
    of the non-empty values are numbers or yes/no words. We preselect the first
    candidate whose header looks like a measurement header, otherwise simply the
    first candidate.

    Returns (best_column_or_None, candidates, scores).
    """
    scores = {}
    candidates = []
    for column in frame.columns:
        if column == smiles_column:
            continue
        frac, n = numeric_or_binary_fraction(frame[column].tolist())
        scores[column] = (frac, n)
        if n > 0 and frac >= threshold:
            candidates.append(column)

    if not candidates:
        return None, [], scores

    hinted = [c for c in candidates if _header_rank(c, ACTIVITY_HEADER_HINTS) > 0]
    if hinted:
        hinted.sort(key=lambda c: _header_rank(c, ACTIVITY_HEADER_HINTS), reverse=True)
        return hinted[0], candidates, scores
    return candidates[0], candidates, scores


# ---------------------------------------------------------------------------
# 3. Labelling
# ---------------------------------------------------------------------------

def looks_binary(values: Sequence) -> tuple:
    """
    Is this column already a yes/no answer?

    Returns (is_binary, mapping) where mapping shows the user how each value
    was read, for example {"active": 1, "inactive": 0}. A column counts as
    binary when every non-empty value is one of the words we recognise and both
    answers actually appear.
    """
    seen = {}
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        text = str(value).strip()
        if not text or text.lower() in ("nan", "none"):
            continue
        low = text.lower()
        if low in TRUE_WORDS:
            seen[text] = 1
        elif low in FALSE_WORDS:
            seen[text] = 0
        else:
            return False, {}
    if not seen:
        return False, {}
    if len(set(seen.values())) < 2:
        # Only actives or only inactives: technically binary, but useless for
        # training. Treat it as binary so the app can explain the problem.
        return True, seen
    return True, seen


def label_binary(values: Sequence) -> tuple:
    """
    Read a yes/no column into labels of 1 (active) and 0 (inactive).

    Returns (labels, mapping, excluded) where labels is a numpy float array
    with NaN for rows we could not read, and excluded counts those rows with
    the reason.
    """
    labels = np.full(len(values), np.nan, dtype=float)
    mapping = {}
    blank = 0
    unreadable = 0
    for i, value in enumerate(values):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            blank += 1
            continue
        text = str(value).strip()
        if not text or text.lower() in ("nan", "none"):
            blank += 1
            continue
        low = text.lower()
        if low in TRUE_WORDS:
            labels[i] = 1.0
            mapping[text] = 1
        elif low in FALSE_WORDS:
            labels[i] = 0.0
            mapping[text] = 0
        else:
            unreadable += 1
    excluded = {"blank": blank, "not a yes/no value": unreadable}
    return labels, mapping, excluded


def to_numeric(values: Sequence) -> np.ndarray:
    """Read a column as numbers, with NaN wherever that is not possible."""
    return pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)


def label_numeric(values: Sequence, direction: str, threshold: float) -> tuple:
    """
    Turn a measurement into labels using a cut-off.

    direction is "at or below" (small numbers mean active, as with MIC in
    micrograms per millilitre, where a low dose that still stops the bacteria
    means a strong antibiotic) or "at or above" (big numbers mean active, as
    with percent inhibition).

    Returns (labels, excluded) where labels is a numpy float array holding 1,
    0 or NaN, and excluded counts the rows we had to skip.
    """
    if direction not in ("at or below", "at or above"):
        raise ValueError(
            'direction must be "at or below" or "at or above", '
            f"got {direction!r}"
        )
    numbers = to_numeric(values)
    labels = np.full(numbers.shape, np.nan, dtype=float)
    usable = ~np.isnan(numbers)
    if direction == "at or below":
        labels[usable] = (numbers[usable] <= threshold).astype(float)
    else:
        labels[usable] = (numbers[usable] >= threshold).astype(float)
    excluded = {"blank or not a number": int((~usable).sum())}
    return labels, excluded


def threshold_for_percent(values: Sequence, direction: str, percent: float) -> float:
    """
    Find the cut-off that labels roughly `percent` of the molecules active.

    With direction "at or below" we take the low end of the measurements, so a
    percent of 5 returns the 5th percentile. With "at or above" we take the
    high end, so 5 returns the 95th percentile.
    """
    numbers = to_numeric(values)
    numbers = numbers[~np.isnan(numbers)]
    if numbers.size == 0:
        raise ValueError("That column has no numbers in it, so I cannot pick a cut-off.")
    percent = float(min(max(percent, 0.0), 100.0))
    if direction == "at or below":
        return float(np.percentile(numbers, percent))
    return float(np.percentile(numbers, 100.0 - percent))


def label_summary(labels: np.ndarray) -> dict:
    """Count actives and inactives, and work out the percentage active."""
    labels = np.asarray(labels, dtype=float)
    usable = labels[~np.isnan(labels)]
    n_active = int((usable == 1).sum())
    n_inactive = int((usable == 0).sum())
    total = n_active + n_inactive
    return {
        "n_active": n_active,
        "n_inactive": n_inactive,
        "n_labelled": total,
        "percent_active": (100.0 * n_active / total) if total else 0.0,
        "enough": n_active >= MIN_PER_CLASS and n_inactive >= MIN_PER_CLASS,
    }


def labels_problem_message(summary: dict) -> "str | None":
    """
    If the labels cannot be used for training, say exactly what to change.

    Returns None when everything is fine.
    """
    if summary["enough"]:
        return None
    active = summary["n_active"]
    inactive = summary["n_inactive"]
    if active == 0:
        return (
            "Your threshold labels 0 molecules as active. "
            "Try a higher threshold or the opposite direction."
        )
    if inactive == 0:
        return (
            "Your threshold labels every molecule as active, so there is nothing "
            "to compare against. Try a lower threshold or the opposite direction."
        )
    if active < MIN_PER_CLASS:
        return (
            f"Only {active} molecules are labelled active, and I need at least "
            f"{MIN_PER_CLASS}. Move the threshold so that more molecules count as "
            "active, or use a larger dataset."
        )
    return (
        f"Only {inactive} molecules are labelled inactive, and I need at least "
        f"{MIN_PER_CLASS}. Move the threshold so that fewer molecules count as "
        "active, or use a larger dataset."
    )


# ---------------------------------------------------------------------------
# 4. Cleaning the molecules
# ---------------------------------------------------------------------------

def clean_molecules(
    frame: pd.DataFrame,
    smiles_column: str,
    labels: np.ndarray,
    name_column=None,
    progress_cb=None,
) -> tuple:
    """
    Produce the final table the model will learn from, plus a cleaning report.

    Steps, in order:
      * drop rows whose label is missing (blank or unreadable measurement);
      * read each SMILES string with RDKit and drop the ones it cannot parse;
      * keep the largest fragment of each molecule, which removes salts;
      * write every molecule in canonical form;
      * drop duplicates by canonical SMILES. If the copies agree on the label
        we keep one of them; if they disagree we drop all of them, because we
        cannot tell which answer is right.

    Returns (clean_frame, report). The clean frame has the columns
    "smiles" (canonical), "label" (1 or 0), "name", and "source_row".
    """
    labels = np.asarray(labels, dtype=float)
    if len(labels) != len(frame):
        raise ValueError("The labels and the table have different lengths.")

    total_rows = int(len(frame))
    blank_label = 0
    blank_smiles = 0
    invalid = []  # (row number shown to the user, the text we could not read)
    records = []

    smiles_values = frame[smiles_column].tolist()
    name_values = (
        frame[name_column].tolist() if name_column in frame.columns else [None] * total_rows
    ) if name_column else [None] * total_rows

    for i in range(total_rows):
        if progress_cb is not None and (i % 1000 == 0 or i == total_rows - 1):
            progress_cb(i + 1, total_rows)

        label = labels[i]
        if np.isnan(label):
            blank_label += 1
            continue

        raw = smiles_values[i]
        text = "" if raw is None else str(raw).strip()
        if not text or text.lower() in ("nan", "none"):
            blank_smiles += 1
            continue

        mol = mol_from_smiles(text)
        if mol is None:
            invalid.append((i + 2, text))  # +2: 1-based, and skip the header row
            continue

        canonical = canonical_smiles(largest_fragment(mol))
        if canonical is None:
            invalid.append((i + 2, text))
            continue

        name = name_values[i]
        if name is None or (isinstance(name, float) and np.isnan(name)):
            name = ""
        records.append(
            {
                "smiles": canonical,
                "label": int(label),
                "name": str(name),
                "source_row": i + 2,
            }
        )

    parsed = pd.DataFrame.from_records(
        records, columns=["smiles", "label", "name", "source_row"]
    )

    duplicates_removed = 0
    conflicts_dropped = 0
    conflicting_examples = []

    if len(parsed) > 0:
        keep_rows = []
        for canonical, group in parsed.groupby("smiles", sort=False):
            if len(group) == 1:
                keep_rows.append(group.iloc[0])
                continue
            if group["label"].nunique() == 1:
                # The copies agree: keep the first one, count the rest as removed.
                keep_rows.append(group.iloc[0])
                duplicates_removed += len(group) - 1
            else:
                # The copies disagree: we cannot tell which label is right.
                conflicts_dropped += len(group)
                if len(conflicting_examples) < 10:
                    conflicting_examples.append(canonical)
        clean = (
            pd.DataFrame(keep_rows).reset_index(drop=True)
            if keep_rows
            else parsed.iloc[0:0].copy()
        )
    else:
        clean = parsed

    summary = label_summary(clean["label"].to_numpy(dtype=float)) if len(clean) else label_summary(
        np.zeros(0)
    )

    report = {
        "total_rows": total_rows,
        "blank_label": blank_label,
        "blank_smiles": blank_smiles,
        "invalid_count": len(invalid),
        "invalid_examples": invalid[:10],
        "duplicates_removed": duplicates_removed,
        "conflicts_dropped": conflicts_dropped,
        "conflicting_examples": conflicting_examples,
        "final_count": int(len(clean)),
        "n_active": summary["n_active"],
        "n_inactive": summary["n_inactive"],
        "percent_active": summary["percent_active"],
        "enough": summary["enough"],
    }
    return clean, report


def read_screening_table(file_like, filename: str) -> tuple:
    """
    Read a screening list (molecules with no labels) for the Batch Screen tab.

    Returns (frame, smiles_column, name_column). Raises ValueError with a
    friendly message when there is no usable column of molecules.
    """
    frame = read_table(file_like, filename)
    smiles_column, scores = detect_smiles_column(frame)
    if smiles_column is None:
        raise ValueError(
            "I could not find a column of molecules (SMILES) in that file. "
            "Check that one column holds SMILES text such as "
            "CC(=O)OC1=CC=CC=C1C(=O)O."
        )
    name_column = None
    for column in frame.columns:
        if column == smiles_column:
            continue
        if _header_rank(column, ("name", "id", "compound", "molecule", "title")) > 0:
            name_column = column
            break
    return frame, smiles_column, name_column


__all__ = [
    "ACTIVITY_HEADER_HINTS",
    "MIN_PER_CLASS",
    "SMILES_HEADER_HINTS",
    "clean_molecules",
    "detect_activity_column",
    "detect_separator",
    "detect_smiles_column",
    "excel_sheet_names",
    "label_binary",
    "label_numeric",
    "label_summary",
    "labels_problem_message",
    "looks_binary",
    "numeric_or_binary_fraction",
    "read_screening_table",
    "read_table",
    "smiles_valid_fraction",
    "threshold_for_percent",
    "to_numeric",
]
