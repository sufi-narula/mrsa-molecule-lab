"""
How long does the whole pipeline take on a realistically large file?

My real dataset is about 39,000 molecules, so this script builds 40,000
synthetic ones, labels about 1.3% of them active, and runs the full pipeline:
clean, scaffold split, build fingerprints, train with the default settings.

IMPORTANT: the molecules and labels here are made up to measure speed and
memory. They are never shown in the app and they mean nothing scientifically.

Run it with:  python scripts/performance_test.py
Options:      --n 40000 --radius 2 --bits 2048 --trees 300 --seed 42
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_io, features, modeling, splitting  # noqa: E402

# Building blocks that join into valid SMILES strings.
#
# Every molecule is built as:  linker + ring + linker + ring [+ linker + ring] + tail
#
# The rings give the molecules real Bemis-Murcko scaffolds, which is what makes
# the scaffold split do meaningful work. The linkers are all divalent, so they
# can sit between two pieces without breaking any atom's valence. That keeps
# almost every generated string chemically valid, and there are well over a
# million possible combinations, which is plenty for 40,000 unique molecules.
RINGS = [
    "c1ccccc1",
    "c1ccncc1",
    "c1ccc2ccccc2c1",
    "C1CCCCC1",
    "C1CCNCC1",
    "c1ccsc1",
    "c1cc[nH]c1",
    "C1CCOC1",
    "c1cnc2ccccc2n1",
    "c1ccc2[nH]ccc2c1",
    "C1CCCC1",
    "c1ccoc1",
]
LINKERS = [
    "C",
    "CC",
    "CCC",
    "O",
    "N",
    "S",
    "CO",
    "CN",
    "C(=O)",
    "NC(=O)",
    "OCC",
    "CC(C)",
    "C(C)",
    "CSC",
]
TAILS = [
    "O",
    "N",
    "C",
    "CC",
    "Cl",
    "F",
    "C(=O)O",
    "C#N",
    "S(=O)(=O)N",
    "C(F)(F)F",
    "OC",
    "NC(=O)C",
    "C(=O)N",
    "CO",
]


def build_synthetic_dataset(n: int, seed: int = 42) -> pd.DataFrame:
    """
    Make n unique, valid molecules with a made-up measurement column.

    Labels are assigned at random with about 1.3% active, matching the hit rate
    of the real 2023 screen, purely so that the timing is realistic.
    """
    rng = np.random.default_rng(seed)
    seen = set()
    rows = []
    attempts = 0
    max_attempts = max(n * 40, 200_000)

    def pick(options):
        return options[int(rng.integers(len(options)))]

    while len(rows) < n and attempts < max_attempts:
        attempts += 1

        pieces = [pick(LINKERS), pick(RINGS), pick(LINKERS), pick(RINGS)]
        if rng.random() < 0.5:
            pieces.extend([pick(LINKERS), pick(RINGS)])
        pieces.append(pick(TAILS))
        smiles = "".join(pieces)

        canonical = features.standardize_smiles(smiles)
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)

        # About 1.3% get a small (active-looking) measurement.
        is_active = rng.random() < 0.013
        value = float(rng.uniform(0.1, 8.0)) if is_active else float(rng.uniform(64.0, 512.0))
        rows.append(
            {
                "name": f"synthetic_{len(rows):06d}",
                "smiles": canonical,
                "measurement": value,
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40_000, help="how many molecules to generate")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--bits", type=int, default=2048)
    parser.add_argument("--trees", type=int, default=modeling.DEFAULT_TREES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 72)
    print("MRSA Molecule Lab: performance test")
    print("Synthetic molecules with random labels. For timing only, not science.")
    print("=" * 72)

    tracemalloc.start()
    overall_start = time.perf_counter()

    # 1. Generate -----------------------------------------------------------
    start = time.perf_counter()
    frame = build_synthetic_dataset(args.n, seed=args.seed)
    generate_seconds = time.perf_counter() - start
    print(f"\n1. Generated {len(frame):,} unique valid molecules in {generate_seconds:.1f} s")
    if len(frame) < args.n:
        print(
            f"   Note: only {len(frame):,} of the requested {args.n:,} unique molecules "
            "could be built from the fragment set."
        )

    # 2. Label and clean ----------------------------------------------------
    start = time.perf_counter()
    labels, _excluded = data_io.label_numeric(
        frame["measurement"].tolist(), "at or below", 50.0
    )
    clean, report = data_io.clean_molecules(frame, "smiles", labels, name_column="name")
    clean_seconds = time.perf_counter() - start
    print(
        f"2. Cleaned in {clean_seconds:.1f} s: {report['final_count']:,} molecules, "
        f"{report['n_active']:,} active ({report['percent_active']:.2f}%), "
        f"{report['duplicates_removed']:,} duplicates removed"
    )

    smiles = clean["smiles"].tolist()
    y = clean["label"].to_numpy(dtype=int)

    # 3. Scaffold split -----------------------------------------------------
    start = time.perf_counter()
    split = splitting.make_split(smiles, y, "scaffold", 0.2, args.seed)
    split_seconds = time.perf_counter() - start
    split_report = split["report"]
    print(
        f"3. Scaffold split in {split_seconds:.1f} s: "
        f"{split_report['n_scaffolds']:,} scaffolds, "
        f"{split_report['n_train']:,} train / {split_report['n_test']:,} test, "
        f"{split_report['shared_scaffolds']} shared scaffolds, "
        f"method = {split_report['method']}"
    )
    if split_report.get("notice"):
        print(f"   {split_report['notice']}")

    train_smiles = [smiles[i] for i in split["train_idx"]]
    test_smiles = [smiles[i] for i in split["test_idx"]]
    y_train = y[split["train_idx"]]
    y_test = y[split["test_idx"]]

    # 4. Fingerprints and training -----------------------------------------
    start = time.perf_counter()
    bundle = modeling.fit_with_safeguard(
        train_smiles,
        y_train,
        model_kind=modeling.MODEL_RANDOM_FOREST,
        radius=args.radius,
        bits=args.bits,
        n_estimators=args.trees,
        seed=args.seed,
    )
    train_seconds = time.perf_counter() - start
    print(
        f"4. Fingerprints + training in {train_seconds:.1f} s "
        f"({bundle['bits']} bits, {bundle['n_estimators']} trees)"
    )
    if bundle["safeguard_notice"]:
        print(f"   MEMORY SAFEGUARD TRIGGERED: {bundle['safeguard_notice']}")
    else:
        print("   Memory safeguard was not needed.")

    # 5. Score the test set -------------------------------------------------
    start = time.perf_counter()
    x_test = features.fingerprint_array(test_smiles, bundle["radius"], bundle["bits"])
    scores = modeling.predict_scores(bundle["model"], x_test)
    evaluation = modeling.evaluate(y_test, scores, 0.5)
    score_seconds = time.perf_counter() - start
    print(f"5. Scored {len(test_smiles):,} test molecules in {score_seconds:.1f} s")

    total_seconds = time.perf_counter() - overall_start
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n" + "-" * 72)
    print(f"TOTAL pipeline time: {total_seconds:.1f} s")
    print(f"Peak Python memory tracked: {peak_bytes / 1e6:.0f} MB")
    print(f"Training fingerprint table: {bundle['x_train'].nbytes / 1e6:.0f} MB")
    if evaluation["single_class"]:
        print("Test set had only one class, so no scores could be computed.")
    else:
        print(
            f"ROC-AUC on random labels: {evaluation['roc_auc']:.3f} "
            "(should sit near 0.5, because the labels are noise)"
        )
    print("-" * 72)
    print(
        "Reminder: these molecules and labels are synthetic. The numbers above are "
        "speed and memory measurements, not scientific results."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
