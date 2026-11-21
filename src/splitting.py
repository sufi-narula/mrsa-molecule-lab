"""
Splitting the molecules into a training set and a test set.

Why this matters more than anything else in the app: if a molecule in the test
set looks almost exactly like one in the training set, the model can score
beautifully without having learned anything useful. It has simply recognised an
old friend.

A "scaffold split" is the honest version. Every molecule has a core skeleton
(its Bemis-Murcko scaffold: strip the dangling bits and keep the rings and the
links between them). We put whole scaffold families on one side or the other,
so the test set contains chemical shapes the model has genuinely never seen.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .features import mol_from_smiles
from rdkit.Chem.Scaffolds import MurckoScaffold

# How many times we retry a scaffold split before giving up and using a random
# one instead.
MAX_ATTEMPTS = 200

# The test set has to contain at least this many of each class, otherwise none
# of the metrics on the Results tab can be computed.
MIN_TEST_PER_CLASS = 2

FALLBACK_NOTICE = (
    "A scaffold split couldn’t give a usable test set for this data, "
    "so a random split was used."
)


#: Prefix of the key given to a molecule whose scaffold could not be computed.
#: These are counted and reported, never quietly treated as real scaffolds.
BROKEN_PREFIX = "__no_scaffold_"

BROKEN_SCAFFOLD_NOTICE = (
    "RDKit could not work out the chemical skeleton for {n} of {total} molecules "
    "on this computer, so those molecules were each put in a group of their own. "
    "A split like that is closer to a random split than a real scaffold split, so "
    "treat the “scaffolds shared” number with suspicion."
)

ALL_BROKEN_NOTICE = (
    "RDKit could not work out the chemical skeleton for ANY molecule on this "
    "computer, so a real scaffold split was impossible and a random split was "
    "used instead. This is a problem with the RDKit installation, not with your "
    "data."
)


def scaffold_key(smiles: str, index: int) -> str:
    """
    The group key for one molecule.

    Normally this is the molecule's Bemis-Murcko scaffold. Molecules with no
    rings at all (ethanol, glycine, urea) have an empty scaffold; lumping all
    of those into one giant group would be wrong, so each ring-free molecule
    gets its own unique key instead.

    If RDKit fails outright (some builds raise "bad any cast" from
    MurckoDecompose), the molecule gets a unique key prefixed with
    BROKEN_PREFIX. That is NOT a silent fallback: scaffold_keys counts those
    keys and the split report shows the count, because a split built from
    failures is really a random split wearing a scaffold split's name.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return f"__unparsable_{index}__"
    try:
        core = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return f"{BROKEN_PREFIX}{index}__"
    if not core:
        return f"__no_rings_{index}__"
    return core


def scaffold_keys(smiles_list) -> list:
    """The group key for every molecule, in the same order as the input."""
    return [scaffold_key(smi, i) for i, smi in enumerate(smiles_list)]


def count_broken_scaffolds(keys) -> int:
    """
    How many molecules RDKit could not decompose at all.

    The app shows this number. If it is large, the "scaffold split" is not
    really a scaffold split and the user deserves to be told so rather than
    shown a reassuring "0 scaffolds shared".
    """
    return sum(1 for key in keys if str(key).startswith(BROKEN_PREFIX))


def group_by_scaffold(smiles_list) -> dict:
    """Map each scaffold key to the list of row positions that share it."""
    groups = defaultdict(list)
    for i, key in enumerate(scaffold_keys(smiles_list)):
        groups[key].append(i)
    return dict(groups)


def _one_scaffold_attempt(groups: dict, n_total: int, target_test: int, seed: int) -> tuple:
    """
    One try at a scaffold split. Returns (train_idx, test_idx).

    The rule about big groups: a scaffold family larger than half the target
    test size would swamp the test set on its own, so it always goes to train.
    """
    big_groups = []
    small_groups = []
    half_target = target_test / 2.0
    for key, members in groups.items():
        if len(members) > half_target:
            big_groups.append(key)
        else:
            small_groups.append(key)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(small_groups))

    test_idx = []
    train_idx = []
    for key in big_groups:
        train_idx.extend(groups[key])

    used = set()
    for position in order:
        key = small_groups[position]
        if len(test_idx) >= target_test:
            break
        test_idx.extend(groups[key])
        used.add(key)

    for key in small_groups:
        if key not in used:
            train_idx.extend(groups[key])

    return sorted(train_idx), sorted(test_idx)


def stratified_split(labels, test_fraction: float = 0.2, seed: int = 42) -> tuple:
    """
    A plain random split that keeps the share of actives the same on both sides.

    Simpler and friendlier to the model than a scaffold split, which is exactly
    why the scores it produces are usually too optimistic.
    """
    labels = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for value in (0, 1):
        members = np.where(labels == value)[0]
        shuffled = rng.permutation(members)
        n_test = int(round(len(shuffled) * test_fraction))
        # Make sure each class contributes at least one molecule to each side
        # whenever it has at least two molecules in total.
        if len(shuffled) >= 2:
            n_test = max(1, min(n_test, len(shuffled) - 1))
        test_idx.extend(shuffled[:n_test].tolist())
        train_idx.extend(shuffled[n_test:].tolist())
    return sorted(train_idx), sorted(test_idx)


def split_report(smiles_list, labels, train_idx, test_idx, method: str, notice=None) -> dict:
    """
    Everything the app displays about a finished split.

    "shared_scaffolds" is the honesty check: for a real scaffold split it must
    be 0, meaning no chemical skeleton appears on both sides.
    """
    labels = np.asarray(labels, dtype=int)
    keys = scaffold_keys(smiles_list)
    train_keys = {keys[i] for i in train_idx}
    test_keys = {keys[i] for i in test_idx}

    # Molecules RDKit could not decompose are not real scaffolds. Count them,
    # and say so loudly, because "0 scaffolds shared" built out of failures
    # would be a reassuring number that means nothing.
    total = int(len(smiles_list))
    broken = count_broken_scaffolds(keys)
    if broken and notice is None:
        if broken == total:
            notice = ALL_BROKEN_NOTICE
        else:
            notice = BROKEN_SCAFFOLD_NOTICE.format(n=broken, total=total)

    return {
        "method": method,
        "notice": notice,
        "broken_scaffolds": broken,
        "n_total": total,
        "n_scaffolds": len(set(keys)),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "train_active": int(labels[train_idx].sum()) if len(train_idx) else 0,
        "train_inactive": int(len(train_idx) - labels[train_idx].sum()) if len(train_idx) else 0,
        "test_active": int(labels[test_idx].sum()) if len(test_idx) else 0,
        "test_inactive": int(len(test_idx) - labels[test_idx].sum()) if len(test_idx) else 0,
        "shared_scaffolds": len(train_keys & test_keys),
    }


def scaffold_split(smiles_list, labels, test_fraction: float = 0.2, seed: int = 42) -> dict:
    """
    Split by scaffold, retrying if the test set turns out to be unusable.

    The algorithm:
      1. work out each molecule's scaffold (ring-free molecules get their own
         unique key);
      2. group the molecules by scaffold;
      3. send any group bigger than half the target test size to train;
      4. shuffle the remaining groups with the seed and add whole groups to
         test until the target test size is reached, everything else to train;
      5. if the test set does not hold at least 2 actives and 2 inactives, try
         again with a new seed, up to 200 times;
      6. if that still fails, fall back to a stratified random split and set a
         visible notice explaining what happened.

    Returns a dictionary with train_idx, test_idx and the display report.
    """
    labels = np.asarray(labels, dtype=int)
    n_total = len(smiles_list)
    if n_total != len(labels):
        raise ValueError("The molecule list and the labels have different lengths.")
    target_test = int(round(n_total * float(test_fraction)))
    target_test = max(1, min(target_test, n_total - 1)) if n_total > 1 else 0

    # Scaffolds are expensive to compute, so do it once and reuse the groups.
    groups = group_by_scaffold(smiles_list)

    for attempt in range(MAX_ATTEMPTS):
        train_idx, test_idx = _one_scaffold_attempt(
            groups, n_total, target_test, seed + attempt
        )
        if not test_idx or not train_idx:
            continue
        test_labels = labels[test_idx]
        if (
            int((test_labels == 1).sum()) >= MIN_TEST_PER_CLASS
            and int((test_labels == 0).sum()) >= MIN_TEST_PER_CLASS
        ):
            report = split_report(
                smiles_list, labels, train_idx, test_idx, method="scaffold"
            )
            report["attempts"] = attempt + 1
            return {"train_idx": train_idx, "test_idx": test_idx, "report": report}

    # Every attempt failed: fall back to a random split and say so.
    train_idx, test_idx = stratified_split(labels, test_fraction, seed)
    report = split_report(
        smiles_list, labels, train_idx, test_idx, method="random (fallback)",
        notice=FALLBACK_NOTICE,
    )
    report["attempts"] = MAX_ATTEMPTS
    return {"train_idx": train_idx, "test_idx": test_idx, "report": report}


def make_split(smiles_list, labels, method: str, test_fraction: float, seed: int) -> dict:
    """
    Split the data with whichever method the user chose.

    method is "scaffold" (the honest, harder test) or "random" (a stratified
    random split).
    """
    if method == "scaffold":
        return scaffold_split(smiles_list, labels, test_fraction, seed)
    train_idx, test_idx = stratified_split(labels, test_fraction, seed)
    report = split_report(
        smiles_list, labels, train_idx, test_idx, method="random (stratified)"
    )
    report["attempts"] = 1
    return {"train_idx": train_idx, "test_idx": test_idx, "report": report}


__all__ = [
    "ALL_BROKEN_NOTICE",
    "BROKEN_PREFIX",
    "BROKEN_SCAFFOLD_NOTICE",
    "FALLBACK_NOTICE",
    "MAX_ATTEMPTS",
    "count_broken_scaffolds",
    "MIN_TEST_PER_CLASS",
    "group_by_scaffold",
    "make_split",
    "scaffold_key",
    "scaffold_keys",
    "scaffold_split",
    "split_report",
    "stratified_split",
]
