"""
Molecules in, numbers out.

A machine learning model cannot read a molecule, it can only read numbers. A
"molecular fingerprint" is the usual translation: we walk over every atom, look
at the small neighbourhood around it, and switch on one bit for each kind of
neighbourhood we find. The result is a long row of 0s and 1s that says which
chemical pieces the molecule contains.

This module also holds the shared RDKit helpers (reading a SMILES string,
removing salts, canonical text) because almost every other module needs them.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

# RDKit prints a warning to the terminal for every molecule it cannot read. We
# hand it messy files on purpose and report bad rows ourselves, so silence it.
RDLogger.DisableLog("rdApp.*")

# Choices the user is allowed to make in the sidebar.
ALLOWED_RADII = (1, 2, 3)
ALLOWED_BITS = (1024, 2048)
DEFAULT_RADIUS = 2
DEFAULT_BITS = 2048


# ---------------------------------------------------------------------------
# Reading and tidying single molecules
# ---------------------------------------------------------------------------

def mol_from_smiles(smiles) -> "Chem.Mol | None":
    """
    Turn one SMILES string into an RDKit molecule, or return None.

    Never raises. Anything unreadable (blank text, a number, nonsense such as
    "hello", an unclosed ring such as "C1CC") simply comes back as None, which
    is how the app counts "invalid" rows.
    """
    if smiles is None:
        return None
    try:
        text = str(smiles).strip()
    except Exception:
        return None
    if not text:
        return None
    try:
        return Chem.MolFromSmiles(text)
    except Exception:
        # RDKit normally returns None instead of raising, but an odd input type
        # must never be allowed to crash the app.
        return None


def largest_fragment(mol: "Chem.Mol") -> "Chem.Mol":
    """
    Keep only the biggest piece of a molecule (a simple way to remove salts).

    Database entries are often salts, for example "CC(=O)O.[Na+]" (acetic acid
    plus a sodium ion). The sodium is not the interesting part, so we keep the
    fragment with the most heavy (non-hydrogen) atoms.
    """
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:
        return mol
    if not frags or len(frags) == 1:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def canonical_smiles(mol: "Chem.Mol") -> "str | None":
    """
    The one official way of writing this molecule as text.

    Two differently written SMILES strings for the same molecule give the same
    canonical SMILES, which is how the cleaning step finds duplicates.
    """
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception:
        return None


def standardize_smiles(smiles) -> "str | None":
    """Read a SMILES string, drop salts, and give back canonical text."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return canonical_smiles(largest_fragment(mol))


def describe_molecule(mol: "Chem.Mol") -> dict:
    """
    A small table of plain chemical properties, shown on the Predict tab.

    These are standard numbers a chemist would recognise. The app puts a short
    plain-language explanation next to the table.
    """
    return {
        "Molecular weight": round(float(Descriptors.MolWt(mol)), 2),
        "logP (greasiness)": round(float(Descriptors.MolLogP(mol)), 2),
        "Hydrogen-bond donors": int(Descriptors.NumHDonors(mol)),
        "Hydrogen-bond acceptors": int(Descriptors.NumHAcceptors(mol)),
        "Ring count": int(Descriptors.RingCount(mol)),
        "Rotatable bonds": int(Descriptors.NumRotatableBonds(mol)),
        "TPSA (polar surface area)": round(float(Descriptors.TPSA(mol)), 2),
    }


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def get_generator(radius: int = DEFAULT_RADIUS, bits: int = DEFAULT_BITS):
    """
    Build a Morgan fingerprint generator (RDKit's modern API).

    radius is how far from each atom we look: radius 1 sees an atom and its
    direct neighbours, radius 2 sees one step further, and so on. bits is how
    many on/off switches the fingerprint has. More bits means fewer accidental
    collisions between different chemical pieces, but more memory.
    """
    if radius not in ALLOWED_RADII:
        raise ValueError(f"radius must be one of {ALLOWED_RADII}, got {radius!r}")
    if bits not in ALLOWED_BITS:
        raise ValueError(f"bits must be one of {ALLOWED_BITS}, got {bits!r}")
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)


def fingerprint_array(
    smiles_list: Sequence[str],
    radius: int = DEFAULT_RADIUS,
    bits: int = DEFAULT_BITS,
    progress_cb=None,
) -> np.ndarray:
    """
    Turn a list of SMILES into one table of 0s and 1s (rows = molecules).

    Stored as uint8 (one byte per bit) to keep memory small: 40,000 molecules
    at 2048 bits is about 80 MB, which fits comfortably.

    Any molecule that cannot be read becomes a row of all zeros. Callers are
    expected to have cleaned their data first (see data_io.clean_molecules), so
    in normal use this does not happen.
    """
    gen = get_generator(radius, bits)
    n = len(smiles_list)
    out = np.zeros((n, bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = mol_from_smiles(smi)
        if mol is not None:
            out[i] = gen.GetFingerprintAsNumPy(mol)
        if progress_cb is not None and (i % 500 == 0 or i == n - 1):
            progress_cb(i + 1, n)
    return out


def fingerprint_one(
    mol: "Chem.Mol",
    radius: int = DEFAULT_RADIUS,
    bits: int = DEFAULT_BITS,
    from_atoms: "Iterable[int] | None" = None,
) -> np.ndarray:
    """
    The fingerprint of a single molecule, as one row ready for the model.

    from_atoms limits which atoms are used as starting points. The explanation
    code uses that to ask "what does the model predict if we ignore this atom?"
    """
    gen = get_generator(radius, bits)
    if from_atoms is None:
        arr = gen.GetFingerprintAsNumPy(mol)
    else:
        arr = gen.GetFingerprintAsNumPy(mol, fromAtoms=list(from_atoms))
    return arr.astype(np.uint8).reshape(1, -1)


def bit_vectors(
    smiles_list: Sequence[str],
    radius: int = DEFAULT_RADIUS,
    bits: int = DEFAULT_BITS,
) -> list:
    """
    RDKit bit-vector objects for the same molecules.

    These are a different shape of the same information. We keep them because
    RDKit's fast Tanimoto similarity works on bit vectors, not numpy arrays.
    Unreadable molecules get an empty (all zeros) bit vector so that the list
    lines up with the input row for row.
    """
    gen = get_generator(radius, bits)
    out = []
    for smi in smiles_list:
        mol = mol_from_smiles(smi)
        if mol is None:
            out.append(DataStructs.ExplicitBitVect(bits))
        else:
            out.append(gen.GetFingerprint(mol))
    return out


def bit_vector_one(
    mol: "Chem.Mol",
    radius: int = DEFAULT_RADIUS,
    bits: int = DEFAULT_BITS,
):
    """The RDKit bit vector of a single molecule, for similarity searches."""
    return get_generator(radius, bits).GetFingerprint(mol)


def cache_key(smiles_list: Sequence[str], radius: int, bits: int) -> str:
    """
    A short fingerprint-of-the-fingerprints used as a cache key.

    Building fingerprints for tens of thousands of molecules takes a few
    seconds, and Streamlit re-runs the whole script on every click. We hash the
    molecule list together with the settings so that we only rebuild when
    something actually changed.
    """
    hasher = hashlib.sha256()
    hasher.update(f"r{radius}|b{bits}|n{len(smiles_list)}|".encode("utf-8"))
    for smi in smiles_list:
        hasher.update(str(smi).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def tanimoto_to_many(query_bv, reference_bvs: Sequence) -> np.ndarray:
    """
    How similar one molecule is to each molecule in a reference list.

    Tanimoto similarity counts the bits that are on in both fingerprints and
    divides by the bits that are on in either. 1.0 means the fingerprints are
    identical, 0.0 means they share nothing.
    """
    if len(reference_bvs) == 0:
        return np.zeros(0, dtype=float)
    sims = DataStructs.BulkTanimotoSimilarity(query_bv, list(reference_bvs))
    return np.asarray(sims, dtype=float)


def nearest_neighbours(query_bv, reference_bvs: Sequence, k: int = 3) -> list:
    """
    The k most similar reference molecules, best first.

    Returns a list of (index, similarity) pairs pointing back into the
    reference list.
    """
    sims = tanimoto_to_many(query_bv, reference_bvs)
    if sims.size == 0:
        return []
    k = int(min(k, sims.size))
    # argsort gives smallest first, so take the last k and reverse them.
    order = np.argsort(sims, kind="stable")[::-1][:k]
    return [(int(i), float(sims[i])) for i in order]


def max_similarity_to_set(query_bvs: Sequence, reference_bvs: Sequence) -> np.ndarray:
    """
    For each query molecule, its similarity to the closest reference molecule.

    The Batch Screen tab uses this to prefer "new-looking" molecules: a low
    number means the molecule is unlike anything the model already knows is
    active.
    """
    if len(reference_bvs) == 0:
        return np.zeros(len(query_bvs), dtype=float)
    out = np.zeros(len(query_bvs), dtype=float)
    refs = list(reference_bvs)
    for i, bv in enumerate(query_bvs):
        sims = DataStructs.BulkTanimotoSimilarity(bv, refs)
        out[i] = max(sims) if sims else 0.0
    return out


__all__ = [
    "ALLOWED_BITS",
    "ALLOWED_RADII",
    "DEFAULT_BITS",
    "DEFAULT_RADIUS",
    "bit_vector_one",
    "bit_vectors",
    "cache_key",
    "canonical_smiles",
    "describe_molecule",
    "fingerprint_array",
    "fingerprint_one",
    "get_generator",
    "largest_fragment",
    "max_similarity_to_set",
    "mol_from_smiles",
    "nearest_neighbours",
    "standardize_smiles",
    "tanimoto_to_many",
]
