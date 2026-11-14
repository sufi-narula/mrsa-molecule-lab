"""
Which atoms made the model say that?

A model that only gives a number is a "black box": you cannot tell what it
noticed. This module opens the box a crack using the simplest honest trick
there is, called occlusion: cover one atom at a time and see how much the
answer moves.

For every atom we rebuild the fingerprint using every atom *except* that one as
a starting point, score it again, and record

    contribution = (score with all atoms) - (score without this atom)

A positive contribution means the atom was pushing the prediction up: removing
it made the molecule look less active to the model. A negative contribution
means the opposite.

This is a rough tool. The MIT and Broad team used a far more careful method
(Monte Carlo tree search) to find the smallest chemical piece responsible for
each prediction.
"""

from __future__ import annotations

import numpy as np

from .features import fingerprint_one, get_generator
from .modeling import predict_scores
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdMolDescriptors  # noqa: F401  (kept for RDKit init order)

SIMPLIFIED_NOTE = (
    "This is a simplified explanation. The MIT and Broad team used a more advanced "
    "method (Monte Carlo tree search) to find the smallest chemical piece that "
    "explains each prediction."
)

UNAVAILABLE_MESSAGE = "Explanation not available for this molecule."

# Colours for the drawing: warm (orange-red) pushes the score up, cool (blue)
# pushes it down. Given as red, green, blue fractions between 0 and 1.
WARM = (0.90, 0.33, 0.13)
COOL = (0.09, 0.40, 0.75)


def atom_contributions(model, mol, radius: int = 2, bits: int = 2048) -> np.ndarray:
    """
    One contribution number per atom, by the occlusion method described above.

    Returns a numpy array with one entry per atom. Raises nothing of its own;
    the caller decides what to do if RDKit or the model complains.
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return np.zeros(0, dtype=float)

    generator = get_generator(radius, bits)
    all_atoms = list(range(n_atoms))

    base_fp = fingerprint_one(mol, radius=radius, bits=bits)
    base_score = float(predict_scores(model, base_fp)[0])

    # Build every "leave one atom out" fingerprint first, then score them in a
    # single call, which is much faster than scoring one row at a time.
    rows = np.zeros((n_atoms, bits), dtype=np.uint8)
    for i in all_atoms:
        others = [a for a in all_atoms if a != i]
        if others:
            rows[i] = generator.GetFingerprintAsNumPy(mol, fromAtoms=others)
        # If the molecule has a single atom, the row stays all zeros, which is
        # the honest answer: with that atom removed there is nothing left.

    without_scores = predict_scores(model, rows)
    return base_score - np.asarray(without_scores, dtype=float)


def _colour_for(value: float, largest: float) -> tuple:
    """
    Pick a colour for one atom, with the strength scaled by its contribution.

    The biggest contribution in the molecule gets the full colour; everything
    else is faded towards white in proportion. That keeps the picture readable
    whatever the size of the numbers.
    """
    if largest <= 0:
        return (1.0, 1.0, 1.0)
    strength = min(1.0, abs(value) / largest)
    # Fade from white towards the target colour.
    target = WARM if value > 0 else COOL
    faded = tuple(1.0 - strength * (1.0 - channel) for channel in target)
    return faded


def explanation_svg(mol, contributions, size: int = 420) -> str:
    """
    Draw the molecule with its atoms coloured by contribution, as SVG text.

    SVG is used rather than a bitmap so that the drawing stays sharp and does
    not depend on RDKit having been built with image support.
    """
    contributions = np.asarray(contributions, dtype=float)
    if contributions.size == 0:
        raise ValueError("There are no atom contributions to draw.")

    largest = float(np.max(np.abs(contributions))) if contributions.size else 0.0
    highlight_atoms = []
    highlight_colours = {}
    radii = {}
    for i, value in enumerate(contributions):
        if largest > 0 and abs(value) < largest * 0.05:
            continue  # too small to be worth colouring
        highlight_atoms.append(i)
        highlight_colours[i] = _colour_for(float(value), largest)
        radii[i] = 0.45

    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    options = drawer.drawOptions()
    options.addStereoAnnotation = False
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colours,
        highlightAtomRadii=radii,
        highlightBonds=[],
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def explain_molecule(model, mol, radius: int = 2, bits: int = 2048) -> dict:
    """
    The whole explanation in one call, and never an exception.

    Returns a dictionary with "ok". When ok is False, "message" holds the
    friendly text the app should show instead of a picture.
    """
    try:
        contributions = atom_contributions(model, mol, radius=radius, bits=bits)
        if contributions.size == 0:
            return {"ok": False, "message": UNAVAILABLE_MESSAGE}
        svg = explanation_svg(mol, contributions)
        largest = float(np.max(np.abs(contributions)))
        if largest == 0.0:
            # The model's answer did not move at all. Say so honestly rather
            # than drawing a colourless picture and implying meaning.
            return {
                "ok": False,
                "message": (
                    "Removing any single atom did not change this prediction, so there "
                    "is nothing to highlight."
                ),
            }
        top_up = int(np.argmax(contributions))
        top_down = int(np.argmin(contributions))
        return {
            "ok": True,
            "svg": svg,
            "contributions": contributions,
            "largest": largest,
            "top_up_atom": top_up,
            "top_up_value": float(contributions[top_up]),
            "top_down_atom": top_down,
            "top_down_value": float(contributions[top_down]),
        }
    except Exception:
        # Any molecule RDKit or the model dislikes gets the friendly message.
        return {"ok": False, "message": UNAVAILABLE_MESSAGE}


def legend_svg(width: int = 420, height: int = 54) -> str:
    """
    A small colour key, drawn by hand as SVG.

    Orange means "this atom pushed the score up", blue means "pushed it down",
    and a pale colour means the effect was small.
    """
    warm = "rgb(%d,%d,%d)" % tuple(int(c * 255) for c in WARM)
    cool = "rgb(%d,%d,%d)" % tuple(int(c * 255) for c in COOL)
    return f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img"
     aria-label="Colour key: blue atoms pushed the score down, orange atoms pushed it up.">
  <defs>
    <linearGradient id="contribution-key" x1="0" x2="1" y1="0" y2="0">
      <stop offset="0%" stop-color="{cool}"/>
      <stop offset="50%" stop-color="#FFFFFF"/>
      <stop offset="100%" stop-color="{warm}"/>
    </linearGradient>
  </defs>
  <rect x="8" y="8" width="{width - 16}" height="16" rx="3"
        fill="url(#contribution-key)" stroke="#CBD5E1"/>
  <text x="8" y="42" font-family="sans-serif" font-size="12" fill="#0F172A">
    pushed the score down
  </text>
  <text x="{width - 8}" y="42" font-family="sans-serif" font-size="12" fill="#0F172A"
        text-anchor="end">
    pushed the score up
  </text>
</svg>
""".strip()


__all__ = [
    "COOL",
    "SIMPLIFIED_NOTE",
    "UNAVAILABLE_MESSAGE",
    "WARM",
    "atom_contributions",
    "explain_molecule",
    "explanation_svg",
    "legend_svg",
]
