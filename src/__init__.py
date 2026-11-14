"""
MRSA Molecule Lab: a mini AI antibiotic predictor.

All of the real work (loading data, cleaning molecules, building fingerprints,
splitting, training, scoring and explaining) lives in this package so that it
can be tested without starting the Streamlit user interface.

The honesty footer that the app shows on every page:
"""

HONESTY_FOOTER = (
    "Educational mini-model. It is not the published MIT and Broad model, "
    "a prediction is not proof that a molecule works as a medicine, "
    "and this is not medical advice."
)

__all__ = ["HONESTY_FOOTER"]
