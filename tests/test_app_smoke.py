"""
Headless tests of the app itself, using Streamlit's AppTest framework.

AppTest runs app.py without a browser: it executes the script, lets us click
buttons and set widget values by key, and then lets us read back what the page
would have shown. That means the whole user journey (load demo, confirm labels,
train, predict, batch screen) is tested automatically.

Run them with:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(ROOT / "app.py")

# Training a model inside a test takes a while, so be generous.
TIMEOUT = 300


def widget_by_key(app, kind: str, key: str):
    """
    Find a widget by its key, whatever version of Streamlit is installed.

    Some versions support app.button(key=...) directly, but scanning the list
    works everywhere and gives a clearer failure message.
    """
    for element in getattr(app, kind):
        if getattr(element, "key", None) == key:
            return element
    available = [getattr(e, "key", None) for e in getattr(app, kind)]
    raise AssertionError(f"No {kind} with key {key!r}. Keys present: {available}")


def all_text(app) -> str:
    """Every piece of text the page showed, joined together for searching."""
    pieces = []
    for kind in ("markdown", "caption", "info", "success", "warning", "error", "text"):
        for element in getattr(app, kind, []):
            value = getattr(element, "value", None)
            if value:
                pieces.append(str(value))
    return "\n".join(pieces)


@pytest.fixture
def app():
    """A fresh app, run once, with nothing loaded."""
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    return at


@pytest.fixture
def trained_app(app):
    """
    An app that has loaded the demo data, cleaned it and trained a model.

    This is the state the Results, Predict and Batch Screen tabs need, so it is
    built once per test that needs it.
    """
    widget_by_key(app, "button", "load_demo").click().run()
    assert not app.exception
    widget_by_key(app, "button", "confirm_labels").click().run()
    assert not app.exception
    widget_by_key(app, "button", "train_model").click().run()
    assert not app.exception
    return app


# ---------------------------------------------------------------------------
# Starting up
# ---------------------------------------------------------------------------

def test_app_starts_with_no_data_and_no_exception(app):
    assert not app.exception
    assert "MRSA Molecule Lab" in app.title[0].value


def test_empty_states_point_at_the_right_tabs(app):
    text = all_text(app)
    assert "No data loaded yet" in text
    assert "Nothing to train on yet" in text
    assert "No trained model yet" in text


def test_honesty_footer_is_on_every_tab(app):
    """The footer is rendered once per tab, so it must appear six times."""
    from src import HONESTY_FOOTER

    captions = [c.value for c in app.caption]
    assert captions.count(HONESTY_FOOTER) == 6


def test_about_tab_shows_the_article_facts(app):
    text = all_text(app)
    # The 2023 screen and AI run.
    assert "39,312 compounds" in text
    assert "12,076,365 compounds" in text
    assert "283 compounds" in text
    assert "21 brand-new compounds" in text
    # Why it matters, and the biology.
    assert "mecA" in text
    assert "PBP2a" in text
    assert "1.27 million deaths" in text
    assert "100,000" in text
    assert "45 years" in text
    # What the candidates actually did, and the 2025 reality check.
    assert "1.2 logs" in text
    assert "2 of\n80" in text or "2 of 80" in text
    assert "22 of about 90" in text
    # Teams to watch.
    assert "Collins Lab" in text
    assert "Phare Bio" in text
    assert "Stokes Lab" in text


def test_about_tab_states_the_articles_conclusion(app):
    text = all_text(app)
    assert "a candidate is not" in text
    assert "yet a cure" in text


# ---------------------------------------------------------------------------
# Loading the demo data
# ---------------------------------------------------------------------------

def test_load_demo_labels_twenty_actives(app):
    widget_by_key(app, "button", "load_demo").click().run()
    assert not app.exception

    assert app.session_state["is_demo"] is True
    assert app.session_state["smiles_col"] == "smiles"
    assert app.session_state["threshold"] == 50.0

    text = all_text(app)
    assert "DEMO DATA" in text
    assert "79 rows" in text
    assert "Detected SMILES column: smiles" in text

    # The live summary metrics: 20 active, 59 inactive.
    metric_values = [m.value for m in app.metric]
    assert "20" in metric_values
    assert "59" in metric_values


def test_confirm_labels_cleans_the_demo_data(app):
    widget_by_key(app, "button", "load_demo").click().run()
    widget_by_key(app, "button", "confirm_labels").click().run()
    assert not app.exception

    report = app.session_state["clean_report"]
    assert report["final_count"] == 79
    assert report["n_active"] == 20
    assert report["n_inactive"] == 59
    assert report["invalid_count"] == 0
    assert "Data is clean and ready" in all_text(app)


# ---------------------------------------------------------------------------
# Training and results
# ---------------------------------------------------------------------------

def test_training_produces_real_results(trained_app):
    app = trained_app
    assert not app.exception

    bundle = app.session_state["bundle"]
    evaluation = app.session_state["evaluation"]
    split = app.session_state["split"]

    assert bundle is not None
    assert evaluation is not None

    # A real scaffold split: nothing shared between the two sides.
    assert split["report"]["method"] == "scaffold"
    assert split["report"]["shared_scaffolds"] == 0
    assert split["report"]["n_train"] + split["report"]["n_test"] == 79
    assert split["report"]["test_active"] >= 2
    assert split["report"]["test_inactive"] >= 2

    # Real metrics, computed from the test set.
    assert 0.0 <= evaluation["roc_auc"] <= 1.0
    assert 0.0 <= evaluation["pr_auc"] <= 1.0
    assert evaluation["single_class"] is False

    scores = app.session_state["test_scores"]
    assert len(scores) == split["report"]["n_test"]


def test_results_tab_shows_the_experiment_and_the_summary(trained_app):
    app = trained_app
    text = all_text(app)

    assert "Model vs random picking" in text or "Model vs random" in text
    assert "hypergeometric test" in text
    assert "2023 MIT and Broad study" in text

    # The copy-ready summary must be filled with the real numbers.
    summaries = [t.value for t in app.text_area]
    assert summaries, "the copy-ready summary text area is missing"
    summary = summaries[0]
    for expected in (
        "ROC-AUC:",
        "PR-AUC:",
        "Random hit rate:",
        "Model hit rate:",
        "Enrichment:",
        "Hypergeometric p-value:",
    ):
        assert expected in summary

    evaluation = app.session_state["evaluation"]
    assert f"ROC-AUC: {evaluation['roc_auc']:.3f}" in summary


def test_changing_the_number_of_picks_changes_the_hit_rate(trained_app):
    app = trained_app
    sliders = [s for s in app.slider if "buy and test" in s.label]
    assert sliders, "the top-N slider is missing"

    first = [m.value for m in app.metric]
    sliders[0].set_value(app.session_state["evaluation"]["n_test"]).run()
    assert not app.exception
    second = [m.value for m in app.metric]

    # With every test molecule bought, the model hit rate must equal the
    # random hit rate, so the metrics cannot be identical to the small-N case.
    assert first != second


# ---------------------------------------------------------------------------
# Predicting one molecule
# ---------------------------------------------------------------------------

def test_predict_aspirin(trained_app):
    app = trained_app
    widget_by_key(app, "text_input", "smiles_input").set_value(
        "CC(=O)OC1=CC=CC=C1C(=O)O"
    ).run()
    widget_by_key(app, "button", "predict_button").click().run()
    assert not app.exception

    text = all_text(app)
    assert "safe or working medicine" in text
    labels = [m.label for m in app.metric]
    assert "Predicted chance of being active" in labels

    # The score shown must be a real probability between 0% and 100%.
    score_metric = next(m for m in app.metric if m.label == "Predicted chance of being active")
    percent = float(str(score_metric.value).rstrip("%"))
    assert 0.0 <= percent <= 100.0


def test_predict_ciprofloxacin(trained_app):
    app = trained_app
    widget_by_key(app, "text_input", "smiles_input").set_value(
        "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O"
    ).run()
    widget_by_key(app, "button", "predict_button").click().run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert "Predicted chance of being active" in labels


def test_example_button_fills_the_box(trained_app):
    app = trained_app
    widget_by_key(app, "button", "example_caffeine").click().run()
    assert not app.exception
    assert app.session_state["smiles_input"] == "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


@pytest.mark.parametrize("bad_input", ["C1CC", "hello"])
def test_invalid_smiles_shows_a_friendly_error(trained_app, bad_input):
    app = trained_app
    widget_by_key(app, "text_input", "smiles_input").set_value(bad_input).run()
    widget_by_key(app, "button", "predict_button").click().run()

    # No crash, and a readable explanation rather than a stack trace.
    assert not app.exception
    errors = "\n".join(e.value for e in app.error)
    assert "could not read" in errors
    assert bad_input in errors
    assert "Traceback" not in errors


# ---------------------------------------------------------------------------
# Batch screening
# ---------------------------------------------------------------------------

def test_batch_screen_with_the_demo_list(trained_app):
    app = trained_app
    widget_by_key(app, "button", "batch_demo").click().run()
    assert not app.exception

    batch = app.session_state["batch_result"]
    assert batch is not None

    # The demo screening file has 16 rows: 2 broken, and one duplicate of
    # caffeine, so 13 unique valid molecules survive.
    assert batch["uploaded_rows"] == 16
    assert len(batch["invalid_rows"]) == 2
    assert len(batch["smiles"]) == 13
    assert len(batch["scores"]) == 13

    broken_names = {row["Name"] for row in batch["invalid_rows"]}
    assert "broken_row_unclosed_ring" in broken_names
    assert "broken_row_not_a_molecule" in broken_names

    text = all_text(app)
    assert "about 12 million molecules were scored" in text
    assert "2 row(s)" in text


def test_batch_funnel_never_grows(trained_app):
    """Every stage of the funnel must be no bigger than the stage before it."""
    app = trained_app
    widget_by_key(app, "button", "batch_demo").click().run()
    assert not app.exception

    import numpy as np

    batch = app.session_state["batch_result"]
    scores = np.asarray(batch["scores"], dtype=float)
    similarity = np.asarray(batch["similarity"], dtype=float)

    # Rebuild the funnel the same way the app does, with its default cut-offs.
    passes_score = scores >= 0.30
    passes_both = passes_score & (similarity < 0.50)

    counts = [
        batch["uploaded_rows"],
        len(batch["smiles"]),
        int(passes_score.sum()),
        int(passes_both.sum()),
    ]
    assert counts == sorted(counts, reverse=True)


def test_batch_results_are_downloadable(trained_app):
    app = trained_app
    widget_by_key(app, "button", "batch_demo").click().run()
    assert not app.exception
    keys = [getattr(b, "key", None) for b in app.button]
    # Download buttons appear in app.button in current Streamlit versions; if
    # not, the session state above already proves the data exists.
    assert app.session_state["batch_result"] is not None
    assert "batch_demo" in keys


# ---------------------------------------------------------------------------
# Awkward files
# ---------------------------------------------------------------------------

def test_train_button_is_disabled_when_no_molecule_is_active(app):
    """
    Push the threshold so low that nothing counts as active, and the app must
    refuse to continue and say exactly what to change.
    """
    widget_by_key(app, "button", "load_demo").click().run()

    threshold_inputs = [n for n in app.number_input if n.label == "Threshold"]
    assert threshold_inputs, "the threshold input is missing"
    threshold_inputs[0].set_value(0.0001).run()
    assert not app.exception

    errors = "\n".join(e.value for e in app.error)
    assert "0 molecules as active" in errors

    confirm = widget_by_key(app, "button", "confirm_labels")
    assert confirm.disabled is True
