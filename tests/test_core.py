"""
Unit tests for everything in src/.

These test the logic, not the user interface: labelling, cleaning, splitting,
and above all the statistics behind the Model vs Random experiment, which are
checked against numbers worked out by hand.

Run them with:  pytest -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the project root importable when pytest is run from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_io, demo, experiment, features, modeling, splitting  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo_frame():
    return demo.load_demo_training()


@pytest.fixture(scope="module")
def demo_clean(demo_frame):
    """The demo data, labelled at the documented threshold and cleaned."""
    labels, _excluded = data_io.label_numeric(
        demo_frame[demo.DEMO_ACTIVITY_COLUMN].tolist(),
        demo.DEMO_DIRECTION,
        demo.DEMO_THRESHOLD,
    )
    clean, report = data_io.clean_molecules(
        demo_frame, demo.DEMO_SMILES_COLUMN, labels, name_column=demo.DEMO_NAME_COLUMN
    )
    return clean, report


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def test_smiles_detector_picks_smiles_not_name(demo_frame):
    """The detector must never choose the column of chemical names."""
    column, scores = data_io.detect_smiles_column(demo_frame)
    assert column == "smiles"
    # The name column must not even qualify: names do not parse as molecules.
    name_fraction, name_checked = scores["name"]
    assert name_checked > 0
    assert name_fraction < 0.8


def test_smiles_detector_reports_a_high_valid_fraction(demo_frame):
    fraction, checked = data_io.smiles_valid_fraction(demo_frame["smiles"].tolist())
    assert checked == 79
    assert fraction == 1.0


def test_smiles_detector_returns_none_when_nothing_qualifies():
    frame = pd.DataFrame({"name": ["aspirin", "caffeine"], "note": ["hello", "world"]})
    column, _scores = data_io.detect_smiles_column(frame)
    assert column is None


def test_activity_detector_finds_the_measurement(demo_frame):
    column, candidates, _scores = data_io.detect_activity_column(demo_frame, "smiles")
    assert column == demo.DEMO_ACTIVITY_COLUMN
    assert demo.DEMO_ACTIVITY_COLUMN in candidates


def test_activity_detector_returns_none_for_a_name_and_smiles_file():
    """A two-column file of names and molecules has no measurement column."""
    frame = pd.DataFrame(
        {"name": ["aspirin", "caffeine"], "smiles": ["CC(=O)O", "CN1C=NC2=C1C(=O)N(C)C2=O"]}
    )
    column, candidates, _scores = data_io.detect_activity_column(frame, "smiles")
    assert column is None
    assert candidates == []


def test_activity_detector_skips_a_text_column():
    frame = pd.DataFrame(
        {
            "smiles": ["CCO", "CCC", "CCCC"],
            "comment": ["looked fine", "odd result", "repeat this"],
            "mic": [1.0, 2.0, 64.0],
        }
    )
    column, candidates, _scores = data_io.detect_activity_column(frame, "smiles")
    assert column == "mic"
    assert "comment" not in candidates


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

def test_label_numeric_at_or_below():
    labels, excluded = data_io.label_numeric([1.0, 10.0, 50.0, 51.0, 100.0], "at or below", 50.0)
    assert labels.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert excluded["blank or not a number"] == 0


def test_label_numeric_at_or_above():
    labels, _excluded = data_io.label_numeric([1.0, 10.0, 50.0, 51.0, 100.0], "at or above", 50.0)
    assert labels.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_label_numeric_excludes_blanks_and_text():
    labels, excluded = data_io.label_numeric([1.0, None, "not a number", 80.0], "at or below", 50.0)
    assert labels[0] == 1.0
    assert math.isnan(labels[1])
    assert math.isnan(labels[2])
    assert labels[3] == 0.0
    assert excluded["blank or not a number"] == 2


def test_label_numeric_rejects_a_bad_direction():
    with pytest.raises(ValueError):
        data_io.label_numeric([1.0], "sideways", 1.0)


@pytest.mark.parametrize(
    "values",
    [
        [0, 1, 1, 0],
        ["yes", "no", "yes", "no"],
        ["Active", "Inactive", "active", "inactive"],
        ["TRUE", "FALSE", "true", "false"],
    ],
)
def test_looks_binary_recognises_the_usual_spellings(values):
    is_binary, mapping = data_io.looks_binary(values)
    assert is_binary
    assert set(mapping.values()) == {0, 1}


def test_looks_binary_rejects_a_measurement_column():
    is_binary, _mapping = data_io.looks_binary([0.25, 8.0, 512.0])
    assert not is_binary


def test_label_binary_maps_words_to_one_and_zero():
    labels, mapping, excluded = data_io.label_binary(["active", "inactive", "", "maybe"])
    assert labels[0] == 1.0
    assert labels[1] == 0.0
    assert math.isnan(labels[2])
    assert math.isnan(labels[3])
    assert mapping["active"] == 1
    assert mapping["inactive"] == 0
    assert excluded["blank"] == 1
    assert excluded["not a yes/no value"] == 1


def test_threshold_for_percent_both_directions():
    values = list(range(1, 101))  # 1 to 100
    low = data_io.threshold_for_percent(values, "at or below", 10)
    high = data_io.threshold_for_percent(values, "at or above", 10)
    # The 10th percentile is near 10, the 90th near 91.
    assert 9.0 <= low <= 11.5
    assert 89.5 <= high <= 92.0


def test_threshold_for_percent_needs_numbers():
    with pytest.raises(ValueError):
        data_io.threshold_for_percent(["a", "b"], "at or below", 5)


def test_demo_labels_are_twenty_active_and_fiftynine_inactive(demo_frame):
    """The documented demo result: 20 active, 59 inactive at a threshold of 50."""
    labels, _excluded = data_io.label_numeric(
        demo_frame[demo.DEMO_ACTIVITY_COLUMN].tolist(),
        demo.DEMO_DIRECTION,
        demo.DEMO_THRESHOLD,
    )
    summary = data_io.label_summary(labels)
    assert summary["n_active"] == 20
    assert summary["n_inactive"] == 59
    assert summary["enough"] is True


def test_labels_problem_message_explains_zero_actives():
    summary = data_io.label_summary(np.array([0.0] * 20))
    message = data_io.labels_problem_message(summary)
    assert message is not None
    assert "0 molecules as active" in message


def test_labels_problem_message_is_none_when_fine():
    summary = data_io.label_summary(np.array([1.0] * 6 + [0.0] * 6))
    assert data_io.labels_problem_message(summary) is None


# ---------------------------------------------------------------------------
# Reading files
# ---------------------------------------------------------------------------

def test_read_table_rejects_an_empty_file():
    with pytest.raises(ValueError):
        data_io.read_table(b"", "empty.csv")


def test_read_table_reads_a_one_column_file():
    """
    A file with no separator at all must come back as exactly one column.

    This used to produce three columns called "Unnamed: 0", "Unnamed: 1" and
    "Unnamed: 2", because pandas' separator sniffer guesses badly when there is
    nothing to sniff.
    """
    frame = data_io.read_table(b"smiles\nCCO\nCCC\n", "one.csv")
    assert list(frame.columns) == ["smiles"]
    assert len(frame) == 2


@pytest.mark.parametrize(
    "header, expected",
    [
        (b"name,smiles,mic\n", ","),
        (b"name\tsmiles\tmic\n", "\t"),
        (b"name;smiles;mic\n", ";"),
        (b"name|smiles|mic\n", "|"),
        (b"smiles\n", ","),          # one column: fall back to a comma
        (b"", ","),                  # nothing at all: fall back to a comma
    ],
)
def test_detect_separator(header, expected):
    assert data_io.detect_separator(header) == expected


def test_read_table_reads_a_tab_separated_file():
    frame = data_io.read_table(b"smiles\tmic\nCCO\t128\nCCC\t1\n", "tabs.tsv")
    assert list(frame.columns) == ["smiles", "mic"]
    assert len(frame) == 2


def test_read_table_reads_an_xlsx_file(tmp_path):
    path = tmp_path / "little.xlsx"
    pd.DataFrame({"smiles": ["CCO", "CCC"], "mic": [128, 1]}).to_excel(path, index=False)
    frame = data_io.read_table(str(path), "little.xlsx")
    assert list(frame.columns) == ["smiles", "mic"]
    assert len(frame) == 2
    assert data_io.excel_sheet_names(str(path))


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def test_clean_molecules_drops_invalid_rows():
    frame = pd.DataFrame(
        {
            "smiles": ["CCO", "hello", "C1CC", "CCC", ""],
            "mic": [1.0, 1.0, 1.0, 100.0, 1.0],
        }
    )
    labels, _excluded = data_io.label_numeric(frame["mic"].tolist(), "at or below", 50.0)
    clean, report = data_io.clean_molecules(frame, "smiles", labels)
    assert report["total_rows"] == 5
    assert report["invalid_count"] == 2
    assert report["blank_smiles"] == 1
    assert report["final_count"] == 2
    assert set(clean["smiles"]) == {"CCO", "CCC"}
    # Invalid examples point at the row number the user sees in a spreadsheet.
    assert report["invalid_examples"][0][0] == 3


def test_clean_molecules_removes_salts():
    frame = pd.DataFrame({"smiles": ["CC(=O)O.[Na+]"], "mic": [1.0]})
    labels, _ = data_io.label_numeric(frame["mic"].tolist(), "at or below", 50.0)
    clean, _report = data_io.clean_molecules(frame, "smiles", labels)
    assert clean["smiles"].iloc[0] == "CC(=O)O"


def test_clean_molecules_keeps_one_of_agreeing_duplicates():
    # The same molecule written two different ways, with the same label.
    frame = pd.DataFrame({"smiles": ["c1ccccc1", "C1=CC=CC=C1"], "mic": [1.0, 2.0]})
    labels, _ = data_io.label_numeric(frame["mic"].tolist(), "at or below", 50.0)
    clean, report = data_io.clean_molecules(frame, "smiles", labels)
    assert report["duplicates_removed"] == 1
    assert report["conflicts_dropped"] == 0
    assert report["final_count"] == 1


def test_clean_molecules_drops_conflicting_duplicates():
    # The same molecule twice, once active and once inactive.
    frame = pd.DataFrame({"smiles": ["c1ccccc1", "C1=CC=CC=C1", "CCO"], "mic": [1.0, 100.0, 1.0]})
    labels, _ = data_io.label_numeric(frame["mic"].tolist(), "at or below", 50.0)
    clean, report = data_io.clean_molecules(frame, "smiles", labels)
    assert report["conflicts_dropped"] == 2
    assert report["final_count"] == 1
    assert clean["smiles"].iloc[0] == "CCO"


def test_clean_molecules_skips_rows_with_no_label():
    frame = pd.DataFrame({"smiles": ["CCO", "CCC"], "mic": [1.0, None]})
    labels, _ = data_io.label_numeric(frame["mic"].tolist(), "at or below", 50.0)
    clean, report = data_io.clean_molecules(frame, "smiles", labels)
    assert report["blank_label"] == 1
    assert report["final_count"] == 1


def test_demo_cleans_to_seventynine_molecules(demo_clean):
    clean, report = demo_clean
    assert report["final_count"] == 79
    assert report["invalid_count"] == 0
    assert report["duplicates_removed"] == 0
    assert report["conflicts_dropped"] == 0
    assert report["n_active"] == 20
    assert report["n_inactive"] == 59
    assert len(clean) == 79


# ---------------------------------------------------------------------------
# Fingerprints and similarity
# ---------------------------------------------------------------------------

def test_fingerprint_array_shape_and_type():
    x = features.fingerprint_array(["CCO", "c1ccccc1"], radius=2, bits=1024)
    assert x.shape == (2, 1024)
    assert x.dtype == np.uint8
    assert x.sum() > 0


def test_fingerprint_is_reproducible():
    first = features.fingerprint_array(["CCO"], radius=2, bits=2048)
    second = features.fingerprint_array(["CCO"], radius=2, bits=2048)
    assert np.array_equal(first, second)


def test_fingerprint_rejects_odd_settings():
    with pytest.raises(ValueError):
        features.get_generator(radius=7, bits=2048)
    with pytest.raises(ValueError):
        features.get_generator(radius=2, bits=333)


def test_cache_key_changes_with_settings_and_molecules():
    base = features.cache_key(["CCO", "CCC"], 2, 2048)
    assert base == features.cache_key(["CCO", "CCC"], 2, 2048)
    assert base != features.cache_key(["CCO", "CCC"], 3, 2048)
    assert base != features.cache_key(["CCO", "CCC"], 2, 1024)
    assert base != features.cache_key(["CCO", "CCCC"], 2, 2048)


def test_identical_molecules_have_similarity_one():
    bvs = features.bit_vectors(["c1ccccc1", "CCO"])
    query = features.bit_vector_one(features.mol_from_smiles("C1=CC=CC=C1"))
    sims = features.tanimoto_to_many(query, bvs)
    assert sims[0] == pytest.approx(1.0)
    assert sims[1] < 1.0


def test_nearest_neighbours_sorted_best_first():
    bvs = features.bit_vectors(["CCO", "CCCCCCCC", "c1ccccc1"])
    query = features.bit_vector_one(features.mol_from_smiles("CCO"))
    neighbours = features.nearest_neighbours(query, bvs, k=3)
    assert len(neighbours) == 3
    assert neighbours[0][0] == 0
    assert neighbours[0][1] == pytest.approx(1.0)
    similarities = [s for _i, s in neighbours]
    assert similarities == sorted(similarities, reverse=True)


def test_standardize_smiles_handles_nonsense():
    assert features.standardize_smiles("hello") is None
    assert features.standardize_smiles("C1CC") is None
    assert features.standardize_smiles("") is None
    assert features.standardize_smiles(None) is None
    assert features.standardize_smiles("CCO") == "CCO"


def test_describe_molecule_returns_sensible_numbers():
    mol = features.mol_from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")  # aspirin
    properties = features.describe_molecule(mol)
    # Aspirin's real molecular weight is about 180.16 g/mol.
    assert properties["Molecular weight"] == pytest.approx(180.16, abs=0.1)
    assert properties["Ring count"] == 1


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def test_ring_free_molecules_get_their_own_groups():
    """Ethanol, glycine and urea have no rings and must not be lumped together."""
    keys = splitting.scaffold_keys(["CCO", "NCC(=O)O", "NC(N)=O"])
    assert len(set(keys)) == 3


def _murcko_works() -> bool:
    """
    Is RDKit's scaffold decomposition usable on this machine?

    Some RDKit builds raise RuntimeError("bad any cast") from MurckoDecompose.
    When that happens the scaffold split quietly degrades, which the app now
    reports loudly — so the tests below check the reporting rather than
    pretending the chemistry worked.
    """
    keys = splitting.scaffold_keys(["Cc1ccccc1"])
    return splitting.count_broken_scaffolds(keys) == 0


def test_molecules_sharing_a_skeleton_share_a_group():
    # Benzene, toluene and aniline all reduce to a benzene scaffold.
    keys = splitting.scaffold_keys(["c1ccccc1", "Cc1ccccc1", "Nc1ccccc1"])
    if not _murcko_works():
        pytest.skip("RDKit's MurckoDecompose is broken in this build")
    assert len(set(keys)) == 1


def test_broken_scaffolds_are_counted_not_hidden(monkeypatch):
    """
    A split must never report a reassuring "0 shared scaffolds" that was built
    out of RDKit failures.

    This is the honesty guard: if the chemistry cannot be computed, the report
    has to say so instead of quietly handing every molecule its own group and
    calling the result a scaffold split.
    """

    def explode(*args, **kwargs):
        raise RuntimeError("bad any cast")

    monkeypatch.setattr(
        splitting.MurckoScaffold, "MurckoScaffoldSmiles", explode
    )

    smiles = ["c1ccccc1", "Cc1ccccc1", "Nc1ccccc1", "c1ccncc1"]
    keys = splitting.scaffold_keys(smiles)
    assert splitting.count_broken_scaffolds(keys) == 4

    labels = np.array([1, 0, 1, 0])
    report = splitting.split_report(smiles, labels, [0, 1], [2, 3], method="scaffold")
    assert report["broken_scaffolds"] == 4
    assert report["notice"] == splitting.ALL_BROKEN_NOTICE


def test_partly_broken_scaffolds_are_reported(monkeypatch):
    """Some molecules failing should still raise a warning, not pass silently."""
    real = splitting.MurckoScaffold.MurckoScaffoldSmiles
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("bad any cast")
        return real(*args, **kwargs)

    monkeypatch.setattr(splitting.MurckoScaffold, "MurckoScaffoldSmiles", flaky)

    smiles = ["c1ccccc1", "Cc1ccccc1"]
    labels = np.array([1, 0])
    report = splitting.split_report(smiles, labels, [0], [1], method="scaffold")
    assert report["broken_scaffolds"] >= 1
    assert report["notice"] is not None
    assert "could not work out the chemical skeleton" in report["notice"]


def test_scaffold_split_shares_no_scaffolds(demo_clean):
    clean, _report = demo_clean
    smiles = clean["smiles"].tolist()
    labels = clean["label"].to_numpy(dtype=int)
    split = splitting.scaffold_split(smiles, labels, test_fraction=0.2, seed=42)
    report = split["report"]
    assert report["method"] == "scaffold"
    assert report["shared_scaffolds"] == 0
    # Every molecule lands on exactly one side.
    assert report["n_train"] + report["n_test"] == len(smiles)
    assert not set(split["train_idx"]) & set(split["test_idx"])


def test_scaffold_split_respects_the_test_size(demo_clean):
    clean, _report = demo_clean
    smiles = clean["smiles"].tolist()
    labels = clean["label"].to_numpy(dtype=int)
    for fraction in (0.1, 0.2, 0.4):
        split = splitting.scaffold_split(smiles, labels, test_fraction=fraction, seed=42)
        target = len(smiles) * fraction
        # Whole scaffold groups move together, so the size is close, not exact.
        assert abs(split["report"]["n_test"] - target) <= max(5, 0.5 * target)


def test_scaffold_split_test_set_has_both_classes(demo_clean):
    clean, _report = demo_clean
    smiles = clean["smiles"].tolist()
    labels = clean["label"].to_numpy(dtype=int)
    for seed in (0, 1, 7, 42, 1234):
        split = splitting.scaffold_split(smiles, labels, test_fraction=0.2, seed=seed)
        report = split["report"]
        assert report["test_active"] >= splitting.MIN_TEST_PER_CLASS
        assert report["test_inactive"] >= splitting.MIN_TEST_PER_CLASS


def test_scaffold_split_is_reproducible(demo_clean):
    clean, _report = demo_clean
    smiles = clean["smiles"].tolist()
    labels = clean["label"].to_numpy(dtype=int)
    first = splitting.scaffold_split(smiles, labels, 0.2, 42)
    second = splitting.scaffold_split(smiles, labels, 0.2, 42)
    assert first["test_idx"] == second["test_idx"]


def test_scaffold_split_falls_back_when_impossible():
    """
    Only one molecule is active, so no scaffold split can ever put 2 actives in
    the test set. The code must give up gracefully and say so.
    """
    smiles = ["CCO", "CCC", "CCCC", "CCCCC", "c1ccccc1", "Cc1ccccc1", "CCOCC", "CCCCCC"]
    labels = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    split = splitting.scaffold_split(smiles, labels, test_fraction=0.25, seed=42)
    report = split["report"]
    assert report["method"] == "random (fallback)"
    assert report["notice"] == splitting.FALLBACK_NOTICE
    assert report["attempts"] == splitting.MAX_ATTEMPTS


def test_stratified_split_keeps_both_classes_on_both_sides():
    labels = np.array([1] * 10 + [0] * 90)
    train_idx, test_idx = splitting.stratified_split(labels, test_fraction=0.2, seed=42)
    assert len(train_idx) + len(test_idx) == 100
    assert not set(train_idx) & set(test_idx)
    assert labels[test_idx].sum() >= 1
    assert labels[train_idx].sum() >= 1
    assert (labels[test_idx] == 0).sum() >= 1


def test_make_split_honours_the_chosen_method(demo_clean):
    clean, _report = demo_clean
    smiles = clean["smiles"].tolist()
    labels = clean["label"].to_numpy(dtype=int)
    scaffold = splitting.make_split(smiles, labels, "scaffold", 0.2, 42)
    random_split = splitting.make_split(smiles, labels, "random", 0.2, 42)
    assert scaffold["report"]["method"] == "scaffold"
    assert random_split["report"]["method"] == "random (stratified)"


# ---------------------------------------------------------------------------
# Statistics: checked against numbers worked out by hand
# ---------------------------------------------------------------------------

def test_wilson_interval_against_hand_calculation():
    """
    8 successes out of 20, z = 1.96.

    p = 0.4, z^2 = 3.8416
    denominator = 1 + 3.8416/20 = 1.19208
    centre = (0.4 + 3.8416/40) / 1.19208 = 0.496040 / 1.19208 = 0.416079
    half width = 1.96 * sqrt(0.4*0.6/20 + 3.8416/1600) / 1.19208
               = 1.96 * sqrt(0.012 + 0.0024010) / 1.19208
               = 1.96 * 0.1200042 / 1.19208 = 0.197276
    so the interval is about (0.21880, 0.61336).
    """
    low, high = experiment.wilson_interval(8, 20)
    assert low == pytest.approx(0.21880, abs=1e-4)
    assert high == pytest.approx(0.61336, abs=1e-4)


def test_wilson_interval_at_the_edges():
    # Every pick a hit: the top of the interval is 1, but the bottom is not 1.
    low, high = experiment.wilson_interval(10, 10)
    assert high == pytest.approx(1.0)
    assert 0.6 < low < 1.0
    # No hits at all: the bottom is 0, the top is not.
    low, high = experiment.wilson_interval(0, 10)
    assert low == pytest.approx(0.0)
    assert 0.0 < high < 0.4
    # Nothing tested at all.
    assert experiment.wilson_interval(0, 0) == (0.0, 0.0)


def test_hypergeometric_p_value_against_hand_calculation():
    """
    A bag of 100 molecules, 10 of them active. We draw 10 and get 5 actives.

    P(at least 5) is the sum of P(exactly k) for k = 5 to 10, where each term
    is C(10, k) * C(90, 10 - k) / C(100, 10).

    C(100, 10) = 17,310,309,456,440, and the numerators are:

        k = 5 :  252 * 43,949,268 = 11,075,215,536
        k = 6 :  210 *  2,555,190 =    536,589,900
        k = 7 :  120 *    117,480 =     14,097,600
        k = 8 :   45 *      4,005 =        180,225
        k = 9 :   10 *         90 =            900
        k = 10:    1 *          1 =              1
                                    --------------
                           total  = 11,626,084,162

    11,626,084,162 / 17,310,309,456,440 = 0.000671628...

    So getting 5 of your 10 picks right, when only 1 in 10 molecules is active,
    happens by luck about 7 times in ten thousand.
    """
    p = experiment.hypergeometric_p_value(5, 100, 10, 10)
    assert p == pytest.approx(11_626_084_162 / 17_310_309_456_440, rel=1e-9)
    assert p == pytest.approx(0.000671628, rel=1e-5)


def test_hypergeometric_p_value_simple_cases():
    # Drawing all 100 must find all 10 actives, so this is certain.
    assert experiment.hypergeometric_p_value(10, 100, 10, 100) == pytest.approx(1.0)
    # Finding 0 or more is always certain.
    assert experiment.hypergeometric_p_value(0, 100, 10, 10) == pytest.approx(1.0)
    # Nothing is active, so nothing can be found.
    assert experiment.hypergeometric_p_value(1, 100, 0, 10) == pytest.approx(1.0)
    # Finding every active in a tiny draw is very unlikely.
    assert experiment.hypergeometric_p_value(10, 1000, 10, 10) < 1e-20


def test_enrichment_against_hand_calculation():
    """
    20 test molecules, 4 of them active, so the random hit rate is 0.2.
    The top 5 contain 3 actives, so the model hit rate is 0.6.
    Enrichment = 0.6 / 0.2 = 3.0.
    """
    # Scores arranged so the top 5 are rows 0..4, of which 3 are active.
    scores = np.array([0.9, 0.85, 0.8, 0.75, 0.7] + [0.1] * 15)
    y_true = np.array([1, 1, 1, 0, 0] + [1] + [0] * 14)
    result = experiment.model_vs_random(scores, y_true, 5)
    assert result["actives_total"] == 4
    assert result["random_hit_rate"] == pytest.approx(0.2)
    assert result["actives_found"] == 3
    assert result["model_hit_rate"] == pytest.approx(0.6)
    assert result["enrichment"] == pytest.approx(3.0)
    assert result["expected_random"] == pytest.approx(1.0)


def test_enrichment_is_none_when_nothing_is_active():
    scores = np.array([0.9, 0.5, 0.1])
    y_true = np.array([0, 0, 0])
    result = experiment.model_vs_random(scores, y_true, 2)
    assert result["enrichment"] is None
    assert result["random_hit_rate"] == 0.0
    assert "nothing to compare" in experiment.enrichment_text(result)


def test_results_change_when_n_changes():
    """Buying more molecules must change the hit rate, not leave it frozen."""
    scores = np.array([0.99, 0.98, 0.97, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01])
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    at_3 = experiment.model_vs_random(scores, y_true, 3)
    at_6 = experiment.model_vs_random(scores, y_true, 6)
    at_10 = experiment.model_vs_random(scores, y_true, 10)
    assert at_3["model_hit_rate"] == pytest.approx(1.0)
    assert at_6["model_hit_rate"] == pytest.approx(0.5)
    assert at_10["model_hit_rate"] == pytest.approx(0.3)
    # With every molecule bought, the model cannot beat random picking.
    assert at_10["enrichment"] == pytest.approx(1.0)
    assert at_3["enrichment"] > at_6["enrichment"] > at_10["enrichment"]


def test_n_pick_is_clamped_to_the_test_set():
    scores = np.array([0.9, 0.1])
    y_true = np.array([1, 0])
    assert experiment.model_vs_random(scores, y_true, 99)["n_pick"] == 2
    assert experiment.model_vs_random(scores, y_true, 0)["n_pick"] == 1


def test_model_vs_random_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        experiment.model_vs_random(np.array([0.5, 0.5]), np.array([1]), 1)


def test_a_useless_model_gives_enrichment_near_one():
    """
    Random scores must not produce an impressive-looking enrichment.

    Averaged over many random orderings the enrichment has to sit close to 1.
    This is the test that would catch the app accidentally flattering itself.
    """
    rng = np.random.default_rng(0)
    n = 2000
    y_true = np.zeros(n, dtype=int)
    y_true[rng.choice(n, size=200, replace=False)] = 1  # 10% active

    enrichments = []
    for trial in range(40):
        scores = np.random.default_rng(trial).random(n)
        result = experiment.model_vs_random(scores, y_true, 200)
        enrichments.append(result["enrichment"])

    assert np.mean(enrichments) == pytest.approx(1.0, abs=0.12)


def test_rank_order_is_stable_on_ties():
    """Equal scores keep their original order, so results are reproducible."""
    scores = np.array([0.5, 0.9, 0.5, 0.9, 0.1])
    assert experiment.rank_order(scores).tolist() == [1, 3, 0, 2, 4]


def test_cumulative_gain_starts_at_zero_and_ends_at_one_hundred():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    y_true = np.array([1, 0, 1, 0])
    x, y = experiment.cumulative_gain(scores, y_true)
    assert x[0] == 0.0 and y[0] == 0.0
    assert x[-1] == pytest.approx(100.0)
    assert y[-1] == pytest.approx(100.0)
    # A good ranking climbs above the diagonal.
    assert y[1] > x[1]


def test_cumulative_gain_with_no_actives():
    x, y = experiment.cumulative_gain(np.array([0.5, 0.4]), np.array([0, 0]))
    assert x.tolist() == [0.0]
    assert y.tolist() == [0.0]


def test_enrichment_text_uses_the_real_numbers():
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.05])
    y_true = np.array([1, 1, 0, 0, 0])
    result = experiment.model_vs_random(scores, y_true, 2)
    text = experiment.enrichment_text(result)
    assert "top 2 picks" in text
    assert "2 were truly active" in text
    assert "100%" in text


def test_verdict_text_is_honest_about_a_weak_model():
    # A model that ranks the actives last.
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    y_true = np.array([1, 1, 0, 0])
    result = experiment.model_vs_random(scores, y_true, 2)
    assert "no better than guessing" in experiment.verdict_text(result)


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------

def test_training_and_scoring_the_demo_data(demo_clean):
    clean, _report = demo_clean
    smiles = clean["smiles"].tolist()
    labels = clean["label"].to_numpy(dtype=int)
    split = splitting.scaffold_split(smiles, labels, 0.2, 42)

    train_smiles = [smiles[i] for i in split["train_idx"]]
    test_smiles = [smiles[i] for i in split["test_idx"]]
    y_train = labels[split["train_idx"]]
    y_test = labels[split["test_idx"]]

    bundle = modeling.fit_with_safeguard(
        train_smiles, y_train, model_kind=modeling.MODEL_RANDOM_FOREST,
        radius=2, bits=2048, n_estimators=100, seed=42,
    )
    assert bundle["bits"] == 2048
    assert bundle["safeguard_notice"] is None
    assert bundle["model"].n_estimators == 100

    x_test = features.fingerprint_array(test_smiles, 2, bundle["bits"])
    scores = modeling.predict_scores(bundle["model"], x_test)
    assert scores.shape == (len(test_smiles),)
    assert np.all((scores >= 0.0) & (scores <= 1.0))

    evaluation = modeling.evaluate(y_test, scores, 0.5)
    assert evaluation["single_class"] is False
    assert 0.0 <= evaluation["roc_auc"] <= 1.0
    assert 0.0 <= evaluation["pr_auc"] <= 1.0
    assert evaluation["baseline_pr_auc"] == pytest.approx(
        y_test.sum() / len(y_test)
    )
    assert evaluation["confusion"]["true_positive"] + evaluation["confusion"][
        "false_negative"
    ] == int(y_test.sum())


def test_logistic_regression_also_trains(demo_clean):
    clean, _report = demo_clean
    smiles = clean["smiles"].tolist()[:40]
    labels = clean["label"].to_numpy(dtype=int)[:40]
    bundle = modeling.fit_with_safeguard(
        smiles, labels, model_kind=modeling.MODEL_LOGISTIC, radius=2, bits=1024, seed=42
    )
    assert bundle["n_estimators"] is None
    scores = modeling.predict_scores(
        bundle["model"], features.fingerprint_array(smiles, 2, 1024)
    )
    assert scores.shape == (40,)


def test_training_is_reproducible_with_the_same_seed(demo_clean):
    clean, _report = demo_clean
    smiles = clean["smiles"].tolist()
    labels = clean["label"].to_numpy(dtype=int)
    x = features.fingerprint_array(smiles, 2, 1024)
    first = modeling.train_random_forest(x, labels, n_estimators=50, seed=42)
    second = modeling.train_random_forest(x, labels, n_estimators=50, seed=42)
    assert np.array_equal(
        modeling.predict_scores(first, x), modeling.predict_scores(second, x)
    )


def test_progress_callback_is_called_while_training(demo_clean):
    clean, _report = demo_clean
    smiles = clean["smiles"].tolist()
    labels = clean["label"].to_numpy(dtype=int)
    x = features.fingerprint_array(smiles, 2, 1024)
    seen = []
    modeling.train_random_forest(
        x, labels, n_estimators=75, seed=42, progress_cb=lambda d, t: seen.append((d, t))
    )
    # 75 trees in chunks of 25 means three updates.
    assert seen == [(25, 75), (50, 75), (75, 75)]


def test_evaluate_handles_a_single_class_test_set():
    evaluation = modeling.evaluate(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
    assert evaluation["single_class"] is True
    assert evaluation["roc_auc"] is None
    assert evaluation["pr_auc"] is None
    assert evaluation["confusion"] is None


def test_evaluate_on_a_perfect_model():
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.99, 0.98, 0.02, 0.01])
    evaluation = modeling.evaluate(y_true, scores, 0.5)
    assert evaluation["roc_auc"] == pytest.approx(1.0)
    assert evaluation["precision"] == pytest.approx(1.0)
    assert evaluation["recall"] == pytest.approx(1.0)
    assert evaluation["f1"] == pytest.approx(1.0)
    assert evaluation["baseline_pr_auc"] == pytest.approx(0.5)


def test_score_band_reports_a_percentile():
    reference = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    band = modeling.score_band(0.45, reference)
    assert band["percentile"] == pytest.approx(80.0)
    assert "Higher than 80%" in band["text"]
    empty = modeling.score_band(0.5, np.zeros(0))
    assert empty["percentile"] is None


def test_safeguard_constants_match_the_notice():
    assert modeling.SAFEGUARD_BITS == 1024
    assert modeling.SAFEGUARD_TREES == 150
    assert "1024 bits and 150 trees" in modeling.SAFEGUARD_NOTICE


def test_safeguard_retries_once_on_memory_error(monkeypatch, demo_clean):
    """
    Force the first fingerprint build to raise MemoryError and check that the
    retry really happens with 1024 bits, and that the notice is set.
    """
    clean, _report = demo_clean
    smiles = clean["smiles"].tolist()
    labels = clean["label"].to_numpy(dtype=int)

    real_fingerprint_array = modeling.fingerprint_array
    calls = []

    def flaky(smiles_list, radius=2, bits=2048, progress_cb=None):
        calls.append(bits)
        if len(calls) == 1:
            raise MemoryError("pretend we ran out of memory")
        return real_fingerprint_array(smiles_list, radius=radius, bits=bits)

    monkeypatch.setattr(modeling, "fingerprint_array", flaky)

    bundle = modeling.fit_with_safeguard(
        smiles, labels, model_kind=modeling.MODEL_RANDOM_FOREST,
        radius=2, bits=2048, n_estimators=300, seed=42,
    )
    assert calls == [2048, 1024]
    assert bundle["bits"] == modeling.SAFEGUARD_BITS
    assert bundle["n_estimators"] == modeling.SAFEGUARD_TREES
    assert bundle["safeguard_notice"] == modeling.SAFEGUARD_NOTICE


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------

def test_atom_contributions_has_one_number_per_atom(demo_clean):
    clean, _report = demo_clean
    smiles = clean["smiles"].tolist()
    labels = clean["label"].to_numpy(dtype=int)
    bundle = modeling.fit_with_safeguard(
        smiles, labels, radius=2, bits=1024, n_estimators=50, seed=42
    )
    from src import explain

    mol = features.mol_from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    contributions = explain.atom_contributions(bundle["model"], mol, radius=2, bits=1024)
    assert contributions.shape == (mol.GetNumAtoms(),)
    assert np.all(np.isfinite(contributions))


def test_explain_molecule_never_raises(demo_clean):
    """Even a hostile input must come back as a friendly message, not a crash."""
    from src import explain

    clean, _report = demo_clean
    smiles = clean["smiles"].tolist()
    labels = clean["label"].to_numpy(dtype=int)
    bundle = modeling.fit_with_safeguard(
        smiles, labels, radius=2, bits=1024, n_estimators=50, seed=42
    )

    good = explain.explain_molecule(
        bundle["model"], features.mol_from_smiles("CCOc1ccccc1"), radius=2, bits=1024
    )
    assert good["ok"] in (True, False)
    if good["ok"]:
        assert good["svg"].strip().startswith("<?xml") or "<svg" in good["svg"]

    # A completely wrong object instead of a molecule.
    broken = explain.explain_molecule(bundle["model"], "not a molecule", radius=2, bits=1024)
    assert broken["ok"] is False
    assert broken["message"] == explain.UNAVAILABLE_MESSAGE


def test_legend_svg_is_svg():
    from src import explain

    assert explain.legend_svg().startswith("<svg")


# ---------------------------------------------------------------------------
# Charts and drawings
# ---------------------------------------------------------------------------

def test_molecule_drawing_produces_svg():
    from src import viz

    svg = viz.mol_svg("CCO")
    assert svg is not None
    assert "<svg" in svg
    assert viz.mol_svg("hello") is None


def test_charts_build_and_close():
    from src import viz

    fig, caption = viz.label_bar_chart(20, 59)
    assert "20 of 79" in caption
    viz.close(fig)

    fig, caption = viz.measurement_histogram(
        np.array([0.25, 1.0, 32.0, 128.0, 512.0]), 50.0, "at or below"
    )
    assert "log scale" in caption
    viz.close(fig)

    result = experiment.model_vs_random(
        np.array([0.9, 0.8, 0.1, 0.05]), np.array([1, 1, 0, 0]), 2
    )
    fig, caption = viz.hit_rate_chart(result)
    assert "truly active" in caption
    viz.close(fig)

    x, y = experiment.cumulative_gain(np.array([0.9, 0.8, 0.1]), np.array([1, 0, 1]))
    fig, caption = viz.gain_curve_chart(x, y)
    viz.close(fig)

    fig, caption = viz.roc_chart([0.0, 0.5, 1.0], [0.0, 0.8, 1.0], 0.9)
    assert "0.900" in caption or "0.9" in caption
    viz.close(fig)

    fig, caption = viz.funnel_chart(
        [("Uploaded", 100), ("Valid", 90), ("Scored high", 20), ("Shortlist", 5)]
    )
    assert "100" in caption
    viz.close(fig)


def test_demo_titles_are_marked():
    from src import viz

    assert viz.title_with_demo("Hit rate", True).endswith("(Demo data)")
    assert viz.title_with_demo("Hit rate", False) == "Hit rate"


# ---------------------------------------------------------------------------
# No fake numbers anywhere
# ---------------------------------------------------------------------------

def test_source_has_no_hardcoded_metrics():
    """
    A guard against the app ever displaying a number it did not compute.

    Allowed: the fixed article facts (which live in one ARTICLE dictionary in
    app.py and are labelled as coming from the paper) and the demo data. This
    test checks that no module in src/ contains a suspicious literal score.
    """
    import re

    suspicious = re.compile(
        r"(roc_auc|pr_auc|hit_rate|enrichment|accuracy|precision|recall|f1)"
        r"\s*=\s*0?\.\d+",
        re.IGNORECASE,
    )
    for path in sorted((ROOT / "src").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        matches = suspicious.findall(text)
        assert not matches, f"{path.name} looks like it hardcodes a metric: {matches}"
