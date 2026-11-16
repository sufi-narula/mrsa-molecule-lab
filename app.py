"""
MRSA Molecule Lab: a mini AI antibiotic predictor.

A Streamlit app that goes with my TKS learn article, "Can AI discover a new
antibiotic to cure MRSA?". Load a table of molecules with lab measurements,
train a small model, and see whether ranking molecules with it would have beaten
picking them at random.

This file is only the user interface. All of the real work lives in src/ so it
can be tested without starting a browser.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from src import HONESTY_FOOTER
from src import data_io, demo, experiment, explain, features, modeling, splitting, viz

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="MRSA Molecule Lab", layout="wide")

# The fixed facts from the 2023 Nature paper, quoted in the About tab and in the
# "Compare with the article" card. These are facts from the published study, not
# results produced by this app.
ARTICLE = {
    # The 2023 Nature screen and AI run
    "screened": 39_312,
    "active_in_screen": 512,
    "screen_hit_rate_percent": 1.3,
    "scored": 12_076_365,
    "filter_1": 3_646,
    "filter_2": 2_209,
    "filter_3": 1_261,
    "tested": 283,
    "new_actives": 21,
    "new_tested": 241,
    "new_hit_rate_percent": 9.0,
    # Why any of this matters
    "amr_deaths_2019": 1_270_000,
    "mrsa_deaths_2019": 100_000,
    "approval_drought_years": 45,
    # What the two leading candidates actually did
    "mouse_log_reduction": 1.2,
    # The 2025 generative AI follow-up: most designs could not be made
    "gonorrhea_designs": 80,
    "gonorrhea_made": 2,
    "mrsa_designs": 90,
    "mrsa_made": 22,
}

EXAMPLE_MOLECULES = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ciprofloxacin": "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",
    "Penicillin G": "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",
    "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
}


# ---------------------------------------------------------------------------
# Small helpers that keep the app working across Streamlit versions
# ---------------------------------------------------------------------------

def show_df(frame, **kwargs) -> None:
    """
    Show a table that fills the width of its column.

    Streamlit renamed this argument (use_container_width became width="stretch"),
    so try the current spelling first and fall back to the older one. That keeps
    the app free of deprecation warnings on new versions and still working on
    older ones.
    """
    try:
        st.dataframe(frame, width="stretch", **kwargs)
    except Exception:
        st.dataframe(frame, use_container_width=True, **kwargs)


def show_svg(svg: str) -> None:
    """
    Put an SVG drawing on the page.

    RDKit puts an XML declaration at the top of its SVG output, which confuses
    the HTML renderer, so strip it first.
    """
    if not svg:
        return
    cleaned = "\n".join(
        line for line in svg.splitlines() if not line.strip().startswith("<?xml")
    )
    try:
        st.html(cleaned)
    except Exception:
        st.markdown(cleaned, unsafe_allow_html=True)


def show_chart(fig, caption: str) -> None:
    """Draw a chart, write its plain-language caption, then free the memory."""
    st.pyplot(fig)
    if caption:
        st.caption(caption)
    viz.close(fig)


def footer() -> None:
    """The honesty footer, shown at the bottom of every tab."""
    st.divider()
    st.caption(HONESTY_FOOTER)


def molecule_grid(rows, columns: int = 4, size: int = 210) -> None:
    """
    A grid of molecule drawings.

    rows is a list of (smiles, caption) pairs. Anything that cannot be drawn is
    skipped quietly rather than leaving a gap or raising.
    """
    rows = list(rows)
    for start in range(0, len(rows), columns):
        chunk = rows[start : start + columns]
        cols = st.columns(columns)
        for col, (smiles, caption) in zip(cols, chunk):
            with col:
                svg = viz.mol_svg(smiles, size=size, caption=str(caption))
                if svg:
                    show_svg(svg)
                else:
                    st.caption("Could not draw this molecule.")


# ---------------------------------------------------------------------------
# Session state: everything the app remembers between clicks
# ---------------------------------------------------------------------------

DEFAULT_STATE = {
    "raw_df": None,          # the table exactly as it was loaded
    "source_name": None,     # the file name, shown to the user
    "is_demo": False,        # whether the demo data is loaded
    "sheet_names": None,     # for .xlsx files
    "file_bytes": None,      # kept so a different sheet can be read later
    "smiles_col": None,
    "activity_col": None,
    "name_col": None,
    "direction": "at or below",
    "threshold": None,
    "percent_target": 5.0,
    "clean_df": None,        # the cleaned, labelled molecules
    "clean_report": None,
    "bundle": None,          # the trained model and its settings
    "split": None,
    "test_scores": None,
    "evaluation": None,
    "train_bitvectors": None,
    "train_bv_key": None,
    "batch_result": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_downstream(keep_data: bool = True) -> None:
    """
    Forget anything that depended on data we just replaced.

    Loading a new file has to clear the old model, otherwise the Results tab
    would quietly show yesterday's numbers next to today's data.
    """
    for key in (
        "clean_df",
        "clean_report",
        "bundle",
        "split",
        "test_scores",
        "evaluation",
        "train_bitvectors",
        "train_bv_key",
        "batch_result",
    ):
        st.session_state[key] = DEFAULT_STATE[key]
    if not keep_data:
        for key in ("raw_df", "source_name", "is_demo", "sheet_names", "file_bytes"):
            st.session_state[key] = DEFAULT_STATE[key]


def demo_banner() -> None:
    """The warning shown wherever demo results appear."""
    if st.session_state.is_demo:
        st.warning(demo.DEMO_BANNER, icon="⚠️")


def have_model() -> bool:
    return st.session_state.bundle is not None and st.session_state.evaluation is not None


def training_bit_vectors():
    """
    The RDKit bit vectors for the training molecules, built once and reused.

    They are needed for every similarity search on the Predict and Batch tabs,
    and building them for a large dataset takes a few seconds, so they are
    cached against the settings that produced them.
    """
    bundle = st.session_state.bundle
    if bundle is None:
        return []
    train_smiles = bundle["train_smiles"]
    key = features.cache_key(train_smiles, bundle["radius"], bundle["bits"])
    if st.session_state.train_bv_key != key:
        with st.spinner("Preparing molecule fingerprints for the similarity search…"):
            st.session_state.train_bitvectors = features.bit_vectors(
                train_smiles, radius=bundle["radius"], bits=bundle["bits"]
            )
            st.session_state.train_bv_key = key
    return st.session_state.train_bitvectors


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("MRSA Molecule Lab")
st.markdown(
    "#### A mini AI antibiotic predictor for my article: "
    "Can AI discover a new antibiotic to cure MRSA?"
)

tab_about, tab_data, tab_train, tab_results, tab_predict, tab_batch = st.tabs(
    ["About", "1 Data", "2 Train", "3 Results", "4 Predict", "5 Batch Screen"]
)


# ===========================================================================
# About
# ===========================================================================

with tab_about:
    st.header("What this app is")
    st.markdown(
        f"""
**The question in my article:** can AI discover a new antibiotic to cure MRSA,
the “superbug” staph infection that resists most common antibiotics?

MRSA is methicillin-resistant *Staphylococcus aureus*, a common skin bacterium
that survives many antibiotics because it carries the **mecA** gene. That gene
makes an altered protein called **PBP2a**, which methicillin barely binds to, so
the drug has almost nothing to grab onto.

This is not a small problem. Drug-resistant infections directly caused about
**{ARTICLE['amr_deaths_2019'] / 1_000_000:.2f} million deaths** worldwide in
2019, and MRSA was responsible for **over {ARTICLE['mrsa_deaths_2019']:,}** of
them. Meanwhile only a few dozen new antibiotics have been approved in
**{ARTICLE['approval_drought_years']} years**.

In 2023, researchers at MIT and the Broad Institute trained an explainable AI on
**{ARTICLE['screened']:,}** lab-tested compounds, of which
**{ARTICLE['active_in_screen']}** (about **{ARTICLE['screen_hit_rate_percent']}%**)
stopped staph growing. They used it to score about **12 million** new compounds,
bought and tested **{ARTICLE['tested']}**, and **{ARTICLE['new_actives']}
brand-new compounds worked** — a hit rate of about
**{ARTICLE['new_hit_rate_percent']:.0f}%** against
**{ARTICLE['screen_hit_rate_percent']}%**.

Two leading candidates came from one new chemical family. They killed MRSA in
laboratory dishes and reduced it in infected mice by about
**{ARTICLE['mouse_log_reduction']} logs**, and resistance appeared slowly.

**My answer:** AI is already great at finding candidates, but a candidate is not
yet a cure. It narrows millions of molecules down to a shortlist a real
laboratory can afford to test, and it does that far faster than brute force. But
every one of those molecules still has to survive years of safety testing,
animal studies and clinical trials, and most will not. AI moves the bottleneck;
it does not remove it.

**What this app does:** it is the small, honest version of that 2023 pipeline,
reproduced in miniature. You load a table of molecules with laboratory
measurements, it cleans them and trains a simple model, tests that model on
molecules it has never seen, and then asks the question that actually matters to
a laboratory: if we could only afford to buy and test a handful of molecules,
would the model’s shortlist have found more real antibiotics than picking at
random? The **4 Predict** tab also reproduces the “explainable AI” idea in
miniature, by showing which atoms pushed each prediction up or down.
        """
    )

    st.subheader("Key facts from the study")
    st.markdown(
        f"""
From Wong and colleagues, published in *Nature* in 2023:

- They screened **{ARTICLE['screened']:,} compounds** against *Staphylococcus
  aureus*, and **{ARTICLE['active_in_screen']}** of them
  (about **{ARTICLE['screen_hit_rate_percent']}%**) stopped it growing.
- Their model then scored **{ARTICLE['scored']:,} compounds**.
- Filters narrowed those down to **{ARTICLE['filter_1']:,}**, then
  **{ARTICLE['filter_2']:,}**, then **{ARTICLE['filter_3']:,}** new-looking
  molecules.
- The team bought and tested **{ARTICLE['tested']} compounds**.
- **{ARTICLE['new_actives']} brand-new compounds** stopped *S. aureus*.
- About **{ARTICLE['new_hit_rate_percent']:.0f}%** of the new-looking compounds
  tested were active, against **{ARTICLE['screen_hit_rate_percent']}%** in the
  original screen.
- The two leading candidates reduced MRSA in infected mice by about
  **{ARTICLE['mouse_log_reduction']} logs**, and resistance appeared slowly.

**The 2025 follow-up, and why it is a reality check.** The same group used
generative AI to design brand-new molecules aimed at gonorrhea and MRSA. Designing
them was the easy part. Only **{ARTICLE['gonorrhea_made']} of
{ARTICLE['gonorrhea_designs']}** gonorrhea designs and
**{ARTICLE['mrsa_made']} of about {ARTICLE['mrsa_designs']}** MRSA designs could
actually be made in a laboratory. An AI can draw a molecule that no chemist knows
how to build.
        """
    )

    st.subheader("The biggest challenges")
    st.markdown(
        """
- **Mice are not humans.** Working in an infected mouse is a real result, and it
  is still a long way from working in a person.
- **Antibiotics earn little money.** A drug you take for ten days, and that
  doctors deliberately save for emergencies, is a hard business case.
- **Many designs can’t be built.** See the 2025 numbers above.
- **Good data is scarce.** These models learn from laboratory screens, and there
  are far fewer of those than you would hope.
- **Bacteria keep evolving.** Resistance appearing slowly is not resistance
  never appearing.
        """
    )

    st.subheader("Teams to watch")
    st.markdown(
        """
- The **Collins Lab** at MIT.
- **Phare Bio**, the non-profit spun out to push these candidates forward.
- The **Stokes Lab** and **Stoked Bio** at McMaster.
        """
    )

    st.subheader("Glossary")
    glossary = [
        ("Antibiotic", "A medicine that kills bacteria or stops them growing. It does nothing to viruses."),
        ("MRSA", "Methicillin-resistant Staphylococcus aureus: a staph bacterium that survives many of the antibiotics doctors would normally reach for."),
        ("mecA and PBP2a", "mecA is the gene that makes MRSA resistant. It builds an altered protein, PBP2a, that methicillin barely binds to, so the antibiotic has almost nothing to grab onto."),
        ("Log reduction", "A way of counting how many bacteria were killed. 1 log means ten times fewer, 2 logs means a hundred times fewer. The 2023 candidates managed about 1.2 logs in infected mice."),
        ("Synthesizability", "Whether a molecule can actually be made in a laboratory. An AI can happily draw something no chemist knows how to build, which is exactly what happened to most of the 2025 designs."),
        ("SMILES", "A way of writing a molecule as a line of text, so a computer can read it. Aspirin is CC(=O)OC1=CC=CC=C1C(=O)O."),
        ("Molecular fingerprint", "A long row of 0s and 1s saying which small chemical pieces a molecule contains. This is how a model “reads” a molecule."),
        ("Training and test data", "The model learns from the training molecules. The test molecules are hidden from it, so scoring them is a fair exam rather than a memory test."),
        ("Scaffold split", "Splitting molecules by their core skeleton, so the test set contains chemical shapes the model has never seen. Harder, and much more honest, than splitting at random."),
        ("ROC-AUC", "Pick one active and one inactive molecule at random: this is how often the model gives the active one the higher score. 0.5 is guessing, 1.0 is perfect."),
        ("PR-AUC", "A score that stays meaningful when actives are rare. Compare it against the share of actives in the test set, which is what “no skill” would get."),
        ("Hit rate", "Out of the molecules you tested, the share that really were active."),
        ("Enrichment", "How many times better the model’s hit rate is than the hit rate of picking molecules at random. This is the number a laboratory cares about."),
        ("Explainable AI", "Methods that show which parts of the input a model reacted to, instead of only giving an answer."),
        ("Black box", "A model that gives an answer without any way to see why. Useful, but hard to trust and hard to learn from."),
    ]
    for term, meaning in glossary:
        st.markdown(f"**{term}.** {meaning}")

    st.subheader("Limitations")
    st.markdown(
        """
- One dataset, and one kind of bacteria.
- Results come from a laboratory dish, not from people.
- This is a deliberately simple model, not the published deep learning one.
- Nothing here checks whether a molecule is toxic, whether it can actually be
  made, or whether it would ever reach the infection in a human body.
- A high score is a reason to run an experiment, never evidence of a medicine.
        """
    )

    st.subheader("Credits")
    st.markdown(
        """
- Wong, F. and colleagues (2023). *Discovery of a structural class of
  antibiotics with explainable deep learning*. Nature.
- Stokes, J. M. and colleagues (2020). *A deep learning approach to antibiotic
  discovery*. Cell. (The halicin study.)
- The same MIT group’s 2025 generative-AI design work on gonorrhea and MRSA,
  as described in my article.
- RDKit, for everything to do with chemistry.
- scikit-learn, for the machine learning.
- Streamlit, for the app itself.
        """
    )
    footer()


# ===========================================================================
# 1 Data
# ===========================================================================

with tab_data:
    st.header("1. Load your molecules")
    st.markdown(
        "Give the app a table with one column of molecules written as SMILES text "
        "and one column of laboratory measurements. Or try the built-in demo data "
        "to see how everything works."
    )
    st.caption(
        "The real 2023 screen (about 39,000 molecules tested against "
        "*Staphylococcus aureus*) is published with the paper’s own code at "
        "github.com/felixjwong/antibioticsai. Download it yourself and check its "
        "licence before using it — nothing real is bundled with this app."
    )

    load_left, load_right = st.columns([3, 2])

    with load_left:
        uploaded = st.file_uploader(
            "Upload a data file",
            type=["csv", "tsv", "txt", "xlsx"],
            help=(
                "A spreadsheet or text table. It needs a column of SMILES strings "
                "and a column of measurements, for example MIC values."
            ),
            key="file_uploader",
        )

    with load_right:
        st.write("")
        st.write("")
        if st.button("Load DEMO dataset", key="load_demo", type="secondary"):
            try:
                frame = demo.load_demo_training()
            except Exception as exc:
                st.error(f"I could not load the demo data: {exc}")
            else:
                reset_downstream()
                st.session_state.raw_df = frame
                st.session_state.source_name = demo.TRAINING_PATH.name
                st.session_state.is_demo = True
                st.session_state.sheet_names = None
                st.session_state.file_bytes = None
                # The demo file has known columns, so preselect them.
                st.session_state.smiles_col = demo.DEMO_SMILES_COLUMN
                st.session_state.activity_col = demo.DEMO_ACTIVITY_COLUMN
                st.session_state.name_col = demo.DEMO_NAME_COLUMN
                st.session_state.direction = demo.DEMO_DIRECTION
                st.session_state.threshold = demo.DEMO_THRESHOLD

    # Handle a newly uploaded file.
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        changed = (
            st.session_state.source_name != uploaded.name
            or st.session_state.file_bytes != file_bytes
        )
        if changed:
            try:
                sheets = (
                    data_io.excel_sheet_names(file_bytes)
                    if uploaded.name.lower().endswith((".xlsx", ".xlsm", ".xltx"))
                    else None
                )
                frame = data_io.read_table(
                    file_bytes, uploaded.name, sheet_name=sheets[0] if sheets else None
                )
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"I could not read that file: {exc}")
            else:
                reset_downstream()
                st.session_state.raw_df = frame
                st.session_state.source_name = uploaded.name
                st.session_state.is_demo = False
                st.session_state.sheet_names = sheets
                st.session_state.file_bytes = file_bytes
                st.session_state.smiles_col = None
                st.session_state.activity_col = None
                st.session_state.name_col = None
                st.session_state.threshold = None

    # Let the user choose a different sheet of an Excel file.
    if st.session_state.sheet_names and len(st.session_state.sheet_names) > 1:
        sheets = st.session_state.sheet_names
        picked = st.selectbox(
            "Which sheet of that workbook?",
            sheets,
            index=0,
            help="Your Excel file has more than one sheet.",
        )
        try:
            frame = data_io.read_table(
                st.session_state.file_bytes, st.session_state.source_name, sheet_name=picked
            )
        except Exception as exc:
            st.error(f"I could not read that sheet: {exc}")
        else:
            if not frame.equals(st.session_state.raw_df):
                reset_downstream()
                st.session_state.raw_df = frame
                st.session_state.smiles_col = None
                st.session_state.activity_col = None
                st.session_state.threshold = None

    raw = st.session_state.raw_df

    if raw is None:
        st.info(
            "No data loaded yet. Upload a file above, or click **Load DEMO dataset** "
            "to try the app straight away.",
            icon="📄",
        )
    else:
        demo_banner()
        st.success(
            f"Loaded **{st.session_state.source_name}**: {len(raw):,} rows and "
            f"{raw.shape[1]} columns."
        )
        st.caption("The first 10 rows:")
        show_df(raw.head(10))

        # -------------------------------------------------------------------
        # Column detection
        # -------------------------------------------------------------------
        st.subheader("Which column is which?")

        detected_smiles, smiles_scores = data_io.detect_smiles_column(raw)
        columns = list(raw.columns)

        if detected_smiles is None and st.session_state.smiles_col is None:
            st.error(
                "I couldn’t find a column of molecules (SMILES). "
                "Pick one manually or check your file."
            )

        current_smiles = st.session_state.smiles_col
        if current_smiles not in columns:
            current_smiles = detected_smiles if detected_smiles in columns else columns[0]
        smiles_choice = st.selectbox(
            "Column of molecules (SMILES)",
            columns,
            index=columns.index(current_smiles),
            help=(
                "SMILES is a molecule written as text, for example "
                "CC(=O)OC1=CC=CC=C1C(=O)O for aspirin. A column of names such as "
                "“aspirin” will not work."
            ),
        )
        st.session_state.smiles_col = smiles_choice

        fraction, checked = smiles_scores.get(smiles_choice, (0.0, 0))
        if checked == 0:
            st.error(f"Column “{smiles_choice}” is empty, so I cannot read molecules from it.")
        elif fraction >= 0.8:
            st.success(
                f"Detected SMILES column: {smiles_choice} "
                f"({fraction * 100:.0f}% of the first {checked} values are valid molecules)"
            )
        else:
            st.error(
                f"Only {fraction * 100:.0f}% of the values in “{smiles_choice}” are valid "
                "molecules, so this is probably the wrong column. "
                + (
                    f"“{detected_smiles}” looks like the right one."
                    if detected_smiles
                    else "None of the columns in this file look like molecules."
                )
            )

        detected_activity, candidates, activity_scores = data_io.detect_activity_column(
            raw, smiles_choice
        )

        other_columns = [c for c in columns if c != smiles_choice]
        if not other_columns:
            st.error(
                "This file has only one column, so there is no measurement column to "
                "learn from. A training file needs molecules **and** a measurement."
            )
            activity_choice = None
        elif detected_activity is None:
            st.error(
                "I couldn’t find a column of measurements. A training file needs a "
                "column of numbers (such as MIC values) or yes/no labels, next to the "
                "molecules. The columns I looked at were: "
                + ", ".join(
                    f"“{c}” ({activity_scores.get(c, (0.0, 0))[0] * 100:.0f}% usable values)"
                    for c in other_columns
                )
                + "."
            )
            activity_choice = None
        else:
            current_activity = st.session_state.activity_col
            if current_activity not in other_columns:
                current_activity = (
                    detected_activity if detected_activity in other_columns else other_columns[0]
                )
            activity_choice = st.selectbox(
                "Column of laboratory measurements",
                other_columns,
                index=other_columns.index(current_activity),
                help=(
                    "The number the laboratory measured, such as an MIC value or a "
                    "percent inhibition. A yes/no column works too."
                ),
            )
            st.session_state.activity_col = activity_choice

            usable_fraction, usable_checked = activity_scores.get(activity_choice, (0.0, 0))
            if usable_checked == 0 or usable_fraction < 0.9:
                st.warning(
                    f"“{activity_choice}” is mostly text or empty "
                    f"({usable_fraction * 100:.0f}% of its values are numbers or yes/no "
                    "words), so it probably is not the measurement column. "
                    + (
                        "These columns look more usable: "
                        + ", ".join(f"“{c}”" for c in candidates)
                        + "."
                        if candidates
                        else "No column in this file looks like a measurement."
                    )
                )

        # An optional name column, used to label the molecules in tables.
        name_options = ["(none)"] + [c for c in columns if c not in (smiles_choice,)]
        current_name = st.session_state.name_col or "(none)"
        if current_name not in name_options:
            current_name = "(none)"
        name_choice = st.selectbox(
            "Column of molecule names (optional)",
            name_options,
            index=name_options.index(current_name),
            help="Only used to label the molecules in tables and charts.",
        )
        st.session_state.name_col = None if name_choice == "(none)" else name_choice

        # -------------------------------------------------------------------
        # Labelling
        # -------------------------------------------------------------------
        labels = None
        excluded = {}

        if activity_choice is not None:
            st.subheader("Turn the measurement into a label")

            raw_values = raw[activity_choice].tolist()
            is_binary, mapping = data_io.looks_binary(raw_values)

            if is_binary:
                st.info(
                    "This column is already a yes/no answer, so no threshold is needed.",
                    icon="✅",
                )
                labels, mapping, excluded = data_io.label_binary(raw_values)
                st.write("I read the values like this:")
                show_df(
                    pd.DataFrame(
                        {
                            "Value in your file": list(mapping.keys()),
                            "Read as": [
                                "active (1)" if v == 1 else "inactive (0)"
                                for v in mapping.values()
                            ],
                        }
                    )
                )
            else:
                st.markdown(
                    "Check the source for the units and which direction means active. "
                    "Example: MIC in µg/mL, lower is stronger, so active = at or below. "
                    "Percent inhibition, higher is stronger, so active = at or above."
                )

                numbers = data_io.to_numeric(raw_values)
                usable_numbers = numbers[~np.isnan(numbers)]

                if usable_numbers.size == 0:
                    st.error(
                        f"“{activity_choice}” has no numbers in it, so I cannot build a "
                        "threshold from it. Pick a different column."
                    )
                else:
                    direction_options = ["at or below", "at or above"]
                    current_direction = st.session_state.direction
                    if current_direction not in direction_options:
                        current_direction = "at or below"
                    direction = st.radio(
                        "Active when the value is:",
                        direction_options,
                        index=direction_options.index(current_direction),
                        horizontal=True,
                        help=(
                            "“At or below” suits potency measurements such as MIC, where a "
                            "small number means a strong antibiotic. “At or above” suits "
                            "measurements such as percent inhibition."
                        ),
                    )
                    if direction != st.session_state.direction:
                        st.session_state.direction = direction
                        st.session_state.threshold = None

                    if st.session_state.threshold is None:
                        # Default: label the most promising 5% as active.
                        st.session_state.threshold = data_io.threshold_for_percent(
                            raw_values, direction, 5.0
                        )

                    threshold_col, percent_col = st.columns(2)
                    with threshold_col:
                        threshold = st.number_input(
                            "Threshold",
                            value=float(st.session_state.threshold),
                            step=float(
                                max(abs(float(st.session_state.threshold)) / 10.0, 0.01)
                            ),
                            format="%.4f",
                            help="Molecules on the active side of this number are labelled active.",
                        )
                        st.session_state.threshold = float(threshold)

                    with percent_col:
                        percent_target = st.number_input(
                            "or set the threshold so that X% are active",
                            min_value=0.1,
                            max_value=99.9,
                            value=float(st.session_state.percent_target),
                            step=0.5,
                            help=(
                                "Type a percentage and press the button, and the app works "
                                "out the matching threshold from your data."
                            ),
                        )
                        st.session_state.percent_target = float(percent_target)
                        if st.button("Use this percentage", key="apply_percent"):
                            st.session_state.threshold = data_io.threshold_for_percent(
                                raw_values, direction, percent_target
                            )
                            st.rerun()

                    labels, excluded = data_io.label_numeric(
                        raw_values, direction, float(st.session_state.threshold)
                    )

                    fig, caption = viz.measurement_histogram(
                        numbers,
                        float(st.session_state.threshold),
                        direction,
                        is_demo=st.session_state.is_demo,
                    )
                    show_chart(fig, caption)

        # -------------------------------------------------------------------
        # Live label summary and the confirm button
        # -------------------------------------------------------------------
        if labels is not None:
            summary = data_io.label_summary(labels)

            st.subheader("How your labels look right now")
            metric_cols = st.columns(3)
            metric_cols[0].metric("Active", f"{summary['n_active']:,}")
            metric_cols[1].metric("Inactive", f"{summary['n_inactive']:,}")
            metric_cols[2].metric("Percent active", f"{summary['percent_active']:.1f}%")

            fig, caption = viz.label_bar_chart(
                summary["n_active"], summary["n_inactive"], is_demo=st.session_state.is_demo
            )
            show_chart(fig, caption)
            st.info(
                "Actives are usually rare. That is why plain accuracy is misleading.",
                icon="💡",
            )

            skipped = {k: v for k, v in excluded.items() if v}
            if skipped:
                st.caption(
                    "Rows left out of the labelling: "
                    + ", ".join(f"{count:,} ({reason})" for reason, count in skipped.items())
                    + "."
                )

            problem = data_io.labels_problem_message(summary)
            if problem:
                st.error(problem)

            if st.button(
                "Confirm labels and clean data",
                key="confirm_labels",
                type="primary",
                disabled=problem is not None,
            ):
                progress = st.progress(0.0, text="Reading molecules…")

                def clean_progress(done, total):
                    progress.progress(done / max(total, 1), text=f"Reading molecules… {done:,}/{total:,}")

                try:
                    clean, report = data_io.clean_molecules(
                        raw,
                        smiles_choice,
                        labels,
                        name_column=st.session_state.name_col,
                        progress_cb=clean_progress,
                    )
                except Exception as exc:
                    progress.empty()
                    st.error(f"Something went wrong while cleaning the molecules: {exc}")
                else:
                    progress.empty()
                    st.session_state.clean_df = clean
                    st.session_state.clean_report = report
                    # A new, cleaned dataset invalidates any model we had.
                    for key in ("bundle", "split", "test_scores", "evaluation", "batch_result"):
                        st.session_state[key] = DEFAULT_STATE[key]

        # -------------------------------------------------------------------
        # Cleaning report
        # -------------------------------------------------------------------
        report = st.session_state.clean_report
        if report is not None:
            st.subheader("Cleaning report")
            st.markdown(
                f"""
- Rows in the file: **{report['total_rows']:,}**
- Skipped because the measurement was blank or unreadable: **{report['blank_label']:,}**
- Skipped because the molecule was blank: **{report['blank_smiles']:,}**
- Skipped because RDKit could not read the molecule: **{report['invalid_count']:,}**
- Duplicate molecules removed (the copies agreed): **{report['duplicates_removed']:,}**
- Duplicate molecules dropped entirely (the copies disagreed): **{report['conflicts_dropped']:,}**
- **Final molecules: {report['final_count']:,}** — {report['n_active']:,} active and
  {report['n_inactive']:,} inactive ({report['percent_active']:.1f}% active)
                """
            )
            if report["invalid_examples"]:
                st.caption("The first molecules RDKit could not read:")
                show_df(
                    pd.DataFrame(
                        report["invalid_examples"], columns=["Row in your file", "Text"]
                    )
                )
            if report["conflicting_examples"]:
                st.caption(
                    "Molecules that appeared more than once with disagreeing labels, so "
                    "all copies were dropped:"
                )
                show_df(pd.DataFrame({"Molecule": report["conflicting_examples"]}))

            if report["final_count"] == 0:
                st.error("Nothing survived cleaning, so there is nothing to train on.")
            elif not report["enough"]:
                st.error(
                    "After cleaning there are not enough molecules on both sides "
                    f"({report['n_active']:,} active, {report['n_inactive']:,} inactive). "
                    f"I need at least {data_io.MIN_PER_CLASS} of each."
                )
            else:
                st.success("Data is clean and ready. Go to the **2 Train** tab next.")

    footer()


# ===========================================================================
# 2 Train
# ===========================================================================

with tab_train:
    st.header("2. Train the model")

    clean = st.session_state.clean_df
    if clean is None or len(clean) == 0:
        st.info(
            "Nothing to train on yet. Go to the **1 Data** tab, load a file or the demo "
            "dataset, and press **Confirm labels and clean data**.",
            icon="🧪",
        )
    else:
        demo_banner()
        st.caption(
            f"Ready to train on {len(clean):,} cleaned molecules "
            f"({int(clean['label'].sum()):,} active)."
        )

        left, right = st.columns(2)

        with left:
            model_kind = st.selectbox(
                "Model",
                list(modeling.MODEL_CHOICES),
                index=0,
                help="Random Forest is the default. Logistic Regression is a simpler check.",
            )
            st.caption(
                "A random forest is hundreds of small decision trees voting together. "
                "Logistic regression is one straight-line rule, useful as a sanity check."
            )

            radius = st.selectbox(
                "Fingerprint radius",
                list(features.ALLOWED_RADII),
                index=list(features.ALLOWED_RADII).index(features.DEFAULT_RADIUS),
            )
            st.caption(
                "How far around each atom the model looks. Bigger means larger chemical "
                "pieces, but fewer of them are shared between molecules."
            )

            bits = st.selectbox(
                "Fingerprint size (bits)",
                list(features.ALLOWED_BITS),
                index=list(features.ALLOWED_BITS).index(features.DEFAULT_BITS),
            )
            st.caption(
                "How many on/off switches describe each molecule. More bits means fewer "
                "accidental collisions, and more memory."
            )

        with right:
            split_label = st.selectbox(
                "How to split the data",
                ["Scaffold split (harder, more honest)", "Random split (stratified)"],
                index=0,
            )
            split_method = "scaffold" if split_label.startswith("Scaffold") else "random"
            st.caption(
                "A scaffold split hides whole families of chemical skeletons from the "
                "model, so the test is about new chemistry rather than memory."
            )

            test_percent = st.slider(
                "Test set size (%)", min_value=10, max_value=40, value=20, step=1
            )
            st.caption(
                "The share of molecules kept back from training and used only as the exam."
            )

            seed = st.number_input(
                "Random seed", min_value=0, max_value=10_000, value=42, step=1
            )
            st.caption(
                "Fixing this number means you get exactly the same split and the same "
                "model every time, so your results are reproducible."
            )

            n_trees = modeling.DEFAULT_TREES
            if model_kind == modeling.MODEL_RANDOM_FOREST:
                n_trees = st.slider(
                    "Number of trees", min_value=50, max_value=600, value=modeling.DEFAULT_TREES,
                    step=50,
                )
                st.caption("More trees is usually a little better and a little slower.")

        if st.button("Train model", key="train_model", type="primary"):
            smiles_list = clean["smiles"].tolist()
            labels = clean["label"].to_numpy(dtype=int)

            status = st.empty()
            progress = st.progress(0.0, text="Working out the scaffolds…")

            try:
                with st.spinner("Splitting the molecules…"):
                    split = splitting.make_split(
                        smiles_list,
                        labels,
                        method=split_method,
                        test_fraction=test_percent / 100.0,
                        seed=int(seed),
                    )

                train_idx = split["train_idx"]
                test_idx = split["test_idx"]
                train_smiles = [smiles_list[i] for i in train_idx]
                test_smiles = [smiles_list[i] for i in test_idx]
                y_train = labels[train_idx]
                y_test = labels[test_idx]

                def feature_progress(done, total):
                    progress.progress(
                        0.45 * done / max(total, 1),
                        text=f"Building fingerprints… {done:,}/{total:,}",
                    )

                def train_progress(done, total):
                    progress.progress(
                        0.45 + 0.5 * done / max(total, 1),
                        text=f"Growing the model… {done}/{total}",
                    )

                bundle = modeling.fit_with_safeguard(
                    train_smiles,
                    y_train,
                    model_kind=model_kind,
                    radius=int(radius),
                    bits=int(bits),
                    n_estimators=int(n_trees),
                    seed=int(seed),
                    feature_progress_cb=feature_progress,
                    train_progress_cb=train_progress,
                )

                progress.progress(0.95, text="Scoring the unseen test molecules…")
                x_test = features.fingerprint_array(
                    test_smiles, radius=bundle["radius"], bits=bundle["bits"]
                )
                scores = modeling.predict_scores(bundle["model"], x_test)
                evaluation = modeling.evaluate(y_test, scores, threshold=0.5)

                bundle.update(
                    {
                        "train_smiles": train_smiles,
                        "test_smiles": test_smiles,
                        "y_train": y_train,
                        "y_test": y_test,
                        "test_names": [clean["name"].iloc[i] for i in test_idx],
                        "split_method": split_method,
                        "test_percent": test_percent,
                    }
                )

                progress.progress(1.0, text="Done.")
                st.session_state.bundle = bundle
                st.session_state.split = split
                st.session_state.test_scores = scores
                st.session_state.evaluation = evaluation
                st.session_state.batch_result = None
                st.session_state.train_bv_key = None
            except MemoryError:
                progress.empty()
                st.error(
                    "This machine ran out of memory even with the smaller settings. Try "
                    "1024 bits, fewer trees, or a smaller dataset."
                )
            except Exception as exc:
                progress.empty()
                st.error(f"Training stopped with a problem: {exc}")
            else:
                progress.empty()
                status.success(
                    f"Trained on {len(train_smiles):,} molecules in "
                    f"{bundle['seconds']:.1f} seconds. Go to the **3 Results** tab."
                )

        bundle = st.session_state.bundle
        split = st.session_state.split
        if bundle is not None and split is not None:
            if bundle.get("safeguard_notice"):
                st.warning(bundle["safeguard_notice"], icon="🧠")

            report = split["report"]
            if report.get("notice"):
                st.warning(report["notice"], icon="⚠️")

            st.subheader("How the data was split")
            cols = st.columns(4)
            cols[0].metric("Different scaffolds", f"{report['n_scaffolds']:,}")
            cols[1].metric(
                "Training molecules",
                f"{report['n_train']:,}",
                f"{report['train_active']:,} active",
            )
            cols[2].metric(
                "Test molecules",
                f"{report['n_test']:,}",
                f"{report['test_active']:,} active",
            )
            cols[3].metric("Scaffolds on both sides", f"{report['shared_scaffolds']:,}")

            if report["method"] == "scaffold" and report["shared_scaffolds"] == 0:
                st.success(
                    "No chemical skeleton appears on both sides, so every test molecule is "
                    "genuinely new chemistry to the model.",
                    icon="✅",
                )
            elif report["shared_scaffolds"] > 0:
                st.info(
                    f"{report['shared_scaffolds']:,} skeletons appear in both the training "
                    "and test sets. That is expected for a random split, and it makes the "
                    "test easier than real drug discovery.",
                    icon="ℹ️",
                )

            st.caption(
                f"Model: {bundle['model_kind']}; fingerprint radius {bundle['radius']}, "
                f"{bundle['bits']} bits"
                + (f"; {bundle['n_estimators']} trees" if bundle["n_estimators"] else "")
                + f"; seed {bundle['seed']}; trained in {bundle['seconds']:.1f} seconds."
            )

    footer()


# ===========================================================================
# 3 Results
# ===========================================================================

with tab_results:
    st.header("3. Results: did the model beat random picking?")

    if not have_model():
        st.info(
            "No trained model yet. Load data on the **1 Data** tab, then press "
            "**Train model** on the **2 Train** tab.",
            icon="📊",
        )
    else:
        demo_banner()
        bundle = st.session_state.bundle
        evaluation = st.session_state.evaluation
        scores = np.asarray(st.session_state.test_scores, dtype=float)
        y_test = np.asarray(bundle["y_test"], dtype=int)

        if evaluation["single_class"]:
            st.error(
                "Every molecule in the test set has the same label, so none of the usual "
                "scores can be calculated. Try a different threshold, a different split, "
                "or a larger dataset."
            )
        else:
            st.subheader("Scores on the unseen test molecules")
            score_cols = st.columns(3)
            score_cols[0].metric(
                "ROC-AUC",
                f"{evaluation['roc_auc']:.3f}",
                help="0.5 is guessing, 1.0 is perfect.",
            )
            score_cols[1].metric(
                "PR-AUC",
                f"{evaluation['pr_auc']:.3f}",
                help="Average precision. Stays meaningful when actives are rare.",
            )
            score_cols[2].metric(
                "PR-AUC with no skill",
                f"{evaluation['baseline_pr_auc']:.3f}",
                help=(
                    "The share of actives in the test set. A model that knows nothing "
                    "scores this, so PR-AUC only counts if it is clearly higher."
                ),
            )

            st.info(
                "Very high scores on a small or easy dataset usually mean an easy test, "
                "not a great model.",
                icon="💡",
            )

            # ---------------------------------------------------------------
            # Confusion matrix at an adjustable cut-off
            # ---------------------------------------------------------------
            st.subheader("What the model gets right and wrong")
            cut_off = st.slider(
                "Probability cut-off for calling a molecule active",
                min_value=0.05,
                max_value=0.95,
                value=0.50,
                step=0.05,
                help=(
                    "The model gives every molecule a probability. This is where you draw "
                    "the line between “active” and “inactive”."
                ),
            )
            at_cutoff = modeling.evaluate(y_test, scores, threshold=cut_off)
            matrix = at_cutoff["confusion"]

            show_df(
                pd.DataFrame(
                    {
                        "Model says inactive": [
                            matrix["true_negative"],
                            matrix["false_negative"],
                        ],
                        "Model says active": [
                            matrix["false_positive"],
                            matrix["true_positive"],
                        ],
                    },
                    index=["Really inactive", "Really active"],
                )
            )
            pr_cols = st.columns(3)
            pr_cols[0].metric(
                "Precision",
                f"{at_cutoff['precision']:.3f}",
                help="Of the molecules the model flags as active, the share that really are.",
            )
            pr_cols[1].metric(
                "Recall",
                f"{at_cutoff['recall']:.3f}",
                help="Of all the really active molecules, the share the model catches.",
            )
            pr_cols[2].metric(
                "F1",
                f"{at_cutoff['f1']:.3f}",
                help="One number balancing precision and recall.",
            )

            with st.expander("Plain accuracy (not recommended)"):
                st.write(
                    f"Accuracy at this cut-off is **{at_cutoff['accuracy']:.3f}**, and you "
                    "should mostly ignore it. In this test set "
                    f"{evaluation['n_active']:,} of {evaluation['n_test']:,} molecules are "
                    f"active, so a model that simply answered “inactive” to everything "
                    f"would score "
                    f"{1 - evaluation['baseline_pr_auc']:.3f} while being completely "
                    "useless. That is why this app leads with hit rate and enrichment."
                )

            # ---------------------------------------------------------------
            # The key experiment
            # ---------------------------------------------------------------
            st.divider()
            st.subheader("Model vs random picking")
            st.markdown(
                "This is the question a real laboratory asks. We can only afford to buy "
                "and test a few molecules. Does the model’s shortlist contain more real "
                "antibiotics than a shortlist picked blindly?"
            )

            n_test = int(len(y_test))
            default_n = max(3, int(round(0.03 * n_test)))
            default_n = min(default_n, n_test)
            # Same guard as the Batch Screen tab: a one-molecule test set gives
            # a slider no range to work with, and Streamlit refuses that.
            if n_test > 1:
                n_pick = st.slider(
                    "How many top-ranked molecules we would buy and test in the lab",
                    min_value=1,
                    max_value=n_test,
                    value=default_n,
                    step=1,
                )
            else:
                n_pick = 1
                st.caption(
                    "There is only one molecule in the test set, so there is nothing "
                    "to choose between. Use a larger dataset for a meaningful comparison."
                )

            result = experiment.model_vs_random(scores, y_test, n_pick)

            metric_cols = st.columns(4)
            metric_cols[0].metric(
                "Random hit rate",
                f"{result['random_hit_rate'] * 100:.1f}%",
                help="The share of the whole test set that is active.",
            )
            metric_cols[1].metric(
                "Model hit rate",
                f"{result['model_hit_rate'] * 100:.1f}%",
                help=(
                    f"95% confidence interval: {result['model_ci_low'] * 100:.1f}% to "
                    f"{result['model_ci_high'] * 100:.1f}%."
                ),
            )
            metric_cols[2].metric(
                "Enrichment",
                "n/a" if result["enrichment"] is None else f"{result['enrichment']:.2f}×",
                help="How many times better than random picking.",
            )
            metric_cols[3].metric(
                "Actives found",
                f"{result['actives_found']} of {result['n_pick']}",
                help=f"Random picking would find about {result['expected_random']:.1f}.",
            )

            st.markdown(
                f"The chance of finding **{result['actives_found']}** or more actives in "
                f"{result['n_pick']} picks by luck alone is "
                f"**{result['p_value']:.4f}** "
                f"({result['p_value'] * 100:.2f}%). This is an exact hypergeometric test."
            )
            st.info(experiment.verdict_text(result), icon="🔎")

            fig, caption = viz.hit_rate_chart(result, is_demo=st.session_state.is_demo)
            show_chart(fig, caption)

            x_gain, y_gain = experiment.cumulative_gain(scores, y_test)
            fig, caption = viz.gain_curve_chart(
                x_gain, y_gain, is_demo=st.session_state.is_demo
            )
            show_chart(fig, caption)

            fpr, tpr = evaluation["roc_curve"]
            fig, caption = viz.roc_chart(
                fpr, tpr, evaluation["roc_auc"], is_demo=st.session_state.is_demo
            )
            show_chart(fig, caption)

            # ---------------------------------------------------------------
            # Compare with the article
            # ---------------------------------------------------------------
            with st.container(border=True):
                st.markdown("#### Compare with the article")
                st.markdown(
                    f"""
**2023 MIT and Broad study:** {ARTICLE['screen_hit_rate_percent']}% of screened
compounds were active, and about {ARTICLE['new_hit_rate_percent']}% of the AI’s
new-looking picks were active ({ARTICLE['new_actives']} of
{ARTICLE['new_tested']} tested).

**Your mini-model on this data:** {result['random_hit_rate'] * 100:.1f}% random
vs {result['model_hit_rate'] * 100:.1f}% model picks.
                    """
                )
                st.caption(
                    "Careful: these are different datasets, a different organism screen and "
                    "a very different model, so this is an illustration of the same idea, "
                    "not a like-for-like comparison."
                )

            # ---------------------------------------------------------------
            # Top predictions
            # ---------------------------------------------------------------
            st.subheader("The model’s top predictions")
            order = result["ranked_order"]
            top_rows = order[: min(20, n_test)]
            table = pd.DataFrame(
                {
                    "Rank": np.arange(1, len(top_rows) + 1),
                    "Name": [str(bundle["test_names"][i]) for i in top_rows],
                    "SMILES": [bundle["test_smiles"][i] for i in top_rows],
                    "Score": [round(float(scores[i]), 4) for i in top_rows],
                    "Really active?": [
                        "yes" if y_test[i] == 1 else "no" for i in top_rows
                    ],
                }
            )
            show_df(table, hide_index=True)

            st.caption("The 12 highest-scoring test molecules:")
            molecule_grid(
                [
                    (
                        bundle["test_smiles"][i],
                        f"#{rank + 1} · {float(scores[i]):.2f} · "
                        f"{'active' if y_test[i] == 1 else 'inactive'}",
                    )
                    for rank, i in enumerate(order[: min(12, n_test)])
                ]
            )

            # ---------------------------------------------------------------
            # Copy-ready summary
            # ---------------------------------------------------------------
            st.subheader("Copy-ready summary")
            split_report = st.session_state.split["report"]
            lines = [
                "MRSA Molecule Lab, my mini-model run",
                "",
                f"Dataset: {st.session_state.source_name}",
                f"Molecules after cleaning: {len(st.session_state.clean_df):,} "
                f"({int(st.session_state.clean_df['label'].sum()):,} active, "
                f"{st.session_state.clean_report['percent_active']:.1f}% active)",
                f"Split: {split_report['method']}, "
                f"{bundle['test_percent']}% held out, seed {bundle['seed']}",
                f"Scaffolds shared between train and test: {split_report['shared_scaffolds']}",
                f"Training molecules: {split_report['n_train']:,} "
                f"({split_report['train_active']:,} active)",
                f"Test molecules: {split_report['n_test']:,} "
                f"({split_report['test_active']:,} active)",
                f"Model: {bundle['model_kind']}, radius {bundle['radius']}, "
                f"{bundle['bits']} bits"
                + (f", {bundle['n_estimators']} trees" if bundle["n_estimators"] else ""),
                "",
                f"ROC-AUC: {evaluation['roc_auc']:.3f}",
                f"PR-AUC: {evaluation['pr_auc']:.3f} "
                f"(no-skill baseline {evaluation['baseline_pr_auc']:.3f})",
                "",
                f"Molecules 'bought' (top N): {result['n_pick']}",
                f"Random hit rate: {result['random_hit_rate'] * 100:.2f}%",
                f"Model hit rate: {result['model_hit_rate'] * 100:.2f}% "
                f"(95% CI {result['model_ci_low'] * 100:.2f}% to "
                f"{result['model_ci_high'] * 100:.2f}%)",
                "Enrichment: "
                + ("n/a" if result["enrichment"] is None else f"{result['enrichment']:.2f}x"),
                f"Actives found: {result['actives_found']} "
                f"(random picking would expect {result['expected_random']:.2f})",
                f"Hypergeometric p-value: {result['p_value']:.5f}",
                "",
                experiment.verdict_text(result),
                "",
                HONESTY_FOOTER,
            ]
            if st.session_state.is_demo:
                lines.insert(1, f"*** {demo.DEMO_BANNER} ***")
            summary_text = "\n".join(lines)

            st.text_area("Numbers for my article", summary_text, height=380)
            st.download_button(
                "Download this summary as a .txt file",
                data=summary_text.encode("utf-8"),
                file_name="mrsa_molecule_lab_summary.txt",
                mime="text/plain",
                key="download_summary",
            )

    footer()


# ===========================================================================
# 4 Predict
# ===========================================================================

with tab_predict:
    st.header("4. Predict one molecule")

    if not have_model():
        st.info(
            "No trained model yet. Load data on the **1 Data** tab, then press "
            "**Train model** on the **2 Train** tab, and come back here.",
            icon="🔬",
        )
    else:
        demo_banner()
        bundle = st.session_state.bundle

        st.caption("Try one of these, or paste your own SMILES text:")
        example_cols = st.columns(len(EXAMPLE_MOLECULES))
        for col, (label, smiles) in zip(example_cols, EXAMPLE_MOLECULES.items()):
            with col:
                if st.button(label, key=f"example_{label.lower().replace(' ', '_')}"):
                    st.session_state.smiles_input = smiles

        smiles_text = st.text_input(
            "SMILES",
            key="smiles_input",
            placeholder="CC(=O)OC1=CC=CC=C1C(=O)O",
            help="A molecule written as text. The example buttons above fill this in.",
        )

        predict_clicked = st.button("Predict", key="predict_button", type="primary")

        if predict_clicked or smiles_text:
            if not smiles_text.strip():
                st.warning("Type or paste a SMILES string first, or press one of the examples.")
            else:
                mol = features.mol_from_smiles(smiles_text)
                if mol is None:
                    st.error(
                        f"I could not read “{smiles_text.strip()}” as a molecule. "
                        "SMILES looks like CC(=O)OC1=CC=CC=C1C(=O)O (aspirin). "
                        "Check for a typo, such as a ring that is opened but never closed."
                    )
                else:
                    mol = features.largest_fragment(mol)
                    x_one = features.fingerprint_one(
                        mol, radius=bundle["radius"], bits=bundle["bits"]
                    )
                    score = float(modeling.predict_scores(bundle["model"], x_one)[0])

                    st.warning(
                        "A high score does not mean this molecule is a safe or working "
                        "medicine.",
                        icon="⚠️",
                    )

                    draw_col, score_col = st.columns([1, 1])
                    with draw_col:
                        svg = viz.mol_svg(mol, size=300)
                        if svg:
                            show_svg(svg)
                    with score_col:
                        st.metric("Predicted chance of being active", f"{score * 100:.1f}%")
                        band = modeling.score_band(score, st.session_state.test_scores)
                        st.write(band["text"])

                    st.subheader("What this molecule is like")
                    properties = features.describe_molecule(mol)
                    show_df(
                        pd.DataFrame(
                            {
                                "Property": list(properties.keys()),
                                "Value": list(properties.values()),
                            }
                        ),
                        hide_index=True,
                    )
                    st.caption(
                        "Molecular weight is how heavy the molecule is; logP is how greasy "
                        "it is; the hydrogen-bond counts and TPSA say how well it sticks to "
                        "water; rotatable bonds say how floppy it is. Medicines that work by "
                        "mouth tend to sit in fairly narrow ranges on all of these."
                    )

                    # -----------------------------------------------------------
                    # Nearest training molecules
                    # -----------------------------------------------------------
                    st.subheader("The most similar molecules the model learned from")
                    train_bvs = training_bit_vectors()
                    query_bv = features.bit_vector_one(
                        mol, radius=bundle["radius"], bits=bundle["bits"]
                    )
                    neighbours = features.nearest_neighbours(query_bv, train_bvs, k=3)

                    if not neighbours:
                        st.caption("There are no training molecules to compare against.")
                    else:
                        rows = []
                        for index, similarity in neighbours:
                            rows.append(
                                (
                                    bundle["train_smiles"][index],
                                    f"{similarity:.2f} similar · "
                                    f"{'active' if bundle['y_train'][index] == 1 else 'inactive'}",
                                )
                            )
                        molecule_grid(rows, columns=3, size=220)
                        show_df(
                            pd.DataFrame(
                                {
                                    "Tanimoto similarity": [
                                        round(s, 3) for _i, s in neighbours
                                    ],
                                    "Label in training data": [
                                        "active" if bundle["y_train"][i] == 1 else "inactive"
                                        for i, _s in neighbours
                                    ],
                                    "SMILES": [
                                        bundle["train_smiles"][i] for i, _s in neighbours
                                    ],
                                }
                            ),
                            hide_index=True,
                        )

                        best_similarity = neighbours[0][1]
                        if best_similarity >= 0.7:
                            st.info(
                                f"Very similar to something the model already saw "
                                f"(similarity {best_similarity:.2f}).",
                                icon="🔁",
                            )
                        elif best_similarity < 0.4:
                            st.warning(
                                f"Unlike the training data (closest similarity "
                                f"{best_similarity:.2f}), so this prediction is less "
                                "reliable.",
                                icon="❓",
                            )
                        else:
                            st.caption(
                                f"Closest training molecule has a similarity of "
                                f"{best_similarity:.2f}, so this is familiar chemistry but "
                                "not a near copy."
                            )

                    # -----------------------------------------------------------
                    # Atom-level explanation
                    # -----------------------------------------------------------
                    st.subheader("Which atoms moved the answer?")
                    with st.spinner("Covering up one atom at a time…"):
                        explanation = explain.explain_molecule(
                            bundle["model"], mol, radius=bundle["radius"], bits=bundle["bits"]
                        )
                    if not explanation["ok"]:
                        st.caption(explanation["message"])
                    else:
                        show_svg(explanation["svg"])
                        show_svg(explain.legend_svg())
                        st.caption(
                            "Each atom is covered up in turn and the molecule is scored "
                            "again. Orange atoms pushed the score up, blue atoms pushed it "
                            "down, and the strength of the colour shows how much."
                        )
                    st.caption(explain.SIMPLIFIED_NOTE)

    footer()


# ===========================================================================
# 5 Batch Screen
# ===========================================================================

with tab_batch:
    st.header("5. Screen a whole list")

    if not have_model():
        st.info(
            "No trained model yet. Load data on the **1 Data** tab and train a model on "
            "the **2 Train** tab first.",
            icon="📥",
        )
    else:
        demo_banner()
        bundle = st.session_state.bundle

        st.markdown(
            "Upload a list of molecules with no labels and the model will rank them, the "
            "way a laboratory would decide what to buy."
        )

        upload_col, demo_col = st.columns([3, 2])
        with upload_col:
            batch_file = st.file_uploader(
                "A CSV or TSV with a SMILES column (names optional)",
                type=["csv", "tsv", "txt"],
                key="batch_uploader",
            )
        with demo_col:
            st.write("")
            st.write("")
            use_demo_list = st.button("Use demo screening list", key="batch_demo")

        source = None
        if use_demo_list:
            source = (demo.demo_screening_bytes(), demo.SCREENING_PATH.name)
        elif batch_file is not None:
            source = (batch_file.getvalue(), batch_file.name)

        # Scoring a long list is slow, so only do it when the file is new.
        # Moving the sliders below must never re-score anything.
        if source is not None:
            fingerprint_of_source = (source[1], len(source[0]), bundle["bits"], bundle["radius"])
            already_done = (
                st.session_state.batch_result is not None
                and st.session_state.batch_result.get("source_fingerprint")
                == fingerprint_of_source
            )
            if already_done:
                source = None

        if source is not None:
            raw_bytes, raw_name = source
            try:
                frame, smiles_column, name_column = data_io.read_screening_table(
                    raw_bytes, raw_name
                )
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"I could not read that file: {exc}")
            else:
                with st.spinner("Reading and scoring the molecules…"):
                    uploaded_rows = int(len(frame))
                    invalid_rows = []
                    seen = {}
                    for i in range(uploaded_rows):
                        raw_smiles = frame[smiles_column].iloc[i]
                        text = "" if raw_smiles is None else str(raw_smiles).strip()
                        name = (
                            str(frame[name_column].iloc[i])
                            if name_column is not None
                            else ""
                        )
                        canonical = features.standardize_smiles(text)
                        if canonical is None:
                            invalid_rows.append(
                                {"Row in your file": i + 2, "Text": text, "Name": name}
                            )
                            continue
                        if canonical not in seen:
                            seen[canonical] = name

                    valid_smiles = list(seen.keys())
                    valid_names = [seen[s] for s in valid_smiles]

                    if not valid_smiles:
                        st.error(
                            "None of the rows in that file could be read as molecules, so "
                            "there is nothing to score."
                        )
                        st.session_state.batch_result = None
                    else:
                        x_batch = features.fingerprint_array(
                            valid_smiles, radius=bundle["radius"], bits=bundle["bits"]
                        )
                        batch_scores = modeling.predict_scores(bundle["model"], x_batch)

                        # Similarity to the ACTIVE training molecules only: that is
                        # what "new-looking" means here.
                        active_smiles = [
                            s
                            for s, label in zip(bundle["train_smiles"], bundle["y_train"])
                            if label == 1
                        ]
                        active_bvs = features.bit_vectors(
                            active_smiles, radius=bundle["radius"], bits=bundle["bits"]
                        )
                        query_bvs = features.bit_vectors(
                            valid_smiles, radius=bundle["radius"], bits=bundle["bits"]
                        )
                        similarity = features.max_similarity_to_set(query_bvs, active_bvs)

                        st.session_state.batch_result = {
                            "source_name": raw_name,
                            "source_fingerprint": (
                                raw_name,
                                len(raw_bytes),
                                bundle["bits"],
                                bundle["radius"],
                            ),
                            "uploaded_rows": uploaded_rows,
                            "smiles": valid_smiles,
                            "names": valid_names,
                            "scores": batch_scores,
                            "similarity": similarity,
                            "invalid_rows": invalid_rows,
                            "n_active_train": len(active_smiles),
                        }

        batch = st.session_state.batch_result
        if batch is not None:
            st.success(
                f"Scored **{len(batch['smiles']):,}** unique molecules from "
                f"**{batch['source_name']}**."
            )

            filter_cols = st.columns(2)
            with filter_cols[0]:
                score_cutoff = st.slider(
                    "Keep molecules scoring at least",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.30,
                    step=0.05,
                    help="The model's predicted chance of being active.",
                )
            with filter_cols[1]:
                similarity_cutoff = st.slider(
                    "Keep molecules less similar than this to any training active",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.50,
                    step=0.05,
                    help=(
                        "Low numbers favour new-looking structures. The real study did the "
                        "same, because a molecule that just copies a known antibiotic is "
                        "not a new discovery."
                    ),
                )

            scores_array = np.asarray(batch["scores"], dtype=float)
            similarity_array = np.asarray(batch["similarity"], dtype=float)

            passes_score = scores_array >= score_cutoff
            passes_both = passes_score & (similarity_array < similarity_cutoff)
            n_pass_score = int(passes_score.sum())
            n_pass_both = int(passes_both.sum())

            # A slider needs a range. When 0 or 1 molecules survive the filters
            # there is nothing to choose between, so we show a line of text
            # instead. (Streamlit refuses a slider whose minimum equals its
            # maximum, which used to crash this tab.)
            if n_pass_both > 1:
                top_n = st.slider(
                    "How many to “buy and test”",
                    min_value=1,
                    max_value=n_pass_both,
                    value=min(10, n_pass_both),
                    step=1,
                )
            elif n_pass_both == 1:
                top_n = 1
                st.caption(
                    "Only one molecule passed both filters, so that is the whole "
                    "shortlist. Lower the score cut-off or allow more similarity to "
                    "see more."
                )
            else:
                top_n = 0
                st.caption(
                    "No molecules passed both filters. Lower the score cut-off, or "
                    "allow molecules that look more like the known actives."
                )
            shortlist_size = min(top_n, n_pass_both)

            stages = [
                ("Rows uploaded", batch["uploaded_rows"]),
                ("Valid, unique molecules", len(batch["smiles"])),
                (f"Score at or above {score_cutoff:.2f}", n_pass_score),
                (f"Not too similar to a known active (< {similarity_cutoff:.2f})", n_pass_both),
                ("Shortlist to test", shortlist_size),
            ]
            fig, caption = viz.funnel_chart(stages, is_demo=st.session_state.is_demo)
            show_chart(fig, caption)

            st.caption(
                "In the real study, about 12 million molecules were scored and 283 were "
                "tested in the lab."
            )

            # The full ranked table, with each molecule's status.
            order = experiment.rank_order(scores_array)
            shortlist_positions = set(
                [i for i in order if passes_both[i]][:shortlist_size]
            )

            def status_for(i: int) -> str:
                if i in shortlist_positions:
                    return "shortlisted"
                if not passes_score[i]:
                    return "score too low"
                if similarity_array[i] >= similarity_cutoff:
                    return "too similar to a known active"
                return "passed filters, outside the top picks"

            ranked = pd.DataFrame(
                {
                    "Rank": np.arange(1, len(order) + 1),
                    "Name": [batch["names"][i] for i in order],
                    "SMILES": [batch["smiles"][i] for i in order],
                    "Score": [round(float(scores_array[i]), 4) for i in order],
                    "Similarity to nearest training active": [
                        round(float(similarity_array[i]), 3) for i in order
                    ],
                    "Status": [status_for(int(i)) for i in order],
                }
            )

            st.subheader("Ranked molecules")
            if len(ranked) > 200:
                st.caption("Showing the first 200 rows. The download holds all of them.")
            show_df(ranked.head(200), hide_index=True)

            st.download_button(
                "Download ranked CSV",
                data=ranked.to_csv(index=False).encode("utf-8"),
                file_name="mrsa_molecule_lab_ranked.csv",
                mime="text/csv",
                key="download_ranked",
            )

            if batch["n_active_train"] == 0:
                st.warning(
                    "The training set contained no active molecules, so the similarity "
                    "filter has nothing to compare against and lets everything through."
                )

            st.caption("The 12 highest-scoring molecules in your list:")
            molecule_grid(
                [
                    (
                        batch["smiles"][i],
                        f"#{rank + 1} · {float(scores_array[i]):.2f}",
                    )
                    for rank, i in enumerate(order[: min(12, len(order))])
                ]
            )

            st.subheader("Rows I could not read")
            if batch["invalid_rows"]:
                show_df(pd.DataFrame(batch["invalid_rows"]), hide_index=True)
                st.caption(
                    f"{len(batch['invalid_rows'])} row(s) in that file were not valid "
                    "molecules and were left out of everything above."
                )
            else:
                st.caption("Every row in that file was a valid molecule.")

    footer()
