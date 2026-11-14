# MRSA Molecule Lab

**A mini AI antibiotic predictor** — the hands-on build that goes with my TKS
learn article.

> **The article’s question:** Can AI discover a new antibiotic to cure MRSA, the
> “superbug” staph infection that resists most common antibiotics?

MRSA is methicillin-resistant *Staphylococcus aureus*, a common skin bacterium
that survives many antibiotics because it carries the **mecA** gene, which makes
an altered protein called **PBP2a** that methicillin barely binds to.
Drug-resistant infections directly caused about 1.27 million deaths worldwide in
2019, MRSA was responsible for over 100,000 of them, and only a few dozen new
antibiotics have been approved in 45 years.

In 2023, researchers at MIT and the Broad Institute (Wong et al., *Nature*)
trained an explainable AI on 39,312 lab-tested compounds, of which 512 (about
1.3%) stopped staph growing, then used it to score about 12 million new
compounds. They bought and tested 283, and 21 brand-new compounds worked — a hit
rate of about 9% against 1.3%. Two leading candidates from one new chemical
family killed MRSA in laboratory dishes and reduced it in infected mice by about
1.2 logs, with resistance appearing slowly.

My conclusion: **AI is already great at finding candidates, but a candidate is
not yet a cure.**

## What this reproduces

This is a miniature, honest reproduction of that 2023 pipeline — two of the
“reproduce an existing project” routes at once:

- **Re-run a mini version of the 2023 MIT and Broad AI.** Train a model on the
  real *S. aureus* screening data, test it on molecules it has not seen, and
  compare its hit rate against random picking. That comparison is the whole
  point of the **3 Results** tab.
- **Reproduce the “explainable AI” idea in miniature.** The **4 Predict** tab
  covers up one atom at a time and re-scores the molecule, then colours each
  atom by how much it pushed the prediction up or down.

It is deliberately *not* a Chemprop re-run: this uses fingerprints plus a random
forest, which trains in seconds on a laptop and is small enough to read and
understand in an afternoon.

This app is the small, honest version of that pipeline. You load a table of
molecules with laboratory measurements; it cleans them, trains a simple model,
tests that model on molecules it has never seen, and then answers the question a
real laboratory cares about: would ranking molecules with the model have found
more real antibiotics than picking them at random?

Every number the app displays is computed from the data you load, at the moment
you load it. Nothing is faked, simulated or hardcoded. The only built-in data is
the small labelled demo set in `data/demo/`, and the fixed facts from the 2023
paper quoted on the About tab.

---

## Running it on GitHub (no installing anything)

If you do not want to install Python on your laptop, GitHub can do all three of
these for you once the code is pushed to a repository.

### 1. Let GitHub run the tests for you

`.github/workflows/tests.yml` runs on every push. It installs the packages on a
fresh Linux machine, runs `pytest -q` on **both** Python 3.11 and 3.12, starts
the app and checks that `http://localhost:8501/_stcore/health` answers, and
fails the build if Streamlit or RDKit print a single deprecation warning.

Watch it under the repository's **Actions** tab. A green tick means the whole
project genuinely works; click into a run to read the real test output.

The performance test (40,000 synthetic molecules) is in the same workflow but
only runs when you start it by hand: **Actions → tests → Run workflow**.

### 2. Use the app in a Codespace

`.devcontainer/devcontainer.json` is set up, so on the repository page click
**Code → Codespaces → Create codespace on main**. GitHub gives you a Linux
machine with Python 3.12, installs everything automatically, and opens VS Code
in your browser. Then in its terminal:

```bash
streamlit run app.py
```

A preview tab opens on port 8501 with the real app in it. Codespaces is free
for a generous number of hours each month on a personal account.

### 3. Put it online for anyone to use

[Streamlit Community Cloud](https://share.streamlit.io) deploys straight from a
GitHub repository, free, and gives you a public link you can put in your
article. Point it at your repo, set the main file to `app.py`, and deploy.
`requirements.txt` and `packages.txt` are already there for it (`packages.txt`
installs the two small Linux graphics libraries that RDKit's drawing code
wants).

> If you deploy it publicly, remember that anyone can upload a file to it. Do
> not deploy a version with a licence-restricted dataset baked in.

---

## Install and run on your own Mac

You need Python 3.11 or 3.12. Check what you have:

```bash
python3 --version
```

If that prints an error or a version below 3.11, install Python 3.12 from
[python.org/downloads](https://www.python.org/downloads/) (or with Homebrew:
`brew install python@3.12`), then open a new terminal.

Create an isolated virtual environment inside this folder and install
everything into it:

```bash
cd "<this folder>"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check that the three big packages imported properly:

```bash
python -c "import rdkit, sklearn, streamlit; print('ok')"
```

Start the app:

```bash
streamlit run app.py
```

It opens at <http://localhost:8501>. Press `Ctrl+C` in the terminal to stop it.

Every time you come back to the project, activate the environment first:

```bash
source .venv/bin/activate
```

---

## How to use it

### First, with the demo data

1. **1 Data** → click **Load DEMO dataset**. 79 molecules load, and the app
   detects the `smiles` column and the `measurement_demo_mic_ug_per_ml` column
   by itself. At the default threshold of 50 you get **20 active** and
   **59 inactive**.
2. Look at the histogram and the label counts, then click
   **Confirm labels and clean data**. Read the cleaning report.
3. **2 Train** → leave the defaults (Random Forest, radius 2, 2048 bits,
   scaffold split, 20% test, seed 42) and click **Train model**. Check that
   “Scaffolds on both sides” is **0**: that is what makes the test honest.
4. **3 Results** → this is the heart of it. Compare the random hit rate with the
   model’s hit rate on its top N picks, read the enrichment and the p-value, and
   look at the three charts. Copy the **Copy-ready summary** into your notes.
5. **4 Predict** → press an example button (or paste your own SMILES) to score
   one molecule, see its properties, its nearest training molecules, and which
   atoms moved the answer.
6. **5 Batch Screen** → click **Use demo screening list** to score 16 molecules
   and watch the funnel narrow them to a shortlist.

> The demo measurements were hand-labelled from general knowledge, **not**
> measured in a laboratory. The app says so in a banner wherever they are used.
> Do not put demo numbers in an article.

### Then, with a real dataset

1. Download the dataset yourself and **check its licence**. Mine is the
   *Staphylococcus aureus* growth-inhibition screen of about 39,000 molecules
   from the 2023 paper, published with the authors’ own code at
   [github.com/felixjwong/antibioticsai](https://github.com/felixjwong/antibioticsai).
   No real dataset is included in this repository, and you should read that
   repo’s licence before using or redistributing the data.
2. **1 Data** → upload the `.csv`, `.tsv`, `.txt` or `.xlsx` file (for Excel you
   can pick the sheet).
3. Check the detected columns and fix them with the dropdowns if needed. The
   app will never preselect a column of chemical *names* as the molecules.
4. Choose the direction and threshold. **Check the source for the units and
   which direction means active.** MIC in µg/mL: lower is stronger, so active =
   *at or below*. Percent inhibition: higher is stronger, so active =
   *at or above*. You can also type a percentage and let the app work out the
   matching threshold.
5. Confirm, train, and read the Results tab. For roughly 39,000 molecules
   expect training to take on the order of a minute or two on a laptop.

---

## Running the tests

With the virtual environment active:

```bash
pytest -q
```

- `tests/test_core.py` — the logic: column detection, labelling, cleaning,
  scaffold splitting (including the retry and the fallback), and the statistics
  (Wilson interval, enrichment and hypergeometric p-value checked against
  hand-worked numbers). It also checks that a *useless* model, given random
  scores, produces an enrichment near 1 — the test that would catch the app
  flattering itself.
- `tests/test_app_smoke.py` — the app itself, headless, through Streamlit’s
  `AppTest`: start with no data, load the demo, confirm, train, predict aspirin
  and ciprofloxacin, show friendly errors for `C1CC` and `hello`, and run the
  batch screen.

Time the full pipeline on a realistically large file (40,000 synthetic
molecules, labels random at about 1.3% active, for speed measurement only):

```bash
python scripts/performance_test.py
```

---

## What is in the project

```
app.py                     the Streamlit interface: header, six tabs, footer
src/data_io.py             reading files, column detection, labelling, cleaning
src/features.py            RDKit helpers, fingerprints, similarity
src/splitting.py           scaffold split and stratified split
src/modeling.py            training with progress, memory safeguard, metrics
src/experiment.py          Model vs Random: hit rate, Wilson CI, hypergeometric
src/explain.py             atom-level explanation images
src/viz.py                 charts and molecule drawings
src/demo.py                loads the demo files
data/demo/                 the small hand-labelled demo dataset
tests/                     unit tests and headless UI tests
scripts/performance_test.py timing on 40,000 synthetic molecules
```

All of the logic lives in `src/` so it can be tested without a browser; `app.py`
is only the interface.

---

## Limitations

- One dataset, and one kind of bacteria.
- Results come from a laboratory dish, not from people.
- This is a deliberately simple model (fingerprints plus a random forest), not
  the published deep learning model.
- Nothing here checks toxicity, or whether a molecule can actually be made.
- A scaffold split is harder and more honest than a random split, but it is
  still much easier than real prospective drug discovery.
- The atom-level explanation is a simple occlusion method. The MIT and Broad
  team used Monte Carlo tree search to find the smallest chemical piece
  responsible for each prediction.
- **No real dataset is included.** Download it yourself and check its licence
  before you use it or redistribute it.

---

## Credits

- Wong, F. et al. (2023). *Discovery of a structural class of antibiotics with
  explainable deep learning.* Nature.
- Stokes, J. M. et al. (2020). *A deep learning approach to antibiotic
  discovery.* Cell. (The halicin study.)
- [RDKit](https://www.rdkit.org/) for the chemistry.
- [scikit-learn](https://scikit-learn.org/) for the machine learning.
- [Streamlit](https://streamlit.io/) for the app.

---

*Educational mini-model. It is not the published MIT and Broad model, a
prediction is not proof that a molecule works as a medicine, and this is not
medical advice.*
