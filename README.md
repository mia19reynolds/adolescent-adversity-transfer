# Cross-Cohort Transfer of Adolescent Adversity Prediction Models

This repository holds the analysis pipeline for an MSc project asking whether a prediction model
built in one adolescent cohort still works in another once the features and the outcome have been
harmonised imperfectly. The source cohort is the UK Millennium Cohort Study (MCS) age-14 sweep;
the target is the 2023 United States National Youth Risk Behavior Survey (YRBS).

Nine model families are compared under:

- unadapted transfer;
- label-free adaptation, which uses target covariates or the source model's own predictions but
  never a target outcome;
- label-using adaptation across target-label budgets from 50 to 2,000;
- post-training probability adjustment;
- calibration, fixed-capacity screening, conformal prediction, subgroup and outcome-sensitivity
  analyses; and
- reverse transfer from YRBS to MCS, as a directional sensitivity analysis.

The outcome counts how many of five shared adversity pillars a respondent reports. The main
analysis cuts at two or more; one or more and three or more are kept as sensitivity analyses. The
harmonised predictor registry and every recoding decision live in the source and in
`spec/harmonisation_spec_v5.csv`, and are not restated here — one description is enough, and two
would eventually disagree.

## Repository structure

| Path | Purpose |
|---|---|
| `notebooks/` | The four ordered analysis notebooks. The tracked source declares `PUBLIC_NOTEBOOK = False`. |
| `src/` | Data construction, recoding, modelling, transfer, evaluation and publication helpers. |
| `tests/` | The test suite. Runs on synthetic fixtures; needs no cohort data and no configured paths. |
| `spec/harmonisation_spec_v5.csv` | Cross-cohort harmonisation decision record. |
| `spec/local_model_settings.csv` | The fixed model configurations, for both cohorts: 54 rows, `cohort,threshold,family,settings,protocol_id,settings_digest`. Written only by `scripts/promote_local_settings.py`. |
| `spec/public_notebook_cells.json` | Cell-level allow-list for outputs kept in public notebook copies. |
| `scripts/` | Making and structurally checking sanitised public notebook copies. |
| `requirements.txt` | Python dependencies and version constraints. |

Participant data, harmonised row-level data, fitted models, person-level scores, working tables
and unreviewed publication candidates are all deliberately absent.

## Where output goes

**The project directory holds source. Everything a run generates is written below
`THESIS_WORK_DIR`** — working tables, diagnostic and publication figures, LaTeX fragments,
candidate tables, checkpoints and search products alike. Person-level scores are the one
exception and go somewhere stricter still: `MCS_DATA_DIR`, because they are person-level.

The intended layout puts the two side by side, with the working root a sibling of the clone
rather than anything inside it:

```
thesis_final/
  project/       source, and later the Git repository
  run_outputs/   all generated run products
```

**Git is initialised only in `project/`, never in the parent directory.** The working root is
not a directory Git should ever see: it holds MCS-derived material, and git history does not
forget.

Nothing generated is written into the repository — not a table, not a figure, not a LaTeX
fragment. A figure or a fragment that belongs in the manuscript is copied there by hand from
the working root once you have reviewed it, which is the same deliberate step every published
table already goes through. `.gitignore` ignores `outputs/`, `publication_outputs/` and
`publication_candidates/` unconditionally, so a stray one cannot be committed by accident.

## Data access

### Millennium Cohort Study

The project uses *Millennium Cohort Study: Age 14, Sweep 6, 2015*, UK Data Service study number
SN 8156. Access has to be obtained independently through the
[UK Data Service study record](https://doc.ukdataservice.ac.uk/doc/8156/mrdoc/UKDA/UKDA_Study_8156_Information.htm),
and the data used under its licence and disclosure conditions.

Nothing here downloads, redistributes or reconstructs the restricted MCS files. Once access is
approved, prepare a single combined Sweep 6 parquet holding the original MCS variables that
`src/recode_mcs.py`, `src/features.py` and `src/outcomes.py` ask for. The feature crosswalk in
Notebook 01 names the source variable behind every retained construct.

### Youth Risk Behavior Survey

The 2023 National YRBS data and documentation are on the
[CDC national YRBS datasets page](https://www.cdc.gov/yrbs/data/national-yrbs-datasets-documentation.html).
The CDC distributes the national file in Access and ASCII formats. Convert it to parquet without
renaming any original variable and save it as `yrbs2023_raw.parquet` at the configured `YRBS_RAW`
location. There is deliberately no downloader in this repository.

Variable wording and response codes are in the
[2023 YRBS Data User's Guide](https://www.cdc.gov/yrbs/media/pdf/2023/2023_National_YRBS_Data_Users_Guide508.pdf).

## Environment

Developed with Python 3.12.13. To build a clean environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

You will also need a Jupyter frontend and a kernel for this environment. Neither is pinned in
`requirements.txt`, because they are tooling for running the analysis rather than part of it.

## Local path configuration

The pipeline writes to two roots outside the repository:

- `MCS_DATA_DIR` — restricted MCS inputs and secure person-level artefacts;
- `THESIS_WORK_DIR` — the open YRBS input and every file a run generates.

Neither may be the repository, a directory inside it, your home directory or a filesystem root,
and symlinks are resolved before that check. Set them in the shell that launches Jupyter:

```bash
export MCS_DATA_DIR=/absolute/path/to/a/dedicated/mcs_directory
export THESIS_WORK_DIR=/absolute/path/to/a/dedicated/working_directory
```

VS Code and JupyterLab do not always pass a shell environment through to a kernel. If that
happens, put each root on the first line of `.mcs_data_dir` and `.thesis_work_dir` in the
repository root instead. Both files hold a path and never data, both are git-ignored, and both
are specific to one machine.

`src/paths.json` declares the relative components under those roots. It is git-ignored because it
describes one machine's storage layout, so create it locally with this shape and change only the
components you need to:

```json
{
  "mcs": {
    "MCS_CORE": ["raw", "mcs6_core.parquet"],
    "MCS_FEATURES": ["derived", "mcs_features.parquet"],
    "MCS_PILLARS": ["derived", "mcs_pillars.parquet"],
    "MCS_ATTRIBUTES": ["derived", "mcs_attributes.parquet"],
    "MCS_SCORES": ["scores"]
  },
  "work": {
    "YRBS_RAW": ["data", "raw", "yrbs2023_raw.parquet"],
    "YRBS_FEATURES": ["data", "derived", "yrbs_features.parquet"],
    "YRBS_PILLARS": ["data", "derived", "yrbs_pillars.parquet"],
    "YRBS_ATTRIBUTES": ["data", "derived", "yrbs_attributes.parquet"]
  }
}
```

`MCS_SCORES` names a directory; everything else names a file. A path is resolved only when the
configured attribute is actually asked for, and an unsafe root is refused before anything is read
or written.

### An optional extra gate

If `src/_mcs_gate.py` is present, no MCS path is handed out unless `MCS_READ_OK=1` is set in the
environment for that invocation. It exists so that a process which merely inherited a shell
environment cannot reach restricted data by accident. The file is git-ignored, so a fresh clone
does not have one and the variable does nothing; add it if you want the extra gate on your own
machine, and export `MCS_READ_OK=1` alongside the two roots above when you do.

## Running the analysis

Activate the environment, configure the two roots, then start Jupyter from the repository root.
Restart the kernel and run each notebook top to bottom, in this order:

1. `01_data_and_features.ipynb` — reads both cohorts, applies the harmonisation and outcome
   definitions, runs the descriptive cohort checks, and writes the canonical feature, pillar and
   attribute artefacts.
2. `02_models_and_transfer.ipynb` — builds the repeated splits; then, in Section D, selects one
   fixed model configuration per family and threshold inside each cohort, by the same procedure in
   both, and writes the two private selection records under the working root. It does **not**
   promote them — Section D stops at a gate; see *Model selection and promotion* below. After
   promotion the same notebook fits the model and adaptation battery, runs the label-budget and
   robustness analyses, and writes the score handoff and the aggregate working tables.
3. `03_evaluation_and_robustness.ipynb` — discrimination, calibration, conformal coverage,
   subgroup and screening-capacity behaviour, conditional uncertainty, and the publication
   candidates.
4. `04_explorations.ipynb` — the reverse-transfer sensitivity analysis.

The tracked notebooks declare:

```python
PUBLIC_NOTEBOOK = False
```

That is internal mode, and it shows the full diagnostic record. The flag is presentation only: it
decides what is displayed or printed, and nothing else. Fitting, scoring, validation and file
writing are identical either way, so both settings produce the same scientific files. Do not edit
the tracked source to `True` for the final run — the public copies handle that, and flipping it by
hand would relabel internal output as public.

Run each notebook completely and in order. The public-copy workflow refuses a notebook whose
executable cells do not all carry strictly increasing execution counts, because outputs beside
broken counts do not describe one coherent run.

## Model selection and promotion

Hyperparameters are selected **once per cohort**, by the same procedure in both, and then held
fixed across all twenty evaluation splits. Section D of `02_models_and_transfer.ipynb` runs that
selection and writes two private records under `THESIS_WORK_DIR`; those records carry the
cross-validated scores behind each choice and stay outside the repository.

Section D then **stops at a promotion gate** before any model is fitted. On a first run the gate
refuses, because the promoted specification does not yet describe the selection just made. Review
the 54-row settings table the section displays, promote with the two commands below, and re-run
from the gate cell — nothing above it is recomputed, because a valid record on disk is loaded
rather than re-searched.

Promotion into the tracked specification is a separate, reviewed act:

```bash
python scripts/promote_local_settings.py                     # dry run: the full diff, nothing written
python scripts/promote_local_settings.py --apply --confirm   # writes spec/local_model_settings.csv
```

The tracked file carries the 54 configurations and two provenance identifiers — `protocol_id` for
the selection procedure and `settings_digest` for the configurations themselves — and no
cross-validated value. No notebook writes it. After promoting, re-run notebooks 02, 03 and 04 in
order; results from either side of a promotion were fitted under different configurations and must
not be mixed.

An earlier model specification was superseded. It was selected by a procedure that is not
reproducible from this repository, and it is not used anywhere in the analysis. The superseded
files are retained privately with the archived earlier run and are not part of this project.

## Checks

The test suite runs on synthetic fixtures. It opens no cohort, no fitted model and no score file,
and it resolves no configured path, so it works on a clone with nothing set up:

```bash
python -m pytest tests/ -q
```

The two structural checkers can also be run over the tracked notebooks at any time:

```bash
python scripts/check_notebook_outputs.py
python scripts/check_public_outputs.py
```

## Creating public notebook copies

An executed internal notebook can hold restricted or disclosure-relevant diagnostics, so it must
not be released as it stands. `scripts/make_public_notebooks.py` writes separate copies, sets the
presentation flag to `True` in the copy, and keeps stored output only in the cell IDs listed in
`spec/public_notebook_cells.json`. It never modifies the source.

Do a dry run first:

```bash
python scripts/make_public_notebooks.py \
  --source . \
  --dest /absolute/path/to/public_notebooks
```

If the retain-and-clear decisions are what you expect, write the copies:

```bash
python scripts/make_public_notebooks.py \
  --source . \
  --dest /absolute/path/to/public_notebooks \
  --apply
```

Then check the copies themselves:

```bash
python scripts/check_notebook_outputs.py /absolute/path/to/public_notebooks/*.ipynb
python scripts/check_public_outputs.py   /absolute/path/to/public_notebooks/*.ipynb
```

These scripts enforce structural rules and scan for known unsafe patterns. **Passing them is not
disclosure clearance.** Any retained MCS-derived aggregate still needs researcher, institutional
and UK Data Service review before it can be released.

## What this repository can and cannot reproduce

It records the complete analysis logic, the harmonisation decisions and the fixed model
configuration. It cannot give you a one-command reproduction from a public clone. MCS access is
controlled independently by the UK Data Service, and the restricted cohort, the derived row-level
artefacts, the person-level scores and the local path configuration are all absent by design. YRBS
is open, but you still have to download it and convert it yourself.

Inside a properly authorised environment, the notebooks are built to run in order from the raw
cohort files. Generated publication candidates are review artefacts, they land under
`THESIS_WORK_DIR` rather than in the repository, and nothing here approves them for release.

## Citation

Anyone using MCS or YRBS should follow the citation requirements attached to the editions they
obtain. In particular, cite UK Data Service study number 8156 and the CDC 2023 National YRBS
dataset and documentation alongside any citation for this project.
