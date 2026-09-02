# Cross-cohort transfer of adolescent adversity prediction models under imperfect harmonisation

This repository holds the analysis pipeline for an MSc project asking a narrow question: if you
build a risk-prediction model in one adolescent cohort, does it still work in another once the
predictors and the outcome have been harmonised imperfectly?

The source cohort is the UK Millennium Cohort Study (MCS), age-14 sweep. The target is the 2023
United States National Youth Risk Behavior Survey (YRBS). The two were designed independently,
for different populations, with different instruments, so nothing about the transfer is
guaranteed to work — which is the point.

## What the two terms mean here

**Cross-cohort transfer** means fitting a model on MCS respondents and evaluating it on YRBS
respondents. Everything downstream is organised around that direction. A separate,
smaller reverse analysis (YRBS to MCS) is kept as a directional sensitivity check, not as a
second headline.

**Imperfect harmonisation** means the shared predictor set and the shared outcome are
*constructed* rather than *found*. No variable appears in both surveys in the same wording, with
the same response options, at the same age. Each harmonised construct is a decision about which
source items are close enough to treat as the same thing, and several of those decisions are
defensible rather than obvious. The consequence is that a drop in performance across the border
confounds two things — genuine population difference, and measurement difference introduced by
the harmonisation itself — and this design cannot separate them. Every reading in the analysis is
written with that limit in view.

## What the pipeline does

The outcome counts how many of five shared adversity pillars a respondent reports. The main
analysis cuts at two or more; one or more and three or more are carried throughout as
sensitivity thresholds rather than as afterthoughts.

Nine model families are fitted and compared under a fixed protocol: twenty repeated stratified
75/25 splits, one hyperparameter configuration per family and threshold selected once inside each
cohort, and the same evaluation metrics everywhere. Against that backdrop the analysis compares:

- unadapted transfer — the source model applied to the target as-is;
- label-free adaptation, which may use target covariates or the source model's own predictions
  but never a target outcome;
- label-using adaptation across target-label budgets from 50 to 2,000, with four of those budgets
  carried through a deeper operational evaluation (see *Label budgets* below);
- post-training probability adjustment;
- calibration, fixed-capacity screening, split-conformal prediction, subgroup evaluation and
  outcome-definition sensitivity;
- reverse transfer from YRBS to MCS.

### Prediction, association, and what this is not

The analysis is **predictive**. It asks how well a fitted model ranks and calibrates on
respondents it has not seen, and how that degrades across a cohort border. Model coefficients are
not interpreted as effects.

Notebook 01 contains a **descriptive association** section that compares univariate
predictor–outcome associations between the two cohorts. That exists to characterise the
harmonisation — to show where the two cohorts behave differently before any model is fitted — and
not to estimate anything.

Nothing here is **causal**. There is no identification strategy, no confounding adjustment and no
counterfactual estimand anywhere in the pipeline. The adaptation regimes change how a model is
fitted; they do not change what it identifies. A model that predicts adversity well is not
evidence about what causes it, and a screening capacity result is a statement about a ranking, not
about an intervention.

## Repository structure

```
notebooks/    the four ordered analysis notebooks
src/          data construction, recoding, modelling, transfer, evaluation, publication helpers
scripts/      promotion, notebook patching, and the public-copy tooling
spec/         harmonisation decisions, fixed model settings, public-output allow-list
tests/        synthetic-fixture test suite
requirements.txt
```

| Path | Purpose |
|---|---|
| `notebooks/01_data_and_features.ipynb` | Reads both cohorts, applies the harmonisation and outcome definitions, runs the descriptive and association checks, writes the canonical feature, pillar and attribute artefacts. |
| `notebooks/02_models_and_transfer.ipynb` | Builds the repeated splits, runs model selection, fits the model and adaptation battery, runs the label-budget and robustness analyses, writes the score handoff and the aggregate tables. |
| `notebooks/03_evaluation_and_robustness.ipynb` | Discrimination, calibration, conformal coverage, subgroup and screening-capacity behaviour, conditional uncertainty, and the publication candidates. |
| `notebooks/04_explorations.ipynb` | The reverse-transfer sensitivity analysis. |
| `src/data.py` | Splits, outcome construction, harmonised frame assembly, cohort association helpers. |
| `src/features.py`, `src/recode_mcs.py`, `src/recode_yrbs.py`, `src/recode_utils.py` | The harmonised predictor registry and the per-cohort recoders behind it. |
| `src/outcomes.py` | The five adversity pillars and the threshold definitions. |
| `src/models.py` | Model families, candidate pools, and the consensus selection procedure. |
| `src/transfer.py` | Every adaptation regime, the label-budget curve, and the cross-fitted recalibration. |
| `src/detail_budgets.py` | The detailed-evaluation label budgets and the vocabulary for keying a frame by one. |
| `src/evaluation.py` | Metrics, subgroup and capacity layers, the respondent bootstrap, conformal prediction. |
| `src/scores.py` | The person-level score handoff contract between notebooks 02 and 03. |
| `src/config.py`, `src/inputs.py` | Path resolution and the working-root writers. |
| `src/publication.py`, `src/paper.py`, `src/tex_tables.py`, `src/tables.py` | Publication candidates, figures and LaTeX fragments. |
| `spec/harmonisation_spec_v5.csv` | The cross-cohort harmonisation decision record. |
| `spec/local_model_settings.csv` | The fixed model configurations for both cohorts: 54 rows, `cohort,threshold,family,settings,protocol_id,settings_digest`. Written only by `scripts/promote_local_settings.py`. |
| `spec/public_notebook_cells.json` | Cell-level allow-list for output kept in public notebook copies. |

Participant data, harmonised row-level data, fitted models, person-level scores, working tables
and unreviewed publication candidates are all deliberately absent from the repository.

## Data access

Neither cohort is distributed here, and nothing in this repository downloads, redistributes or
reconstructs either one.

### Millennium Cohort Study (restricted)

The project uses *Millennium Cohort Study: Age 14, Sweep 6, 2015*, UK Data Service study number
SN 8156. It is Safeguarded Tier 1a: access must be obtained independently through the
[UK Data Service study record](https://doc.ukdataservice.ac.uk/doc/8156/mrdoc/UKDA/UKDA_Study_8156_Information.htm)
under an accepted application, and used under that licence and its disclosure conditions.

Once access is approved, prepare a single combined Sweep 6 parquet holding the original MCS
variables that `src/recode_mcs.py`, `src/features.py` and `src/outcomes.py` ask for, and place it
at the configured `MCS_CORE` location. The feature crosswalk in notebook 01 names the source
variable behind every retained construct.

### Youth Risk Behavior Survey (open)

The 2023 National YRBS data and documentation are on the
[CDC national YRBS datasets page](https://www.cdc.gov/yrbs/data/national-yrbs-datasets-documentation.html).
The CDC distributes the national file in Access and ASCII formats. Convert it to parquet **without
renaming any original variable** and save it as `yrbs2023_raw.parquet` at the configured
`YRBS_RAW` location. There is deliberately no downloader in this repository.

Variable wording and response codes are in the
[2023 YRBS Data User's Guide](https://www.cdc.gov/yrbs/media/pdf/2023/2023_National_YRBS_Data_Users_Guide508.pdf).

## Environment

Developed with Python 3.12.13.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

You will also need a Jupyter frontend and a kernel for this environment. Neither is pinned in
`requirements.txt`, because they are tooling for running the analysis rather than part of it.

Two things in `requirements.txt` are worth reading before you change them. The `scikit-learn`
ceiling is load-bearing: version 1.10 removes the `penalty=` argument and defaults `l1_ratio` to
0.0, which would silently turn every L1 logistic in this pipeline into an L2 one. And `POT` and
`tabpfn` are pinned but never imported — they are there so that an environment built from this
file can still reproduce the two rejected methods reported in the non-surviving-methods appendix.

The library versions actually observed during a run are recorded at runtime in
`$THESIS_WORK_DIR/tables/s3_library_versions.csv`.

## Local path configuration

Nothing a run generates is written into the repository. Two roots are configured outside it:

- `MCS_DATA_DIR` — restricted MCS inputs and secure person-level artefacts;
- `THESIS_WORK_DIR` — the open YRBS input and every file a run generates.

Neither may be the repository, a directory inside it, your home directory or a filesystem root,
and symlinks are resolved before that check. Set them in the shell that launches Jupyter:

```bash
export MCS_DATA_DIR=/absolute/path/to/a/dedicated/mcs_directory
export THESIS_WORK_DIR=/absolute/path/to/a/dedicated/working_directory
```

The intended layout puts the working root beside the clone rather than inside it:

```
thesis_final/
  project/       source, and the Git repository if you initialise one
  run_outputs/   all generated run products
```

Git should be initialised only in `project/`, never in the parent. The working root holds
MCS-derived material and git history does not forget. `.gitignore` excludes `outputs/`,
`publication_outputs/` and `publication_candidates/` unconditionally so that a stray generated
file cannot be committed by accident.

VS Code and JupyterLab do not always pass a shell environment through to a kernel. If that
happens, put each root on the first line of `.mcs_data_dir` and `.thesis_work_dir` in the
repository root instead. Both files hold a path and never data, both are git-ignored, and both are
specific to one machine.

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

If `src/_mcs_gate.py` is present, no MCS path is handed out unless `MCS_READ_OK=1` is set for that
invocation. It exists so that a process which merely inherited a shell environment cannot reach
restricted data by accident. The file is git-ignored, so a fresh clone does not have one and the
variable does nothing; add it if you want the extra gate on your own machine, and export
`MCS_READ_OK=1` alongside the two roots when you do.

## Harmonisation

Harmonisation is the part of this project most likely to affect how you read the results, so it is
worth understanding before the modelling.

The single source of truth for the predictor schema is `features.FEATURE_MAP`; the per-cohort
recoders that build it are `src/recode_mcs.py` and `src/recode_yrbs.py`, sharing helpers in
`src/recode_utils.py`. The outcome pillars and thresholds are in `src/outcomes.py`. The decision
record — which candidate constructs were kept, which were dropped, which were parked, and why —
is `spec/harmonisation_spec_v5.csv`. Those files are the description; this README deliberately
does not restate the mapping, because two descriptions eventually disagree.

Two structural consequences are worth stating plainly:

**Equivalence is claimed nowhere.** A harmonised construct is a decision that two differently
worded items are close enough to be treated as one predictor. Some of those decisions are near
enough to exact; others involve collapsing response scales, aligning different reference periods,
or accepting that one cohort asks about a behaviour and the other about its frequency. The
project does not assert measurement invariance, and none was tested.

**Missingness and coverage differ by cohort and by construct**, so a predictor can be well
measured on one side and thin on the other. `data.missingness_audit` reports per-feature and
per-subgroup missingness across cohorts, and the recoders are written so that a construct which
cannot be built raises rather than returning a frame quietly short of a predictor.

Because of both, a performance drop across the border should be read as *transfer under this
harmonisation*, not as a clean estimate of population difference.

One preprocessing choice is worth knowing before you read any number. `data.standardise_cohort`
median-imputes and z-scores each frame **against itself**, test frames included. Between cohorts
that is deliberate and is reported as the simplest label-free adaptation: the source scaler is
never refitted on the target. Within one cohort it means a model fitted in training units predicts
on a test frame centred on its own mean, so how a test row is transformed depends on the other
test rows. No outcome is involved, so this is not label leakage, but the protocol is transductive
and that is a property of the design rather than an oversight. It is recorded in
`data.standardise_cohort` and in `SplitBundle`, and it applies identically to every regime, so it
does not favour one arm over another.

## Running the analysis

Activate the environment, configure the two roots, then start Jupyter from the repository root.
Restart the kernel and run each notebook top to bottom, in order: 01, 02, 03, 04.

Notebook 02 is the expensive one. It fits the full nine-family battery across twenty seeds and
three thresholds, plus the label-budget curve, and it should be expected to run for hours rather
than minutes. Notebooks 01, 03 and 04 are comparatively quick. All four require the restricted MCS
input; none of them will run from a bare clone.

The tracked notebooks declare:

```python
PUBLIC_NOTEBOOK = False
```

That is internal mode, and it shows the full diagnostic record. The flag is presentation only: it
decides what is displayed or printed, and nothing else. Fitting, scoring, validation and file
writing are identical either way, so both settings produce the same scientific files. Do not edit
the tracked source to `True` for the final run — the public-copy tooling handles that, and
flipping it by hand would relabel internal output as public.

Run each notebook completely and in order. The public-copy workflow refuses a notebook whose
executable cells do not all carry strictly increasing execution counts, because outputs beside
broken counts do not describe one coherent run.

### Model selection and promotion

Hyperparameters are selected **once per cohort**, by the same procedure in both, and then held
fixed across all twenty evaluation splits. Section D of `02_models_and_transfer.ipynb` runs that
selection and writes two private records under `THESIS_WORK_DIR`; those records carry the
cross-validated scores behind each choice and stay outside the repository.

Section D then **stops at a promotion gate** before any model is fitted. On a first run the gate
refuses, because the promoted specification does not yet describe the selection just made. Review
the 54-row settings table the section displays, promote with the commands below, and re-run from
the gate cell — nothing above it is recomputed, because a valid record on disk is loaded rather
than re-searched.

```bash
python scripts/promote_local_settings.py                     # dry run: the full diff, nothing written
python scripts/promote_local_settings.py --apply --confirm   # writes spec/local_model_settings.csv
```

The tracked file carries the 54 configurations and two provenance identifiers — `protocol_id` for
the selection procedure and `settings_digest` for the configurations themselves — and no
cross-validated value. No notebook writes it. After promoting, re-run notebooks 02, 03 and 04 in
order: results from either side of a promotion were fitted under different configurations and must
not be mixed.

The selection is honest with respect to the evaluation splits — no outer test partition of either
cohort enters it — but the three development seeds' training partitions do overlap the twenty
evaluation seeds' training partitions. Notebook 02 records this, and
`transfer.nested_target_sensitivity_scores` implements the nested alternative that would remove the
overlap. It is off by default because it changes the estimand from "one fixed configuration
evaluated twenty times" to "a selection procedure evaluated twenty times", which is not the claim
the analysis makes.

### Label budgets

The label-budget curve spans k = 50 to 2,000 across all nine families and is the main evidence on
how much target labelling buys.

Four of those budgets are additionally carried through the detailed operational evaluation —
subgroup performance, fixed-capacity screening, conformal coverage, the respondent bootstrap and
cross-fitted recalibration. They are declared in `src/detail_budgets.py`:

```python
PRIMARY_DETAIL_BUDGET      = 500
SENSITIVITY_DETAIL_BUDGETS = (100, 200, 300)
DETAIL_BUDGETS             = (100, 200, 300, 500)
```

**k = 500 is the primary point**; the other three are secondary sensitivity points. k = 300 was
chosen to describe the transition between k = 200 and k = 500, once the discrimination-budget
curve showed where that transition sat. The detailed evaluation covers the four focal pipelines
only — target-only on
`L1_LR`, and full revision on `RF`, `HistGB` and `CatBoost` — not all nine families. The samples
are nested prefixes of the k = 500 draw, so a movement between budgets reflects fewer labels
rather than different ones.

The notebook cells for these budgets are applied by a guarded patcher rather than by hand:

```bash
python scripts/patch_detail_budgets.py --check   # classify only; writes nothing
python scripts/patch_detail_budgets.py           # apply, with byte-exact backups
```

It identifies cells by stable ID, refuses a notebook that is partially patched, unfamiliar or
carrying stored output, writes through a temporary sibling, verifies the result before replacing
the original, and is idempotent. If the notebooks in your clone already contain
`detail_budgets.DETAIL_BUDGETS`, the patch is applied and `--check` will say so.

## Outputs and how to read them

Everything a run generates is written below `THESIS_WORK_DIR`, with one exception: person-level
scores go to `MCS_DATA_DIR`, because they are person-level. Nothing is written into the
repository — not a table, not a figure, not a LaTeX fragment.

Notebook 02 writes five artefacts that have downstream readers:

| File | Grain | Read by |
|---|---|---|
| `regime_battery.csv` | one row per regime × family × arm × threshold × seed, and per budget for the label-using regimes | notebook 04 |
| `regime_battery_summary.csv` | across-seed mean, SD and percentiles | notebook 03 |
| `loo_sensitivity_summary.csv` | per-seed rows, despite the name | notebook 03 |
| `outcome_variants_summary.csv` | per-seed rows, despite the name | notebook 03 |
| `yrbs_scores.parquet` | one row per YRBS test respondent per seed, per regime, per budget | notebook 03 |

Notebook 03 turns the person-level handoff into the evaluation layers and writes publication
*candidates* — `capacity_results.csv`, `subgroup_results.csv`, `uncertainty_intervals.csv`,
`conformal_results.csv`, `threshold_sensitivity.csv`, `target_gap_results.csv` — under
`publication_candidates/` in the working root, rounded to four decimal places. Notebook 02 writes
`label_budget_results.csv` the same way. A candidate is a review artefact and nothing about
writing one implies it may be released.

### How cross-cohort performance is assessed

Performance is read against two **local references**, each a model developed *and* evaluated
inside one cohort under that cohort's own consensus selection: `mcs_internal` on the MCS side and
`yrbs_local` on the YRBS side. Neither is a ceiling — a transfer procedure exceeding one is a
possible result, not a contradiction. From those, `evaluation.add_reference_gaps` builds:

```
transfer_loss         mcs_internal − this row      what crossing the border costs
target_resource_gap   yrbs_local − unadapted       what a target-developed model would add
adaptation_gain       this row − unadapted         what an adaptation recovers
target_gap_recovered  adaptation_gain / target_resource_gap
```

Discrimination is reported as AUC and PR-AUC with the prevalence null beside every PR-AUC.
Calibration is reported as slope, intercept, Brier score and expected calibration error. Screening
behaviour is reported at fixed review capacities of 10% and 20% — precision, recall, false-positive
rate and realised flag rate, always beside the no-skill line, because a precision at a fixed
capacity is unreadable without it. The label-budget curve uses 5% and 15% instead; those are that
experiment's own operating points and the two sets should not be quoted against each other.

Five cautions apply throughout:

**Spread is not uncertainty.** The `split_sd` reported beside an across-seed mean describes how
the estimate moves across twenty *overlapping* 75/25 splits. They share most of their training
rows and about a quarter of each test set, so they are positively correlated and this spread
understates what a fresh sample would do. It is not a standard error. The only interval-bearing
quantities are notebook 03's respondent bootstrap, which resamples adolescents while holding the
fitted models, splits and adaptation samples fixed — and even those exclude model-development
uncertainty and ignore the YRBS survey design, which can understate uncertainty where responses
cluster within schools.

**No survivor averages — in the evaluation layers.** Across the subgroup, fixed-capacity,
conformal and bootstrap summaries, a metric that could not be estimated on every seed is left
blank with a recorded reason rather than averaged over whichever seeds worked. A blank in those
tables is a labelled gap, not a missing value.

**The label-budget curve is the exception, and it differs in two further ways.**
`transfer.label_budget_curve` averages the seeds that produced a value and records how many in
`n_seeds`, rather than blanking an incomplete cell; it reports population standard deviations
(`ddof=0`) where every other summary in the project reports sample ones (`ddof=1`), which makes
its spreads about 2.5% smaller at twenty seeds; and the `(pool)` rows carrying `n_pos_ksample`
record each split's positive count once per family, so their `n_seeds` reads nine times the number
of splits. None of this affects the curve's means or the nine-family comparison it exists to
support, but anyone quoting a spread from `label_budget_results.csv` beside a spread from
elsewhere in the report should reconcile the two conventions first.

**Cross-cohort metric comparisons are limited to AUC.** Precision and recall at a fixed capacity
are bounded by outcome prevalence, and the two cohorts differ in prevalence, so those quantities
are comparable *within* a cohort and not across the border. `mcs_internal` is reported at the MCS
prevalence null and everything else at the YRBS one.

**Transportability is not established by any of this.** The evaluation describes how one
harmonised model behaves on one target sample from one survey year. It says nothing about other
target populations, other harmonisations, or other years, and the measurement differences
described above sit inside every number.

## Checks

The test suite runs on synthetic fixtures. It opens no cohort, no fitted model and no score file,
and it resolves no configured path, so it works on a clone with nothing set up:

```bash
python -m pytest tests/ -q
```

The two structural checkers can be run over the tracked notebooks at any time. With no arguments
they default to `notebooks/*.ipynb`:

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

If the retain-and-clear decisions are what you expect, write the copies with `--apply` (add
`--overwrite` only if you are deliberately replacing an existing destination file):

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

## Reproducibility and governance

Randomness is controlled rather than incidental. The twenty evaluation splits are
`train_test_split` at seeds 0–19, stratified on the outcome with missing outcomes counted as
negatives for the cut only, and model fitting takes that same seed. The label-budget ordering is
drawn at the split's own seed, so the k = 500 prefix is the slice every label-using number is
computed on; the conformal division and the respondent bootstrap each take a separately declared
seed. Given the same inputs and the same promoted settings, a run reproduces.

Two identifiers carry provenance. `protocol_id` names the *selection procedure*; `settings_digest`
names the *selected configurations*. Both are stamped on `spec/local_model_settings.csv` and
checked when it is loaded — a file written under a different procedure, or edited since promotion,
is refused whole rather than repaired. Results fitted either side of a promotion are not
comparable and must not be mixed.

On governance: MCS Sweep 6 is UK Data Service Safeguarded Tier 1a, and this repository is written
so that no row-level MCS material and no fitted model reaches version control. MCS person-level
scores are excluded from the score handoff at source, and the exact MCS confusion counts and
denominators are blanked before the two battery files are written — the rates on those rows
remain, and remain MCS-derived. Those are structural safeguards, not clearance: release of any
MCS-derived aggregate requires researcher, institutional and UK Data Service review.

## What this repository can and cannot reproduce

It records the complete analysis logic, the harmonisation decisions and the fixed model
configuration. It cannot give you a one-command reproduction from a public clone. MCS access is
controlled independently by the UK Data Service, and the restricted cohort, the derived row-level
artefacts, the person-level scores and the local path configuration are all absent by design. YRBS
is open, but you still have to download and convert it yourself.

Inside a properly authorised environment the notebooks are built to run in order from the raw
cohort files. Generated publication candidates are review artefacts, they land under
`THESIS_WORK_DIR`, and nothing here approves them for release.
