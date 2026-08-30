"""Model construction, consensus selection and screening for notebook 02.

The module covers nine families and the fixed and tuned arms. Tuned settings come from
``spec/local_model_settings.csv``, which carries one fixed configuration per
``(cohort, threshold, family)`` — 54 in total. Both cohorts' configurations are chosen by the
SAME procedure, ``consensus_select`` below: three development seeds, five-fold stratified inner
cross-validation, AUC, mean-of-seed-means with a standard-deviation then pool-order tie rule.

A MODEL USES THE SETTINGS SELECTED IN THE COHORT IT IS TRAINED ON. That is the one rule the
role boundary reduces to, and it holds in both directions: MCS-trained models (the MCS local
reference, every forward transfer and adaptation regime) take the MCS mapping, and YRBS-trained
models (the YRBS local reference, and the reverse YRBS-to-MCS transfer model) take the YRBS one.

Neither selection reads an outer test partition, and neither reads the other cohort — see
``mcs_dev_frames`` and ``yrbs_dev_frames``, which are the only cohort-aware code here.

Several functions fit models and say so in their docstrings. MCS scores are not persisted.
"""

import warnings
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd
from sklearn.exceptions import ConvergenceWarning


from regime_names import FAMILIES, LR_FAMS, THRESHOLDS

# Without these filters the notebook emits three warnings per logistic fit — hundreds of lines
# during a battery run — none of which says anything about the model.
#
# THE SUPPRESSION IS INFORMED, NOT LAZY. sklearn >= 1.8 deprecates `penalty=` in favour of
# `l1_ratio`, and because `l1_ratio` now defaults to 0.0 it also warns that `penalty='l1'`
# conflicts with `l1_ratio=0.0` — and `l1_ratio=0.0` means pure L2. If sklearn resolved that
# conflict in favour of the default, every "L1" model here would silently be fitting L2.
#
# IT DOES NOT. Measured on sklearn 1.9.0, 40 features of which 3 informative:
#     penalty='l1', solver='liblinear'   37 zero coefficients, |b|_1 = 2.980
#     l1_ratio=1  (true L1)              37 zero coefficients, |b|_1 = 2.980   <- identical
#     l1_ratio=0  (true L2)               0 zero coefficients, |b|_1 = 5.643
#     penalty='l2'                        0 zero coefficients, |b|_1 = 5.643   <- identical
# `penalty=` wins. The L1 models are fitting L1 and the warning is about a future removal.
#
# THAT STOPS BEING TRUE AT sklearn 1.10, where `penalty=` is removed. At that point the
# default `l1_ratio=0.0` takes over and every L1 model becomes L2 with no warning at all.
# requirements.txt pins `<1.10` for exactly this reason. Do not lift that pin without
# re-running the check above and switching the estimators to `l1_ratio`.
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*'penalty' was deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*Inconsistent values: penalty=.*")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# The battery's model seed. A fitting constant, so it lives with the estimators that take it;
# `transfer.py` imports it from here. Seed variation in the paper enters only through
# `data.build_splits`, never through this — see the SELECTION_SEED note below.
RANDOM_STATE = 42

# The screening search draws candidates and fits its probe estimators at a FIXED seed of its
# own; the battery's model seed is RANDOM_STATE. Two names because they are two protocols.
SELECTION_SEED = 0

Family = Literal["L1_LR", "L2_LR", "EN_LR", "RF", "ET", "HistGB", "LightGBM", "XGB", "CatBoost"]
Arm = Literal["untuned", "tuned"]
Threshold = Literal[1, 2, 3]

# key: (label, canonical, native_nan). `canonical` is what decides whether the untuned arm
# uses FROZEN or the library default — it is not cosmetic.
FAM: Mapping[str, tuple] = {
    "L1_LR": ("L1 logistic", False, False), "L2_LR": ("L2 logistic", True, False),
    "EN_LR": ("Elastic-net logistic", False, False), "RF": ("Random forest", False, False),
    "ET": ("Extra trees", False, False), "HistGB": ("HistGradientBoosting", False, True),
    "LightGBM": ("LightGBM", False, True), "XGB": ("XGBoost", True, True),
    "CatBoost": ("CatBoost", True, True),
}
WARM_FAMS = {"XGB", "CatBoost", "LightGBM"}
FROZEN: Mapping[str, Mapping[str, Any]] = {
    "L2_LR": {},
    "XGB": dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8),
    "CatBoost": dict(iterations=300, depth=4, learning_rate=0.05, boosting_type="Plain"),
}


def make_estimator(family: Family, params: Mapping[str, Any] | None = None, *,
                   seed: int, njobs: int = -1):
    """Construct an unfitted estimator for one family.

    The single factory every regime calls, so that 'the XGB base' means one thing across
    naive transfer, fine-tuning, leaf refresh and the ensembles. Watch two details that
    have already caused reconciliation gaps: CatBoost `thread_count` (1 in the battery, -1 in the
    frozen conformal config, and the -1 setting is not bit-reproducible), and XGB's
    search-time `random_state=0` versus the battery's fixed 42.

    `seed` is REQUIRED and not defaulted. CatBoost is the one family whose fit depends on the
    split seed — it takes `random_seed=seed` — rather than on the fixed `RANDOM_STATE`, so
    defaulting it would silently make every CatBoost cell seed-0's model. Every other family
    ignores it.
    """
    p = dict(params or {})
    if family == "L1_LR":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, random_state=RANDOM_STATE, **p)
    if family == "L2_LR":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **p)
    if family == "EN_LR":
        from sklearn.linear_model import LogisticRegression
        p.setdefault("l1_ratio", 0.5)
        return LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000, random_state=RANDOM_STATE, **p)
    if family == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=njobs, **p)
    if family == "ET":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=njobs, **p)
    if family == "HistGB":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE, **p)
    if family == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, deterministic=True, verbose=-1, **p)
    if family == "XGB":
        import xgboost as xgb
        return xgb.XGBClassifier(eval_metric="auc", random_state=RANDOM_STATE, **p)
    if family == "CatBoost":
        from catboost import CatBoostClassifier
        base = dict(eval_metric="AUC", loss_function="Logloss", random_seed=seed,
                    allow_writing_files=False, verbose=0, thread_count=1)
        base.update(p); return CatBoostClassifier(**base)
    raise ValueError(family)


# The three per-family branches below, their order and their preprocessing reproduce the
# screening pipeline's unstandardised path rather than being a fresh design; rung 1 of the
# baseline ladder is only meaningful if it is the arrangement that was actually screened.
def source_scaled_estimator(family: Family, params: Mapping[str, Any] | None = None, *,
                            seed: int, njobs: int = -1):
    """An estimator whose preprocessing is FITTED WITH IT, so it travels with the model.

    THE DIFFERENCE FROM EVERYTHING ELSE IN THIS PIPELINE. `data.standardise_cohort` fits an
    imputer and a scaler on whichever frame it is handed, so the standardised frames in a
    `SplitBundle` are each centred on themselves — the target frames included. A model built
    here instead carries its own imputer and scaler, fitted on the source training rows, and
    applies them unchanged to whatever it is asked to predict on. No target-cohort statistic
    enters at any point.

    That is what makes it the rung BENEATH unadapted transfer rather than a variant of it.
    Unadapted transfer is already one label-free adaptation deep: the target frame has been
    re-centred on target statistics before the source model ever sees it. This is the transfer
    with no adaptation of any kind, and it is the comparison that says what cohort
    standardisation is worth.

    Three branches, by what each family needs on the raw feature scale:

      linear (L1/L2/EN)   median-impute, then z-score, then the estimator
      bagged (RF, ET)     median-impute only — neither takes NaN
      boosted             the bare estimator; HistGB, LightGBM, XGB and CatBoost handle NaN
                          natively, and imputing for them would change the model rather than
                          just its inputs

    SO THE THREE BRANCHES ARE NOT COMPARABLE TO EACH OTHER ON MISSINGNESS, and the archived
    code says so in the same words. Two boosted families see the missingness pattern; the other
    seven see a median. Read this rung against `unadapted` within a family, never across them.

    Returns an unfitted sklearn `Pipeline` for the first two branches and a bare estimator for
    the third. Both expose `fit` / `predict_proba`, which is all any caller here uses.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    est = make_estimator(family, params, seed=seed, njobs=njobs)
    if family in LR_FAMS:
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)
    if family in ("RF", "ET"):
        return make_pipeline(SimpleImputer(strategy="median"), est)
    return est                       # native NaN handling — used bare, as the archive had it


def select_params(family: Family, arm: Arm, threshold: Threshold,
                  tuned: Mapping[tuple, Mapping[str, Any]] | None = None
                  ) -> tuple[Mapping[str, Any], str]:
    """Return (params, source_label) for a family/arm/threshold cell.

    source_label distinguishes 'frozen', 'library-default' and 's3-tuned' so every
    reported row can say where its hyperparameters came from.

    Mirrors the transfer battery's main loop exactly: in the untuned arm a CANONICAL family
    (L2_LR, XGB, CatBoost — FAM[fam][1] is True) takes its FROZEN dict and everything else
    takes the library default `{}`; in the tuned arm every family takes TUNED[(t, fam)].

    `tuned` is a fixed selection table keyed (threshold, family). Omit it and the MCS mapping is
    loaded from `spec/local_model_settings.csv` by `mcs_settings` below — MCS, because every
    caller of this function is configuring a model trained on MCS. Pass the mapping explicitly if
    you already hold it.

    THE YRBS MAPPING NEVER ARRIVES HERE. A YRBS-trained model takes `yrbs_settings()` at its own
    call site (`transfer.reference_scores`, `transfer.reverse_transfer_scores`), which is what
    keeps the two cohorts' configurations from crossing.
    """
    canon = FAM[family][1]
    if arm == "untuned":
        return (FROZEN[family] if canon else {}), ("frozen" if canon else "library_default")
    if tuned is None:
        tuned = mcs_settings()
    return tuned[(threshold, family)], MCS_HYPERPARAMETER_SOURCE


# ---------------------------------------------------------------- the tracked specification
#
# ONE FILE, FIFTY-FOUR ROWS, BOTH COHORTS. `spec/local_model_settings.csv` carries
# `cohort,threshold,family,settings,protocol_id,settings_digest` and nothing else. One file
# because promotion is then a single `os.replace`, which either happens or does not; two files
# would need two replacements, and two replacements are not one transaction — a crash between
# them would leave the MCS and YRBS halves from different selections with nothing to detect it.
#
# NO RESULT QUANTITY IS TRACKED. The three seed-level CV AUCs, their mean and their standard
# deviation stay in the private working records under $THESIS_WORK_DIR. The MCS ones are
# MCS-derived and need separate disclosure review before they could go anywhere else; none of
# them is needed to state which configuration a model was fitted under.

COHORTS: Sequence[str] = ("mcs", "yrbs")
SETTINGS_COLUMNS: Sequence[str] = (
    "cohort", "threshold", "family", "settings", "protocol_id", "settings_digest")

# The per-cohort labels a metric row carries in `hyperparameter_source`, so that every reported
# row says which cohort chose its configuration.
MCS_HYPERPARAMETER_SOURCE = "mcs_consensus3"
YRBS_HYPERPARAMETER_SOURCE = "yrbs_consensus3"


def load_local_settings(path=None) -> dict:
    """Both cohorts' fixed configurations: `{"mcs": {...27...}, "yrbs": {...27...}}`.

    Each inner mapping is keyed `(threshold:int, family:str)` — NEVER by seed. The whole point of
    a fixed mapping is that the twenty evaluation splits share one configuration per cell, so a
    seed in the key would silently turn the reported across-split spread into a search artefact.

    REFUSES RATHER THAN REPAIRS. A `protocol_id` that is not the live one means the file was
    written under a different selection procedure; a `settings_digest` that does not match the
    rows means the file has been edited since it was promoted. Either way the configurations are
    somebody else's answer and are refused whole.

    The `settings` column holds a Python dict repr, so `ast.literal_eval` parses it; `json.loads`
    would fail on the single quotes.

    SIDE EFFECTS: reads and parses one tracked CSV. Fits nothing, writes nothing.
    """
    import ast

    import pandas as pd

    if path is None:
        import config as _C
        path = Path(_C.LOCAL_MODEL_SETTINGS)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} is not present. It is the tracked specification holding both cohorts' "
            f"fixed configurations, and it is written only by "
            f"scripts/promote_local_settings.py after a reviewed run of Notebook 02's "
            f"selection section. Nothing here re-derives it, and no run may write it.")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)

    absent = [c for c in SETTINGS_COLUMNS if c not in frame.columns]
    if absent:
        raise ValueError(
            f"{path.name} is missing column(s) {absent}. The tracked specification carries "
            f"exactly {list(SETTINGS_COLUMNS)}.")
    unexpected = [c for c in frame.columns if c not in set(SETTINGS_COLUMNS)]
    if unexpected:
        raise ValueError(
            f"{path.name} carries unexpected column(s) {unexpected}. Nothing but the declared "
            f"six belongs in it — a cross-validated score in the tracked specification would be "
            f"an MCS-derived result in the repository.")

    live = protocol_id()
    declared = sorted(set(frame["protocol_id"]))
    if declared != [live]:
        raise ValueError(
            f"{path.name} declares protocol_id {declared}; the live procedure is {live!r}. The "
            f"configurations in it were chosen by a different selection procedure — a different "
            f"candidate pool, development-seed set, fold count, objective or tie rule — so they "
            f"are that procedure's answer and not this one's. Re-run Notebook 02's selection "
            f"section and promote.")

    digests = sorted(set(frame["settings_digest"]))
    if len(digests) != 1:
        raise ValueError(
            f"{path.name} carries {len(digests)} distinct settings_digest values. One file "
            f"describes one promotion, so the digest is the same on every row.")
    recomputed = settings_digest(frame[list(SETTINGS_COLUMNS[:4])])
    if digests[0] != recomputed:
        raise ValueError(
            f"{path.name} records settings_digest {digests[0]!r} but its rows hash to "
            f"{recomputed!r}. The file has been edited since it was promoted, so what it "
            f"declares and what it holds are different sets of configurations.")

    if frame.duplicated(["cohort", "threshold", "family"]).any():
        dup = frame[frame.duplicated(["cohort", "threshold", "family"], keep=False)]
        raise ValueError(
            f"{path.name} has duplicate (cohort, threshold, family) keys:\n"
            f"{dup[['cohort', 'threshold', 'family']].to_string(index=False)}")

    out: dict = {cohort: {} for cohort in COHORTS}
    for _, row in frame.iterrows():
        cohort = str(row["cohort"])
        if cohort not in out:
            raise ValueError(
                f"{path.name} carries cohort {cohort!r}; the declared cohorts are "
                f"{list(COHORTS)}.")
        family = str(row["family"])
        if family not in set(FAMILIES):
            raise ValueError(
                f"{path.name} carries family {family!r}, which is not one of the nine in "
                f"`regime_names.FAMILIES`.")
        threshold = int(str(row["threshold"]).replace(">=", ""))
        if threshold not in set(THRESHOLDS):
            raise ValueError(
                f"{path.name} carries threshold {row['threshold']!r}, which is not one of "
                f"{list(THRESHOLDS)}.")
        out[cohort][(threshold, family)] = ast.literal_eval(row["settings"])

    expected = len(THRESHOLDS) * len(FAMILIES)
    for cohort in COHORTS:
        if len(out[cohort]) != expected:
            raise ValueError(
                f"{path.name} carries {len(out[cohort])} {cohort.upper()} configuration(s) for a "
                f"{len(THRESHOLDS)} x {len(FAMILIES)} (threshold x family) grid; expected "
                f"{expected}. Every family needs a configuration at every threshold, in both "
                f"cohorts.")
    return out


def mcs_settings(path=None) -> Mapping[tuple, Mapping[str, Any]]:
    """The MCS fixed configurations, keyed `(threshold, family)`.

    What every MCS-TRAINED model is fitted under: the MCS local reference, the canonical source
    models, and every forward transfer and adaptation regime built from them.
    """
    return load_local_settings(path)["mcs"]


def yrbs_settings(path=None) -> Mapping[tuple, Mapping[str, Any]]:
    """The YRBS fixed configurations, keyed `(threshold, family)`.

    What every YRBS-TRAINED model is fitted under: the YRBS local reference, and the reverse
    YRBS-to-MCS transfer model in notebook 04. It reaches no forward adaptation procedure.
    """
    return load_local_settings(path)["yrbs"]


# The target-side selections notebook 02 saved, and how to ask for them
#
# WHAT THIS IS FOR. `select_target_params` below runs a five-fold search inside every
# (family, threshold, seed) cell's YRBS training partition. That search is the expensive part of
# notebook 02 and it is not repeated: notebook 02 writes its result and notebook 04's reverse
# reading loads it. THE MODELS STILL HAVE TO BE FITTED — what is saved is a configuration, not a
# fitted model, and every role in the reverse reading fits on its own training partition.
#
# NOTEBOOK 02 READS IT BACK, AND THE PREPROCESSING VERSION IS WHAT MAKES THAT SAFE. The search
# is the expensive part of a run, so notebook 02 loads a complete saved selection rather than
# repeating it. A selection is a choice made against a feature schema: a file written under a
# different representation of the predictors describes a search that no longer exists, and the
# loader refuses it rather than letting it reach a fit. See `data.PREPROCESSING_VERSION`.
TARGET_SETTINGS_FILENAME = "yrbs_resource_rich_settings.json"
TARGET_SETTINGS_SUBDIR = "tables"
TARGET_SETTINGS_FORMAT = 2

# The eight fields every saved record carries. They are `select_target_params`' own return keys
# plus the three that identify the cell, so a loaded record can be passed straight back into the
# code that would have produced it.
TARGET_SETTINGS_FIELDS: Sequence[str] = (
    "family", "threshold", "seed", "status", "folds", "candidates", "cv_auc", "params",
)


def check_preprocessing(payload: Mapping[str, Any], where: str) -> None:
    """Refuse a saved artefact whose preprocessing is not the one the live schema describes.

    Checks both halves of the declaration: the version name, and the model feature schema it
    was written against. The name alone would pass a schema that had changed without the name
    being bumped; the schema alone would pass two different preprocessings that happened to
    produce the same columns. Neither is worth the ambiguity, and both are cheap.

    An artefact with no declaration at all is a legacy artefact and is refused by the same
    route, because "no version recorded" is not evidence that the live one applies.
    """
    import data as D

    version = payload.get("preprocessing_version")
    if version != D.PREPROCESSING_VERSION:
        raise ValueError(
            f"{where} declares preprocessing_version {version!r}; the live schema is "
            f"{D.PREPROCESSING_VERSION!r}. The selections in it were made against a different "
            f"representation of the predictors, so they are that run's answer and not this "
            f"one's. Re-run the selection rather than reusing them.")
    columns = payload.get("model_features")
    if list(columns or ()) != list(D.MODEL_FEATURE_COLUMNS):
        raise ValueError(
            f"{where} records a model feature schema of {len(list(columns or ()))} column(s) "
            f"that is not the live `data.MODEL_FEATURE_COLUMNS` "
            f"({len(D.MODEL_FEATURE_COLUMNS)} columns, in a declared order). A configuration "
            f"chosen over different predictors is not a configuration for these.")


def target_settings_payload(settings: Mapping[tuple, Mapping[str, Any]], *,
                            families: Sequence[str], thresholds: Sequence[int],
                            seeds: Sequence[int]) -> dict:
    """The saved object for a complete target-side selection, ready to serialise.

    `settings` is keyed `(family, threshold, seed)` and holds `select_target_params`' own
    records. The declared grid must be covered exactly: a missing cell would save an incomplete
    selection that a later run would then load as if it were complete, and an unexpected cell
    means the grid the search ran over is not the grid being declared.

    Carries the preprocessing declaration `check_preprocessing` reads back, so what the
    selections were made under travels with them rather than being inferred later.
    """
    import data as D

    wanted = [(str(f), int(t), int(s)) for f in families for t in thresholds for s in seeds]
    keyed = {(str(f), int(t), int(s)): record for (f, t, s), record in settings.items()}
    absent = [key for key in wanted if key not in keyed]
    extra = [key for key in keyed if key not in set(wanted)]
    if absent or extra:
        raise ValueError(
            f"the selection covers the declared grid imperfectly: {len(absent)} cell(s) missing "
            f"and {len(extra)} beyond it. A partial selection is not saved, because a later run "
            f"would load it as a complete one.")

    records = []
    for key in wanted:
        family, threshold, seed = key
        record = keyed[key]
        cv_auc = record.get("cv_auc")
        params = record.get("params")
        records.append(dict(
            family=family, threshold=threshold, seed=seed,
            status=str(record["status"]), folds=int(record["folds"]),
            candidates=int(record["candidates"]),
            cv_auc=(None if cv_auc is None or cv_auc != cv_auc else float(cv_auc)),
            params=(None if not params else _json_safe(params))))

    return {
        "format_version": TARGET_SETTINGS_FORMAT,
        "preprocessing_version": D.PREPROCESSING_VERSION,
        "model_features": list(D.MODEL_FEATURE_COLUMNS),
        "selection_method": "five-fold AUC selection within each YRBS training partition",
        "families": [str(f) for f in families],
        "thresholds": [int(t) for t in thresholds],
        "seeds": [int(s) for s in seeds],
        "settings": records,
    }


def save_target_settings(settings: Mapping[tuple, Mapping[str, Any]], *,
                         families: Sequence[str], thresholds: Sequence[int],
                         seeds: Sequence[int], path=None):
    """Write a complete target-side selection under the working root, and read it back.

    Returns the path written. The read-back goes through `load_target_settings`, so what is
    verified is that the file the loader will see is the selection that was made — a written
    file that the reader would refuse is a lost search, and it is cheaper to find out here.

    SIDE EFFECTS: creates the `tables` directory under the working root if it is absent, and
    replaces the settings file. Writes nothing inside the repository and fits nothing.
    """
    import json
    from pathlib import Path as _Path

    payload = target_settings_payload(settings, families=families, thresholds=thresholds,
                                      seeds=seeds)
    location = _Path(path) if path is not None else target_settings_path()
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    reopened = load_target_settings(location)
    target_settings_coverage(reopened, families=families, thresholds=thresholds, seeds=seeds)
    return location


def _json_safe(value):
    """A selected parameter as an ordinary JSON value, or a refusal naming its type.

    Numpy scalars reach here because the search spaces hold them; a non-finite float does not,
    because a configuration that is not a number cannot be reconstructed from the file.
    """
    import numpy as np

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("a selected model parameter is not finite, so it cannot be saved")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("a selected parameter dictionary has a non-text key")
        return {key: _json_safe(item) for key, item in value.items()}
    raise TypeError(f"cannot save a selected parameter of type {type(value).__name__}")


def target_settings_path():
    """Where notebook 02 wrote the saved selections. Resolves the working root on the call."""
    import config as _C
    return _C.work_path(TARGET_SETTINGS_SUBDIR) / TARGET_SETTINGS_FILENAME


def load_target_settings(path=None) -> Mapping[tuple, Mapping[str, Any]]:
    """The saved target-side selections, keyed `(family:str, threshold:int, seed:int)`.

    Returns one record per cell in the shape `select_target_params` returns — `params`,
    `status`, `folds`, `candidates`, `cv_auc` — so a caller consumes a loaded selection and a
    freshly computed one through the same code. `params` is `None` wherever nothing was
    selected, and `status` says why.

    REFUSES RATHER THAN REPAIRS. An unknown format version, a preprocessing version that is not
    the live one, a missing field, a duplicate key or a record whose status and parameters
    disagree is an error naming the cell. A saved selection that cannot be read is a reason to
    stop, not a reason to fall back to a default configuration: the whole point of the file is
    that a caller uses the configuration the target-side search chose and no other.

    NOTHING IS MIGRATED. A selection made under a superseded preprocessing is a valid record of
    what was chosen then and is not a configuration for now, so it is refused whole rather than
    read cell by cell.

    SIDE EFFECTS: reads and parses one JSON file under the working root. Fits nothing, writes
    nothing.
    """
    import json
    from pathlib import Path as _Path

    location = _Path(path) if path is not None else target_settings_path()
    if not location.exists():
        raise FileNotFoundError(
            f"{location.name} is not under the working root. It is written by notebook 02's "
            f"target-side search, which is where it is recomputed when it is absent; nothing "
            f"else re-derives it. Notebook 04 reads it and stops here rather than searching, "
            f"because a selection made in the reverse reading would be that reading's answer "
            f"and not the one the forward analysis reports.")
    payload = json.loads(location.read_text(encoding="utf-8"))

    version = payload.get("format_version")
    if version != TARGET_SETTINGS_FORMAT:
        raise ValueError(
            f"{location.name} declares format_version {version!r}; this reader understands "
            f"{TARGET_SETTINGS_FORMAT}. The layout of a saved selection has changed, so the "
            f"records cannot be read as they stand.")
    check_preprocessing(payload, location.name)
    records = payload.get("settings")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{location.name} carries no `settings` list, so it holds no selection.")

    out: dict = {}
    for record in records:
        missing = [f for f in TARGET_SETTINGS_FIELDS if f not in record]
        if missing:
            raise ValueError(
                f"{location.name}: a saved record is missing field(s) {missing}. Every record "
                f"carries all {len(TARGET_SETTINGS_FIELDS)}.")
        key = (str(record["family"]), int(record["threshold"]), int(record["seed"]))
        if key in out:
            raise ValueError(
                f"{location.name}: {key} appears more than once, so which configuration the "
                f"cell used is undetermined.")
        status, params = str(record["status"]), record["params"]
        if status == TARGET_SELECTION_SELECTED and not params:
            raise ValueError(
                f"{location.name}: {key} reports {TARGET_SELECTION_SELECTED!r} and carries no "
                f"parameters. A selection with nothing selected cannot be used, and there is "
                f"no configuration to fall back to that would still be this one.")
        if status != TARGET_SELECTION_SELECTED and params:
            raise ValueError(
                f"{location.name}: {key} reports {status!r} and yet carries parameters. A cell "
                f"that could not be searched has nothing to have chosen.")
        cv_auc = record["cv_auc"]
        out[key] = dict(params=(dict(params) if params else None), status=status,
                        folds=int(record["folds"]), candidates=int(record["candidates"]),
                        cv_auc=(float("nan") if cv_auc is None else float(cv_auc)))
    return out


def target_settings_coverage(settings: Mapping[tuple, Mapping[str, Any]], *,
                             families: Sequence[str], thresholds: Sequence[int],
                             seeds: Sequence[int]) -> dict:
    """Check a loaded selection covers a declared grid, and count how the cells came out.

    Returns `{"expected": n, "selected": n, "non_estimable": n}` over the declared grid alone,
    so a file holding more cells than are wanted is not counted as coverage of them.

    A MISSING CELL IS AN ERROR AND A NON-ESTIMABLE ONE IS NOT. The first means the file does not
    describe the run being asked for; the second is a recorded outcome of the search, and the
    caller keeps the cell and reports the reason rather than borrowing a configuration from a
    neighbouring seed or threshold.
    """
    wanted = [(f, int(t), int(s)) for f in families for t in thresholds for s in seeds]
    absent = [key for key in wanted if key not in settings]
    if absent:
        raise ValueError(
            f"the saved selection is missing {len(absent)} of {len(wanted)} declared "
            f"(family, threshold, seed) cells, e.g. {absent[:4]}. Nothing here fills a gap: a "
            f"configuration borrowed from another cell is that cell's answer, not this one's.")
    selected = sum(1 for key in wanted
                   if settings[key]["status"] == TARGET_SELECTION_SELECTED)
    return {"expected": len(wanted), "selected": selected,
            "non_estimable": len(wanted) - selected}


# Selection: the search spaces, once
# THE KEY ORDER IS PART OF THE SPECIFICATION. `ParameterSampler` consumes the dict in insertion
# order, so REORDERING THESE CHANGES WHICH 40 CONFIGS ARE DRAWN at random_state=0 — the grids
# would still be the same grids, and the search would no longer be the same search.
TREE_SEARCH_DIST: Mapping[str, Mapping[str, list]] = {
    "CatBoost": dict(depth=[2, 4, 6], learning_rate=[0.02, 0.05, 0.1], iterations=[200, 300, 500],
                     l2_leaf_reg=[1, 3, 10], boosting_type=["Plain", "Ordered"]),
    "XGB": dict(max_depth=[2, 3, 4, 6], learning_rate=[0.02, 0.05, 0.1], n_estimators=[200, 300, 500],
                subsample=[0.7, 0.8, 1.0], colsample_bytree=[0.7, 0.8, 1.0],
                min_child_weight=[1, 5, 10], reg_lambda=[1, 5, 10]),
    "HistGB": dict(max_iter=[200, 300, 500], learning_rate=[0.02, 0.05, 0.1],
                   max_depth=[None, 2, 4, 6], l2_regularization=[0, 1, 10]),
    "LightGBM": dict(n_estimators=[300, 500], learning_rate=[0.02, 0.05, 0.1], num_leaves=[7, 15, 31],
                     min_child_samples=[5, 20, 50], reg_lambda=[0, 1, 10]),
    "RF": dict(n_estimators=[300, 500], max_depth=[None, 4, 8, 16],
               min_samples_leaf=[1, 5, 20], max_features=["sqrt", 0.5]),
    "ET": dict(n_estimators=[300, 500], max_depth=[None, 4, 8, 16],
               min_samples_leaf=[1, 5, 20], max_features=["sqrt", 0.5]),
}

LR_SEARCH_GRID: Mapping[str, list] = {
    "L1_LR": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "L2_LR": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "EN_LR": [{"C": c, "l1_ratio": l} for c in (0.01, 0.1, 1.0) for l in (0.15, 0.5, 0.85)],
}

SEARCH_N_ITER = 40
CV_FOLDS, CV_SHUFFLE_SEED = 5, 0


# THE ONE POOL. Both cohorts' selections and the optional nested sensitivity all call this, so
# "the two cohorts searched the same space in the same order" is true by construction rather than
# by two lists that happen to agree. The nine per-family sizes are 4, 4, 9 and 40 x 6 = 257 per
# threshold, and `tests/test_consensus_selection.py` pins them.
def candidate_pool(family: Family) -> list:
    """One family's candidate list, constructed deterministically.

    For the LR families this is the full grid, in declared order. For the six tree/GBM families
    it is `ParameterSampler(dist, n_iter=min(40, grid_size), random_state=SELECTION_SEED)`.

    THE ORDER IS PART OF THE SPECIFICATION, because it is the last tie-break in
    `rank_candidates`: where two candidates have the same three-seed mean AUC and the same
    standard deviation, the one earlier in this list wins. Reordering `TREE_SEARCH_DIST` changes
    which forty configurations are drawn and therefore which one a tie resolves to.
    """
    import numpy as np
    from sklearn.model_selection import ParameterSampler
    if family in LR_FAMS:
        return [dict(d) for d in LR_SEARCH_GRID[family]]
    dist = TREE_SEARCH_DIST[family]
    grid_size = int(np.prod([len(v) for v in dist.values()]))
    return [dict(d) for d in ParameterSampler(dist, n_iter=min(SEARCH_N_ITER, grid_size),
                                              random_state=SELECTION_SEED)]


# ================================================================ consensus selection
#
# ONE PROCEDURE, RUN ONCE PER COHORT. `consensus_select` is the whole of the scientific core and
# it has no cohort concept at all: it takes prepared training frames keyed `(threshold, seed)`
# and nothing else. That is what makes the information boundary a property of the INTERFACE
# rather than of discipline — an outer test frame and the other cohort are not merely unused
# here, they are not in scope, so no code path can reach one.
#
# The two adapters below (`mcs_dev_frames`, `yrbs_dev_frames`) are the only cohort-aware code,
# and each reads exactly two named keys out of a split bundle.
#
# THE PROTOCOL, in full:
#
#   development seeds   exactly DEV_SEEDS = (0, 1, 2)
#   data                that cohort's OUTER-TRAINING partition for that seed and threshold
#   inner folds         StratifiedKFold(5, shuffle=True, random_state=<development seed>),
#                       built ONCE per (threshold, seed) and shared by every candidate
#   preprocessing       each fold half through `data.standardise_cohort` SEPARATELY (see
#                       `prepare_folds` for why this convention and not a pipeline)
#   estimator           `make_estimator(family, candidate, seed=<development seed>)`
#   objective           AUC on each fold's held-out half
#   aggregation         mean fold AUC WITHIN each development seed; then the mean of those
#                       three seed-level means, and their POPULATION sd (ddof=0)
#   eligibility         estimable on ALL THREE development seeds, or not a candidate at all
#   ranking             highest three-seed mean AUC; exact tie -> lowest three-seed sd;
#                       still tied -> candidate-pool order
#
# THIS IS A REPLACEMENT, NOT A RECONSTRUCTION. It does not reproduce, recover or approximate any
# earlier selection, and nothing here should be read as evidence about one.

DEV_SEEDS: tuple = (0, 1, 2)
OBJECTIVE = "auc"
TIE_RULE = "mean_desc,sd_asc,pool_order"
PROTOCOL_VERSION = "consensus3"
SELECTION_SELECTED = "selected"

# Why a candidate is not eligible, and why a cell could not be selected. Both are recorded
# rather than inferred from an absence.
NOT_ESTIMABLE_FOLDS = "non_estimable:too_few_minority_for_five_folds"
NOT_ESTIMABLE_SCORED = "non_estimable:no_candidate_scored_on_every_development_seed"


def candidate_pool_fingerprint(families: Sequence[str] = FAMILIES) -> str:
    """A digest of the candidate space, so a changed pool invalidates a saved record.

    Covers every family's pool in order, the draw seed and the draw size — which between them
    determine both WHICH candidates exist and, because pool order is the last tie-break, which
    one a tie resolves to.
    """
    import hashlib

    payload = [f"selection_seed={SELECTION_SEED}", f"n_iter={SEARCH_N_ITER}"]
    for family in families:
        payload.append(family)
        for candidate in candidate_pool(family):
            payload.append(repr(sorted(candidate.items())))
    return "sha256:" + hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


def protocol_id(families: Sequence[str] = FAMILIES) -> str:
    """The identifier of the selection PROCEDURE — not of what it selected.

    Two different selections run under the same procedure share this string and differ in
    `settings_digest`. A matching `protocol_id` alone is therefore NOT sufficient to conclude
    that two sets of results are comparable; both identifiers have to agree.
    """
    fingerprint = candidate_pool_fingerprint(families).split(":")[-1][:12]
    return (f"{PROTOCOL_VERSION}-seeds{''.join(str(s) for s in DEV_SEEDS)}"
            f"-cv{CV_FOLDS}-{OBJECTIVE}-{fingerprint}")


def settings_digest(rows) -> str:
    """A digest of the SELECTED CONFIGURATIONS — not of the procedure that chose them.

    `rows` is anything iterable of `(cohort, threshold, family, settings)`, or a DataFrame
    carrying those four columns. The digest is taken over sorted rows, and the settings mapping
    is rendered as canonical JSON. That makes it independent of dictionary insertion order and
    of JSON's `sort_keys=True`, while parsing the Python-dict text stored in the promoted CSV
    makes that text and the in-memory mapping hash alike.
    """
    import ast
    import hashlib
    import json

    if hasattr(rows, "itertuples"):
        rows = [(r.cohort, r.threshold, r.family, r.settings) for r in rows.itertuples()]

    def canonical_settings(value) -> str:
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                pass
        return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)

    rendered = []
    for cohort, threshold, family, settings in rows:
        rendered.append("|".join((str(cohort), str(threshold), str(family),
                                  canonical_settings(settings))))
    rendered.sort()
    return "sha256:" + hashlib.sha256("\n".join(rendered).encode("utf-8")).hexdigest()


def inner_folds(y, *, folds: int = CV_FOLDS, seed: int):
    """Stratified inner folds over one cohort's outer-training rows, or `None` if unsupported.

    THE CONDITION IS ON THE SMALLER OUTCOME CLASS, not on the row count: stratified folds put at
    least one member of each class in every fold. Returning `None` is how "no search is possible
    here" is reported; the caller records it and emits no configuration.

    One definition, used by both cohorts' consensus selection and by the optional nested
    sensitivity, so none of the three can drift into a different notion of an inner fold.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    labels = np.asarray(y, float)
    labels = labels[~np.isnan(labels)].astype(int)
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    if int(min(np.bincount(labels, minlength=2))) < folds:
        return None
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def prepare_folds(X, y, *, folds: int = CV_FOLDS, seed: int):
    """The fold halves every candidate is scored on, or `None` where no fold is supported.

    Returns a list of `(X_fit, y_fit, X_held, y_held)`, one per fold. BUILT ONCE per cohort,
    threshold and development seed and reused across all 257 candidates, so every candidate is
    compared on identical partitions and identical transformed frames — a candidate that got its
    own fold draw would be compared against a different question.

    PREPROCESSING, AND WHAT IT IS AND IS NOT. Each fold half goes through
    `data.standardise_cohort` SEPARATELY: the fitting half is median-imputed and z-scored against
    itself, and the held-out half is median-imputed and z-scored against itself.

    That is the convention every frame in this pipeline is fitted and evaluated under —
    `build_splits` standardises each frame against itself, and a source model fitted on `Xm_cs`
    predicts on `Xm_te_cs`, which was centred on the test frame's own statistics. An sklearn
    pipeline would instead fit the scaler on the fold's fitting half and apply it to the held-out
    half, which is a DIFFERENT transformation from any that a reported model is evaluated under,
    and selecting under it would answer a different question from the one the analysis asks.

    IT IS NOT CONVENTIONAL INDUCTIVE PREPROCESSING AND MUST NOT BE DESCRIBED AS SUCH. Because the
    held-out half is standardised against itself, how one held-out row is transformed depends on
    the other rows in that held-out batch: the validation transformation is TRANSDUCTIVE WITHIN
    THE HELD-OUT COVARIATE BATCH. What it does guarantee, and what the leakage boundary needs, is
    that no held-out row contributes anything to the fitting half's model or to the fitting half's
    transformation.

    A residual property, recorded rather than corrected: a fold's held-out half is one fifth of a
    training partition and so a SMALLER batch than the outer test frame the reported models are
    evaluated on, so the transductive standardisation is estimated from fewer rows here than
    there.
    """
    import numpy as np

    import data as D

    splitter = inner_folds(y, folds=folds, seed=seed)
    if splitter is None:
        return None
    labels = np.asarray(y, float).astype(int)
    prepared = []
    for fit_rows, held_rows in splitter.split(X, labels):
        prepared.append((D.standardise_cohort(X.iloc[fit_rows]), labels[fit_rows],
                         D.standardise_cohort(X.iloc[held_rows]), labels[held_rows]))
    return prepared


def score_candidate(family: Family, params: Mapping[str, Any], prepared, *, seed: int):
    """One candidate's mean fold AUC on ONE development seed's folds, or `None`.

    `None` means the candidate could not be scored here at all — every fold's held-out half was
    single-class. It is not zero and it is not a low score: `rank_candidates` treats it as
    "this seed did not contribute", and a candidate that fails to score on any one development
    seed is ineligible entirely rather than being ranked on the seeds that worked.

    SIDE EFFECTS: fits `len(prepared)` probe models in memory and discards them.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    scores = []
    for X_fit, y_fit, X_held, y_held in prepared:
        if np.unique(y_held).size < 2:
            continue
        probe = make_estimator(family, params, seed=seed)
        probe.fit(X_fit, y_fit)
        scores.append(float(roc_auc_score(y_held, probe.predict_proba(X_held)[:, 1])))
    if not scores:
        return None
    return float(np.mean(scores))


def rank_candidates(pool: Sequence[Mapping[str, Any]], per_seed_means: Sequence[Sequence],
                    *, dev_seeds: Sequence[int] = DEV_SEEDS) -> list:
    """Every candidate, aggregated and ordered by the selection rule.

    `per_seed_means[i]` is candidate `i`'s per-development-seed mean AUCs, in `dev_seeds` order,
    with `None` wherever that seed could not score it.

    AGGREGATION. Mean within seed has already happened; here the three seed-level means are
    summarised by their mean and their POPULATION standard deviation (`ddof=0`). Population
    rather than sample because the three development seeds are the whole declared set, not a
    sample from a larger one — and because with n=3 the choice only rescales the tie-break, which
    is order-preserving either way.

    ELIGIBILITY. A candidate is eligible only where every one of the three development seeds
    produced a mean. A candidate ranked on two seeds would be competing under a different
    protocol from its neighbours.

    ORDER. `(-mean, sd, pool_index)`. Highest three-seed mean first; an EXACT tie on the mean
    (tested by float equality, not a tolerance — a tolerance would make the rule depend on an
    arbitrary epsilon) is broken by the lower standard deviation; a remaining exact tie by
    position in `candidate_pool`, which is deterministic.

    Returns a list of dicts in rank order, each carrying `rank`, `pool_index`, `eligible`,
    `per_seed_mean_auc`, `mean_auc`, `sd_auc` and `params`. INELIGIBLE CANDIDATES ARE RETAINED,
    ranked last, so the saved record shows what was considered and not merely what won.
    """
    import numpy as np

    scored, unscored = [], []
    for index, (params, means) in enumerate(zip(pool, per_seed_means)):
        record = dict(pool_index=index, params=dict(params),
                      per_seed_mean_auc=[None if m is None else float(m) for m in means])
        if any(m is None for m in means):
            record.update(eligible=False, mean_auc=None, sd_auc=None)
            unscored.append(record)
            continue
        values = np.asarray([float(m) for m in means], dtype=float)
        record.update(eligible=True, mean_auc=float(values.mean()),
                      sd_auc=float(values.std(ddof=0)))
        scored.append(record)

    scored.sort(key=lambda r: (-r["mean_auc"], r["sd_auc"], r["pool_index"]))
    unscored.sort(key=lambda r: r["pool_index"])
    ordered = scored + unscored
    for position, record in enumerate(ordered, start=1):
        record["rank"] = position
    return ordered


def _tie_broken_on(ranking: Sequence[Mapping]) -> str:
    """Which clause of the tie rule actually decided the winner.

    Recorded because "the mean decided it" and "three candidates tied on the mean and the sd, and
    pool order decided it" are very different statements about how firm a selection is, and the
    second is invisible once only the winner is stored.
    """
    eligible = [r for r in ranking if r["eligible"]]
    if not eligible:
        return "none"
    best = eligible[0]
    on_mean = [r for r in eligible if r["mean_auc"] == best["mean_auc"]]
    if len(on_mean) == 1:
        return "mean"
    on_sd = [r for r in on_mean if r["sd_auc"] == best["sd_auc"]]
    return "sd" if len(on_sd) == 1 else "pool_order"


def select_cell(family: Family, dev_frames: Mapping, *, threshold: int,
                dev_seeds: Sequence[int] = DEV_SEEDS, folds: int = CV_FOLDS) -> dict:
    """One `(threshold, family)` cell's fixed configuration, with its full candidate ranking.

    `dev_frames` maps `(threshold, seed) -> (X, y)` and is the ONLY data this sees. It holds one
    cohort's outer-training frames and nothing else — no test partition, no other cohort, no
    bundle to reach either through.

    Returns a record carrying `params`, `status`, `folds`, `candidates`, `dev_seeds`,
    `per_seed_mean_auc`, `mean_auc`, `sd_auc`, `tie_broken_on` and `ranking`.

    SIDE EFFECTS: fits `candidates x folds x len(dev_seeds)` probe models and discards them.
    """
    pool = candidate_pool(family)
    prepared_by_seed, unusable = {}, []
    for seed in dev_seeds:
        X, y = dev_frames[(int(threshold), int(seed))]
        prepared = prepare_folds(X, y, folds=folds, seed=int(seed))
        if prepared is None:
            unusable.append(int(seed))
        prepared_by_seed[int(seed)] = prepared

    if unusable:
        return dict(params=None, status=NOT_ESTIMABLE_FOLDS, folds=0, candidates=len(pool),
                    dev_seeds=[int(s) for s in dev_seeds], per_seed_mean_auc=None,
                    mean_auc=None, sd_auc=None, tie_broken_on="none",
                    unusable_dev_seeds=unusable, ranking=[])

    per_seed_means = []
    for params in pool:
        per_seed_means.append([score_candidate(family, params, prepared_by_seed[int(seed)],
                                               seed=int(seed))
                               for seed in dev_seeds])

    ranking = rank_candidates(pool, per_seed_means, dev_seeds=dev_seeds)
    eligible = [r for r in ranking if r["eligible"]]
    if not eligible:
        return dict(params=None, status=NOT_ESTIMABLE_SCORED, folds=int(folds),
                    candidates=len(pool), dev_seeds=[int(s) for s in dev_seeds],
                    per_seed_mean_auc=None, mean_auc=None, sd_auc=None,
                    tie_broken_on="none", ranking=ranking)

    winner = eligible[0]
    return dict(params=dict(winner["params"]), status=SELECTION_SELECTED, folds=int(folds),
                candidates=len(pool), dev_seeds=[int(s) for s in dev_seeds],
                per_seed_mean_auc=list(winner["per_seed_mean_auc"]),
                mean_auc=winner["mean_auc"], sd_auc=winner["sd_auc"],
                tie_broken_on=_tie_broken_on(ranking), ranking=ranking)


def consensus_select(dev_frames: Mapping, *, families: Sequence[str] = FAMILIES,
                     thresholds: Sequence[int] = THRESHOLDS,
                     dev_seeds: Sequence[int] = DEV_SEEDS,
                     folds: int = CV_FOLDS, objective: str = OBJECTIVE,
                     progress=None) -> dict:
    """The whole selection for ONE cohort: one fixed configuration per (threshold, family).

    `dev_frames` maps `(threshold, seed) -> (X, y)`; build it with `mcs_dev_frames` or
    `yrbs_dev_frames`, which are the only functions that know what a cohort is.

    A CELL THAT CANNOT BE SELECTED STOPS THE RUN. There is no fallback, no dropped cell, no
    reduced fold count and no selection on fewer seeds: each of those would put a configuration
    chosen under a different protocol into a column that says it was chosen under this one. The
    error names every affected cell so the decision about what to do next is made deliberately
    and once, rather than silently and per cell.

    Returns `{(threshold, family): record}`, exactly `len(thresholds) x len(families)` cells.
    """
    if objective != OBJECTIVE:
        raise ValueError(
            f"objective={objective!r}: this selector computes {OBJECTIVE!r} and nothing else. "
            f"Both cohorts and the reported primary metric use it, and a second objective here "
            f"would make the two cohorts' references incomparable.")
    declared = [int(s) for s in dev_seeds]
    if declared != list(DEV_SEEDS):
        raise ValueError(
            f"dev_seeds={declared} is not the declared development-seed set {list(DEV_SEEDS)}. "
            f"The protocol fixes three development seeds; selecting on any other set is a "
            f"different procedure and must not be recorded under this one's identifier.")

    absent = [(int(t), int(s)) for t in thresholds for s in declared
              if (int(t), int(s)) not in dev_frames]
    if absent:
        raise ValueError(
            f"the development frames are missing {len(absent)} (threshold, seed) cell(s), e.g. "
            f"{absent[:4]}. Every declared threshold needs every development seed.")
    beyond = [key for key in dev_frames
              if key not in {(int(t), int(s)) for t in thresholds for s in declared}]
    if beyond:
        raise ValueError(
            f"the development frames carry {len(beyond)} cell(s) beyond the declared grid, e.g. "
            f"{beyond[:4]}. A frame outside the declared development seeds must not be reachable "
            f"from a selection that reports itself as a three-seed consensus.")

    out, failed = {}, []
    for threshold in thresholds:
        for family in families:
            record = select_cell(family, dev_frames, threshold=int(threshold),
                                 dev_seeds=declared, folds=folds)
            out[(int(threshold), str(family))] = record
            if record["status"] != SELECTION_SELECTED:
                failed.append((int(threshold), str(family), record["status"]))
        if progress is not None:
            progress(int(threshold))

    if failed:
        detail = "\n    ".join(f">={t} {f}: {s}" for t, f, s in failed)
        raise RuntimeError(
            f"{len(failed)} of {len(out)} (threshold, family) cell(s) had no candidate estimable "
            f"on all {len(declared)} development seeds:\n    {detail}\n"
            f"Nothing is substituted here. Selecting on fewer seeds, widening the folds or "
            f"dropping the cell would each produce a configuration chosen under a different "
            f"protocol from every other cell, and the reported specification would no longer "
            f"describe one procedure. Decide how to proceed deliberately.")
    return out


# ---------------------------------------------------------------- the cohort adapters
#
# THE ONLY COHORT-AWARE CODE IN THE SELECTION PATH, and each is deliberately tiny. Each reads
# exactly two named keys from each development-seed bundle and returns frames, so the selector
# never receives a bundle and cannot reach a key neither adapter asked for.
#
# The keys are the OUTER-TRAINING frames with NaN-outcome rows already dropped. No outer test
# frame, no outcome from one, and nothing from the other cohort is read by either adapter.

MCS_TRAINING_KEYS: Sequence[str] = ("Xm_trm", "ym_trm")
YRBS_TRAINING_KEYS: Sequence[str] = ("Xy_trm", "yy_trm")


def _dev_frames(splits: Mapping, keys: Sequence[str], *, cohort: str,
                thresholds: Sequence[int], dev_seeds: Sequence[int]) -> dict:
    """`{(threshold, seed): (X, y)}` for one cohort, reading `keys` and nothing else."""
    declared = [int(s) for s in dev_seeds]
    outside = [s for s in declared if s not in set(DEV_SEEDS)]
    if outside:
        raise ValueError(
            f"{cohort} development frames were asked for seed(s) {outside}, which are not in the "
            f"declared development-seed set {list(DEV_SEEDS)}. Selection reads development seeds "
            f"only; an evaluation seed's training partition is not a development partition.")
    features, outcome = keys
    out = {}
    for threshold in thresholds:
        for seed in declared:
            bundle = splits[(int(seed), int(threshold))]
            out[(int(threshold), int(seed))] = (bundle[features], bundle[outcome])
    return out


def mcs_dev_frames(splits: Mapping, *, thresholds: Sequence[int] = THRESHOLDS,
                   dev_seeds: Sequence[int] = DEV_SEEDS) -> dict:
    """MCS outer-training frames for the development seeds, and nothing else.

    Reads `Xm_trm` and `ym_trm` from each development-seed bundle. The MCS outer-test frames
    (`Xm_te`, `ymte`, `Xm_te_cs`) and every YRBS key are never named here, so an MCS selection
    built through this adapter cannot reach one.
    """
    return _dev_frames(splits, MCS_TRAINING_KEYS, cohort="MCS", thresholds=thresholds,
                       dev_seeds=dev_seeds)


def yrbs_dev_frames(splits: Mapping, *, thresholds: Sequence[int] = THRESHOLDS,
                    dev_seeds: Sequence[int] = DEV_SEEDS) -> dict:
    """YRBS outer-training frames for the development seeds, and nothing else.

    Reads `Xy_trm` and `yy_trm` from each development-seed bundle. The YRBS outer-test frames
    (`Xy_te`, `yte`, `Xy_te_cs`, `Xy_te_cs2`, `y_idx`) and every MCS key are never named here.
    """
    return _dev_frames(splits, YRBS_TRAINING_KEYS, cohort="YRBS", thresholds=thresholds,
                       dev_seeds=dev_seeds)


def settings_from_record(records: Mapping) -> dict:
    """`{(threshold, family): params}` from a selection, refusing an unselected cell.

    The one adapter from a selection record into the shape every consumer takes. Keyed
    `(threshold, family)` — no seed — so a downstream caller cannot accidentally receive a
    seed-specific mapping.
    """
    unselected = sorted(key for key, record in records.items()
                        if record.get("status") != SELECTION_SELECTED)
    if unselected:
        raise ValueError(
            f"{len(unselected)} cell(s) carry no selected configuration, e.g. {unselected[:4]}. "
            f"A fixed mapping has to be complete: there is nothing to fall back to that would "
            f"still be this procedure's answer.")
    return {(int(t), str(f)): dict(record["params"]) for (t, f), record in records.items()}


def compare_settings(new_map: Mapping, other_map: Mapping) -> "pd.DataFrame":
    """Per-cell agreement between two fixed mappings. PURE, and post hoc by construction.

    Takes both mappings as arguments and reads no file, so it cannot be reached from inside a
    selection and cannot influence one.

    TWO USES, BOTH AFTER A SELECTION HAS FINISHED. The promotion gate in Notebook 02's selection
    section calls it on every internal run — once per cohort, comparing the settings reconstructed
    from that cohort's private record against the promoted specification — and refuses to fit
    anything until all 27 cells agree. It was also used once, during migration, to report how many
    cells a superseded specification disagreed with.

    THE COMPARISON IS TOLERANCE-AWARE AND KEYED ON THE UNION. `_cfg_equal` decides agreement, so a
    float that survived a CSV round trip is not reported as a change. Keys present on one side only
    appear as rows with `matches=False`, which is why a caller that requires both a row count and
    `matches.all()` detects a short, long or mis-keyed specification as well as a changed one.
    """
    rows = []
    for key in sorted(set(new_map) | set(other_map)):
        threshold, family = key
        left, right = new_map.get(key), other_map.get(key)
        rows.append(dict(
            threshold=f">={threshold}", family=family,
            matches=bool(left is not None and right is not None and _cfg_equal(left, right)),
            selected=str(left) if left is not None else "",
            other=str(right) if right is not None else ""))
    return pd.DataFrame(rows)


# ------------------------------------------------- the nested per-seed target sensitivity
#
# NOT THE HEADLINE, AND OFF BY DEFAULT. This selects on YRBS inside EVERY evaluation seed, and is
# computed by the run that uses it.
#
# WHAT IT IS FOR. The headline YRBS local reference takes one fixed configuration, chosen by the
# consensus procedure above and held across all twenty splits. This asks a different question:
# whether a fully nested redevelopment procedure — a fresh selection inside each of the twenty
# training partitions — reaches somewhere different. It is a sensitivity analysis and supplies no
# headline reference, no headline gap and no anchor.
#
# THE INFORMATION BOUNDARY. What is selected here reaches `yrbs_resource_rich` and nothing else.
# No transfer procedure, no reference and no sensitivity battery receives it.
#
# PREPROCESSING. Identical to `prepare_folds` above — each fold half standardised against itself
# — so the sensitivity and the headline differ in their seed protocol and in nothing else.

TARGET_SELECTION_SELECTED = "selected"


def target_selection_folds(y, *, folds: int = CV_FOLDS, seed: int):
    """Stratified inner folds over the outer YRBS training rows, or `None` if none are supported.

    Delegates to `inner_folds`, which is the one definition, so the nested sensitivity and the
    consensus selection cannot drift apart on what an inner fold is.
    """
    return inner_folds(y, folds=folds, seed=seed)


def select_target_params(family: Family, splits, *, seed: int, folds: int = CV_FOLDS) -> dict:
    """Choose one family's hyperparameters inside ONE seed's YRBS training partition.

    Returns a record — never a bare dict of parameters — so a caller cannot use the result
    without seeing whether a search happened:

        params      the selected configuration, or None when nothing was selected
        status      "selected", or "non_estimable:<reason>"
        folds       the fold count used, or 0
        candidates  how many configurations this FAMILY offers: 4 for L1-LR and L2-LR,
                    9 for EN-LR, 40 for each tree and boosting family. Never the 257 total
        cv_auc      the winning configuration's mean inner-CV AUC, or NaN

    THE OUTER TEST FRAME IS NOT READ. Selection touches `Xy_trm` and `yy_trm` only — the outer
    training rows with an observed outcome. No outer-test outcome enters selection or fitting.
    Outer-test FEATURES do enter their own cohort standardisation at evaluation time, as they do
    for every YRBS regime in this pipeline; that convention is shared, not something this
    benchmark introduces.

    SAME GRID, SAME OBJECTIVE AS THE SOURCE SIDE. `candidate_pool` and five-fold AUC are the
    source selection's, so the two references differ in which cohort chose the configuration
    rather than in how the choice was made.

    NO FALLBACK. A family that cannot be searched here returns `params=None` and a reason. It
    does not silently receive the MCS-selected configuration, which would put a source-chosen
    number in a target-chosen column.

    SIDE EFFECTS: none. Fits `candidates x folds` probe models in memory and discards them.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    import data as D

    pool = candidate_pool(family)
    n_candidates = len(pool)
    X = splits["Xy_trm"]
    y = splits["yy_trm"]

    splitter = target_selection_folds(y, folds=folds, seed=seed)
    if splitter is None:
        return dict(params=None, status="non_estimable:too_few_minority_for_five_folds",
                    folds=0, candidates=n_candidates, cv_auc=float("nan"))

    y_int = np.asarray(y, float).astype(int)
    # Each half is standardised against itself, which is the convention every YRBS frame in this
    # pipeline is evaluated under. Done once per fold rather than once per candidate.
    prepared = []
    for fit_rows, held_rows in splitter.split(X, y_int):
        prepared.append((D.standardise_cohort(X.iloc[fit_rows]), y_int[fit_rows],
                         D.standardise_cohort(X.iloc[held_rows]), y_int[held_rows]))

    best_params, best_auc = None, -np.inf
    for candidate in pool:
        scores = []
        for X_fit, y_fit, X_held, y_held in prepared:
            if np.unique(y_held).size < 2:
                continue
            probe = make_estimator(family, candidate, seed=seed)
            probe.fit(X_fit, y_fit)
            scores.append(float(roc_auc_score(y_held, probe.predict_proba(X_held)[:, 1])))
        if not scores:
            continue
        mean_auc = float(np.mean(scores))
        # First maximum wins, and `candidate_pool` is deterministic, so a tie resolves the same
        # way on every run.
        if mean_auc > best_auc:
            best_params, best_auc = dict(candidate), mean_auc

    if best_params is None:
        return dict(params=None, status="non_estimable:no_candidate_scored",
                    folds=int(splitter.get_n_splits()), candidates=n_candidates,
                    cv_auc=float("nan"))
    return dict(params=best_params, status=TARGET_SELECTION_SELECTED,
                folds=int(splitter.get_n_splits()), candidates=n_candidates,
                cv_auc=best_auc)


# ================================================================ the private selection record
#
# WHERE THE CROSS-VALIDATED NUMBERS LIVE, AND WHY THEY LIVE ONLY THERE. A selection record holds
# every candidate's three seed-level mean AUCs, their mean and their standard deviation — 771
# candidate records per cohort (257 per threshold x 3 thresholds). Those are cohort-derived
# results. The MCS ones are MCS-derived and need separate disclosure review before they could go
# anywhere else, so the record is written under $THESIS_WORK_DIR and NEVER into the repository,
# a publication candidate, a LaTeX fragment or a manuscript table.
#
# The tracked specification promoted from it carries the configurations and two provenance
# identifiers, and no score at all.
#
# WHAT A RECORD MAY NOT CONTAIN, and what the schema below is checked against: any respondent
# count, row count, prevalence, row index, participant identifier, prediction or path. Candidate
# counts, fold counts, candidate-pool sizes, ranks and pool indices ARE permitted — they are
# properties of the search space and the protocol, not of anybody's records.

SELECTION_RECORD_FORMAT = 1
SELECTION_RECORD_SUBDIR = "tables"
SELECTION_RECORD_FILENAME: Mapping[str, str] = {
    "mcs": "mcs_consensus_selection.json",
    "yrbs": "yrbs_consensus_selection.json",
}

# The provenance every record declares, and every field of it is compared on load. A record whose
# procedure differs in ANY of these is a different experiment and is refused whole.
SELECTION_PROVENANCE: Sequence[str] = (
    "format_version", "cohort", "preprocessing_version", "model_features", "development_seeds",
    "folds", "objective", "tie_rule", "candidate_pool_fingerprint", "candidate_pool_sizes",
    "fitting_convention", "protocol_id", "settings_digest", "thresholds", "families",
)
SELECTION_SETTING_FIELDS: Sequence[str] = (
    "threshold", "family", "status", "folds", "candidates", "dev_seeds", "per_seed_mean_auc",
    "mean_auc", "sd_auc", "tie_broken_on", "params",
)
SELECTION_RANKING_FIELDS: Sequence[str] = (
    "threshold", "family", "rank", "pool_index", "eligible", "per_seed_mean_auc", "mean_auc",
    "sd_auc", "params",
)
SELECTION_TOP_FIELDS: Sequence[str] = (*SELECTION_PROVENANCE, "settings", "ranking")

FITTING_CONVENTION = (
    "make_estimator; StratifiedKFold(5, shuffle=True, random_state=<development seed>) built "
    "once per threshold and seed; each fold half standardised against itself via "
    "data.standardise_cohort")

# Names that would mean a respondent-level quantity had reached the record. Checked explicitly,
# in addition to the field allow-list, because the allow-list only rejects what it does not know
# about and this says plainly what must never be introduced.
FORBIDDEN_RECORD_FIELDS: frozenset = frozenset({
    "respondent_count", "n_rows", "analytic_n", "n_people", "denominator", "prevalence",
    "row_index", "participant_id", "predictions",
})


def selection_record_path(cohort: str):
    """Where one cohort's private selection record lives. Resolves the working root on the call."""
    import config as _C
    if cohort not in SELECTION_RECORD_FILENAME:
        raise ValueError(f"cohort must be one of {list(COHORTS)}, got {cohort!r}")
    return _C.work_path(SELECTION_RECORD_SUBDIR) / SELECTION_RECORD_FILENAME[cohort]


def selection_record_payload(records: Mapping, *, cohort: str,
                             families: Sequence[str] = FAMILIES,
                             thresholds: Sequence[int] = THRESHOLDS,
                             dev_seeds: Sequence[int] = DEV_SEEDS) -> dict:
    """The saved object for one cohort's complete selection, ready to serialise.

    Carries the full candidate ranking, not only the winners: 771 candidate records per cohort.
    What makes that worth its size is that a selection is only auditable if the runners-up are
    visible — a tie broken on pool order looks identical to a decisive win once everything but
    the winner has been discarded.
    """
    import data as D

    if cohort not in set(COHORTS):
        raise ValueError(f"cohort must be one of {list(COHORTS)}, got {cohort!r}")
    wanted = [(int(t), str(f)) for t in thresholds for f in families]
    keyed = {(int(t), str(f)): record for (t, f), record in records.items()}
    absent = [key for key in wanted if key not in keyed]
    extra = [key for key in keyed if key not in set(wanted)]
    if absent or extra:
        raise ValueError(
            f"the {cohort.upper()} selection covers the declared grid imperfectly: {len(absent)} "
            f"cell(s) missing and {len(extra)} beyond it. A partial selection is not saved, "
            f"because a later run would load it as a complete one.")

    settings, ranking = [], []
    for threshold, family in wanted:
        record = keyed[(threshold, family)]
        settings.append(dict(
            threshold=threshold, family=family, status=str(record["status"]),
            folds=int(record["folds"]), candidates=int(record["candidates"]),
            dev_seeds=[int(s) for s in record["dev_seeds"]],
            per_seed_mean_auc=_json_safe(record["per_seed_mean_auc"]),
            mean_auc=_json_safe(record["mean_auc"]), sd_auc=_json_safe(record["sd_auc"]),
            tie_broken_on=str(record["tie_broken_on"]),
            params=(None if not record["params"] else _json_safe(record["params"]))))
        for entry in record.get("ranking", ()):
            ranking.append(dict(
                threshold=threshold, family=family, rank=int(entry["rank"]),
                pool_index=int(entry["pool_index"]), eligible=bool(entry["eligible"]),
                per_seed_mean_auc=_json_safe(entry["per_seed_mean_auc"]),
                mean_auc=_json_safe(entry["mean_auc"]), sd_auc=_json_safe(entry["sd_auc"]),
                params=_json_safe(entry["params"])))

    digest = settings_digest(
        (cohort, f">={row['threshold']}", row["family"], row["params"]) for row in settings)
    return {
        "format_version": SELECTION_RECORD_FORMAT,
        "cohort": str(cohort),
        "preprocessing_version": D.PREPROCESSING_VERSION,
        "model_features": list(D.MODEL_FEATURE_COLUMNS),
        "development_seeds": [int(s) for s in dev_seeds],
        "folds": int(CV_FOLDS),
        "objective": OBJECTIVE,
        "tie_rule": TIE_RULE,
        "candidate_pool_fingerprint": candidate_pool_fingerprint(families),
        "candidate_pool_sizes": {str(f): len(candidate_pool(f)) for f in families},
        "fitting_convention": FITTING_CONVENTION,
        "protocol_id": protocol_id(families),
        "settings_digest": digest,
        "thresholds": [int(t) for t in thresholds],
        "families": [str(f) for f in families],
        "settings": settings,
        "ranking": ranking,
    }


def save_selection_record(records: Mapping, *, cohort: str,
                          families: Sequence[str] = FAMILIES,
                          thresholds: Sequence[int] = THRESHOLDS,
                          dev_seeds: Sequence[int] = DEV_SEEDS, path=None):
    """Write one cohort's private selection record, and read it back. Returns the path.

    The read-back goes through `load_selection_record`, so what is verified is that the file the
    loader will see is the selection that was made — a written file the reader would refuse is a
    lost search, and it is cheaper to find out here.

    SIDE EFFECTS: creates the `tables` directory under the working root if absent, and replaces
    the record. Writes nothing inside the repository and fits nothing.
    """
    import json
    from pathlib import Path as _Path

    payload = selection_record_payload(records, cohort=cohort, families=families,
                                       thresholds=thresholds, dev_seeds=dev_seeds)
    location = _Path(path) if path is not None else selection_record_path(cohort)
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    load_selection_record(cohort=cohort, path=location)
    return location


def load_selection_record(*, cohort: str, path=None) -> dict:
    """One cohort's private selection record, refused unless its provenance is the live one.

    REFUSES RATHER THAN REPAIRS, on every provenance field: an unknown format version, another
    cohort, a superseded preprocessing version or model feature schema, a different
    development-seed set, fold count, objective or tie rule, a changed candidate pool, a
    protocol identifier that is not the live one, a settings digest that does not match the rows
    it accompanies, a missing field, a duplicate cell, or a status and parameters that disagree.

    NOTHING IS TOPPED UP AND NOTHING IS MIGRATED. A record short of a cell is refused whole
    rather than completed by a fresh partial search: half a grid searched under one candidate
    pool and half under another is not one selection.

    SIDE EFFECTS: reads and parses one JSON file under the working root. Fits nothing, writes
    nothing.
    """
    import json
    from pathlib import Path as _Path

    import data as D

    location = _Path(path) if path is not None else selection_record_path(cohort)
    if not location.exists():
        raise FileNotFoundError(
            f"{location.name} is not under the working root. It is written by Notebook 02's "
            f"selection section, which is where it is recomputed when it is absent; nothing "
            f"else re-derives it.")
    payload = json.loads(location.read_text(encoding="utf-8"))

    unexpected = [k for k in payload if k not in set(SELECTION_TOP_FIELDS)]
    if unexpected:
        raise ValueError(
            f"{location.name} carries unexpected top-level field(s) {sorted(unexpected)}. The "
            f"record schema is closed: a field nothing declared is a field nothing checked, and "
            f"this file must be provably free of respondent-level material.")
    absent = [k for k in SELECTION_TOP_FIELDS if k not in payload]
    if absent:
        raise ValueError(f"{location.name} is missing field(s) {absent}.")

    expected = {
        "format_version": SELECTION_RECORD_FORMAT,
        "cohort": str(cohort),
        "preprocessing_version": D.PREPROCESSING_VERSION,
        "model_features": list(D.MODEL_FEATURE_COLUMNS),
        "development_seeds": [int(s) for s in DEV_SEEDS],
        "folds": int(CV_FOLDS),
        "objective": OBJECTIVE,
        "tie_rule": TIE_RULE,
        "candidate_pool_fingerprint": candidate_pool_fingerprint(),
        "protocol_id": protocol_id(),
    }
    for field, want in expected.items():
        got = payload.get(field)
        if isinstance(want, list):
            got = list(got or ())
        if got != want:
            raise ValueError(
                f"{location.name} declares {field}={got!r}; the live value is {want!r}. The "
                f"selections in it were made under a different procedure or a different "
                f"representation of the predictors, so they are that run's answer and not this "
                f"one's. Re-run the selection rather than reusing them.")

    settings = payload.get("settings")
    if not isinstance(settings, list) or not settings:
        raise ValueError(f"{location.name} carries no `settings` list, so it holds no selection.")

    seen, out = set(), {}
    for row in settings:
        missing = [f for f in SELECTION_SETTING_FIELDS if f not in row]
        if missing:
            raise ValueError(
                f"{location.name}: a saved cell is missing field(s) {missing}. Every cell "
                f"carries all {len(SELECTION_SETTING_FIELDS)}.")
        stray = [f for f in row if f not in set(SELECTION_SETTING_FIELDS)]
        if stray:
            raise ValueError(
                f"{location.name}: a saved cell carries undeclared field(s) {sorted(stray)}.")
        key = (int(row["threshold"]), str(row["family"]))
        if key in seen:
            raise ValueError(
                f"{location.name}: {key} appears more than once, so which configuration the "
                f"cell used is undetermined.")
        seen.add(key)
        status, params = str(row["status"]), row["params"]
        if status == SELECTION_SELECTED and not params:
            raise ValueError(
                f"{location.name}: {key} reports {SELECTION_SELECTED!r} and carries no "
                f"parameters. A selection with nothing selected cannot be used, and there is no "
                f"configuration to fall back to that would still be this one.")
        if status != SELECTION_SELECTED and params:
            raise ValueError(
                f"{location.name}: {key} reports {status!r} and yet carries parameters. A cell "
                f"that could not be searched has nothing to have chosen.")
        out[key] = dict(
            params=(dict(params) if params else None), status=status,
            folds=int(row["folds"]), candidates=int(row["candidates"]),
            dev_seeds=[int(s) for s in row["dev_seeds"]],
            per_seed_mean_auc=row["per_seed_mean_auc"], mean_auc=row["mean_auc"],
            sd_auc=row["sd_auc"], tie_broken_on=str(row["tie_broken_on"]))

    wanted = {(int(t), str(f)) for t in payload["thresholds"] for f in payload["families"]}
    if seen != wanted:
        raise ValueError(
            f"{location.name} covers {len(seen)} of the {len(wanted)} declared "
            f"(threshold, family) cells. A partial record is refused whole: nothing here fills a "
            f"gap, because a configuration borrowed from another cell is that cell's answer.")

    recomputed = settings_digest(
        (str(cohort), f">={row['threshold']}", row["family"], row["params"]) for row in settings)
    if payload["settings_digest"] != recomputed:
        raise ValueError(
            f"{location.name} records settings_digest {payload['settings_digest']!r} but its "
            f"selected configurations hash to {recomputed!r}. The file has been edited since it "
            f"was written.")

    for entry in payload.get("ranking", ()):
        stray = [f for f in entry if f not in set(SELECTION_RANKING_FIELDS)]
        if stray:
            raise ValueError(
                f"{location.name}: a ranking row carries undeclared field(s) {sorted(stray)}.")
    return {"payload": payload, "records": out}


def selection_coverage(records: Mapping, *, families: Sequence[str] = FAMILIES,
                       thresholds: Sequence[int] = THRESHOLDS) -> dict:
    """Check a loaded selection covers the declared grid, and count how the cells came out.

    A MISSING CELL IS AN ERROR. Unlike the nested sensitivity, a non-selected cell is an error
    too: the fixed specification has to be complete, and `consensus_select` already refuses to
    return one that is not.
    """
    wanted = [(int(t), str(f)) for t in thresholds for f in families]
    absent = [key for key in wanted if key not in records]
    if absent:
        raise ValueError(
            f"the selection is missing {len(absent)} of {len(wanted)} declared "
            f"(threshold, family) cells, e.g. {absent[:4]}.")
    selected = sum(1 for key in wanted if records[key]["status"] == SELECTION_SELECTED)
    if selected != len(wanted):
        raise ValueError(
            f"{len(wanted) - selected} of {len(wanted)} cells carry no configuration. A fixed "
            f"specification has to be complete.")
    return {"expected": len(wanted), "selected": selected,
            "tie_broken_on": {basis: sum(1 for key in wanted
                                         if records[key]["tie_broken_on"] == basis)
                              for basis in ("mean", "sd", "pool_order")}}


def selection_review_frame(loaded: Mapping) -> "pd.DataFrame":
    """The internal review table, read from a SAVED record rather than recomputed.

    Ten columns exactly: cohort, threshold, family, the three development seeds' mean AUCs,
    the three-seed mean and standard deviation, which clause of the tie rule decided the cell,
    and the selected parameters.

    READ, NOT RECOMPUTED. The point of the table is to show what was written to the record and
    will be promoted from it; a recomputed table could differ from the file and nobody would
    know which one the specification came from.

    MCS-DERIVED AND INTERNAL. These are cross-validated performance values. They are displayed in
    Notebook 02's selection section, whose cells are all absent from that notebook's public
    allow-list, and they reach no tracked specification, no publication candidate, no LaTeX
    fragment and no manuscript table.
    """
    payload = loaded["payload"]
    cohort = str(payload["cohort"])
    seeds = [int(s) for s in payload["development_seeds"]]
    rows = []
    for row in payload["settings"]:
        means = list(row["per_seed_mean_auc"] or [None] * len(seeds))
        entry = dict(cohort=cohort, threshold=f">={int(row['threshold'])}",
                     family=str(row["family"]))
        for position, seed in enumerate(seeds):
            entry[f"seed_{seed}_auc"] = means[position] if position < len(means) else None
        entry.update(mean_auc=row["mean_auc"], sd_auc=row["sd_auc"],
                     tie_broken_on=row["tie_broken_on"], params=str(row["params"]))
        rows.append(entry)
    columns = (["cohort", "threshold", "family"] + [f"seed_{s}_auc" for s in seeds]
               + ["mean_auc", "sd_auc", "tie_broken_on", "params"])
    return pd.DataFrame(rows, columns=columns)


def _cfg_equal(a: Mapping, b: Mapping) -> bool:
    """Config equality: numerics compared with tolerance, everything else exactly.

    The tolerance matters because a config round-trips through `str()` into the tracked
    spec and back through `ast.literal_eval`. `max_features` is the case that forces the
    mixed handling: it takes either `'sqrt'` or `0.5`.
    """
    def _num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    if set(a) != set(b):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if _num(va) and _num(vb):
            if abs(float(va) - float(vb)) > 1e-9:
                return False
        elif va != vb:
            return False
    return True


