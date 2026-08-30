"""
src/transfer.py — the seventeen transfer regimes, k-budget sweeps, calibration.

Backs the second half of notebook 02_models_and_transfer.ipynb.

CONTENTS — five separable concerns, kept in one file because the procedures are the
single point every regime flows through. Splitting would either duplicate it or
create a circular import.

    1. REGIME VOCABULARY + FITTING PRIMITIVES
       LABEL_FREE, LABEL_USING, REFERENCE, NOT_RUN, then fit_offset_intercept,
       fit_nonneg_logistic, source_raw_logit, stacked_ensemble, raw_l1_head,
       source_params

    2. THE BATCH PATH — start here
       <procedure>_scores   one procedure over one (family, threshold, seed) cell.
                            Eighteen of them, each fitting what its own method needs.
       battery_cell         one cell's metric rows and YRBS scores; notebook 02 owns the loop.

    3. THE REGIME DEFINITIONS
       Each `<procedure>_scores` function is the single definition of its regime,
       and they are what runs. `source_model` is the shared MCS fit. The helpers
       that remain beside it (bbse_correct, warm_finetune, leaf_refresh, rule_head,
       leaf_membership_head) are the shared pieces it calls. There is ONE definition
       per regime: an earlier arrangement kept freestanding per-regime duplicates
       that were meant to be "kept in sync", and nothing enforced the sync.

    4. BACKFILL + OUTCOME SENSITIVITY
       run_backfill, consolidate_backfill, leave_one_pillar_out,
       outcome_variant_battery

    5. LABEL-BUDGET SWEEP + CALIBRATION
       label_budget_curve, budget_summary

THE SEVENTEEN REGIMES, exactly as S4 enumerates them.

  SIX LABEL-FREE (no YRBS label is touched):
    unadapted                 transfer the MCS-fitted model unchanged
    bbse                      black-box shift estimation, prior correction (AUC-invariant)
    importance_weight         Shimodaira density-ratio reweighting of MCS training rows
    quantile_map              marginal CDF matching of target features onto source
    pseudo_label              one round of self-training on confident YRBS predictions,
                              selected by decile of the unlabelled pool
    pseudo_label_thresh       three rounds of the same idea, selected by a fixed confidence
                              threshold on the same unlabelled pool

  ELEVEN LABEL-USING (consume k YRBS labels):
    target_only               fresh fit on the k slice
    fine_tune                 full-model revision: warm-start for XGB/CatBoost/LightGBM,
                              weighted refit on MCS+k otherwise. This is what the paper
                              calls `full_revision` — it is NOT a literal pipeline key.
    isotonic_recal            isotonic recalibration of transferred scores
    raw_l1_head               L1-logistic head on the unstandardised model features
    coef_freeze_intercept     LR only: freeze coefficients, refit intercept
    platt_frozen              Platt scaling with frozen source model
    leaf_refresh_global       tree families: relearn leaf values, keep split structure
    sign_support              LR only: constrain coefficients to source signs
    feature_set               LR only: refit on the source-selected feature subset
    ensemble_same_family      stacked ensemble within a family
    ensemble_catboost_source  stacked ensemble against the CatBoost source model

  Plus TWO LOCAL REFERENCE regimes, which are not transfer methods: `mcs_internal` and
  `yrbs_local`. Each is a model developed and evaluated inside one cohort, configured by that
  cohort's own consensus selection. Neither is a ceiling.

  `rank_mean_ensemble` is not run. The unweighted mean / rank-mean
  score combiners survive separately in the backfill, not as an S4 regime.

  And TWO REVERSE ROLES, which are not regimes and are not part of the battery:
  `reverse_transfer` and `mcs_local_reference` read the same comparison with the cohorts
  swapped, for notebook 04's supplementary section. See `reverse_transfer_scores`.

  And ONE SENSITIVITY REGIME, off by default and supplying no headline quantity:
  `yrbs_resource_rich`, the nested per-split target redevelopment. See
  `nested_target_sensitivity_scores`.

REGIME APPLICABILITY IS GATED BY FAMILY and the gate is a deliberate key omission, not a
bug: coef_freeze_intercept / sign_support / feature_set are LR-only; leaf_refresh_global
is tree-only; ensemble_catboost_source excludes CatBoost itself. Preserve the gate — a
consolidated loop that runs every regime on every family will silently invent cells.

TIER 1a: MCS trains the bases, and every forward transfer metric is computed on the YRBS test
slice. The three reverse roles evaluate inside MCS instead, so their score vectors are MCS
row-level and are aggregated where they are produced. No MCS scores are persisted anywhere.
"""

from typing import Literal, Mapping, Sequence

import numpy as np

from models import RANDOM_STATE          # the battery's model seed, owned by models.py
import pandas as pd

Family = str
Arm = Literal["untuned", "tuned"]
Threshold = Literal[1, 2, 3]

LABEL_FREE: Sequence[str] = (
    "unadapted", "bbse", "importance_weight", "quantile_map", "pseudo_label", "pseudo_label_thresh",
)
LABEL_USING: Sequence[str] = (
    "target_only", "fine_tune", "isotonic_recal", "raw_l1_head", "coef_freeze_intercept",
    "platt_frozen", "leaf_refresh_global", "sign_support", "feature_set",
    "ensemble_same_family", "ensemble_catboost_source",
)
REFERENCE: Sequence[str] = ("mcs_internal", "yrbs_local")


SEEDS: Sequence[int] = tuple(range(20))
THRESHOLDS: Sequence[int] = (1, 2, 3)
LINEAGE = "cs"
# The reported arm. Everything that fits a model here runs under it — the battery, structure
# transfer, and the two outcome-sensitivity batteries, which used the untuned arm until the
# notebook stopped carrying two. A sensitivity analysis on a configuration the paper does not
# report answers a question nobody asked.
ARM: Arm = "tuned"
K = 500
ALPHA = 5.0
# AUC == naive by construction: both are monotone transforms of the naive score. Reported
# with the flag so a flat AUC reads as "invariant by construction", not as a null result.
MONOTONE = frozenset({"bbse", "isotonic_recal"})
# Stated, not substituted. S4 does not run these and neither should a consolidated loop.
NOT_RUN: Sequence[str] = ("rank_mean_ensemble", "leaf_refresh", "rule_extraction(R1)",
                          "leaf_membership_head(F1)", "conformal")


def _lclip(p):
    from scipy.special import logit
    return logit(np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6))


def fit_offset_intercept(eta, y) -> float:
    """1-DOF binomial GLM: minimise deviance of sigma(alpha + eta) over a scalar alpha.

    `eta` is a FIXED offset — the source model's contribution, frozen. Only the intercept
    moves. Used by R1 (`coef_freeze_intercept`, eta = X.beta) and R2
    (`leaf_refresh_global`, eta = the source raw logit). statsmodels is intentionally not
    used; L-BFGS-B on the deviance is the source's choice and reproduces its numbers.
    """
    from scipy.optimize import minimize
    from scipy.special import expit
    eta = np.asarray(eta, float); y = np.asarray(y, float)

    def nll(a):
        p = np.clip(expit(a[0] + eta), 1e-9, 1 - 1e-9)
        return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(minimize(nll, np.zeros(1), method="L-BFGS-B").x[0])


def fit_nonneg_logistic(X, y) -> tuple:
    """Logistic regression with NON-NEGATIVE coefficients and a free intercept. R4.

    No L1 or L2 penalty: the SIGN CONSTRAINT is the only regulariser, which is the point of
    `sign_support` — it asks whether the source's coefficient signs alone carry enough
    structure. Returns `(params[coefs..., intercept], any_coef_at_bound)`.
    """
    from scipy.optimize import minimize
    from scipy.special import expit
    X = np.asarray(X, float); y = np.asarray(y, float); p = X.shape[1]

    def nll(w):
        pr = np.clip(expit(X @ w[:-1] + w[-1]), 1e-9, 1 - 1e-9)
        return -float(np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr)))
    res = minimize(nll, np.zeros(p + 1), method="L-BFGS-B",
                   bounds=[(0.0, None)] * p + [(None, None)])
    return res.x, bool(np.any(np.abs(res.x[:p]) < 1e-8))


def source_raw_logit(family: Family, model, X):
    """The raw logit / margin of a source model, per family. Feeds R2 and R3.

    Each library exposes it differently and the differences are not cosmetic: taking
    `predict_proba` and logit-ing it instead would clip at the 1e-6 bound and flatten the
    tails, which is exactly where `platt_frozen` gets its leverage. RF/ET have no margin, so
    the source clips their probability — that asymmetry is the source's and is kept.
    """
    from scipy.special import logit
    if family == "XGB":
        return np.asarray(model.predict(X, output_margin=True), float)
    if family == "LightGBM":
        return np.asarray(model.predict(X, raw_score=True), float)
    if family == "CatBoost":
        return np.asarray(model.predict(X, prediction_type="RawFormulaVal"), float)
    if family == "HistGB":
        return np.asarray(model.decision_function(X), float)
    if family in ("RF", "ET"):
        return logit(np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6))
    return np.asarray(model.decision_function(X), float)      # LR families: x.beta + b


def stacked_ensemble(source_model, family: Family, params, Xk_df, yk, Xte_df, seed: int) -> tuple:
    """R6/R7 core. Returns `(stacked, convex, lambda)`.

    A single stratified 80/20 holdout INSIDE the k slice — deliberately not out-of-fold. The
    source records that as a decision, not an oversight: with k=500 an OOF scheme would leave
    each fold's meta-features fitted on 400 records and the meta-learner is only 2-parameter.

    Two combiners are returned because they answer different questions. `stacked` is an
    unpenalised 2-feature logit on the clipped source and target logits — it can invert or
    down-weight either. `convex` is a lambda-grid blend of the PROBABILITIES, constrained to
    the simplex, selected on holdout log-likelihood. The stacked one is primary; the convex
    one is carried in `extra` as the interpretable comparator.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import models as PM
    tr, ho = train_test_split(np.arange(len(Xk_df)), test_size=0.2,
                              stratify=yk.values, random_state=seed)
    Xtr, ytr = Xk_df.iloc[tr], yk.iloc[tr]
    Xho, yho = Xk_df.iloc[ho], yk.iloc[ho].values.astype(float)
    m_ho = PM.make_estimator(family, params, seed=seed); m_ho.fit(Xtr, ytr.astype(int))
    p_tgt_ho = m_ho.predict_proba(Xho)[:, 1]; p_src_ho = source_model.predict_proba(Xho)[:, 1]
    meta = LogisticRegression(penalty=None, fit_intercept=True, solver="lbfgs")
    meta.fit(np.c_[_lclip(p_src_ho), _lclip(p_tgt_ho)], yho.astype(int))
    m_full = PM.make_estimator(family, params, seed=seed); m_full.fit(Xk_df, yk.astype(int))
    p_tgt_te = m_full.predict_proba(Xte_df)[:, 1]
    p_src_te = source_model.predict_proba(Xte_df)[:, 1]
    stacked = meta.predict_proba(np.c_[_lclip(p_src_te), _lclip(p_tgt_te)])[:, 1]
    best_l, best_ll = 0.0, -np.inf
    for lam in np.round(np.arange(0.0, 1.01, 0.1), 1):
        b = np.clip(lam * p_src_ho + (1 - lam) * p_tgt_ho, 1e-9, 1 - 1e-9)
        ll = float(np.sum(yho * np.log(b) + (1 - yho) * np.log(1 - b)))
        if ll > best_ll:
            best_ll, best_l = ll, float(lam)
    return stacked, best_l * p_src_te + (1 - best_l) * p_tgt_te, best_l


def raw_l1_head(Xk_raw, yk, Xte_raw, seed: int):
    """F4 analogue: L1-LR on RAW features fit on k labels, C from an 80/20 within-slice split.

    FAMILY- AND ARM-INDEPENDENT BY CONSTRUCTION — it never touches the source model. It is
    therefore computed ONCE per (threshold, seed) and the same array replicated across all
    nine families, with the rows flagged `family_independent=True`. Recomputing it per family
    would be 9x the work for bit-identical output; treating the nine copies as independent
    evidence would be worse.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def pipe(Cv):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(penalty="l1", solver="liblinear", C=Cv,
                                                max_iter=5000, random_state=seed))
    try:
        Xtr, Xval, ytr, yval = train_test_split(Xk_raw, yk, test_size=0.2, stratify=yk,
                                                random_state=seed)
    except ValueError:
        Xtr, Xval, ytr, yval = Xk_raw, Xk_raw, yk, yk
    best_C, best = 0.01, -np.inf
    for Cv in (0.01, 0.1, 1.0):
        pp = pipe(Cv).fit(Xtr, ytr)
        try:
            a = roc_auc_score(yval, pp.predict_proba(Xval)[:, 1])
        except ValueError:
            a = -np.inf
        if a > best:
            best, best_C = a, Cv
    return pipe(best_C).fit(Xk_raw, yk).predict_proba(Xte_raw)[:, 1]


def source_params(source_family: Family, arm: Arm, threshold: Threshold, *, tuned=None):
    """Arm/threshold-appropriate hyperparameters for a CROSS-FAMILY source model.

    R1/R4/R5 fit an L1_LR source regardless of the target family; R7 fits a CatBoost source.
    Those source models take the hyperparameters the SOURCE family would have had in this arm,
    not the target family's — mirroring the main loop's logic exactly.
    """
    import models as PM
    if arm == "untuned":
        return PM.FROZEN[source_family] if PM.FAM[source_family][1] else {}
    tuned = tuned if tuned is not None else PM.mcs_settings()
    return tuned[(int(threshold), source_family)]


def quantile_match(X_src, X_tgt):
    """Marginal CDF matching — earlier finding: hurts here because large marginal gaps
    are in low-importance features (sleep, social media). Kept for reproducibility."""
    Xt = X_tgt.copy()
    for c in X_tgt.columns:
        src = X_src[c].dropna().sort_values().values
        if len(src) == 0: continue
        ranks = X_tgt[c].rank(pct=True)
        Xt[c] = np.quantile(src, ranks.fillna(0.5).clip(0, 1))
    return Xt


# THE REGIME KEY SET IS THE CONTRACT. Every procedure below must emit exactly the regimes the
# battery declares for its (family, arm, threshold) cell — no extras, none missing — and
# `run_procedure` in notebook 02 checks that cell by cell rather than over the finished frame.
# The `ft_mechanism` strings are the same kind of contract: 'warm_start' for the families in
# WARM_FAMS, 'weighted_refit(alpha=5)' for the rest, '' where full revision does not apply.
# UNVERIFIED: needs MCS data to check the SCORES. Every value returned comes from a model
#             fitted on MCS training rows.
# ---------------------------------------------------------------------------
# The procedures, one function each.
#
# Every one takes the fitted source model and the split bundle and returns
# `{regime: (y_eval, scores, extra)}`. `mcs_internal` is keyed to `ymte`; every
# other regime to `yte`. AN ABSENT KEY MEANS "NOT APPLICABLE" and is the deliberate
# convention: the consumer unpacks every value as a `(y, p, extra)`
# triple, so a `None` would raise, and omitting the key is the clean skip. The family gates
# live where the keys are produced — LR-only for R1/R4/R5, tree-only for R2,
# `ensemble_catboost_source` excludes CatBoost. A loop that ran every procedure on every family
# would silently invent cells that were never computed and they would look real in the grid.
#
# THEY ARE SEPARATE BECAUSE THE NOTEBOOK RUNS THEM SEPARATELY. Each refits its own source model
# through `source_model`, which costs time and nothing else: `make_estimator` pins CatBoost to
# `thread_count=1` and every family refits bit-identically at a fixed seed, so a procedure run
# on its own gives the same numbers as the same procedure run in a batch.


def source_model(family: Family, params, seed: int, splits):
    """The source model. Fitted on the MCS training side, in `Xm_cs` units.

    Every procedure starts from one of these. Refitting per procedure is deterministic — the
    factory fixes the seed and pins CatBoost's `thread_count` — so which procedures a run
    happens to include cannot change any of their scores.
    """
    import models as PM
    base = PM.make_estimator(family, params, seed=seed)
    base.fit(splits["Xm_cs"], splits["ym_trm"].astype(int))
    return base


def fit_source_models(*, splits, tuned, families, thresholds, seeds, quiet: bool = False):
    """Fit the canonical MCS source model for every (family, threshold, seed). Returns a dict.

    A plain dict keyed `(family, threshold, seed)`, not a provider object: the notebook holds
    it, procedures take one model out of it, and nothing else knows it exists.

    THESE ARE THE MODELS EVERY LATER REGIME ADAPTS. Fitting one per procedure instead means
    roughly nine copies of each, all identical — the same fit repeated because no one held the
    first. Fitting them once here is the reason Section E exists.

    TIER 1a: the models are fitted on MCS training rows and are restricted artefacts. They are
    held in memory for the session and **never written to disk**, because a tree ensemble can
    leak training rows through its leaf structure. Nothing in this function persists anything.

    The reported figure is SERIALISED SIZE — `len(pickle.dumps(model))` summed, each measured
    and discarded — which is a proxy for what the dict costs and not a measurement of process
    memory. Reading it as resident set size would overstate what has been established. It is
    printed rather than capped: a figure far above expectation is a finding to hand back, not
    something to silence with eviction.
    """
    import pickle
    import time
    import models as PM
    total_bytes, t0 = 0, time.time()
    out = {}
    for t in thresholds:
        for fam in families:
            params, _ = PM.select_params(fam, "tuned", int(t), tuned)
            for seed in seeds:
                m = source_model(fam, params, seed, splits[(seed, int(t))])
                out[(fam, int(t), seed)] = m
                total_bytes += len(pickle.dumps(m))
    if not quiet:
        print(f"  {len(out)} canonical source models | "
              f"{total_bytes / 1024**2:,.0f} MB serialised (pickle size, a proxy for the "
              f"dict's cost — not process memory) | {time.time() - t0:.0f}s")
    return out


def source_for(source_models, family, threshold, seed):
    """One model out of `fit_source_models`' dict, or a message naming the missing key."""
    key = (family, int(threshold), seed)
    try:
        return source_models[key]
    except KeyError:
        raise KeyError(
            f"no canonical source model for {key}. `fit_source_models` was called with a "
            f"narrower scope than this procedure is being run over, or Section E has not been "
            f"run. Refitting here would silently reintroduce the duplicate fits Section E "
            f"exists to remove.") from None


def unadapted_scores(splits, *, family, params, seed, base=None, **_) -> dict:
    """Unadapted transfer: the source model applied to the YRBS test frame, unchanged.

    The baseline every other procedure is read against, and the cheapest thing an importing
    authority could do — no target labels, no correction, no refit.
    """
    base = base if base is not None else source_model(family, params, seed, splits)
    return {"unadapted": (splits["yte"], base.predict_proba(splits["Xy_te_cs"])[:, 1], {})}


def source_scaled_scores(splits, *, family, params, seed, **_) -> dict:
    """Transfer with no target-cohort statistic anywhere in it — the rung below unadapted.

    Fits `models.source_scaled_estimator` on the RAW MCS training frame, so the imputer and
    the scaler are fitted on source rows, and applies the whole thing to the RAW YRBS test
    frame. Compare it with `unadapted_scores`, which differs in exactly one respect: there,
    both frames were standardised against themselves first, which is a label-free adaptation
    the target cohort's own distribution supplies. The difference between the two rows is what
    cohort standardisation buys.

    NOT PART OF THE PUBLISHED RESULTS. Nothing under `outputs/` contains this regime; the
    published battery is cohort-standardised throughout. It is restored so the baseline ladder
    can be stated rather than assumed, and notebook 02 runs it only when asked.

    Rows carry `lineage="raw"`, which is what distinguishes them in `regime_battery.csv` from
    every other row in the frame.
    """
    import models as PM
    est = PM.source_scaled_estimator(family, params, seed=seed)
    est.fit(splits["Xm_trm"], splits["ym_trm"].astype(int))
    return {"source_scaled": (splits["yte"],
                              est.predict_proba(splits["Xy_te"])[:, 1],
                              {"lineage": "raw"})}


def reference_scores(splits, *, family, params, seed, base=None,
                     target_params=None, **_) -> dict:
    """The two local references every transfer number is read against.

    NEITHER IS A CEILING. Each is a model developed and evaluated inside one cohort, and a
    transfer procedure exceeding one is a possible result rather than an inconsistency.

    `mcs_internal`   the MCS local reference: the source model, configured by the MCS consensus
                     selection, on held-out MCS records. What it achieves without crossing a
                     border.
    `yrbs_local`     the YRBS local reference: a model fitted on the labelled YRBS training pool
                     and configured by the YRBS consensus selection, on held-out YRBS records.
                     What the target cohort reaches developing its own model.

    THE TWO ARE SYMMETRIC BY CONSTRUCTION. Both configurations come from the same procedure —
    three development seeds, five-fold inner CV, AUC, mean of the seed-level means — run
    separately inside each cohort's own outer-training partitions. So the distance between them,
    read through unadapted transfer, is not confounded by one side having had a different kind of
    development budget from the other.

    A MODEL USES THE SETTINGS SELECTED IN THE COHORT IT IS TRAINED ON. `params` is the MCS
    mapping's entry and configures the MCS-trained model; `target_params` is the YRBS mapping's
    and configures the YRBS-trained one. `params` is NEVER used for `yrbs_local`: a target
    reference that quietly fell back to the source-selected configuration would be measuring the
    source's development choices under the target's name.

    `yrbs_local` IS EMITTED ONLY WHEN `target_params` IS PASSED, so a caller that has not loaded
    the YRBS mapping gets no target reference rather than a mis-configured one.

    NO PER-SEED SELECTION HAPPENS HERE. Both configurations are fixed across the twenty
    evaluation splits, so there is no per-cell search status to record and no cell that could
    fail to be configured. The twenty-seed nested target search is a separate, off-by-default
    sensitivity — see `nested_target_sensitivity_scores`.

    TIER 1a — `mcs_internal` SCORES ARE MCS ROW-LEVEL. They must not be persisted or printed.
    The row carries is_mcs=True and only aggregates are written; the score parquet branch
    excludes it. Aggregate immediately; do not hold, log or save.

    PREVALENCE NULL. `mcs_internal` is reported at the MCS null; every other regime at the
    YRBS one. The two are not comparable, so a PR-AUC quoted across that boundary is
    meaningless where an AUC is fine. Both nulls are on the row, in `prevalence`.

    SCALER. `Xm_te_cs` is the MCS test frame standardised against ITSELF, not against the
    MCS training side — `standardise_cohort` fits on whatever frame it is given. So `base`
    is fitted in `Xm_cs` units and scored in `Xm_te_cs` units; the two scalers are close but
    not identical. The same holds for `yrbs_local`, which fits on the YRBS-train scaling
    and predicts on the YRBS-test one. `Xy_te_cs2` is equal to `Xy_te_cs` cell for cell,
    so the `_cs2` key changes nothing on the test side. Recorded, not corrected: see
    SplitBundle's docstring.
    """
    import models as PM
    base = base if base is not None else source_model(family, params, seed, splits)
    out = {"mcs_internal": (splits["ymte"],
                            base.predict_proba(splits["Xm_te_cs"])[:, 1], {})}  # aggregates only

    if target_params is None:
        return out
    if not target_params:
        raise ValueError(
            f"reference_scores: target_params for {family} at seed {seed} is empty. The YRBS "
            f"local reference takes its configuration from the YRBS consensus selection alone; "
            f"an empty mapping would silently fit library defaults under a name that says a "
            f"configuration was selected.")

    # The per-regime label. `metric_rows` reads it from `extra`, so `mcs_internal` keeps the
    # MCS-selected label the caller passed and only this row carries the YRBS one.
    local = PM.make_estimator(family, target_params, seed=seed)
    local.fit(splits["Xy_tr_cs2"], splits["yy_trm"].astype(int))
    out["yrbs_local"] = (splits["yte"], local.predict_proba(splits["Xy_te_cs2"])[:, 1],
                         {"hyperparameter_source": PM.YRBS_HYPERPARAMETER_SOURCE})
    return out


def nested_target_sensitivity_scores(splits, *, family, seed, target_params=None,
                                     target_selection=None, **_) -> dict:
    """The nested per-split target redevelopment — A SENSITIVITY ANALYSIS, NOT A REFERENCE.

    Emits `yrbs_resource_rich`: the same YRBS training rows as the local reference, but with the
    configuration re-selected inside THIS seed's own YRBS training partition rather than taken
    from the fixed consensus specification. Its question is whether a fully nested redevelopment
    procedure reaches somewhere different from the fixed local specification.

    IT SUPPLIES NO HEADLINE QUANTITY. No reference-relative column is computed against it, it is
    not a member of `evaluation.REPORTING_GROUPS["reference"]`, and it is off by default. It is
    kept in a function of its own precisely so that it cannot be emitted by accident alongside
    the two local references.

    Where the search was not estimable the regime is still emitted with an all-NaN score vector,
    so the cell is recorded as attempted rather than disappearing from the sensitivity's grid.

    NO FALLBACK. A cell whose search did not succeed receives no configuration at all; giving it
    the consensus one would put a fixed-specification result in a nested-search column.
    """
    import models as PM

    if target_selection is None:
        raise ValueError(
            "nested_target_sensitivity_scores: no per-seed selection record was passed. This "
            "regime exists to report what a per-split search chose, so it cannot be run without "
            "one.")
    diagnostics = {
        "hyperparameter_source": "yrbs_cv_per_seed",
        "target_selection_status": target_selection["status"],
        "target_selection_folds": target_selection["folds"],
        "target_selection_candidates": target_selection["candidates"],
        "target_selection_cv_auc": target_selection["cv_auc"],
    }
    if target_selection["status"] != PM.TARGET_SELECTION_SELECTED:
        # ATTEMPTED, NOT ESTIMATED. An all-NaN vector makes `metric_rows` blank every measured
        # metric, mark the row degenerate and write no person-level rows, which is how a cell
        # that could not be estimated stays visible instead of vanishing from the grid.
        return {"yrbs_resource_rich": (
            splits["yte"], np.full(len(np.asarray(splits["yte"])), np.nan),
            {**diagnostics, "flag": target_selection["status"]})}
    if target_params is None:
        raise ValueError(
            f"nested_target_sensitivity_scores: the selection for {family} at seed {seed} "
            f"reports {PM.TARGET_SELECTION_SELECTED!r} but no target_params were passed. Falling "
            f"back to the fixed consensus configuration would report the local reference under "
            f"the sensitivity's name.")

    nested = PM.make_estimator(family, target_params, seed=seed)
    nested.fit(splits["Xy_tr_cs2"], splits["yy_trm"].astype(int))
    return {"yrbs_resource_rich": (splits["yte"],
                                   nested.predict_proba(splits["Xy_te_cs2"])[:, 1],
                                   diagnostics)}


# ---------------------------------------------------------------- the reverse direction
#
# THE FORWARD ANALYSIS FITS IN MCS AND EVALUATES IN YRBS. Everything above is that direction.
# The three functions below read it the other way round — fit in YRBS, evaluate in MCS — so
# that "transfer is difficult here" can be separated from "transfer out of MCS is difficult".
# One direction cannot make that separation and neither can two, but which direction transfers
# better constrains which explanations survive.
#
# THE CORRESPONDENCE IS EXACT, ROLE FOR ROLE, and `regime_names.REVERSE_ROLES` records it:
# `reverse_transfer` mirrors `unadapted`, and `mcs_local_reference` mirrors `yrbs_local`. Read
# the pair the same way in both directions: the transfer arm is a model developed in one cohort
# and evaluated in the other, and the local reference is what the evaluating cohort reaches
# developing its own model. NEITHER REFERENCE IS A CEILING.
#
# ONE OF THE TWO IS FITTED HERE AND THE OTHER IS REUSED. `mcs_local_reference` is the battery's
# own `mcs_internal` — the same fit, the same split, the same metrics — so it is read from
# notebook 02's per-seed frame under validation rather than computed a second time. See
# `mcs_local_reference_rows`.
#
# THE MIRROR IS NOW EXACT, WHICH IT PREVIOUSLY WAS NOT. Both cohorts' configurations come from
# the same consensus procedure run inside their own outer-training partitions, and both are fixed
# across the twenty splits. So `reverse_transfer` is a fixed YRBS-developed model evaluated
# twenty times in MCS, exactly as `unadapted` is a fixed MCS-developed model evaluated twenty
# times in YRBS, and the two directions are comparable without a caveat about which side had a
# per-split search. A MODEL USES THE SETTINGS SELECTED IN THE COHORT IT IS TRAINED ON, here as
# everywhere else.
#
# TIER 1a — EVERY SCORE VECTOR BELOW IS MCS ROW-LEVEL, because all three roles predict on
# `Xm_te_cs`. None may be persisted or printed. `metric_rows` is deliberately NOT reused: it
# builds person-level score frames for every regime except `mcs_internal`, and every role here
# would qualify. `reverse_metric_rows` aggregates on the spot and returns rows only.


def reverse_transfer_scores(splits, *, family, seed, yrbs_params) -> dict:
    """The YRBS-configured role for one (family, threshold, seed) cell.

    `reverse_transfer`  the YRBS training side, configured by the YRBS consensus selection,
                        scored on the MCS test frame. The exact mirror of `unadapted`.

    THE OTHER ROLE IS NOT FITTED HERE. `mcs_local_reference` is the MCS training side under the
    MCS consensus selection, which is what the forward battery already computes and stores as
    `mcs_internal`. `mcs_local_reference_rows` validates those rows and reuses them, so this
    function fits one model per cell and the reverse reading contains no second definition of a
    quantity the battery already carries.

    THE CONFIGURATION COMES FROM THE COHORT THE MODEL IS TRAINED ON, in this direction as in the
    forward one. `reverse_transfer` is trained on YRBS, so it takes `yrbs_params` —
    `models.yrbs_settings()[(threshold, family)]`. Handing it the MCS mapping would make it a
    model configured by one cohort's development budget and trained on another's, under a name
    that says only the training cohort changed; it is refused rather than accepted.

    NO PER-SEED SELECTION AND NO UNCONFIGURED CELL. `yrbs_params` comes from the fixed
    specification, which is complete by construction, so every cell is configured and there is
    no search status to record.

    SCALER. `Xm_te_cs` is the MCS test frame standardised against ITSELF, and `Xy_tr_cs2` is
    the YRBS training side standardised against itself. Both are the bundle's own frames, so
    the reverse direction inherits the same cohort-standardisation convention the forward
    direction is evaluated under and introduces none of its own.
    """
    import models as PM

    if not yrbs_params:
        raise ValueError(
            f"reverse_transfer_scores: no YRBS configuration was passed for {family} at seed "
            f"{seed}. This model is trained on YRBS, so it takes the YRBS consensus selection — "
            f"`models.yrbs_settings()[(threshold, family)]`. Falling back to the MCS mapping, or "
            f"to library defaults, would report a differently developed model under a name that "
            f"says only the training cohort changed.")

    # The YRBS-side fit, on the frames `reference_scores` fits its target-trained arm on —
    # `Xy_tr_cs2` with the NaN-outcome rows already dropped — and scored on the MCS test frame
    # rather than the YRBS one. That single substitution is what reverses the direction.
    reverse = PM.make_estimator(family, yrbs_params, seed=seed)
    reverse.fit(splits["Xy_tr_cs2"], splits["yy_trm"].astype(int))
    return {"reverse_transfer": (
        splits["ymte"], reverse.predict_proba(splits["Xm_te_cs"])[:, 1],
        {"hyperparameter_source": PM.YRBS_HYPERPARAMETER_SOURCE})}


# The third role, reused rather than refitted
#
# `mcs_local_reference` IS THE BATTERY'S `mcs_internal`. Both are `source_model(family, params,
# seed, splits)` under the tracked MCS-selected configuration, scored on `Xm_te_cs` against
# `ymte` — one estimand, one split, one preprocessing, one set of metric definitions. Refitting
# it here would be a second computation of a number notebook 02 already produced, on the same
# rows, and the only thing a second computation could establish is that the two agree.
#
# SO THE ROWS ARE VALIDATED RATHER THAN TRUSTED. What must hold is checkable on the frame:
# the regime, the reported arm, the cohort-standardised lineage, the MCS-evaluated flag, the
# hyperparameter source, and one row per declared cell with every seed present. Anything short
# of that stops the read with a message naming what was wrong — never a quiet recomputation,
# never a default, never a partial grid.

MCS_LOCAL_ROLE = "mcs_local_reference"
MCS_LOCAL_SOURCE_REGIME = "mcs_internal"
# The label the forward battery writes on an MCS-configured row. It moves in lockstep with
# `models.MCS_HYPERPARAMETER_SOURCE`: this is a validation gate, so a stale value here silently
# makes notebook 04 refuse a perfectly good battery.
MCS_LOCAL_HYPERPARAMETER_SOURCE = "mcs_consensus3"
MCS_LOCAL_IDENTITY: Sequence[str] = (
    "regime", "family", "label", "arm", "hyperparameter_source", "threshold", "lineage",
    "seed", "is_mcs", "degenerate",
)


def mcs_local_reference_rows(battery, *, families, thresholds, seeds) -> list:
    """The battery's per-seed `mcs_internal` rows, validated, as `mcs_local_reference` rows.

    `battery` is notebook 02's per-seed frame — one row per regime x family x arm x threshold x
    seed. Returns rows in `reverse_metric_rows`' schema, so the two sources concatenate into one
    frame without either knowing about the other.

    SIX CHECKS, AND EACH ONE REFUSES RATHER THAN REPAIRS:

      schema        every identity column and every reported metric is present
      regime        rows are `mcs_internal`, and each carries the MCS-evaluated flag
      arm           the reported arm alone; the untuned arm answers a different question
      lineage       cohort-standardised, which is what every published number here is
      configuration the MCS-selected label, so the row cannot be a target-configured one
      coverage      exactly one row per declared cell, every seed present, no duplicates

    A frame that fails any of them is a frame describing a different run, a different arm or a
    different grid, and the caller is told which. Nothing here recomputes, substitutes or
    accepts a partial grid: a reused number has to be the number it is standing in for.
    """
    from collections import Counter

    import pandas as pd

    required = (*MCS_LOCAL_IDENTITY, "prevalence_null", *REVERSE_METRICS)
    absent = [c for c in required if c not in getattr(battery, "columns", ())]
    if absent:
        raise ValueError(
            f"the per-seed battery frame is missing column(s) {absent}, so its rows cannot be "
            f"read as the metric rows this reading reports. It is written by notebook 02 as "
            f"regime_battery.csv and carries all {len(required)}.")

    rows = battery[battery["regime"].astype(str) == MCS_LOCAL_SOURCE_REGIME]
    if rows.empty:
        raise ValueError(
            f"the per-seed battery frame carries no {MCS_LOCAL_SOURCE_REGIME!r} rows, so there "
            f"is nothing to reuse as the local MCS reference. It comes from notebook 02's "
            f"reference section; a battery written without that section cannot supply it.")

    rows = rows[rows["arm"].astype(str) == ARM]
    if rows.empty:
        raise ValueError(
            f"every {MCS_LOCAL_SOURCE_REGIME!r} row in the battery is outside the reported arm "
            f"({ARM!r}). The untuned arm is a different configuration and answers a different "
            f"question, so it cannot stand in for the local MCS reference.")

    unexpected = sorted(set(rows["lineage"].astype(str)) - {LINEAGE})
    if unexpected:
        raise ValueError(
            f"the reused rows carry lineage {unexpected}, not {LINEAGE!r}. Everything this "
            f"reading reports is cohort-standardised, and a raw-scale row is a different "
            f"preprocessing rather than a different presentation of the same one.")

    if not rows["is_mcs"].astype(bool).all():
        raise ValueError(
            f"a {MCS_LOCAL_SOURCE_REGIME!r} row does not carry the MCS-evaluated flag, so the "
            f"regime column and the evaluation cohort disagree. The reused rows must be the "
            f"ones evaluated inside MCS.")

    labels = sorted(set(rows["hyperparameter_source"].astype(str))
                    - {MCS_LOCAL_HYPERPARAMETER_SOURCE})
    if labels:
        raise ValueError(
            f"the reused rows carry hyperparameter source(s) {labels}, not "
            f"{MCS_LOCAL_HYPERPARAMETER_SOURCE!r}. The local MCS reference is defined by its "
            f"configuration having been chosen inside MCS; a row configured anywhere else is a "
            f"different role.")

    wanted = {(str(f), f">={int(t)}", int(s))
              for f in families for t in thresholds for s in seeds}
    present = list(zip(rows["family"].astype(str), rows["threshold"].astype(str),
                       rows["seed"].astype(int)))
    duplicated = sorted(k for k, n in Counter(present).items() if n > 1 and k in wanted)
    if duplicated:
        raise ValueError(
            f"{len(duplicated)} declared cell(s) appear more than once among the reused rows, "
            f"e.g. {duplicated[:4]}. Which of them is the local MCS reference is undetermined, "
            f"and averaging them would weight one seed twice.")

    absent_cells = sorted(wanted - set(present))
    if absent_cells:
        by_cell = Counter((f, t) for f, t, _ in absent_cells)
        raise ValueError(
            f"the battery is short {len(absent_cells)} of {len(wanted)} declared "
            f"(family, threshold, seed) cells for the local MCS reference, over "
            f"{len(by_cell)} (family, threshold) cell(s), e.g. {absent_cells[:4]}. A cell "
            f"missing a seed cannot be summarised over {len(list(seeds))} of them, and nothing "
            f"here fills the gap or reports a mean over the seeds that survived.")

    keep = pd.Series(present, index=rows.index).isin(wanted)
    rows = rows[keep.to_numpy()]

    out = []
    for record in rows.to_dict("records"):
        degenerate = bool(record["degenerate"])
        auc = float(record["auc"]) if record["auc"] == record["auc"] else float("nan")
        if not degenerate and auc != auc:
            raise ValueError(
                f"the reused row for {(record['family'], record['threshold'], record['seed'])} "
                f"reports no AUC and is not marked degenerate, so it is neither an estimate nor "
                f"a recorded failure to estimate one.")
        note = record.get("note", "")
        out.append(dict(
            family=str(record["family"]), label=str(record["label"]), role=MCS_LOCAL_ROLE,
            threshold=str(record["threshold"]), seed=int(record["seed"]), lineage=LINEAGE,
            hyperparameter_source=MCS_LOCAL_HYPERPARAMETER_SOURCE,
            target_selection_status="", target_selection_cv_auc=float("nan"),
            degenerate=degenerate, note=("" if note != note else str(note)),
            prevalence_null=float(record["prevalence_null"]),
            **{m: float(record[m]) for m in REVERSE_METRICS}))
    return out


# the row.
REVERSE_METRICS: Sequence[str] = (
    "auc", "prauc", "prauc_lift", "brier", "ece", "cal_slope", "cal_intercept",
)


def reverse_metric_rows(sc: dict, *, family: Family, threshold: int, seed: int) -> list:
    """One cell's `{role: (y, p, extra)}` into metric rows. No score frame, by construction.

    The metric definitions are the pipeline's own — `evaluation.metrics`, the same function
    `metric_rows` calls — so an AUC, a PR-AUC, a Brier score, an ECE and the two calibration
    statistics mean here exactly what they mean in the forward battery. What differs is what
    leaves the function: rows only, carrying the rates and neither the counts nor the
    denominators behind them.

    A role whose score vector carries no finite value on any evaluable row is recorded with
    every measured metric blank and `degenerate=True`, which is `metric_rows`' own treatment of
    a cell that was attempted and could not be estimated.
    """
    import evaluation as E
    import models as PM

    rows = []
    for role, (y, p, extra) in sc.items():
        _y = np.asarray(y, dtype=np.float64)
        _p = np.asarray(p, dtype=np.float64)
        _eval = ~np.isnan(_y)
        if not _eval.any():
            raise ValueError(
                f"{role} at {(family, threshold, seed)}: no evaluable outcome in the MCS test "
                f"frame, so no metric is defined on it.")
        if not np.isin(_y[_eval], (0.0, 1.0)).all():
            raise ValueError(
                f"{role} at {(family, threshold, seed)}: the evaluable outcome takes a value "
                f"other than 0 or 1, so it is not the binary label this pipeline evaluates.")
        null = float(np.nanmean(_y))
        if np.isfinite(_p[_eval]).any():
            measured = E.metrics(_y, _p, prevalence=null)
            values = {m: measured[m] for m in REVERSE_METRICS}
            degenerate = False
        else:
            values = {m: np.nan for m in REVERSE_METRICS}
            degenerate = True
        rows.append(dict(
            family=family, label=PM.FAM[family][0], role=role,
            threshold=f">={threshold}", seed=seed, lineage=LINEAGE,
            hyperparameter_source=extra.get("hyperparameter_source",
                                            PM.MCS_HYPERPARAMETER_SOURCE),
            target_selection_status=extra.get("target_selection_status", ""),
            target_selection_cv_auc=extra.get("target_selection_cv_auc", float("nan")),
            degenerate=degenerate, note=extra.get("flag", ""),
            prevalence_null=null, **values))
    return rows



def prior_correction_scores(splits, *, family, params, seed, base=None, **_) -> dict:
    """Prior correction (BBSE): rescale the source model's probabilities for the target rate.

    Transforms an existing model's output rather than training anything, so it needs the
    source model. Strictly increasing while both weights are positive, which is why its rank
    invariance is a property of the estimate and gets checked rather than assumed.
    """
    base = base if base is not None else source_model(family, params, seed, splits)
    s_unadapted = base.predict_proba(splits["Xy_te_cs"])[:, 1]
    bbse_scores, bbse_info = bbse_correct(
        base.predict_proba(splits["Xm_te_cs"])[:, 1], splits["ymte"],
        base.predict_proba(splits["Xy_tr_cs"])[:, 1], s_unadapted, return_diagnostics=True)
    return {"bbse": (splits["yte"], bbse_scores, bbse_info)}


def pseudo_label_scores(splits, *, family, params, seed, base=None, **_) -> dict:
    """Pseudo-labelling: the source model's most confident target deciles join its training set.

    Needs the source model to produce the pseudo-labels, then trains a fresh model on MCS plus
    those records. Confident means the top and bottom decile of the pool by predicted score.
    """
    import models as PM
    base = base if base is not None else source_model(family, params, seed, splits)
    Xm_cs, ym = splits["Xm_cs"], splits["ym_trm"]
    pool_scores = pd.Series(base.predict_proba(splits["Xy_tr_cs"])[:, 1],
                            index=splits["Xy_tr_cs"].index)
    nd = max(1, int(0.10 * len(pool_scores)))
    top, bot = pool_scores.nlargest(nd).index, pool_scores.nsmallest(nd).index
    Xps = splits["Xy_tr_cs"].loc[list(top) + list(bot)]
    yps = np.r_[np.ones(len(top)), np.zeros(len(bot))]
    mp_ = PM.make_estimator(family, params, seed=seed)
    mp_.fit(pd.concat([Xm_cs, Xps], ignore_index=True), np.r_[ym.values, yps].astype(int))
    return {"pseudo_label": (splits["yte"], mp_.predict_proba(splits["Xy_te_cs"])[:, 1], {})}


def quantile_map_scores(splits, *, family, params, seed, **_) -> dict:
    """Quantile mapping: MCS features mapped onto the YRBS pool's marginals, then trained.

    Aligns each feature's whole marginal rather than just its location and scale. Trains a
    fresh model on the mapped frame and never touches the source model.
    """
    import data as D
    import models as PM
    ym = splits["ym_trm"]
    Xm_map = quantile_match(splits["Xy_tr"], splits["Xm_trm"])
    Xm_map_cs = D.standardise_cohort(Xm_map)
    Xy_te_cs_q = D.standardise_cohort(splits["Xy_te"])       # == splits["Xy_te_cs"]
    mq = PM.make_estimator(family, params, seed=seed); mq.fit(Xm_map_cs, ym.astype(int))
    return {"quantile_map": (splits["yte"], mq.predict_proba(Xy_te_cs_q)[:, 1], {})}


def importance_weight_scores(splits, *, family, params, seed, base=None, **_) -> dict:
    """Importance weighting: density ratios from a domain classifier reweight MCS training.

    Corrects the joint under a covariate-shift assumption. Trains a fresh weighted model; the
    source model is fitted ONLY on the fallback path, for families whose estimator takes no
    `sample_weight`, where the method degenerates to unadapted transfer and says so.
    """
    from sklearn.linear_model import LogisticRegression
    import models as PM
    Xm_cs, ym = splits["Xm_cs"], splits["ym_trm"]
    Xall = pd.concat([Xm_cs, splits["Xy_tr_cs"]], ignore_index=True)
    dom = np.r_[np.zeros(len(Xm_cs)), np.ones(len(splits["Xy_tr_cs"]))]
    dc = LogisticRegression(max_iter=500).fit(Xall.values, dom)
    pm_ = dc.predict_proba(Xm_cs.values)[:, 1]
    w = np.clip(pm_ / (1 - pm_ + 1e-8), 0.01, 100.0)
    miw = PM.make_estimator(family, params, seed=seed)
    try:
        miw.fit(Xm_cs, ym.astype(int), sample_weight=w)
        return {"importance_weight": (splits["yte"],
                                      miw.predict_proba(splits["Xy_te_cs"])[:, 1], {})}
    except TypeError:
        base = base if base is not None else source_model(family, params, seed, splits)
        return {"importance_weight": (splits["yte"],
                                      base.predict_proba(splits["Xy_te_cs"])[:, 1],
                                      {"flag": "no_sample_weight_support"})}


def _k_slice(splits, k: int = K):
    """The k=500 anchor and whether it can be used. Every label-using procedure starts here.

    Returns `(Xk, yk, effective_k, usable)`. `usable` is False when the drawn slice carries one
    class, in which case a procedure cannot be estimated and falls back to unadapted transfer
    with a flag rather than inventing a number.
    """
    s_cs = splits.k_slice(k)
    Xk, yk = s_cs[splits.feat_cols], s_cs["y"]
    return Xk, yk, len(Xk), yk.nunique() >= 2


def fine_tune_scores(splits, *, family, params, seed, k: int = K, base=None, **_) -> dict:
    """Full revision: every parameter re-estimated, from the source model plus k target labels.

    Warm-starts from the source model for the families that support it; the rest get a fresh
    weighted refit on MCS plus the k labels, the target rows carried at weight ALPHA. The
    weighted branch trains a genuinely new model and needs no source model at all.
    """
    import models as PM
    Xk, yk, eff_cs, cs_ok = _k_slice(splits, k)
    if cs_ok and family in PM.WARM_FAMS:
        base = base if base is not None else source_model(family, params, seed, splits)
        fscore = warm_finetune(family, params, base, Xk, yk, seed)(splits["Xy_te_cs"])
        return {"fine_tune": (splits["yte"], fscore,
                              {"ft_mechanism": "warm_start", "effective_k": eff_cs})}
    if cs_ok:
        Xm_cs, ym = splits["Xm_cs"], splits["ym_trm"]
        Xb = pd.concat([Xm_cs, Xk], ignore_index=True)
        yb = np.r_[ym.values, yk.values].astype(int)
        wgt = np.r_[np.ones(len(Xm_cs)), np.full(len(Xk), ALPHA)]
        mf = PM.make_estimator(family, params, seed=seed); mf.fit(Xb, yb, sample_weight=wgt)
        return {"fine_tune": (splits["yte"], mf.predict_proba(splits["Xy_te_cs"])[:, 1],
                              {"ft_mechanism": f"weighted_refit(alpha={ALPHA:g})",
                               "effective_k": eff_cs})}
    base = base if base is not None else source_model(family, params, seed, splits)
    return {"fine_tune": (splits["yte"], base.predict_proba(splits["Xy_te_cs"])[:, 1],
                          {"ft_mechanism": "skipped_single_class", "flag": "degenerate"})}


def target_only_scores(splits, *, family, params, seed, base=None, k: int = K, **_) -> dict:
    """Trains on the k target labels alone — the source cohort is not used.

    What an authority could build from a small labelled sample of its own and nothing else.

    `base` IS NOT USED BY THE METHOD. The normal path fits on the k labels and never touches a
    source model. It is taken only for the degenerate path, where the drawn slice carries one
    class and there is nothing to fit: that path falls back to unadapted transfer, which needs
    the source model, and refitting one there would be a second copy of a model the caller
    already holds.
    """
    import models as PM
    Xk, yk, eff_cs, cs_ok = _k_slice(splits, k)
    if cs_ok:
        mt = PM.make_estimator(family, params, seed=seed); mt.fit(Xk, yk.astype(int))
        return {"target_only": (splits["yte"], mt.predict_proba(splits["Xy_te_cs"])[:, 1],
                                {"effective_k": eff_cs})}
    base = base if base is not None else source_model(family, params, seed, splits)
    return {"target_only": (splits["yte"], base.predict_proba(splits["Xy_te_cs"])[:, 1],
                            {"flag": "degenerate_single_class"})}


# Post-adaptation cross-fitted logistic recalibration
# SOURCE: working_draft.tex:285 (§V-F). "Logistic recalibration followed target-only fitting and
#         full revision across all families and budgets. Using the same k labelled YRBS records,
#         it fits an intercept and slope relating model scores to outcomes. Cross-fitting divides
#         these records into folds to avoid in-sample scores. Each fold is predicted by a model
#         adapted on the other folds; the out-of-fold predictions fit the recalibration mapping.
#         The final model is adapted on all k records before the mapping is applied to test
#         scores. Five folds were used, reduced to three when the smaller outcome class could
#         not support five."
# UNVERIFIED: needs MCS and YRBS data to check the SCORES. The fold construction, the
#             out-of-fold coverage and the fitting/evaluation separation are checked on
#             synthetic data in tests/test_crossfit_recal.py.

# The manuscript's fold counts. Five, reduced to three, and no further: below three the
# smaller class cannot populate three folds either, and the procedure reports rather than
# reducing again.
RECAL_FOLDS_PRIMARY = 5
RECAL_FOLDS_FALLBACK = 3

# Tolerance for CLASSIFYING a fitted slope's direction. It is a diagnostic threshold and
# carries no statistical claim: the slope is not constrained and nothing is rounded to it.
# Justification for the value — the logit of a clipped probability spans roughly +/-13.8, so a
# coefficient of 1e-8 moves the linear predictor by about 1.4e-7 across the whole score range,
# which is far below any reported precision. Predictions from such a mapping are constant for
# every practical purpose, and `recal_score_range` is reported beside the slope so a
# numerically non-zero slope that still yields a flat mapping is visible rather than inferred.
RECAL_SLOPE_TOL = 1e-8

# The four pipelines the draft carries through §VI-C and §VI-D as (regime, family) pairs.
# A pair, not two independent lists: `fine_tune` is defined for all nine families and
# `target_only` for all nine, but only these four combinations are reported.
FOCAL_PIPELINES: Sequence[tuple] = (
    ("target_only", "L1_LR"),
    ("fine_tune", "RF"),
    ("fine_tune", "HistGB"),
    ("fine_tune", "CatBoost"),
)


def recal_fold_count(yk, *, primary: int = RECAL_FOLDS_PRIMARY,
                     fallback: int = RECAL_FOLDS_FALLBACK) -> int:
    """How many folds this label slice supports: `primary`, `fallback`, or 0 for neither.

    THE CONDITION IS ON THE SMALLER OUTCOME CLASS, not on the slice size. Stratified folds put
    at least one member of each class in every fold, so a slice of 500 with four positives
    supports neither five folds nor three. Returning 0 is how that is reported; the caller
    records it rather than silently dropping to two folds or fitting in-sample.
    """
    y = np.asarray(yk, float)
    y = y[~np.isnan(y)].astype(int)
    if y.size == 0 or np.unique(y).size < 2:
        return 0
    minority = int(min(np.bincount(y, minlength=2)))
    if minority >= primary:
        return primary
    if minority >= fallback:
        return fallback
    return 0


def focal_adapter(regime: str, family: Family, params, base, seed: int, splits, Xk, yk):
    """The focal adaptation, fitted on `(Xk, yk)`, returned as a callable `score(X) -> p1`.

    THIS IS THE SAME ARITHMETIC AS `target_only_scores` AND `fine_tune_scores`, factored so the
    fit can be applied to a held-out fold as well as to the test frame. It is deliberately not a
    new method: same estimator factory, same params, same seed, same weighting, same warm-start
    branch. Cross-fitting needs a model that can predict something other than `Xy_te_cs`, which
    is the only reason this exists.

    `Xk`/`yk` are whatever subset the caller is fitting on — the whole k slice for the final
    model, or K-1 folds of it during cross-fitting.
    """
    import models as PM
    if regime == "target_only":
        mt = PM.make_estimator(family, params, seed=seed)
        mt.fit(Xk, yk.astype(int))
        return lambda X, _m=mt: _m.predict_proba(X)[:, 1]
    if regime != "fine_tune":
        raise ValueError(
            f"focal_adapter is defined for 'target_only' and 'fine_tune', not {regime!r}. "
            f"The focal pipelines are {list(FOCAL_PIPELINES)}.")
    if family in PM.WARM_FAMS:
        if base is None:
            raise ValueError(
                f"fine_tune on {family} warm-starts from the canonical source model and none "
                f"was passed. Section D fits them; pass one rather than refitting here.")
        return warm_finetune(family, params, base, Xk, yk, seed)
    Xm_cs, ym = splits["Xm_cs"], splits["ym_trm"]
    Xb = pd.concat([Xm_cs, Xk], ignore_index=True)
    yb = np.r_[ym.values, yk.values].astype(int)
    wgt = np.r_[np.ones(len(Xm_cs)), np.full(len(Xk), ALPHA)]
    mf = PM.make_estimator(family, params, seed=seed)
    mf.fit(Xb, yb, sample_weight=wgt)
    return lambda X, _m=mf: _m.predict_proba(X)[:, 1]


def crossfit_logistic_recal_scores(splits, *, family, params, seed, regime: str,
                                   k: int = K, base=None, **_) -> dict:
    """Adapt on the k target labels, then recalibrate on OUT-OF-FOLD predictions of those labels.

    Emits `{regime}_logistic_recal`. The regime name carries the distinction, so no separate
    flag column is needed to tell a recalibrated row from a raw one.

    WHY OUT-OF-FOLD. Fitting the mapping on the same records the model was adapted on reads the
    adaptation's own training scores, which are optimistic, and the correction inherits that
    optimism. `platt_frozen` and `isotonic_recal` do exactly that, deliberately, on the FROZEN
    SOURCE model — they are a different experiment and this is not a renaming of either.

    THE EVALUATION SET ENTERS NEITHER FIT. Folds are drawn inside the k slice, which
    `SplitBundle.k_slice` takes from the training half; `Xy_te_cs` is touched once, at the end,
    to score the final model. Nothing measured on it feeds back.

    NON-ESTIMABLE CELLS RETURN NaN SCORES, NOT THE UNCORRECTED MODEL'S. A cell whose k slice
    carries one class, or whose smaller class cannot populate three folds, has no recalibration
    to report. Returning the raw adapted scores under a recalibrated regime name would put an
    uncorrected number in a corrected column; returning NaN makes `metric_rows` blank the
    metrics, mark the row degenerate and write no score rows for it. The raw focal result stays
    available under its own regime, where it belongs.

    `recal_status` records which of the three outcomes this cell had.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    Xk, yk, eff_cs, cs_ok = _k_slice(splits, k)
    key = f"{regime}_logistic_recal"
    nan_scores = np.full(len(np.asarray(splits["yte"])), np.nan)
    if not cs_ok:
        return {key: (splits["yte"], nan_scores,
                      {"flag": "non-estimable: the k slice carries one outcome class",
                       "recal_status": "single_class_slice",
                       "effective_k": eff_cs, "recal_folds": 0})}

    n_folds = recal_fold_count(yk)
    if n_folds == 0:
        return {key: (splits["yte"], nan_scores,
                      {"flag": "non-estimable: the smaller outcome class cannot populate "
                               "three folds",
                       "recal_status": "too_few_minority_for_three_folds",
                       "effective_k": eff_cs, "recal_folds": 0})}

    yk_int = yk.astype(int).to_numpy()
    oof = np.full(len(Xk), np.nan)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fit_idx, held_idx in folds.split(Xk, yk_int):
        scorer = focal_adapter(regime, family, params, base, seed, splits,
                               Xk.iloc[fit_idx], yk.iloc[fit_idx])
        oof[held_idx] = scorer(Xk.iloc[held_idx])
    if np.isnan(oof).any():
        raise RuntimeError(
            f"{key} at {(family, seed)}: {int(np.isnan(oof).sum())} of {len(oof)} k records "
            f"received no out-of-fold prediction. Every record must be held out exactly once.")

    # Intercept and slope relating the model's score to the outcome, on the logit scale.
    mapping = LogisticRegression(penalty=None, fit_intercept=True, solver="lbfgs")
    mapping.fit(_lclip(oof).reshape(-1, 1), yk_int)

    # The final model is adapted on ALL k records, then the mapping is applied to its test
    # scores. This fit is identical to the raw focal procedure's — same factory, params, seed
    # and training rows — and is repeated here so the procedure is self-contained.
    final = focal_adapter(regime, family, params, base, seed, splits, Xk, yk)
    raw_te = final(splits["Xy_te_cs"])
    corrected = mapping.predict_proba(_lclip(raw_te).reshape(-1, 1))[:, 1]
    if corrected.shape != np.shape(raw_te):
        raise RuntimeError(f"{key}: recalibration changed the score vector's length")

    # `_lclip` bounds the score before taking its logit, so any evaluation score at or beyond
    # those bounds maps to the same logit and the mapping ties them. A positive slope therefore
    # means the order is NOT REVERSED; it does not mean the ranking is untouched. The clipped
    # fraction says how much room there was for new ties.
    raw_arr = np.asarray(raw_te, float)
    lo_b, hi_b = 1e-6, 1 - 1e-6
    clipped = int(((raw_arr <= lo_b) | (raw_arr >= hi_b)).sum())

    return {key: (splits["yte"], corrected,
                  {"effective_k": eff_cs, "recal_folds": n_folds,
                   "recal_status": "estimated",
                   "recal_clipped_n": clipped,
                   "recal_clipped_fraction": float(clipped / raw_arr.size),
                   "recal_score_range": float(corrected.max() - corrected.min()),
                   "recal_intercept": float(mapping.intercept_[0]),
                   # THE SLOPE IS NOT CONSTRAINED POSITIVE. An unpenalised two-parameter
                   # logistic can come out with a negative coefficient, which reverses the
                   # ranking, or with one at zero, which makes every prediction the same. The
                   # manuscript specifies "an intercept and slope relating model scores to
                   # outcomes" and does not define an increasing-only recalibrator, so the
                   # fitted sign is reported rather than imposed.
                   "recal_slope": float(mapping.coef_[0][0])})}


def isotonic_recal_scores(splits, *, family, params, seed, k: int = K, base=None, **_) -> dict:
    """Refits a monotone link on the source model's scores using k target labels.

    Monotone NON-decreasing, so its flat segments tie scores: AUC can legitimately fall and
    can never rise. Two parameters' worth of re-estimation in spirit — it moves calibration,
    not the ranking. Needs the source model, whose scores it is recalibrating.
    """
    from sklearn.isotonic import IsotonicRegression
    Xk, yk, eff_cs, cs_ok = _k_slice(splits, k)
    base = base if base is not None else source_model(family, params, seed, splits)
    s_unadapted = base.predict_proba(splits["Xy_te_cs"])[:, 1]
    if cs_ok:
        iso = IsotonicRegression(out_of_bounds="clip").fit(base.predict_proba(Xk)[:, 1],
                                                           yk.values)
        return {"isotonic_recal": (splits["yte"], iso.transform(s_unadapted),
                                   {"effective_k": eff_cs})}
    return {"isotonic_recal": (splits["yte"], s_unadapted,
                               {"flag": "degenerate_single_class"})}


def raw_l1_head_scores(splits, *, rawl1=None, **_) -> dict:
    """An L1 head fitted on RAW target features and k labels, ignoring the source cohort.

    Family- and arm-independent by construction, so the caller computes it once per
    (threshold, seed) and passes it in. Absent `rawl1` the key is omitted rather than
    recomputed per family, which would produce copies that are only accidentally identical.
    """
    if rawl1 is None:
        return {}
    return {"raw_l1_head": (splits["yte"], rawl1, {"family_independent": True})}


def platt_frozen_scores(splits, *, family, params, seed, k: int = K, base=None, **_) -> dict:
    """A two-parameter logistic refit on the source model's raw logit, using k target labels.

    The cheapest end of the label-using ordering: intercept and slope only, so it cannot move a
    ranking. Needs the source model, whose logit it is rescaling.
    """
    from sklearn.linear_model import LogisticRegression
    Xk, yk, eff_cs, cs_ok = _k_slice(splits, k)
    if not cs_ok:
        return {}
    base = base if base is not None else source_model(family, params, seed, splits)
    lg_k = source_raw_logit(family, base, Xk)
    lg_te = source_raw_logit(family, base, splits["Xy_te_cs"])
    pl = LogisticRegression(penalty=None, fit_intercept=True, solver="lbfgs")
    pl.fit(lg_k.reshape(-1, 1), yk.astype(int))
    return {"platt_frozen": (splits["yte"], pl.predict_proba(lg_te.reshape(-1, 1))[:, 1],
                             {"effective_k": eff_cs})}


def leaf_refresh_global_scores(splits, *, family, params, seed, k: int = K, base=None, **_) -> dict:
    """A one-parameter global intercept offset on the source model's logit. Trees only.

    NOT the leaf refresh of the write-up, despite the name — that one is in the
    structure-transfer section. One degree of freedom, so the ranking is untouched.
    """
    from scipy.special import expit
    import models as PM
    if family in PM.LR_FAMS:
        return {}
    Xk, yk, eff_cs, cs_ok = _k_slice(splits, k)
    if not cs_ok:
        return {}
    base = base if base is not None else source_model(family, params, seed, splits)
    lg_k = source_raw_logit(family, base, Xk)
    lg_te = source_raw_logit(family, base, splits["Xy_te_cs"])
    a2 = fit_offset_intercept(lg_k, yk.values)
    return {"leaf_refresh_global": (splits["yte"], expit(a2 + lg_te),
                                    {"effective_k": eff_cs,
                                     "note": "1-DOF analogue of leaf_refresh (cohort-std)"})}


def _frozen_l1_source(splits, *, arm, threshold, seed, tuned, l1_source=None):
    """The L1-LR coefficients the three coefficient-transfer procedures freeze.

    NOT the cell's own family model: these procedures transfer a sparse linear source's
    coefficients into an LR target, so the source is L1_LR whatever the family under test is.

    `l1_source` is the canonical L1_LR model for this (threshold, seed), if the caller holds
    one. It is the same object the battery's own L1_LR cell uses — `source_params` and
    `select_params` return identical settings on the tuned arm — so passing it and fitting a
    fresh one give the same coefficients. Read-only: only `coef_` is taken.
    """
    import models as PM
    if l1_source is None:
        l1_source = PM.make_estimator(
            "L1_LR", source_params("L1_LR", arm, int(threshold), tuned=tuned), seed=seed)
        l1_source.fit(splits["Xm_cs"], splits["ym_trm"].astype(int))
    return l1_source.coef_.ravel()


def coef_freeze_intercept_scores(splits, *, family, params, seed, arm=ARM, threshold=1, l1_source=None,
                                 tuned=None, k: int = K, **_) -> dict:
    """Freeze the L1-LR source coefficients and re-estimate the intercept alone. LR targets only.

    One parameter. Uses no model fitted on this family — the coefficients come from a fresh
    L1_LR source fit.
    """
    from scipy.special import expit
    import models as PM
    if family not in PM.LR_FAMS:
        return {}
    Xk, yk, eff_cs, cs_ok = _k_slice(splits, k)
    if not cs_ok:
        return {}
    beta = _frozen_l1_source(splits, arm=arm, threshold=threshold, seed=seed,
                             tuned=tuned, l1_source=l1_source)
    eta_k = Xk.values @ beta
    eta_te = splits["Xy_te_cs"].values @ beta
    a1 = fit_offset_intercept(eta_k, yk.values)
    return {"coef_freeze_intercept": (splits["yte"], expit(a1 + eta_te),
                                      {"effective_k": eff_cs})}


def sign_support_scores(splits, *, family, params, seed, arm=ARM, threshold=1, l1_source=None,
                        tuned=None, k: int = K, **_) -> dict:
    """Keep the source's active set and coefficient SIGNS, re-estimate magnitudes on k labels.

    LR targets only. More freedom than freezing the coefficients, less than a full refit — the
    source constrains which features enter and in which direction, nothing more.
    """
    from scipy.special import expit
    import models as PM
    if family not in PM.LR_FAMS:
        return {}
    Xk, yk, eff_cs, cs_ok = _k_slice(splits, k)
    if not cs_ok:
        return {}
    beta = _frozen_l1_source(splits, arm=arm, threshold=threshold, seed=seed,
                             tuned=tuned, l1_source=l1_source)
    active = np.where(np.abs(beta) > 1e-8)[0]
    if active.size == 0:
        active = np.arange(beta.size)         # all-zero source -> keep all columns
    signs = np.sign(beta[active]); signs[signs == 0] = 1.0
    Xk_a = Xk.values[:, active] * signs
    Xte_a = splits["Xy_te_cs"].values[:, active] * signs
    w4, any_bound = fit_nonneg_logistic(Xk_a, yk.values)
    return {"sign_support": (splits["yte"], expit(Xte_a @ w4[:-1] + w4[-1]),
                             {"effective_k": eff_cs, "any_coef_bound": any_bound})}


def feature_set_scores(splits, *, family, params, seed, arm=ARM, threshold=1, l1_source=None,
                       tuned=None, k: int = K, **_) -> dict:
    """Keep only the source's active features and retrain freely on k target labels.

    LR targets only. NOT the harmonised feature schema, despite the name. Everything except
    which columns to use is re-estimated, so this is a fresh model on a restricted frame.
    """
    import models as PM
    if family not in PM.LR_FAMS:
        return {}
    Xk, yk, eff_cs, cs_ok = _k_slice(splits, k)
    if not cs_ok:
        return {}
    beta = _frozen_l1_source(splits, arm=arm, threshold=threshold, seed=seed,
                             tuned=tuned, l1_source=l1_source)
    active = np.where(np.abs(beta) > 1e-8)[0]
    if active.size == 0:
        active = np.arange(beta.size)
    m5 = PM.make_estimator(family, params, seed=seed)
    m5.fit(Xk.iloc[:, active], yk.astype(int))
    return {"feature_set": (splits["yte"],
                            m5.predict_proba(splits["Xy_te_cs"].iloc[:, active])[:, 1],
                            {"effective_k": eff_cs})}


def ensemble_same_family_scores(splits, *, family, params, seed, k: int = K, base=None, **_) -> dict:
    """Convex blend of the source model with a target-trained model of the same family."""
    Xk, yk, eff_cs, cs_ok = _k_slice(splits, k)
    if not cs_ok:
        return {}
    base = base if base is not None else source_model(family, params, seed, splits)
    st6, cx6, lam6 = stacked_ensemble(base, family, params, Xk, yk, splits["Xy_te_cs"], seed)
    return {"ensemble_same_family": (splits["yte"], st6,
                                     {"effective_k": eff_cs, "convex_scores": cx6,
                                      "convex_lambda": lam6})}


def ensemble_catboost_source_scores(splits, *, family, params, seed, arm=ARM, threshold=1,
                                    tuned=None, cb_source=None, k: int = K, **_) -> dict:
    """Blend a CatBoost SOURCE model with a target-trained model of the cell's own family.

    Excluded for CatBoost itself, where it would be the same-family ensemble. The CatBoost
    source is fitted fresh here rather than borrowed from another cell.
    """
    import models as PM
    if family == "CatBoost":
        return {}
    Xk, yk, eff_cs, cs_ok = _k_slice(splits, k)
    if not cs_ok:
        return {}
    # The canonical CatBoost source for this (threshold, seed) if the caller holds one — the
    # same settings `select_params` gives CatBoost's own cell. Read-only: `stacked_ensemble`
    # only calls `predict_proba` on it.
    cb_src = cb_source
    if cb_src is None:
        cb_src = PM.make_estimator("CatBoost",
                                   source_params("CatBoost", arm, int(threshold), tuned=tuned),
                                   seed=seed)
        cb_src.fit(splits["Xm_cs"], splits["ym_trm"].astype(int))
    st7, cx7, lam7 = stacked_ensemble(cb_src, family, params, Xk, yk, splits["Xy_te_cs"], seed)
    return {"ensemble_catboost_source": (splits["yte"], st7,
                                         {"effective_k": eff_cs, "convex_scores": cx7,
                                          "convex_lambda": lam7})}


def pseudo_label_thresh_scores(splits, *, family, params, seed, base=None, **_) -> dict:
    """Three rounds of confidence-threshold self-training on the unlabelled target pool.

    LABEL-FREE. The adaptation pool is `Xy_tr_cs`, the YRBS training covariates, and no YRBS
    training outcome is read at any point — not for drawing the pool, not for fitting, not for
    eligibility, and not for stopping. `splits["yte"]` is returned as the held-out evaluation
    outcome and reaches nothing that is fitted here. The hyperparameters are the
    source-selected ones; no target-side selection enters.

    Each round scores the whole pool with the current model, keeps the records it is confident
    about — probability >= 0.8 read as positive, <= 0.2 as negative — and refits a fresh
    estimator on the MCS training records plus those pseudo-labelled target records. Three
    rounds. A round that selects nothing stops the loop and keeps the last model fitted, which
    is the source model when that happens in round one.

    Distinct from `pseudo_label`, which is a single round and selects by decile rather than by
    a fixed confidence threshold. Neither carries `k` or `effective_k`: there is no label
    budget to report, because no target label is consumed.
    """
    import models as PM
    base = base if base is not None else source_model(family, params, seed, splits)
    Xm_cs, ym = splits["Xm_cs"], splits["ym_trm"]
    pool = splits["Xy_tr_cs"]
    p_prev = base.predict_proba(pool)[:, 1]
    final8 = base
    for _ in range(3):
        pos = p_prev >= 0.8
        keep = pos | (p_prev <= 0.2)
        if not keep.any():
            break
        yc = np.r_[ym.values, np.where(pos, 1.0, 0.0)[keep]].astype(int)
        final8 = PM.make_estimator(family, params, seed=seed)
        final8.fit(pd.concat([Xm_cs, pool[keep]], ignore_index=True), yc)
        p_prev = final8.predict_proba(pool)[:, 1]
    return {"pseudo_label_thresh": (splits["yte"],
                                    final8.predict_proba(splits["Xy_te_cs"])[:, 1],
                                    {"note": "3-round conf-thresh self-training (cohort-std), "
                                             "label-free"})}


# The range a persisted probability must lie in. A vector outside it is not a probability and
# cannot be joined to an evaluation that treats it as one.
SCORE_RANGE = (0.0, 1.0)


def score_vector_usable(y, p) -> tuple:
    """Whether a prediction vector can be persisted as person-level scores. `(usable, reason)`.

    THIS IS ABOUT THE VECTOR, NOT ABOUT ANY METRIC. A single-class evaluation slice leaves
    ROC-AUC undefined while Brier, the calibration diagnostics, the fixed-capacity counts and
    every later subgroup calculation remain computable from the same scores — so an undefined
    metric is not a reason to discard the predictions. What makes a vector unusable is the
    vector: a length that cannot be indexed, no evaluable row at all, a non-finite score on a
    row that will be evaluated, or a value outside [0, 1].

    The all-NaN vector `crossfit_logistic_recal_scores` emits when no mapping can be fitted is
    unusable by this rule, which is how a non-estimable cell comes to write nothing.
    """
    yarr = np.asarray(y, float)
    parr = np.asarray(p, float)
    if parr.shape != yarr.shape:
        return False, f"score vector is {parr.shape}, outcome vector is {yarr.shape}"
    evaluable = ~np.isnan(yarr)
    if not evaluable.any():
        return False, "no evaluable row"
    pe = parr[evaluable]
    n_bad = int((~np.isfinite(pe)).sum())
    if n_bad:
        return False, f"{n_bad} of {pe.size} evaluable rows carry no finite score"
    lo, hi = SCORE_RANGE
    if pe.min() < lo or pe.max() > hi:
        return False, f"scores span [{pe.min():.3g}, {pe.max():.3g}], outside [{lo}, {hi}]"
    return True, ""


def metric_rows(sc: dict, S, *, family: Family, arm, threshold: int, seed: int, source: str,
                k: int = K, keep_scores: bool = True) -> tuple:
    """Turn one procedure's `{regime: (y, p, extra)}` into metric rows and YRBS score frames.

    `keep_scores=False` computes the metrics and returns no score frames. The length check
    against the YRBS test index still runs, because a procedure whose scores cannot be indexed
    by it is a defect whether or not this run keeps them. Notebook 02 keeps the score frames
    for the one regime that has a downstream reader and discards the rest, so the default is
    the permissive one and the caller says when it does not want them.

    The row is wide and every field on it is load-bearing somewhere downstream — the
    prior-correction diagnostics, the two monotonicity flags kept apart, the solver status — so
    it is built here rather than at each of the eighteen call sites.

    Returns `(rows, score_frames, skipped)`. A score frame carries eight columns —
    `family, arm, threshold, regime, seed, row_id, score, y_true`. Notebook 02 drops `arm`
    before writing, because it is constant across every score-bearing regime, so the persisted
    handoff is the other seven.

    THE PERSISTED SCORES ARE YRBS ONLY. `mcs_internal` is excluded and must stay excluded: it
    is evaluated inside MCS, so its scores are MCS row-level material and persisting them
    would breach the Tier 1a licence. Every other regime scores the YRBS test set, which is
    open CDC data, and all of them are kept — otherwise notebook 03 can only read its numbers
    back from a summary rather than compute them from what the models produced.
    """
    import evaluation as E
    import models as PM

    rows, score_frames, skipped_scores = [], [], []
    for regime, (y, p, extra) in sc.items():
        # AN INFEASIBLE CELL IS RECORDED, NOT RAISED. `bbse_correct` returns an all-NaN score
        # vector for the four solver statuses where no estimate exists — deliberately, so a
        # failure is not filed as "no shift detected". Handing that to `E.metrics` raised
        # `ValueError: Input contains NaN` from inside sklearn, several frames below anything
        # that names the procedure, and took the whole run with it. The declared behaviour is
        # to report infeasibility; this is where it gets reported.
        #
        # The test is "no finite score on any evaluable row", which is the shape that path
        # produces. A PARTIALLY non-finite vector is not covered and still raises: nothing here
        # is specified to produce one, so it would be a surprise rather than a declared outcome.
        _y = np.asarray(y, dtype=np.float64)
        _p = np.asarray(p, dtype=np.float64)
        _eval = ~np.isnan(_y)
        # THE OUTCOME IS A BINARY LABEL, and the score handoff stores it as one. `astype(int8)`
        # further down would turn a 0.9 into a 0 without complaining, so the vector is checked
        # here — before any metric is computed, so the failure names the outcome rather than
        # arriving from inside sklearn as an unsupported format.
        if _eval.any() and not np.isin(_y[_eval], (0.0, 1.0)).all():
            raise ValueError(
                f"{regime} at {(family, threshold, seed)}: the evaluable outcome takes a value "
                f"other than 0 or 1, so it is not the binary label this pipeline evaluates")
        if _eval.any() and not np.isfinite(_p[_eval]).any():
            # The row keeps the same key set as a real one, and `metrics` decides that key set
            # rather than a list retyped here that would drift the moment it gains a column.
            # A distinct-valued placeholder makes every branch inside `metrics` well defined;
            # each measured value is then blanked, and only the description of the evaluable
            # slice survives.
            _keep = {"n_test", "prevalence", "n_pos", "prevalence_null"}
            m = E.metrics(_y, np.linspace(0.01, 0.99, _p.size),
                          prevalence=float(np.nanmean(_y)))
            m = {k: (v if k in _keep else np.nan) for k, v in m.items()}
            extra = {**extra, "degenerate": True,
                     "flag": extra.get("flag") or "no finite score: metrics not computed"}
        else:
            m = E.metrics(_y, _p, prevalence=float(np.nanmean(_y)))
        rows.append(dict(
            family=family, label=PM.FAM[family][0], arm=arm,
            # PER REGIME, NOT PER CALL. `reference_scores` returns three regimes at once and
            # only one of them is configured from the target side, so the label has to be able
            # to differ within a single call. Everything else takes the caller's `source`.
            hyperparameter_source=extra.get("hyperparameter_source", source),
            regime=regime, threshold=f">={threshold}",
            # `k` AND `effective_k` ARE NUMERIC WITH NULLS. They carry a number for the
            # label-using regimes and nothing for the rest. An empty string makes the
            # column `object`, which parquet cannot write — every sharded battery run
            # failed at its first checkpoint that way. So does None when NO label-using
            # regime is present to coerce the column, which is reachable now that each
            # procedure runs on its own. NaN is float64 at birth either way, and matches
            # `regime_battery.csv` (9,040 nulls).
            # `lineage` is the procedure's, not the module's. Everything published here is
            # cohort-standardised and takes the LINEAGE default; `source_scaled` is fitted and
            # scored on the raw feature scale and says so on its own rows, so a frame holding
            # both can be told apart on the column rather than on the regime name.
            lineage=extra.get("lineage", LINEAGE),
            k=(k if regime in LABEL_USING else np.nan), seed=seed,
            is_mcs=(regime == "mcs_internal"),
            ft_mechanism=extra.get("ft_mechanism", ""),
            effective_k=extra.get("effective_k", np.nan),
            family_independent=extra.get("family_independent", False),
            # Theory versus observation, kept apart. The first says the method
            # is non-decreasing by construction; it never said a run's scores
            # were. The second is measured from the scores this cell produced.
            monotone_by_construction=(regime in MONOTONE),
            strict_rank_preservation_expected=extra.get(
                "strict_rank_preservation_expected", regime in MONOTONE),
            solver_status=extra.get("solver_status", ""),
            valid_correction=extra.get("valid_correction", True),
            weights_strictly_positive=extra.get("weights_strictly_positive", True),
            target_prevalence_unconstrained=extra.get(
                "target_prevalence_unconstrained", float("nan")),
            target_prevalence_constrained=extra.get(
                "target_prevalence_constrained", float("nan")),
            target_predicted_positive_rate=extra.get(
                "target_predicted_positive_rate", float("nan")),
            source_prevalence=extra.get("source_prevalence", float("nan")),
            tpr=extra.get("tpr", float("nan")),
            fpr=extra.get("fpr", float("nan")),
            w0=extra.get("w0", float("nan")),
            w1=extra.get("w1", float("nan")),
            recal_folds=extra.get("recal_folds", np.nan),
            recal_status=extra.get("recal_status", ""),
            # The target-side search's own diagnostics, kept apart from the recalibration and
            # BBSE fields above. Overloading `recal_status` would make two different procedures'
            # failures indistinguishable in one column.
            target_selection_status=extra.get("target_selection_status", ""),
            target_selection_folds=extra.get("target_selection_folds", np.nan),
            target_selection_candidates=extra.get("target_selection_candidates", np.nan),
            target_selection_cv_auc=extra.get("target_selection_cv_auc", float("nan")),
            recal_intercept=extra.get("recal_intercept", float("nan")),
            recal_clipped_fraction=extra.get("recal_clipped_fraction", float("nan")),
            recal_score_range=extra.get("recal_score_range", float("nan")),
            recal_slope=extra.get("recal_slope", float("nan")),
            n_distinct_before=extra.get("n_distinct_before", -1),
            n_distinct_after=extra.get("n_distinct_after", -1),
            new_tie_fraction=extra.get("new_tie_fraction", float("nan")),
            degenerate=bool(extra.get("degenerate", False)),
            note=extra.get("flag", ""), **m))
        # Keep YRBS scores for regimes evaluated on the target test set.
        #
        # `mcs_internal` is excluded and MUST STAY EXCLUDED — it is the
        # source reference, evaluated inside MCS, so its scores are MCS
        # row-level material and persisting them would breach the Tier 1a
        # licence. Everything else scores YRBS, which is open CDC data.
        #
        # Naive alone used to be kept, which left eighteen of the nineteen
        # regimes with no per-person output — so notebook 03 had to read their
        # metrics back from a summary CSV instead of computing them from what
        # the models produced. Keeping them all is what makes the notebook the
        # thing that runs the analysis.
        #
        # A regime whose row set does not match the YRBS test index cannot be
        # indexed by it. Those are skipped and reported, never mis-joined.
        # WHETHER SCORES ARE PERSISTED IS DECIDED ON THE VECTOR, not on whether a metric came
        # out degenerate. A length that cannot be indexed by the YRBS test index is a defect and
        # is reported to the caller; an unusable vector — no finite score where one is needed,
        # or a value outside [0, 1] — writes no frame, and the metric row above is the record
        # that the cell was attempted. An undefined metric on a usable vector changes nothing:
        # the predictions are still persisted.
        if regime != "mcs_internal":
            yarr = np.asarray(y, float)
            if len(yarr) != len(S["y_idx"]):
                skipped_scores.append((family, arm, threshold, regime, len(yarr)))
            elif keep_scores and score_vector_usable(yarr, np.asarray(p, float))[0]:
                ymask = ~np.isnan(yarr)
                yv = yarr[ymask]
                pv = np.asarray(p, float)[ymask]
                # THE OUTCOME TRAVELS WITH THE PREDICTION. Notebook 03 checks each cell's
                # y_true against the cohort's own value before it resamples anything, so a
                # score frame that carried predictions alone could not be validated. `int8` is
                # the storage type for a binary label and is safe because the vector was
                # checked against {0, 1} above. `score` stays float64 — the vector the metrics
                # were computed from, restricted to the evaluable rows and not rounded on the
                # way out.
                score_frames.append(pd.DataFrame(dict(
                    family=family, arm=arm, threshold=f">={threshold}", regime=regime,
                    seed=seed, row_id=S["y_idx"][ymask],
                    score=pv.astype(np.float64, copy=False),
                    y_true=yv.astype(np.int8))))
    return rows, score_frames, skipped_scores


# The method: Lipton et al. 2018, black-box shift estimation; Saerens et al. 2002, prior
# correction.

# Numerical tolerance for calling an estimate "at the boundary" rather than outside it. It is a
# floating-point allowance, nothing more — it carries no statistical claim about whether the
# estimate differs from 0 or 1, and widening it to reclassify cells would be exactly that claim
# made silently.
PREVALENCE_TOL = 1e-9
# Below this the source classifier cannot separate the classes well enough to invert, because
# q_hat = (m - FPR) / (TPR - FPR) divides by it. Declared here rather than tuned per run.
MIN_TPR_MINUS_FPR = 1e-6


def estimate_target_prevalence(source_validation_probabilities,
                               source_validation_labels,
                               unlabelled_target_pool_probabilities,
                               *, threshold: float = 0.5) -> dict:
    """Estimate the TARGET outcome prevalence without seeing a single target label.

    Hard-label BBSE. Everything it reads is either source-side and labelled, or target-side and
    UNLABELLED:

        TPR, FPR  from source validation predictions against source validation outcomes
        m         from target-pool predictions alone — no target outcome is touched
        pi        source prevalence

        q_raw = (m - FPR) / (TPR - FPR)

    which is the Rogan-Gladen estimator, and is algebraically the same estimator as solving
    the joint confusion system C w = mu: verified to 7e-15 over 2,000 random interior cases.
    It is written in this form because the failure modes are legible here — the denominator is
    the identifiability condition and the numerator is what puts the estimate out of range —
    whereas the matrix form hides both inside a solve.

    THE SIGNATURE IS THE GUARANTEE. There is no parameter through which a target outcome could
    enter, so no feasibility decision, status or fallback below can depend on one. Target labels
    belong to the evaluation layer, which runs after this and never feeds back into it.

    Returns the estimate, the quantities behind it, and a status. It does not correct anything.
    """
    p_val = np.asarray(source_validation_probabilities, float)
    y_val = np.asarray(source_validation_labels, float)
    p_pool = np.asarray(unlabelled_target_pool_probabilities, float)

    out = {"tpr": float("nan"), "fpr": float("nan"),
           "target_predicted_positive_rate": float("nan"),
           "source_prevalence": float("nan"),
           "target_prevalence_unconstrained": float("nan"),
           "target_prevalence_constrained": float("nan"),
           "prevalence_tol": PREVALENCE_TOL,
           "min_tpr_minus_fpr": MIN_TPR_MINUS_FPR,
           "solver_status": "numerical_failure", "valid_correction": False}

    labelled = ~np.isnan(y_val) & np.isfinite(p_val)
    yv = y_val[labelled].astype(int)
    yhat = (p_val[labelled] >= threshold).astype(int)
    n_pos, n_neg = int((yv == 1).sum()), int((yv == 0).sum())
    if n_pos == 0 or n_neg == 0:
        out["solver_status"] = "insufficient_source_classes"
        return out

    finite_pool = p_pool[np.isfinite(p_pool)]
    if finite_pool.size == 0:
        out["solver_status"] = "insufficient_target_predictions"
        return out

    tpr = float((yhat[yv == 1] == 1).mean())
    fpr = float((yhat[yv == 0] == 1).mean())
    m = float((finite_pool >= threshold).mean())
    pi = float((yv == 1).mean())
    out.update(tpr=tpr, fpr=fpr, target_predicted_positive_rate=m, source_prevalence=pi)

    if not (tpr - fpr) > MIN_TPR_MINUS_FPR:
        # TPR == FPR is Lipton's identifiability condition failing: the confusion matrix is
        # singular and the target prevalence is not recoverable from predictions at all.
        out["solver_status"] = "not_identifiable"
        return out
    if pi <= 0.0 or pi >= 1.0:
        out["solver_status"] = "insufficient_source_classes"
        return out

    q_raw = (m - fpr) / (tpr - fpr)
    q_con = min(1.0, max(0.0, q_raw))
    out["target_prevalence_unconstrained"] = q_raw
    out["target_prevalence_constrained"] = q_con

    if q_raw < -PREVALENCE_TOL:
        # m below the source false-positive rate. No target prevalence in [0, 1] produces that
        # under label shift, so the hard-BBSE model does not fit these predictions.
        out["solver_status"] = "incompatible_low"
    elif q_raw > 1.0 + PREVALENCE_TOL:
        out["solver_status"] = "incompatible_high"
    elif q_raw <= PREVALENCE_TOL:
        out["solver_status"] = "boundary_low"
    elif q_raw >= 1.0 - PREVALENCE_TOL:
        out["solver_status"] = "boundary_high"
    else:
        out["solver_status"] = "interior"
        out["valid_correction"] = True
    return out


def bbse_correct(source_validation_probabilities, source_validation_labels,
                 unlabelled_target_pool_probabilities, probabilities_to_correct,
                 *, return_diagnostics: bool = False):
    """Saerens prior correction, applied at a target prevalence estimated without target labels.

        w1 = q / pi                w0 = (1 - q) / (1 - pi)
        p' = p.w1 / ((1-p).w0 + p.w1)          dp'/dp = w0.w1 / ((1-p).w0 + p.w1)^2

    so the map is STRICTLY INCREASING exactly while both weights are strictly positive, which
    is to say while 0 < q < 1. That is the only condition under which the method preserves a
    ranking, and it is a property of the estimate rather than of the method.

    AT THE BOUNDARY THE MAP IS CONSTANT, and this returns the constant explicitly: all zeros at
    q = 0, all ones at q = 1. The previous implementation reached the same place by clipping a
    negative weight and then dividing by `denominator + 1e-12`, which at q = 1 left a spread of
    about 4e-07 — enough for AUC to look preserved while every score was pinned at one. A
    constant is the honest output; a constant with floating-point dust on it is not.

    WHAT REPLACED THE CLIP. `np.clip(np.linalg.solve(C, mu), 0, None)` was not a constrained
    solve — it disagrees with NNLS and gives a worse residual — and it discarded the fact that
    it had truncated anything. Clipping now happens on the prevalence, where it is a boundary
    of [0, 1] with a name, and the unconstrained value is reported beside it.

    `return_diagnostics=True` returns (corrected, info). Statuses other than `interior` mean the
    output is the operational result of the declared procedure, NOT a valid correction, and
    `info["valid_correction"]` is False for every one of them.
    """
    p_test = np.asarray(probabilities_to_correct, float)
    info = estimate_target_prevalence(source_validation_probabilities,
                                      source_validation_labels,
                                      unlabelled_target_pool_probabilities)

    n_before = int(np.unique(p_test[np.isfinite(p_test)]).size)
    status = info["solver_status"]

    if status in ("not_identifiable", "insufficient_source_classes",
                  "insufficient_target_predictions", "numerical_failure"):
        # No estimate exists. Returning the uncorrected scores would file a failure as "no
        # shift detected", so the scores are NaN and the status says why.
        corrected = np.full_like(p_test, np.nan, dtype=float)
        w0 = w1 = float("nan")
    else:
        q = info["target_prevalence_constrained"]
        pi = info["source_prevalence"]
        w1, w0 = q / pi, (1.0 - q) / (1.0 - pi)
        if q <= 0.0:
            corrected = np.zeros_like(p_test, dtype=float)
        elif q >= 1.0:
            corrected = np.ones_like(p_test, dtype=float)
        else:
            num1 = p_test * w1
            num0 = (1.0 - p_test) * w0
            corrected = num1 / (num0 + num1)          # no epsilon: both weights are positive

    n_after = int(np.unique(corrected[np.isfinite(corrected)]).size)
    strictly_positive = bool(np.isfinite(w0) and np.isfinite(w1) and w0 > 0.0 and w1 > 0.0)
    info.update({
        "w0": float(w0), "w1": float(w1),
        "weights_strictly_positive": strictly_positive,
        "degenerate": bool(status != "interior"),
        "n_distinct_before": n_before, "n_distinct_after": n_after,
        "new_tie_fraction": float(1.0 - n_after / n_before) if n_before else float("nan"),
        "strict_rank_preservation_expected": strictly_positive,
        # sum_y p_source(y) . w_y must equal 1. Here it does by construction — (1-pi).w0 + pi.w1
        # = (1-q) + q — because q is used directly rather than reconstructed from weights that
        # had been truncated. Under the old clip it reached 1.0625 without anything noticing.
        "implied_prior_sum": float((1.0 - info["source_prevalence"]) * w0
                                   + info["source_prevalence"] * w1)
        if np.isfinite(w0) and np.isfinite(w1) else float("nan"),
        "flag": "" if status == "interior" else f"{status}: not a valid correction",
    })
    return (corrected, info) if return_diagnostics else corrected


# Label-using adaptation
def warm_finetune(family: Family, params, base, Xk, yk, seed):
    """The warm-start half of full revision, for the three families that support it.

    XGB       +50 trees at lr=0.02, depth 4, continuing from base.get_booster()
    CatBoost  +100 iterations at lr=0.02, continuing from init_model=base
    LightGBM  +100 estimators at lr=0.02, continuing from base.booster_

    Returns a callable score(X) -> p1, matching S4's _p1_holder. Any other family raises,
    exactly as in the source — the weighted-pooled-refit branch for the linear and bagged
    families lives in `fine_tune_scores` and is not part of warm_finetune.
    """
    import models as PM
    if family == "XGB":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=50, learning_rate=0.02, max_depth=4, random_state=RANDOM_STATE)
        m.fit(Xk, yk, xgb_model=base.get_booster())
        return lambda X: m.predict_proba(X)[:, 1]
    if family == "CatBoost":
        m = PM.make_estimator("CatBoost", dict(params, iterations=100, learning_rate=0.02), seed=seed)
        m.fit(Xk, yk.astype(int), init_model=base)
        return lambda X: m.predict_proba(X)[:, 1]
    if family == "LightGBM":
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(n_estimators=100, learning_rate=0.02, random_state=RANDOM_STATE,
                           n_jobs=1, deterministic=True, verbose=-1)
        m.fit(Xk, yk, init_model=base.booster_)
        return lambda X: m.predict_proba(X)[:, 1]
    raise ValueError(family)


_REFRESH_FLAGS = {"process_type": "update", "updater": "refresh", "refresh_leaf": True}
# XGBClassifier kwarg -> low-level booster param, so refreshed leaves reuse the base's
# shrinkage and L2 rather than silently reverting to xgboost's default eta=0.3.
_CLS_TO_BOOSTER = {"learning_rate": "eta", "reg_lambda": "lambda", "reg_alpha": "alpha"}
# The frozen raw recipe (run_leaf_refresh.py base_params), kept so the raw cell reproduces.
LEAF_REFRESH_RAW_ROUNDS = 300


def _booster_params(cls_params) -> dict:
    """Translate an XGBClassifier config into booster params for the refresh pass.

    `n_estimators` is `num_boost_round`, not a booster param, so it is dropped;
    learning_rate / reg_lambda / reg_alpha are renamed. Skipping this rename is the specific
    way to get a plausible-looking wrong answer: the refresh silently runs at eta=0.3.
    """
    out = {"objective": "binary:logistic"}
    for kk, vv in (cls_params or {}).items():
        if kk == "n_estimators":
            continue
        out[_CLS_TO_BOOSTER.get(kk, kk)] = vv
    return out


# UNVERIFIED: needs MCS data to check.
def leaf_refresh(base, splits, seed: int, *, params=None, family: Family = "XGB",
                 lineage: Literal["cs", "raw"] = "cs", k: int = K) -> tuple:
    """Keep the MCS tree structure, relearn leaf values on k YRBS labels.

    THE DELICATE INVARIANT, and the reason this function asserts rather than warns:
    `process_type='update'` is only meaningful if the refreshed model keeps EXACTLY the base's
    tree count. `num_boost_round` is set to the base tree count and
    `len(booster.get_dump())` is compared before and after. Tuned XGB has 500 trees at >=1 and
    300 at >=2; untuned has 300 — the assert tracks whichever it is rather than hard-coding a
    number. If the count changes, the model is no longer "the same trees with new leaves" and
    the comparison against `fine_tune` (which deliberately ADDS trees) stops meaning anything.

    Returns `(scores, n_trees_before, n_trees_after, effective_k)`; the tree counts are meant
    to be recorded on the row, not discarded after the assert.

    TWO LINEAGES, and both are wanted. `cs` is the primary — S4's naive XGB base per seed, so
    leaf-refresh vs fine_tune is a clean "refresh leaves" vs "warm +50 trees" comparison on
    the same base. `raw` reproduces the frozen `leaf_refresh.csv` numbers so the published raw
    values are not orphaned; the source treats a raw miss above 0.005 as a report-first
    finding rather than absorbing it.

    NOT AN S4 REGIME. `leaf_refresh` is in S4's `NOT_RUN` list — it is tree-internal with no
    cross-family analogue. What the battery carries instead is `leaf_refresh_global` (R2), the
    1-DOF intercept-offset analogue that every tree family can do. They are different methods
    and the grid keeps them apart.

    The bundle supplies both the pool that defines `k_slice` and the frame to score on. `base`
    is the fitted `XGBClassifier` rather than a raw `Booster`, for the `cs` path.
    """
    import xgboost as xgb
    feat_cols = splits.feat_cols
    if family != "XGB":
        raise ValueError(
            f"leaf_refresh is XGBoost-only (process_type='update' is an xgboost mechanism); "
            f"got {family!r}. The cross-family analogue in the battery is "
            f"`leaf_refresh_global` (R2), computed by `leaf_refresh_global_scores`.")
    k_sample = splits.k_slice(k, lineage=lineage)
    Xk, yk = k_sample[feat_cols], k_sample["y"]
    if yk.nunique() < 2:
        return None, None, None, len(Xk)
    if lineage == "cs":
        booster = base.get_booster()
        refresh_params = _booster_params(params)
        X_te = splits["Xy_te_cs"]
    else:
        raw_params = {"objective": "binary:logistic", "eval_metric": "auc", "max_depth": 4,
                      "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "seed": seed}
        dtrain = xgb.DMatrix(splits["Xm_trm"], label=splits["ym_trm"].values)
        booster = xgb.train(raw_params, dtrain, num_boost_round=LEAF_REFRESH_RAW_ROUNDS)
        refresh_params = raw_params
        X_te = splits["Xy_te"]
    n_before = len(booster.get_dump())
    dk = xgb.DMatrix(Xk, label=yk.values)
    refreshed = xgb.train(dict(refresh_params, **_REFRESH_FLAGS), dk,
                          num_boost_round=n_before, xgb_model=booster)
    n_after = len(refreshed.get_dump())
    if n_after != n_before:
        raise RuntimeError(
            f"leaf-refresh changed tree count {n_before}->{n_after} (seed {seed}) — "
            "num_boost_round must equal the base tree count; refusing to proceed.")
    return refreshed.predict(xgb.DMatrix(X_te)), n_before, n_after, len(Xk)


# The rule extraction itself lives in `src/rules.py`; these are the head's own settings.
RULE_BUDGET = 300
RULE_C_GRID: Sequence[float] = (0.01, 0.1, 1.0)
RULE_CV_FOLDS = 3


# UNVERIFIED: needs MCS data to check.
def rule_head(base, splits, seed: int, *, n_rules: int = RULE_BUDGET,
              lineage: Literal["cs", "raw"] = "cs", k: int = K) -> dict:
    """Extract root-to-leaf rules from the MCS booster, binarise, fit an L1 head on k labels.

    THE QUESTION IT ANSWERS: does MCS-learned INTERACTION STRUCTURE transfer at all, or are
    the interactions jurisdiction-specific? The rules carry the structure; the L1 head relearns
    only how much each one is worth. If the structure transfers, a few hundred YRBS labels
    should be enough to reweight it.

    Top-`n_rules` root-to-leaf paths by |leaf value| x cover, deduplicated across trees by
    condition set (`src/rules.extract_rules_from_xgb`). C is chosen from {0.01, 0.1, 1.0}
    by stratified 3-fold CV INSIDE the k slice — never on the test set.

    ONE DESIGN CHOICE THAT IS A KNOWN DEVIATION, flagged in `src/rules.py:13-15` and worth
    carrying into the paper: a rule does NOT fire on a NaN input. XGBoost's own
    default-direction routing is not replicated. The source calls the cost small in practice
    because rules typically span several features, but it is a deviation from what the booster
    itself would do, not an implementation detail.

    Returns a dict with the scores AND the bookkeeping the row needs — `n_rules` actually
    extracted (which can be below the budget), the chosen C, how many coefficients survived
    L1, and whether the head converged. The convergence flag matters: an unconverged L1 head
    on 300 correlated binary features is a real possibility and the source records it rather
    than assuming.

    The k slice and the evaluation frame both come from the bundle, as in `leaf_refresh`.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from rules import apply_rules, extract_rules_from_xgb

    feat_cols = splits.feat_cols
    X_te = splits["Xy_te_cs"] if lineage == "cs" else splits["Xy_te"]
    rules = extract_rules_from_xgb(base, top_k=n_rules)
    R_te = apply_rules(rules, X_te)
    k_sample = splits.k_slice(k, lineage=lineage)
    Xk, yk = k_sample[feat_cols], k_sample["y"]
    if yk.nunique() < 2:
        return None
    R_k = apply_rules(rules, Xk)

    cv = StratifiedKFold(RULE_CV_FOLDS, shuffle=True, random_state=seed)
    best_C, best = RULE_C_GRID[0], -np.inf
    for Cv in RULE_C_GRID:
        clf = LogisticRegression(penalty="l1", solver="liblinear", C=Cv, max_iter=5000,
                                 random_state=seed)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = cross_val_score(clf, R_k.values, yk.values, cv=cv, scoring="roc_auc").mean()
        except ValueError:
            s = -np.inf
        if s > best:
            best, best_C = s, Cv
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always")
        head = LogisticRegression(penalty="l1", solver="liblinear", C=best_C, max_iter=5000,
                                  random_state=seed)
        head.fit(R_k.values, yk.values)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in wlog)
    return dict(scores=head.predict_proba(R_te.values)[:, 1], n_rules=len(rules),
                chosen_C=best_C, n_selected=int((head.coef_ != 0).sum()),
                converged=converged, effective_k=len(Xk))


# UNVERIFIED: needs MCS data to check.
def extract_leaves(cb, X_train_cs, budget: int = RULE_BUDGET) -> tuple:
    """Retain leaves that FIRE in MCS-train; rank by |leaf value| x cover; cap to `budget`.

    Leaves with zero cover on the training data are dropped before ranking. That is not
    tidying: a leaf no training row reaches has an arbitrary value, and including it would
    give the L1 head a column that is all-zero on MCS and possibly non-zero on YRBS.
    """
    li = cb.calc_leaf_indexes(X_train_cs)
    n, n_trees = li.shape
    leaf_counts = np.asarray(cb.get_tree_leaf_counts(), dtype=int)
    leaf_values = np.asarray(cb.get_leaf_values(), dtype=float)
    offsets = np.concatenate([[0], np.cumsum(leaf_counts)]).astype(int)
    retained = []
    for tt in range(n_trees):
        vals, counts = np.unique(li[:, tt], return_counts=True)
        cover_map = dict(zip(vals.tolist(), counts.tolist()))
        for leaf in range(int(leaf_counts[tt])):
            cov = cover_map.get(leaf, 0)
            if cov == 0:
                continue
            w = abs(leaf_values[offsets[tt] + leaf]) * (cov / n)
            retained.append(((tt, leaf), w))
    n_retained = len(retained)
    retained.sort(key=lambda x: -x[1])
    return [tl for tl, _ in retained[:budget]], n_retained


def leaf_matrix(cb, X_cs, capped) -> pd.DataFrame:
    """Binary leaf-membership indicators for the capped leaf set, aligned to `X_cs`."""
    li = cb.calc_leaf_indexes(X_cs)
    cols = {f"L{i:04d}": (li[:, tt] == leaf).astype(np.int8)
            for i, (tt, leaf) in enumerate(capped)}
    return pd.DataFrame(cols, index=X_cs.index)


# UNVERIFIED: needs MCS data to check.
def leaf_membership_head(cb, splits, seed: int, *, budget: int = RULE_BUDGET,
                         k: int = K) -> dict:
    """CatBoost leaf-membership indicators -> L1 head. The oblivious-tree analogue of `rule_head`.

    CatBoost's trees are OBLIVIOUS — every level splits on the same feature — so a
    root-to-leaf path is not the interpretable conjunction it is in XGBoost, and `rule_head`'s
    extraction does not carry over. Leaf membership is the analogue that does: one indicator
    per retained leaf, which is the same "keep the structure, relearn the weights" question
    asked in the representation CatBoost actually has.

    C is chosen on a within-slice 80/20 split, NOT by 3-fold CV — a deliberate difference from
    `rule_head` and the frozen runner's choice, kept as-is.

    The disjointness of the k sample and the test set is asserted here, as in the source
    (backfill_f1_leaf_head.py:117). It costs nothing and catches a class of error that would
    otherwise inflate every label-using regime at once.

    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    feat_cols = splits.feat_cols
    capped, n_retained = extract_leaves(cb, splits["Xm_cs"], budget)
    R_te = leaf_matrix(cb, splits["Xy_te_cs"], capped)
    k_sample = splits.k_slice(k)
    Xk_cs, yk = k_sample[feat_cols], k_sample["y"]
    if yk.nunique() < 2:
        return None
    if set(k_sample.index) & set(splits["Xy_te"].index):
        raise RuntimeError(f"k-sample intersects test set at seed {seed}")
    R_k = leaf_matrix(cb, Xk_cs, capped)

    try:
        Rtr, Rval, ytr, yval = train_test_split(R_k, yk, test_size=0.2, stratify=yk,
                                                random_state=seed)
    except ValueError:
        Rtr, Rval, ytr, yval = R_k, R_k, yk, yk
    best_C, best = RULE_C_GRID[0], -np.inf
    for Cv in RULE_C_GRID:
        clf = LogisticRegression(penalty="l1", solver="liblinear", C=Cv, max_iter=5000,
                                 random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Rtr.values, ytr.values)
        try:
            a = roc_auc_score(yval.values, clf.predict_proba(Rval.values)[:, 1])
        except ValueError:
            a = -np.inf
        if a > best:
            best, best_C = a, Cv
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always")
        head = LogisticRegression(penalty="l1", solver="liblinear", C=best_C, max_iter=5000,
                                  random_state=seed)
        head.fit(R_k.values, yk.values)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in wlog)
    return dict(scores=head.predict_proba(R_te.values)[:, 1], n_retained=n_retained,
                chosen_C=best_C, n_selected=int((head.coef_ != 0).sum()),
                converged=converged, effective_k=len(Xk_cs))


# NOT the same metric lists as the main battery's. The backfill mechanisms report a narrower
# set, and reusing the battery's names here would imply a comparison that is not being made.
BACKFILL_SUMMARY_METRICS: Sequence[str] = (
    "auc", "prauc", "prauc_lift", "brier", "ece", "cal_slope", "cal_intercept",
    "dec_precision", "dec_recall", "dec_f1", "dec_mcc",
    "qui_precision", "qui_recall", "qui_f1", "qui_mcc",
)
BACKFILL_PCT_METRICS: Sequence[str] = ("auc", "prauc_lift", "brier", "dec_precision", "dec_recall")
BACKFILL_GROUP: Sequence[str] = ("family", "method", "arm", "hyperparameter_source",
                                 "threshold", "lineage", "k")


# VERIFIED: BACKFILL_GROUP and BACKFILL_SUMMARY_METRICS were diffed against the header of
#           $THESIS_WORK_DIR/tables/backfill_summary.csv — all 7 group columns present, and every one
#           of the 15 metrics appears as both `<m>_mean` and `<m>_sd`; the 5 PCT_METRICS
#           appear as `<m>_plo`/`<m>_phi`. Zero declared-but-absent, zero present-but-
#           undeclared. That column check is what the row shape rests on: the original row
#           builder is not available to read, so the header is the evidence, not the code.
# UNVERIFIED: needs MCS data to check the numbers — every method refits.
def run_backfill(methods: Sequence[str], *, splits: Mapping[tuple, object],
                 thresholds=THRESHOLDS, arms=(ARM,), k: int = K,
                 tuned=None, sources=None) -> pd.DataFrame:
    """Re-run the freeze-week mechanisms on the consolidated S4 protocol.

    The frozen runners are mixed-lineage, untuned, >=1-only and carry mean/sd rather than the
    full battery. The backfill puts each mechanism on the cohort-standardised pipeline, both
    arms, both thresholds, full per-seed metric battery — so they become cell-for-cell
    comparable with S4.

    WHAT MAKES THE COMPARISON PAIRED, and it is the point of the whole exercise: the backfill's
    base model IS S4's naive base for that (family, arm, seed), and its k=500 slice IS
    `pool_cs.sample(500, seed)` — the same slice `fine_tune` and `target_only` consume. So
    "leaf refresh vs warm +50 trees" is a comparison of two mechanisms on one base and one
    label budget, not a comparison of two pipelines.

    THREE OF THE METHODS ALSO RUN A `repro` CELL in the RAW lineage, and the miss tolerance is
    a REPORT-FIRST threshold rather than a gate: if the raw reproduction misses the frozen
    number by more than 0.005, the finding is that the published raw values cannot be quoted —
    which is reported, not silently absorbed and not silently fatal.

    Returns the per-seed long frame. `consolidate_backfill` below is the summary pass, and it
    uses a DIFFERENT metric list from the S4 battery — `BACKFILL_SUMMARY_METRICS` has 15
    entries where `evaluation.SUMMARY_METRICS` has 17 (the backfill drops
    `dec_specificity` and `qui_specificity`). That is the source's shape and changing it would
    break the diff against `backfill_summary.csv`.

    The splits are passed in, keyed `(seed, threshold)`, so all seven methods share one
    construction.

    `sources` is the canonical source-model dict. This function's base is
    `make_estimator(family, select_params(family, "tuned", t, tuned)).fit(Xm_cs, ym_trm)` — the
    same estimator, params, seed and training frames as `fit_source_models`, so passing the dict
    reuses the model instead of refitting an identical one. Omit it and the base is fitted here,
    which is what the function did before and produces the same numbers.

    THREE MECHANISMS, NOT SEVEN. `leaf_refresh`, `rule_head` and `leaf_membership` are the
    ones with a traceable transcription. Four more mechanisms from the same round of work are
    not reimplemented here, and naming them is not this function's job.
    """
    import evaluation as E
    import models as PM

    known = ("leaf_refresh", "rule_head", "leaf_membership")
    unknown = [m for m in methods if m not in known]
    if unknown:
        raise ValueError(f"unknown backfill method(s) {unknown}; known: {list(known)}")

    rows = []
    for method in methods:
        family = "CatBoost" if method == "leaf_membership" else "XGB"
        for t in thresholds:
            t = int(t)
            for seed in sorted({s for (s, tt) in splits if tt == t}):
                S = splits[(seed, t)]
                prev = float(np.nanmean(np.asarray(S["yte"], float)))
                for arm in arms:
                    params, hsrc = PM.select_params(family, arm, t, tuned)
                    if sources is not None and arm == ARM:
                        base = source_for(sources, family, t, seed)
                    else:
                        base = PM.make_estimator(family, params, seed=seed)
                        base.fit(S["Xm_cs"], S["ym_trm"].astype(int))
                    if method == "leaf_refresh":
                        p, nb, na, effk = leaf_refresh(base, S, seed, params=params,
                                                       lineage="cs", k=k)
                        if p is None:
                            continue
                        extra = dict(tree_count_before=nb, tree_count_after=na,
                                     effective_k=effk, repro=False)
                    elif method == "rule_head":
                        r = rule_head(base, S, seed, lineage="cs", k=k)
                        if r is None:
                            continue
                        p = r["scores"]
                        extra = dict(n_rules=r["n_rules"], chosen_C=r["chosen_C"],
                                     n_selected=r["n_selected"], converged=r["converged"],
                                     effective_k=r["effective_k"], repro=False)
                    else:  # leaf_membership
                        r = leaf_membership_head(base, S, seed, k=k)
                        if r is None:
                            continue
                        p = r["scores"]
                        extra = dict(n_retained=r["n_retained"], chosen_C=r["chosen_C"],
                                     n_selected=r["n_selected"], converged=r["converged"],
                                     effective_k=r["effective_k"])
                    m = E.metrics(S["yte"], p, prevalence=prev)
                    rows.append(dict(family=family,
                                     method=("rule_extraction" if method == "rule_head" else method),
                                     arm=arm, hyperparameter_source=hsrc,
                                     threshold=f">={t}", lineage="cs", k=k, seed=seed,
                                     **extra, **m))
    return pd.DataFrame(rows)


def consolidate_backfill(per_seed: pd.DataFrame) -> pd.DataFrame:
    """20-seed mean/sd + 2.5/97.5 percentiles, in `backfill_summary.csv`'s shape.

    Same aggregation shape as `regime_battery_summary.csv` but a DIFFERENT metric list and
    a different group key — the backfill groups on `method` where the battery groups on
    `regime`, and it drops the two specificity columns. Both facts are the source's.
    """
    grp = [c for c in BACKFILL_GROUP if c in per_seed.columns]
    agg = {}
    for m in BACKFILL_SUMMARY_METRICS:
        if m in per_seed.columns:
            agg[f"{m}_mean"] = (m, "mean")
            agg[f"{m}_sd"] = (m, "std")
    summary = per_seed.groupby(grp, dropna=False).agg(
        n_seeds=("seed", "nunique"), **agg).reset_index()
    for m in BACKFILL_PCT_METRICS:
        if m not in per_seed.columns:
            continue
        for q, lab in ((2.5, "lo"), (97.5, "hi")):
            summary = summary.merge(
                per_seed.groupby(grp, dropna=False)[m].quantile(q / 100)
                .round(4).rename(f"{m}_p{lab}").reset_index(), on=grp, how="left")
    return summary


# Sensitivity analyses
LOO_PILLARS: Mapping[str, str] = {
    "loo_sexual":    "ace_sexual_abuse",
    "loo_emotional": "ace_emotional_abuse",
    "loo_physical":  "ace_physical_abuse",
    "loo_mental":    "ace_household_mental_illness",
    "loo_substance": "ace_household_substance",
}
LOO_VARIANTS: Sequence[str] = ("full",) + tuple(LOO_PILLARS)


# The outcome constructor is `outcomes.compose_outcome_loo`; the split convention is stated in
# the docstring below and is the detail that is easiest to get backwards.
# The variant names and dropped-pillar strings are `LOO_PILLARS` and `LOO_VARIANTS` above;
# nothing here invents a variant, and a stored result keyed on a name this module does not
# declare is a mismatch to reconcile rather than to accommodate.
# UNVERIFIED: needs MCS data to check the AUCs.
# SIDE EFFECTS: FITS MODELS and PRINTS progress. One fit per (dropped pillar x
# family x seed) — expensive, and the cost scales with the number of pillars.
# Writes nothing; the caller persists the returned frame.
def leave_one_pillar_out(mcs_pillars: pd.DataFrame, yrbs_pillars: pd.DataFrame, *,
                         X_mcs: pd.DataFrame, X_yrbs: pd.DataFrame,
                         seeds: Sequence[int] = SEEDS,
                         families: Sequence[str] = ("L1_LR", "XGB", "CatBoost"),
                         tuned=None, variants: Sequence[str] = LOO_VARIANTS,
                         splits: Mapping[tuple, object] | None = None,
                         sources=None) -> pd.DataFrame:
    """B1: drop each ACE pillar in turn and re-measure transfer.

    Decomposes the transfer gap into pillar non-equivalence versus relational shift. If
    dropping ONE particular pillar collapses the gap, that pillar is carrying the
    non-equivalence; if the gap is stable across all five drops, the gap is relational and not
    a labelling artefact. Both outcomes are reportable; the second is the paper's finding.

    THE SPLIT CONVENTION IS THE LOAD-BEARING DETAIL, and it is easy to get backwards. Every
    variant is stratified on the FULL-5 OUTCOME, not on its own leave-one-out outcome
    (leave_one_pillar_out.py:28-38). All six variants therefore share the IDENTICAL MCS-test
    and YRBS-test partition for a given seed, so the only thing varying across variants is the
    label definition — which is precisely what the decomposition requires. Stratifying each
    variant on its own outcome would confound "the pillar matters" with "the split moved".

    `splits` is the shared partition, keyed `(seed, threshold)`. The one this needs is the
    `>=1` bundle built from the FULL-FIVE outcome, which is exactly `build_splits(seed,
    make_outcome(pillars, 1), ..., threshold=1)` — the same construction, the same seed and the
    same feature frames as the notebook's own `splits[(seed, 1)]`, so passing that in reuses the
    partition rather than rebuilding an identical one. Omit it and the splits are built here,
    which is what the function did before and gives the same bundles.

    THE SPLIT IS SHARED ACROSS VARIANTS AND IS NOT RE-STRATIFIED PER VARIANT. That is what makes
    the comparison a comparison of label definitions. It is the opposite convention from
    `outcome_variant_battery`, which stratifies each variant on its own outcome; the two answer
    related but different sensitivity questions and are not interchangeable.

    `variants` narrows which of the six are run, so the caller can run one per cell and
    concatenate afterwards. `sources` supplies the canonical source model, which is valid for
    the `full` variant alone: its fit is `make_estimator(fam, params, seed).fit(Xm_cs, ym_trm)`
    on the full-five outcome, identical to `fit_source_models`. Every other variant trains on a
    different label and must fit its own model.

    Reports BOTH sides per cell — `mcs_auc_internal` and `yrbs_auc_transfer` — because the gap
    between them is the quantity of interest, not either alone.

    A >=1-only analysis, so it takes no `thresholds`: the leave-one-out outcome at >=2 over
    four pillars is a different construct, not a threshold variant. It needs the pillar frames
    and both feature matrices.
    """
    import outcomes as BO
    from sklearn.metrics import roc_auc_score
    import data as D
    import models as PM

    def outcome(pillars, variant):
        if variant == "full":
            return D.make_outcome(pillars, 1)
        return BO.compose_outcome_loo(pillars[list(D.SHARED_PILLARS)], LOO_PILLARS[variant],
                                      threshold=1, strict=True)

    unknown = [v for v in variants if v not in LOO_VARIANTS]
    if unknown:
        raise ValueError(f"unknown leave-one-out variant(s) {unknown}; "
                         f"known: {list(LOO_VARIANTS)}")
    needed = sorted({*variants, "full"})          # "full" also supplies the shared stratifier
    mcs_y = {v: outcome(mcs_pillars, v) for v in needed}
    yrbs_y = {v: outcome(yrbs_pillars, v) for v in needed}
    rows = []
    for seed in seeds:
        # ONE split per seed, stratified on the FULL-5 outcome, shared by every variant
        if splits is not None:
            S = splits[(seed, 1)]
        else:
            S = D.build_splits(seed, mcs_y["full"], yrbs_y["full"], X_mcs=X_mcs, X_yrbs=X_yrbs,
                               threshold=1)
        for variant in variants:
            ym_v = mcs_y[variant].reindex(S["Xm_trm"].index)
            ymte_v = mcs_y[variant].reindex(S["Xm_te"].index).values
            yte_v = yrbs_y[variant].reindex(S["Xy_te"].index).values
            fit_mask = ~ym_v.isna()
            for fam in families:
                params, _ = PM.select_params(fam, ARM, 1, tuned)
                try:
                    if sources is not None and variant == "full":
                        # The full-five fit at >=1 IS the canonical source model.
                        base = source_for(sources, fam, 1, seed)
                    else:
                        base = PM.make_estimator(fam, params, seed=seed)
                        base.fit(S["Xm_cs"][fit_mask.values], ym_v[fit_mask].astype(int))
                    pi = base.predict_proba(S["Xm_te_cs"])[:, 1]
                    pt = base.predict_proba(S["Xy_te_cs"])[:, 1]
                    mi, mt = ~np.isnan(ymte_v), ~np.isnan(yte_v)
                    internal = float(roc_auc_score(ymte_v[mi], pi[mi]))
                    transfer = float(roc_auc_score(yte_v[mt], pt[mt]))
                except Exception as exc:
                    # aggregate context only — never the message, which could carry data
                    print(f"    WARN {variant}/{fam} seed {seed}: "
                          f"{type(exc).__name__} during fit/eval — recorded NaN, continuing")
                    internal = transfer = float("nan")
                rows.append(dict(
                    variant=variant,
                    dropped_pillar=("" if variant == "full" else LOO_PILLARS[variant]),
                    model=fam, seed=seed,
                    mcs_auc_internal=internal, yrbs_auc_transfer=transfer,
                    transfer_gap=(internal - transfer
                                  if internal == internal and transfer == transfer
                                  else float("nan"))))
    return pd.DataFrame(rows)


# Each value is a key of `outcomes.CATEGORY_SETS`, which is where the pillar membership lives.
E10_OUTCOMES: Mapping[str, str] = {
    "abuse_cluster": "felitti_abuse", "household_cluster": "felitti_household",
    "pillar_emotional": "pillar_emotional", "pillar_physical": "pillar_physical",
    "pillar_mental": "pillar_mental", "pillar_substance": "pillar_substance",
}
E10_PARENT: Mapping[str, str] = {
    "pillar_emotional": "abuse_cluster", "pillar_physical": "abuse_cluster",
    "pillar_mental": "household_cluster", "pillar_substance": "household_cluster",
}


# The six outcome names and their `CATEGORY_SETS` keys are declared in `E10_OUTCOMES` above.
# A variant that is excluded is recorded with its reason rather than dropped, so the excluded
# set is part of the result and not an absence.
# UNVERIFIED: needs MCS data to check the numbers.
def outcome_variant_battery(mcs_pillars: pd.DataFrame, yrbs_pillars: pd.DataFrame, *,
                            X_mcs: pd.DataFrame, X_yrbs: pd.DataFrame,
                            seeds: Sequence[int] = SEEDS,
                            families: Sequence[str] = ("L1_LR", "XGB", "CatBoost"),
                            tuned=None, yrbs_tuned=None,
                            variants: Sequence[str] | None = None) -> pd.DataFrame:
    """All symmetric outcome variants — category clusters and per-pillar targets.

    THE SYMMETRY ASSERTION IS A HALT, NOT A WARNING, and this function keeps it that way. For
    each outcome the MCS-side and YRBS-side pillar lists must be IDENTICAL. If they are not,
    the run would train on one construct and evaluate on another — the model would be answering
    a different question in each jurisdiction and the transfer number would be meaningless
    rather than merely noisy. Six outcomes pass; the asymmetric ones are excluded by name.

    WHAT IS EXCLUDED AND WHY, recorded rather than dropped (outcome_sensitivity.py:22-26):
      `pillar_sexual`            prevalence guard (<0.05)
      `felitti_neglect`          no MCS pillar exists — CATEGORY_SETS maps it to []
      `hughes_relational`, `hughes_maltreatment`, `hughes_mental_violence`,
      `felitti_household`-specialist    cohort-asymmetric (see ASYMMETRIC_CATEGORIES)
      `y_broad` / `A.x-broad`    MCS separation was never extracted, so y_broad == y_harmonised
      flat-5 >=4                 prevalence guard
    A silently absent variant reads as "not interesting"; a recorded exclusion reads as
    "checked, and here is why not". The source writes them to their own CSV for that reason.

    `felitti_abuse == hughes_relational == abuse_cluster` and
    `felitti_household == hughes_household == household_cluster` on five symmetric pillars, so
    each is emitted ONCE. Emitting the taxonomy aliases separately would triple-count the same
    result under three names.

    A >=1-only analysis. The variants are already narrower constructs, and a >=2 cut on a
    two-pillar cluster is the conjunction rather than a threshold sweep. Needs the pillar
    frames and both feature matrices.

    EACH VARIANT GETS ITS OWN PARTITION, stratified on its own outcome. That is the opposite
    convention from `leave_one_pillar_out`, which holds one full-five partition fixed across
    variants. Neither is changed here: they answer related but different sensitivity questions,
    and which convention this one should use is a methodological decision, not a structural one.

    `variants` narrows which outcomes are run, so the caller can run one per cell and
    concatenate afterwards. The symmetry assertion is applied to the whole declared set either
    way, because an asymmetric composition is a defect in the vocabulary rather than in the
    selection.
    """
    import outcomes as BO
    from sklearn.metrics import roc_auc_score
    import data as D
    import models as PM

    # --- symmetry assertion: HALT, do not warn ---
    asym = []
    for name, key in E10_OUTCOMES.items():
        cols = BO.CATEGORY_SETS[key]
        mcs_cols = [c for c in cols if c in mcs_pillars.columns]
        yrbs_cols = [c for c in cols if c in yrbs_pillars.columns]
        if mcs_cols != yrbs_cols:
            asym.append((name, mcs_cols, yrbs_cols))
    if asym:
        raise ValueError(
            "outcome_variant_battery: asymmetric composition for "
            + "; ".join(f"{n}: MCS={a} vs YRBS={b}" for n, a, b in asym)
            + ". Running these would train on one construct and test on another. "
              "They belong in the excluded list, not in the battery.")

    chosen = list(E10_OUTCOMES) if variants is None else list(variants)
    unknown = [v for v in chosen if v not in E10_OUTCOMES]
    if unknown:
        raise ValueError(f"unknown outcome variant(s) {unknown}; known: {list(E10_OUTCOMES)}")

    tuned = tuned if tuned is not None else PM.mcs_settings()
    yrbs_tuned = yrbs_tuned if yrbs_tuned is not None else PM.yrbs_settings()

    rows = []
    for name in chosen:
        key = E10_OUTCOMES[name]
        cols = BO.CATEGORY_SETS[key]
        ym = BO.compose_outcome(mcs_pillars[cols], threshold=1, strict=True)
        yy = BO.compose_outcome(yrbs_pillars[cols], threshold=1, strict=True)
        for seed in seeds:
            S = D.build_splits(seed, ym, yy, X_mcs=X_mcs, X_yrbs=X_yrbs, threshold=1)
            prev = float(np.nanmean(np.asarray(S["yte"], float)))
            for fam in families:
                # A model uses the settings selected in the cohort it is trained on, here as
                # everywhere: the MCS-trained base takes the MCS mapping, the YRBS-trained local
                # reference takes the YRBS one.
                params, hsrc = PM.select_params(fam, ARM, 1, tuned)
                base = PM.make_estimator(fam, params, seed=seed)
                base.fit(S["Xm_cs"], S["ym_trm"].astype(int))
                local = PM.make_estimator(fam, yrbs_tuned[(1, fam)], seed=seed)
                local.fit(S["Xy_tr_cs2"], S["yy_trm"].astype(int))
                for role, y, p, src in (
                        ("unadapted", S["yte"], base.predict_proba(S["Xy_te_cs"])[:, 1], hsrc),
                        ("mcs_internal", S["ymte"],
                         base.predict_proba(S["Xm_te_cs"])[:, 1], hsrc),
                        ("yrbs_local", S["yte"],
                         local.predict_proba(S["Xy_te_cs2"])[:, 1],
                         PM.YRBS_HYPERPARAMETER_SOURCE)):
                    ya = np.asarray(y, float); m = ~np.isnan(ya)
                    rows.append(dict(outcome=name, category_set=key, role=role, model=fam,
                                     hyperparameter_source=src, lineage="cs",
                                     threshold=">=1", seed=seed,
                                     auc=float(roc_auc_score(ya[m], np.asarray(p, float)[m])),
                                     prevalence=prev, is_mcs=(role == "mcs_internal"),
                                     parent=E10_PARENT.get(name, "")))
    return pd.DataFrame(rows)


BUDGETS: Sequence[int] = (50, 100, 200, 300, 500, 750, 1000, 1500, 2000)
ANCHOR_K, MAX_K = 500, 2000
CEILING_EPS = 0.005
SLOPE_BAND = (0.9, 1.1)
BUDGET_REGIMES: Sequence[str] = ("target_only", "full_revision")

# k-budget sweeps
BUDGET_METRIC_KEYS: Sequence[str] = (
    "auc", "prauc", "brier", "ece", "cal_slope", "cal_intercept",
    "precision_at_5", "recall_at_5", "precision_at_15", "recall_at_15", "prevalence",
)
BUDGET_ARM = ARM


# The emitted key grid is (family, regime, threshold, k, metric) over `BUDGETS` and
# `BUDGET_METRIC_KEYS`, plus the (pool) rows. The regime vocabulary is the PAPER spelling —
# {naive, target_only, full_revision} — because that is what the stored curve is keyed on, and
# `regime_names` is where the translation happens.
# UNVERIFIED: needs MCS data to check the values.
def label_budget_curve(splits: Mapping[tuple, object], *,
                       families=None, regimes=("unadapted", "target_only", "full_revision"),
                       thresholds=THRESHOLDS, budgets=BUDGETS, arm: Arm = BUDGET_ARM,
                       tuned=None, sources=None) -> pd.DataFrame:
    """Performance across target-label budgets k, all nine families, both thresholds.

    NESTED, ANCHORED BUDGETS — this is the part that must be reproduced exactly. k runs over
    [50 .. 2000] as nested prefixes of a single ordering built so that the k=500 prefix IS
    S4's published anchor `pool_cs.sample(500, seed)`:

        ordered = [pool.sample(500, seed)] ++ [(pool - those).sample(1500, seed)]

    so every k<=500 is a prefix of the published point and k>500 extends it. If the ordering
    is rebuilt any other way — one `sample(2000)` call, say — the k=500 column stops
    reconciling with the battery and the curve silently disagrees with the paper. The
    construction lives on the bundle as `SplitBundle.nested_draw()`, in one place.

    `regimes` selects which curves are computed, so the caller can run one per cell.
    `unadapted` does not vary with k; it is emitted flat at every budget when it is asked for,
    because dropping it would falsely imply a family never calibrates — which is exactly the
    RF/ET result the paper turns on, so the flat reference is what makes that reading legible.
    A caller that already holds the unadapted result can leave it out here and put the flat
    reference beside the curves itself.

    `sources` is the canonical source-model dict. The base this fits is
    `make_estimator(fam, select_params(fam, "tuned", t, tuned)).fit(Xm_cs, ym_trm)` — the same
    estimator, params, seed and training frames as `fit_source_models` — so passing the dict
    reuses it. The per-budget models are NOT reusable and are fitted here: they are fitted on
    k target records and differ by k, which is what the curve measures.

    THE VOCABULARY, and this is the one thing about this function that is not arithmetic
    (decision 3). The manuscript says `full_revision`; the pipeline key is
    `fine_tune`. `full_revision` is CANONICAL because it is the name in the manuscript. The
    mapping lives in `src/regime_names.py` and is applied HERE, at the boundary: the regime
    is resolved to its pipeline key to compute, and the row is labelled with the paper name —
    which is what `$THESIS_WORK_DIR/label_budget_curve.csv` already stores. Nowhere else translates.

    THE ASSUMPTION UNDER THAT NAME, which is inherited and unresolved: nothing in the
    manuscript disambiguates full-model revision from the R5 feature-subset refit, and both are
    label-using revisions. `full_revision` is read as the former here. If it turns out to mean
    R5, the fix is one line in `src/regime_names.py`.

    This is the single largest contributor to `$THESIS_WORK_DIR/all_results.csv`: 5,346 of 7,686 rows.

    Budget-lineage only, so it takes no `lineage` keyword.
    """
    import regime_names as RN
    import evaluation as E
    import models as PM

    families = list(families) if families is not None else list(PM.FAMILIES)
    acc: dict = {}
    npos: dict = {}
    for t in thresholds:
        t = int(t)
        for seed in sorted({s for (s, tt) in splits if tt == t}):
            S = splits[(seed, t)]
            feat_cols = S.feat_cols
            yte = S["yte"]
            prev = float(np.nanmean(np.asarray(yte, float)))
            ordered = (S.nested_draw(anchor_k=ANCHOR_K, max_k=MAX_K)
                       if hasattr(S, "nested_draw") else None)
            if ordered is None:
                raise TypeError("label_budget_curve needs a SplitBundle (for nested_draw), "
                                "not a plain split dict — the anchored ordering is what makes "
                                "k=500 reconcile with the battery.")
            for fam in families:
                params, _ = PM.select_params(fam, arm, t, tuned)
                if sources is not None and arm == ARM:
                    base = source_for(sources, fam, t, seed)
                else:
                    base = PM.make_estimator(fam, params, seed=seed)
                    base.fit(S["Xm_cs"], S["ym_trm"].astype(int))
                mn = (_budget_metric_row(E, yte, base.predict_proba(S["Xy_te_cs"])[:, 1], prev)
                      if "unadapted" in regimes else None)
                for k in budgets:
                    sl = ordered.iloc[:min(k, len(ordered))]
                    Xk, yk = sl[feat_cols], sl["y"]
                    npos.setdefault((t, k), []).append(float(yk.sum()))
                    # unadapted is flat in k, recorded at every budget as the reference
                    if mn is not None:
                        for key in BUDGET_METRIC_KEYS:
                            acc.setdefault((fam, "unadapted", t, k), {}) \
                               .setdefault(key, []).append(mn[key])
                    for reg_label in regimes:
                        if reg_label == "unadapted":
                            continue
                        reg_key = RN.to_pipeline(reg_label)
                        if yk.nunique() < 2:
                            continue
                        try:
                            p = _budget_regime_scores(reg_key, fam, params, base, seed, S, Xk, yk)
                            mr = _budget_metric_row(E, yte, p, prev)
                        except Exception as exc:
                            print(f"    WARN {fam}/{reg_label} >={t} k={k} seed={seed}: "
                                  f"{type(exc).__name__} — cell suppressed, continuing")
                            continue
                        for key in BUDGET_METRIC_KEYS:
                            acc.setdefault((fam, reg_label, t, k), {}) \
                               .setdefault(key, []).append(mr[key])

    rows = []
    for (fam, reg_label, t, k), md in acc.items():
        for metric in BUDGET_METRIC_KEYS:
            v = np.asarray([x for x in md.get(metric, []) if x == x], float)
            rows.append(dict(family=fam, regime=reg_label, threshold=f">={t}", k=k,
                             metric=metric,
                             mean=(round(float(v.mean()), 6) if v.size else np.nan),
                             sd=(round(float(v.std()), 6) if v.size else np.nan),
                             n_seeds=int(v.size)))
    for (t, k), v in npos.items():
        rows.append(dict(family="(pool)", regime="pool", threshold=f">={t}", k=k,
                         metric="n_pos_ksample", mean=round(float(np.mean(v)), 1),
                         sd=round(float(np.std(v)), 1), n_seeds=len(v)))
    return (pd.DataFrame(rows)
            .sort_values(["family", "regime", "threshold", "k", "metric"])
            .reset_index(drop=True))


def _budget_regime_scores(reg_key, family, params, base, seed, S, Xk, yk):
    """One budget cell's scores. `reg_key` is the PIPELINE key, already translated."""
    import models as PM
    if reg_key == "unadapted":
        return base.predict_proba(S["Xy_te_cs"])[:, 1]
    if reg_key == "target_only":
        mt = PM.make_estimator(family, params, seed=seed)
        mt.fit(Xk, yk.astype(int))
        return mt.predict_proba(S["Xy_te_cs"])[:, 1]
    if family in PM.WARM_FAMS:                        # fine_tune == the paper's full_revision
        return warm_finetune(family, params, base, Xk, yk, seed)(S["Xy_te_cs"])
    Xb = pd.concat([S["Xm_cs"], Xk], ignore_index=True)
    yb = np.r_[S["ym_trm"].values, yk.values].astype(int)
    wgt = np.r_[np.ones(len(S["Xm_cs"])), np.full(len(Xk), ALPHA)]
    mf = PM.make_estimator(family, params, seed=seed)
    mf.fit(Xb, yb, sample_weight=wgt)
    return mf.predict_proba(S["Xy_te_cs"])[:, 1]


def _budget_metric_row(E, yte, p, prev) -> dict:
    """The same metric definitions the per-family calibration run used, which is what makes
    the k=500 column of the curve reconcile with that run cell for cell.

    Note the operating points here are 5% and 15% — NOT the battery's decile/quintile. The
    curve reports the two policy-relevant capacities (Hello Baby 5%, AFST 15%); the battery
    reports 10%/20%. Mixing them up produces numbers that look comparable and are not.
    """
    m = E.metrics(yte, p, prevalence=prev)
    ym_, pm_ = E._mask(yte, p)
    oc5 = E.operating_point_confusion(ym_, pm_, capacity=0.05, tag="p5")
    oc15 = E.operating_point_confusion(ym_, pm_, capacity=0.15, tag="p15")
    return {"auc": m["auc"], "prauc": m["prauc"], "brier": m["brier"], "ece": m["ece"],
            "cal_slope": m["cal_slope"], "cal_intercept": m["cal_intercept"],
            "precision_at_5": oc5["p5_precision"], "recall_at_5": oc5["p5_recall"],
            "precision_at_15": oc15["p15_precision"], "recall_at_15": oc15["p15_recall"],
            "prevalence": m["prevalence"]}


# Seven of the nine columns are computable from the per-k means the curve stores — ceiling x2,
# flip, calib_usable x2, prec5_delta x2. The two crossover_* columns are not, and are emitted
# as NaN rather than guessed at; the docstring says why.
def budget_summary(curve: pd.DataFrame) -> pd.DataFrame:
    """Per family/threshold: crossover k, ceiling k, flip k, calibration-usable k, and the
    5%-precision delta from naive at k=500. 162 rows of all_results.csv.

    Four of the five derivations need only the per-k MEANS that `label_budget_curve.csv`
    stores, and all four are transcribed here verbatim:

      ceiling_*        smallest k whose mean AUC is within CEILING_EPS (0.005) of that
                       regime's own k=2000 mean
      flip             smallest k where target_only's mean AUC exceeds full_revision's
      calib_usable_*   smallest k whose mean calibration slope first lands in [0.9, 1.1]
      prec5_delta_*    (regime - naive) mean precision at 5% capacity, at k=500, rounded to 4

    THE FIFTH IS NOT COMPUTABLE HERE, and that is a property of the frozen file rather
    than a gap in the transcription. `crossover_*` is a paired one-sided Wilcoxon of each
    (regime, k) cell against naive, Holm-corrected within (family, threshold) over all 18
    tests (label_budget_curve.py:274-288). It needs the twenty PER-SEED AUC values per
    cell. `$THESIS_WORK_DIR/label_budget_curve.csv` stores mean, sd and n_seeds only, and no other
    file on disk carries the per-seed label-budget arrays — `all_results.csv` ingests the
    same means. So the two crossover columns are emitted as NaN rather than guessed at. They
    cannot be recovered from the stored file, and this pipeline does not recompute them.

    `curve` is the long-form frame: columns family, regime, threshold, k, metric, mean,
    sd, n_seeds — i.e. `$THESIS_WORK_DIR/label_budget_curve.csv` as read.
    """
    def mean_of(fam, reg, t, k, metric):
        r = curve[(curve.family == fam) & (curve.regime == reg)
                  & (curve.threshold == t) & (curve.k == k) & (curve.metric == metric)]
        return float(r["mean"].iloc[0]) if len(r) else float("nan")

    def has(fam, reg, t, k, metric):
        return mean_of(fam, reg, t, k, metric) == mean_of(fam, reg, t, k, metric)

    fams = [f for f in curve.family.unique() if f not in ("(pool)",)]
    # Thresholds come from the curve, never from a literal. The frozen
    # label_budget_summary.csv carries >=1, >=2 AND >=3 — nine families x three thresholds,
    # 27 rows — so a hardcoded pair silently republishes it two thirds the size.
    thresholds = sorted(curve.threshold.unique())
    out = []
    for fam in fams:
        for t in thresholds:
            unadapted_p5 = mean_of(fam, "unadapted", t, ANCHOR_K, "precision_at_5")
            # (2) ceiling: smallest k within 0.005 AUC of own k=2000
            ceiling = {}
            for rl in BUDGET_REGIMES:
                top = mean_of(fam, rl, t, MAX_K, "auc")
                ceiling[rl] = min([k for k in BUDGETS if has(fam, rl, t, k, "auc")
                                   and abs(mean_of(fam, rl, t, k, "auc") - top) < CEILING_EPS],
                                  default=None)
            # (3) flip: smallest k where target_only mean AUC > full_revision mean AUC
            flip = next((k for k in BUDGETS
                         if has(fam, "target_only", t, k, "auc") and has(fam, "full_revision", t, k, "auc")
                         and mean_of(fam, "target_only", t, k, "auc")
                         > mean_of(fam, "full_revision", t, k, "auc")), None)
            # (4) calibration-usable: smallest k where mean cal_slope first enters [0.9, 1.1]
            calib = {}
            for rl in BUDGET_REGIMES:
                calib[rl] = min([k for k in BUDGETS if has(fam, rl, t, k, "cal_slope")
                                 and SLOPE_BAND[0] <= mean_of(fam, rl, t, k, "cal_slope") <= SLOPE_BAND[1]],
                                default=None)
            # (5) precision@5% movement at k=500 (regime - naive)
            d_to = mean_of(fam, "target_only", t, ANCHOR_K, "precision_at_5") - unadapted_p5
            d_fr = mean_of(fam, "full_revision", t, ANCHOR_K, "precision_at_5") - unadapted_p5
            out.append(dict(family=fam, threshold=t,
                            crossover_target_only=np.nan,        # see the docstring
                            crossover_full_revision=np.nan,      # see the docstring
                            ceiling_target_only=ceiling["target_only"],
                            ceiling_full_revision=ceiling["full_revision"],
                            flip_target_over_full=flip,
                            calib_usable_target_only=calib["target_only"],
                            calib_usable_full_revision=calib["full_revision"],
                            prec5_delta_naive_to_target_only_k500=round(d_to, 4),
                            prec5_delta_naive_to_full_revision_k500=round(d_fr, 4)))
    return pd.DataFrame(out)


def _clip(p):
    return np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)




# ---------------------------------------------------------------------------
# The calibration-correction sweep
#
# WHY THIS EXISTS HERE. The frozen `outputs/paper/data/calibration.csv` had no producer in this
# repository, so the sweep could be read and not recomputed: a reuse of this code against a
# different extract would have answered its own question with the published MCS -> YRBS 2023
# numbers. This function is that producer.
CORRECTIONS: Sequence[str] = ("none", "logistic", "isotonic", "beta")
# (update_n, calib_n), summing to K. The 500/0 arm has no held-out fold, so the correction is
# fitted on the records the model was updated on — an optimistic baseline, kept because it makes
# the overfitting cost visible rather than hiding it.
RATIOS: Sequence[tuple] = ((500, 0), (400, 100), (350, 150), (300, 200))
# Cal-slope and cal-intercept mean something only for these two. Fitting a logistic slope to an
# isotonically- or beta-corrected score does not, and the source reports NA there.
SLOPE_ONLY = frozenset({"none", "logistic"})
SWEEP_REGIMES: Mapping[str, str] = {"unadapted": "unadapted", "target_only": "target_only",
                                    "full_revision": "fine_tune"}
SWEEP_METRICS: Sequence[str] = ("brier", "ece", "auc", "prauc")
# What `n_seeds` carries where a metric does not apply. Verbatim from the published table, so a
# recomputed sweep and the frozen one say the same thing in the same words.
NOT_MEANINGFUL = "NA (slope on non-logistic correction is not meaningful)"


def recalibrate(scores, method: str, calibration_slice=None):
    """Post-hoc calibration of transferred scores.

    All four are monotone in the score, so AUC is invariant and only the calibration metrics
    move. The pipeline's logistic recalibration is linear in log-odds and cannot remove an
    inverse-sigmoid distortion — the RF/ET failure mode, slopes 1.3-1.6 under label-using
    regimes. `isotonic` and `beta` (Kull 2017) are the matched non-linear corrections.

    `calibration_slice` is the `(p_cal, y_cal)` pair from the held-out fold carved out of the
    k=500 budget, never the model-update records except in the deliberately optimistic 500/0
    arm. `method='none'` returns the scores unchanged and ignores it.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from scipy.special import logit

    if method == "none":
        return np.asarray(scores, float)
    if calibration_slice is None:
        raise ValueError(f"method={method!r} needs calibration_slice=(p_cal, y_cal)")
    p_cal, y_cal = calibration_slice
    if method == "logistic":                 # Platt: intercept + slope on the linear predictor
        lr = LogisticRegression().fit(logit(_clip(p_cal)).reshape(-1, 1), y_cal)
        return lr.predict_proba(logit(_clip(scores)).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip").fit(_clip(p_cal), y_cal)
        return iso.transform(_clip(scores))
    if method == "beta":                     # Kull 2017: LR on [log p, -log(1-p)]
        def feats(p):
            p = _clip(p)
            return np.c_[np.log(p), -np.log(1 - p)]
        lr = LogisticRegression().fit(feats(p_cal), y_cal)
        return lr.predict_proba(feats(scores))[:, 1]
    raise ValueError(method)


# SOURCE: calibration_correction_sweep.py::seed_records
def _sweep_seed(family: Family, params, S, seed: int, k: int, *, base=None,
                regimes=None) -> list:
    """One (family, threshold, seed) cell -> (regime, correction, update_n, calib_n, metric, value).

    `base` is the canonical source model when the caller holds one; it is the same fit this
    would otherwise perform. `regimes` narrows which of `SWEEP_REGIMES` are computed.
    """
    import evaluation as EV
    import models as PM

    yte, Xte = S["yte"], S["Xy_te_cs"]
    prev = float(np.nanmean(np.asarray(yte, float)))
    feat_cols = S.feat_cols
    slice_k = S.k_slice(k)                   # the published anchor, one definition
    if base is None:
        base = PM.make_estimator(family, params, seed=seed)
        base.fit(S["Xm_cs"], S["ym_trm"].astype(int))
    wanted = SWEEP_REGIMES if regimes is None else {r: SWEEP_REGIMES[r] for r in regimes}

    out = []
    for regime, reg_key in wanted.items():
        for update_n, calib_n in RATIOS:
            update_fold, held_fold = slice_k.iloc[:update_n], slice_k.iloc[update_n:]
            if reg_key == "unadapted":
                scorer = lambda X: base.predict_proba(X)[:, 1]           # noqa: E731
                calib = held_fold if calib_n > 0 else slice_k            # whole slice at 500/0
                un = 0
            else:
                Xu, yu = update_fold[feat_cols], update_fold["y"]
                if yu.nunique() < 2:
                    continue
                if reg_key == "target_only":
                    mt = PM.make_estimator(family, params, seed=seed)
                    mt.fit(Xu, yu.astype(int))
                    scorer = lambda X, _m=mt: _m.predict_proba(X)[:, 1]  # noqa: E731
                elif family in PM.WARM_FAMS:
                    scorer = warm_finetune(family, params, base, Xu, yu, seed)
                else:
                    Xb = pd.concat([S["Xm_cs"], Xu], ignore_index=True)
                    yb = np.r_[S["ym_trm"].values, yu.values].astype(int)
                    wgt = np.r_[np.ones(len(S["Xm_cs"])), np.full(len(Xu), ALPHA)]
                    mf = PM.make_estimator(family, params, seed=seed)
                    mf.fit(Xb, yb, sample_weight=wgt)
                    scorer = lambda X, _m=mf: _m.predict_proba(X)[:, 1]  # noqa: E731
                calib = held_fold if calib_n > 0 else update_fold        # optimistic at 500/0
                un = update_n

            y_cal = calib["y"].to_numpy().astype(int)
            if np.unique(y_cal).size < 2:
                continue
            cn = len(calib)
            p_test, p_cal = scorer(Xte), scorer(calib[feat_cols])
            for kind in CORRECTIONS:
                m = EV.metrics(yte, recalibrate(p_test, kind, (p_cal, y_cal)),
                               prevalence=prev)
                for metric in SWEEP_METRICS:
                    out.append((regime, kind, un, cn, metric, m[metric]))
                # Slope and intercept are emitted for EVERY correction so the grid stays
                # rectangular, but carry NaN where they are not meaningful. The published
                # table does the same: a missing row reads as "not run", a NaN row with its
                # reason reads as "run, and the quantity does not apply".
                if kind in SLOPE_ONLY:
                    out.append((regime, kind, un, cn, "cal_slope", m["cal_slope"]))
                    out.append((regime, kind, un, cn, "cal_intercept", m["cal_intercept"]))
                else:
                    out.append((regime, kind, un, cn, "cal_slope", np.nan))
                    out.append((regime, kind, un, cn, "cal_intercept", np.nan))
    return out


def calibration_correction_sweep(splits: Mapping[tuple, object], *,
                                 families: Sequence[str] = None,
                                 thresholds: Sequence[int] = (1, 2, 3),
                                 seeds: Sequence[int] = range(20),
                                 tuned=None, k: int = K, sources=None,
                                 regimes: Sequence[str] | None = None) -> pd.DataFrame:
    """Four post-hoc corrections x three regimes x four update/calibrate ratios, across seeds.

    Answers what the pipeline's default logistic recalibration cannot: whether a matched
    non-linear correction recovers calibration on transferred scores, and what it costs. Every
    correction is monotone, so AUC and PR-AUC are carried as a MONOTONICITY CHECK — they must
    match the `none` column, and a difference beyond isotonic's tie-induced noise is a defect.

    THE REGIME KEYS ARE THIS PIPELINE'S, NOT THE FROZEN TABLE'S. `outputs/paper/data/calibration.csv`
    was published before `naive` was renamed to `unadapted` and still carries the old key;
    `regime_names.RENAMED_KEYS` holds the pair. Translate on the comparison path, never here —
    a producer that emitted a retired key to match a file would be encoding one run's history
    into every future run.

    Fits per cell: one base, plus one update per ratio for each label-using regime asked for.
    `sources` supplies the base from the canonical dict — the identical fit — leaving only the
    update fits, which are fitted on different update folds and cannot be shared. `regimes`
    narrows which of `SWEEP_REGIMES` are computed, so the caller can run one per cell.

    Returns one row per family x regime x threshold x correction x update_n x calib_n x metric,
    which is the grain `tables.py` declares for `calibration`.
    """
    import models as PM
    import regime_names as RN

    families = list(families) if families is not None else list(RN.FAMILIES)
    tuned = tuned if tuned is not None else PM.mcs_settings()
    if regimes is not None:
        unknown = [r for r in regimes if r not in SWEEP_REGIMES]
        if unknown:
            raise ValueError(f"unknown sweep regime(s) {unknown}; "
                             f"known: {list(SWEEP_REGIMES)}")
    cells: dict = {}
    for family in families:
        for t in thresholds:
            params, _ = PM.select_params(family, ARM, t, tuned)
            for seed in seeds:
                S = splits[(seed, int(t))]
                base = (source_for(sources, family, t, seed)
                        if sources is not None else None)
                for rec in _sweep_seed(family, params, S, int(seed), k, base=base,
                                       regimes=regimes):
                    regime, kind, un, cn, metric, val = rec
                    cells.setdefault((family, f">={t}", regime, kind, un, cn, metric),
                                     []).append(val)

    rows = []
    for (family, threshold, regime, correction, un, cn, metric), vals in cells.items():
        v = np.asarray(vals, float)
        applies = bool(np.isfinite(v).any())
        rows.append(dict(family=family, model_class=RN.FAMILY_CLASS[family], regime=regime,
                         threshold=threshold, correction=correction, update_n=un, calib_n=cn,
                         metric=metric,
                         mean=round(float(np.nanmean(v)), 6) if applies else np.nan,
                         sd=round(float(np.nanstd(v)), 6) if applies else np.nan,
                         n_seeds=int(v.size) if applies else NOT_MEANINGFUL))
    return (pd.DataFrame(rows)
            .sort_values(["family", "regime", "threshold", "correction", "calib_n", "metric"])
            .reset_index(drop=True))
