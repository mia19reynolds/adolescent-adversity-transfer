"""
src/evaluation.py — discrimination, calibration, conformal, fairness, operational.

Backs the first half of notebook 03_evaluation_and_robustness.ipynb.

CONTENTS — four kinds of function live here, and the difference matters when you
are working out whether a call is cheap.

    A. PURE METRICS — compute from arrays you pass in. No file, no fit.
       metrics, ece, cal_slope_intercept, operating_point_confusion

    B. CALIBRATION READ-OUT — calibration_summary, the slope/ECE/AUC slice of the battery.
       (No significance machinery: the twenty splits overlap, so the tests it once ran are
       not supported on this design. The significance table is read frozen.)

    C. FROZEN-FILE READERS — each takes an optional `frame=`; when it is None
       they read a default CSV under $THESIS_WORK_DIR/tables/. Pass the frame in if you
       already have it and the read is skipped. Roughly fourteen functions,
       including subgroup_scores, subgroup_coverage, subgroup_calibration,
       subgroup_discrimination, subgroup_panels, per_family_calibration,
       regime_grid, conformal_cell_audit, build_canonical_table.

    D. WRITERS — write_subgroup_by_model is the only one, and it says so.

Every function in C and D carries a `SIDE EFFECTS:` line naming the file it
reads or writes. A function in A or B that grows a file read has changed
category and its docstring must say so.

`build_canonical_table` is 365 lines and is deliberately not split. It is the single
authority for $THESIS_WORK_DIR/tables/CANONICAL.csv, and that file is frozen.

METRIC REPORTING CONVENTIONS, all of which the paper depends on:

  * AUC is the PRIMARY metric. The outcome here is MAJORITY-POSITIVE at the reported cuts, so
    the child-welfare PRM literature's preference for AUPRC — which rests on LOW prevalence —
    does not apply. AUC is also the only metric comparable to published ACE/welfare PRM
    comparators.
  * PR-AUC is supplementary and is NEVER comparable across cohorts or across subgroup
    cells, because the no-skill null is the prevalence, and the prevalence differs by
    cohort, by threshold and by subgroup cell. Every PR-AUC must be reported with its own
    null and its lift (= PR-AUC / null) on the same row — which is why every row carries a
    `prevalence` column rather than this file carrying a number.
  * Cells are NEVER ranked by PR-AUC. They are ranked by AUC.
  * Small-cell rule: n < 50 in the evaluation split is flagged and suppressed. The flag
    must survive into summaries and figures, not be silently dropped.

TIER 1a: every metric in this module is computed on the YRBS test slice. MCS-side scores
are not persisted under the licence. MCS aggregate counts are SDC-rounded to the nearest
10 and any cell with n < 50 has its prevalence suppressed — a rate on a tiny cell is
disclosive.
"""

from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd


from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

def _work_tables() -> Path:
    """The working table directory, outside the repository.

    This defaulted to `<repo>/$THESIS_WORK_DIR/tables` until the working root moved out; that
    path no longer exists, so every caller relying on the default would have raised.

    With no working root configured — a clone — it returns a path that does not resolve.
    Callers pass it to `inputs.resolve`, which falls back to the tracked public copy, so a
    clone reads published data instead of failing on an absent working tree.
    """
    import config as _C
    try:
        return Path(_C.WORK_TABLES)
    except RuntimeError:
        return Path("/nonexistent-working-root")


def _work_root() -> Path:
    """The working root itself. The budget-lineage analysis files live at its TOP level
    (mirroring the retired `outputs/` layout); the s4 tables live under `tables/`. Same
    clone fallback as `_work_tables`."""
    import config as _C
    try:
        return Path(_C.work_path())
    except RuntimeError:
        return Path("/nonexistent-working-root")


def _repo_root() -> Path:
    """The repository root, from this file's location — never from the working directory.

    EVERY DEFAULT PATH IN THIS MODULE RESOLVES THROUGH HERE. A bare `Path("outputs")` default
    is cwd-relative: it works from a shell at the repo root and raises FileNotFoundError from
    anywhere else, including a notebook, where VS Code sets cwd to `notebooks/`. The files were
    always present; only the caller's directory differed. `diagnostics._root` and
    `manuscript._root` do the same thing for the same reason.

    An explicit argument still wins, so a caller can point any of these at a scratch tree.
    """
    return Path(__file__).resolve().parents[1]

Threshold = Literal[1, 2, 3]

ECE_BINS = 10


# 1. The metric battery
def _mask(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float); m = ~np.isnan(y)
    return y[m], p[m]


def ece(y, p, bins=ECE_BINS):
    """Equal-width 10-bin expected calibration error. This is the project's one definition."""
    edges = np.linspace(0, 1, bins + 1); tot = len(y); e = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        m = (p >= lo) & (p <= hi) if b == 0 else (p > lo) & (p <= hi)
        if m.sum():
            e += (m.sum() / tot) * abs(y[m].mean() - p[m].mean())
    return float(e)


def cal_slope_intercept(y, p):
    """Calibration slope (unpenalised logistic on the logit) and intercept (Brent root).

    The intercept is the shift that makes the model's predicted positives add up to the observed
    ones. `outcome_imbalance` is that difference at a given shift, and the root of it is the
    intercept; a sign change across the search bracket is what tells us a root exists inside it.
    """
    p = np.clip(p, 1e-6, 1 - 1e-6); lp = logit(p)
    try:
        slope = float(LogisticRegression(C=1e8, solver="lbfgs", max_iter=2000)
                      .fit(lp.reshape(-1, 1), y).coef_[0][0])
    except (ValueError, RuntimeError):
        slope = np.nan

    def outcome_imbalance(shift):
        return float(np.sum(y - expit(shift + lp)))

    try:
        brackets_a_root = outcome_imbalance(-15) * outcome_imbalance(15) < 0
        inter = brentq(outcome_imbalance, -15, 15) if brackets_a_root else np.nan
    except (ValueError, RuntimeError):
        inter = np.nan
    return slope, float(inter) if inter == inter else np.nan


def metrics(y_true, scores, *, prevalence: float | None = None) -> dict:
    """The full per-cell metric battery, computed once and reused everywhere.

    AUC, PR-AUC, the prevalence null, lift, Brier, ECE, calibration slope and intercept.
    Reimplemented in at least six near-identical copies across the frozen runners
    (`threshfree`, `ece`, `cal_slope_intercept`, `_auc`, `_brier`); S4's is canonical.
    Consolidating these is the highest-value deduplication in the repo.

    `prevalence` is S4's `prev_null` — the prevalence of THIS eval slice, which is what
    prauc_lift divides by. battery_machinery.build_row passes float(nanmean(y)).

    NO INTERVAL IS PRODUCED HERE. A per-seed bootstrap over one split's evaluable rows
    describes one fitted model on one evaluation sample, and the reported statistic is a mean
    over twenty overlapping splits. Attaching the first to the second is a category error, and
    the columns that did it — `auc_boot_lo`, `auc_boot_hi`, and the `auc_lo` / `auc_hi` that
    `summarise_seeds` built from them — have been removed rather than relabelled. An interval
    that corresponds to the reported mean is a separate design question.
    """
    y, p = _mask(y_true, scores); n = y.size
    prev_null = prevalence
    cs, ci = cal_slope_intercept(y, p)
    prauc = float(average_precision_score(y, p))
    row = dict(auc=float(roc_auc_score(y, p)),
               prauc=prauc, prevalence_null=prev_null,
               prauc_lift=prauc / prev_null if prev_null else np.nan,
               brier=float(np.mean((p - y) ** 2)), ece=ece(y, p), cal_slope=cs, cal_intercept=ci,
               n_test=n, prevalence=float(y.mean()), n_pos=int(y.sum()))
    row.update(operating_point_confusion(y, p, capacity=0.10, tag="dec"))
    row.update(operating_point_confusion(y, p, capacity=0.20, tag="qui"))
    return row


def operating_point_confusion(y_true, scores, *, capacity: float, tag: str = "op") -> dict:
    """Precision, recall, specificity, F1 and MCC at a top-q% screening capacity.

    Operating points are justified, not tuned on test: 5% is the Hello Baby PRM
    convention, 15% is the AFST mandatory screen-in threshold (score >= 18/20 ventiles),
    10% is an intermediate. The no-skill precision equals the prevalence — print it next
    to every precision so lift is legible.

    `tag` prefixes every key and defaults to "op". The metric battery depends on the `dec_`
    and `qui_` prefixes to build one flat row.

    THE SELECTION IS NOT TAKEN HERE. `cohort_capacity_flags` is the one implementation, so the
    decile and quintile columns of notebook 02's battery and every capacity table in notebook
    03 flag the same respondents from the same scores. A second local `argsort` is how the two
    came to disagree wherever scores tied at the boundary.
    """
    y = np.asarray(y_true, float); p = np.asarray(scores, float)
    n = y.size
    flag = cohort_capacity_flags(p, (capacity,))[capacity]
    TP = int(np.sum(flag & (y == 1))); FP = int(np.sum(flag & (y == 0)))
    FN = int(np.sum(~flag & (y == 1))); TN = int(np.sum(~flag & (y == 0)))
    prec = TP / (TP + FP) if (TP + FP) else np.nan
    rec = TP / (TP + FN) if (TP + FN) else np.nan
    spec = TN / (TN + FP) if (TN + FP) else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and prec + rec > 0) else np.nan
    den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = (TP * TN - FP * FN) / den if den > 0 else np.nan
    return {f"{tag}_TP": TP, f"{tag}_FP": FP, f"{tag}_FN": FN, f"{tag}_TN": TN,
            f"{tag}_precision": prec, f"{tag}_recall": rec, f"{tag}_specificity": spec,
            f"{tag}_f1": f1, f"{tag}_mcc": mcc, f"{tag}_flag_rate": (TP + FP) / n}


def calibration_summary(frame=None, *, arm: str = "tuned",
                        regimes=("mcs_internal", "unadapted")) -> pd.DataFrame:
    """Calibration slope, intercept and ECE beside AUC — where transfer hurts most.

    Discrimination survives the crossing; calibration does not. Reporting them together is
    the point: a model that ranks well and is badly calibrated is usable for triage and
    unusable for a probability. Pass the live battery summary after a run; with no frame the
    stored battery summary is read (tracked public copy first) and the result describes the
    published numbers — `attrs["source"]` says which happened.
    """
    import regime_names as RN
    if frame is None:
        from inputs import resolve
        p = resolve("paper/data/transfer_grid.csv", allow_frozen=True)
        frame, prov = pd.read_csv(p), f"frozen {p.name}"
    else:
        prov = "this session's run"
    b = frame[(frame["arm"] == arm) & (frame["regime"].isin(regimes))]
    out = (b[["family", "threshold", "regime", "auc_mean", "ece_mean",
              "cal_slope_mean", "cal_intercept_mean"]]
           .assign(regime=lambda d: d["regime"].map(lambda r: RN.SHORT_DISPLAY.get(r, r)))
           .round(4).sort_values(["threshold", "family", "regime"]).reset_index(drop=True))
    out.attrs["source"] = prov
    return out


def regime_grid(summary: pd.DataFrame | None = None, *, thresholds: Sequence[str],
                regimes: Sequence[str], families: Sequence[str] | None = None,
                arms: Sequence[str] = ("tuned",)) -> pd.DataFrame:
    """Every family x regime x arm x threshold cell in one table.

    `thresholds` and `regimes` are REQUIRED and their order is preserved. Neither is inferred
    from whichever rows happen to be present: a threshold that produced nothing must read as an
    empty requested block, not as a threshold nobody asked for.

    `summary` is passed in rather than read from disk, which is what makes the function
    checkable without a working-directory assumption. Omitting it falls back to the battery
    summary notebook 02 wrote.
    """
    if summary is None:
        from inputs import resolve as _resolve
        summary = pd.read_csv(_resolve("regime_battery_summary.csv"))
    return _build_regime_grid(summary, thresholds=thresholds, regimes=regimes,
                              families=families, arms=arms)


GRID_FAMS: Sequence[str] = ("L1_LR", "L2_LR", "EN_LR", "RF", "ET", "HistGB", "LightGBM", "XGB", "CatBoost")
GRID_REFERENCE: Sequence[str] = ("mcs_internal", "yrbs_local")
GRID_LABEL_FREE: Sequence[str] = ("unadapted", "bbse", "importance_weight", "quantile_map",
                                  "pseudo_label", "pseudo_label_thresh")
GRID_LABEL_USING: Sequence[str] = ("target_only", "fine_tune", "isotonic_recal", "raw_l1_head",
                                   "coef_freeze_intercept", "platt_frozen", "leaf_refresh_global",
                                   "sign_support", "feature_set", "ensemble_same_family",
                                   "ensemble_catboost_source")
# The reporting groups, named. Scientific selection belongs at the call site: a caller states
# which groups it is rendering and the grid renders those, rather than a fixed internal list
# quietly deciding. The two post-adaptation recalibration regimes are their OWN group — they are
# label-using but they are not members of Section H's mechanism comparison, and
# folding them in would change what that comparison compares.
REPORTING_GROUPS: Mapping[str, Sequence[str]] = {
    # THE TWO LOCAL REFERENCES, and only those two. The nested per-split target redevelopment
    # (`yrbs_resource_rich`) is a sensitivity analysis with its own group, so it cannot be
    # rendered into a reference block by a caller that asked for references.
    "reference": ("mcs_internal", "yrbs_local"),
    "nested_target_sensitivity": ("yrbs_resource_rich",),
    "unadapted": ("unadapted",),
    "label_free": ("quantile_map", "importance_weight", "bbse", "pseudo_label",
                   "pseudo_label_thresh"),
    "label_using": ("target_only", "fine_tune", "isotonic_recal", "raw_l1_head",
                    "coef_freeze_intercept", "platt_frozen", "leaf_refresh_global",
                    "sign_support", "feature_set", "ensemble_same_family",
                    "ensemble_catboost_source"),
    "post_adaptation_recalibration": ("target_only_logistic_recal", "fine_tune_logistic_recal"),
}


def reporting_regimes(groups: Sequence[str]) -> list:
    """The regimes of the named groups, in declared order, with no duplicate."""
    unknown = [g for g in groups if g not in REPORTING_GROUPS]
    if unknown:
        raise ValueError(f"unknown reporting group(s) {unknown}; "
                         f"known: {list(REPORTING_GROUPS)}")
    out = []
    for g in groups:
        for r in REPORTING_GROUPS[g]:
            if r not in out:
                out.append(r)
    return out


GROUP_OF_REGIME: Mapping[str, str] = {r: g for g, rs in REPORTING_GROUPS.items() for r in rs}

GRID_CLASS = {**{r: "REFERENCE" for r in GRID_REFERENCE},
              **{r: "LABEL_FREE" for r in GRID_LABEL_FREE},
              **{r: "LABEL_USING" for r in GRID_LABEL_USING}}


# ROW ORDER IS PART OF THE OUTPUT. The grid is emitted in the caller's threshold and regime
# order so that two runs, or a run and a stored table, line up row for row without a re-sort.
def _build_regime_grid(summary: pd.DataFrame, *, thresholds: Sequence[str],
                       regimes: Sequence[str], families: Sequence[str] | None = None,
                       arms: Sequence[str] = ("tuned",)) -> pd.DataFrame:
    """The grid construction, threshold-generic.

    Two details that look cosmetic and are not:
      * rows are emitted arm -> regime -> threshold, and **sorted within each block by lift
        descending**, which is why the row order is not alphabetical;
      * `delta_unadapted_*` is NaN for a reference regime by construction, not because the
        value is missing. A reference is not a transfer regime and has no delta.

    AN ABSENT CELL AND A BLANK ONE ARE DIFFERENT THINGS. A (family, regime) the battery never
    ran produces no row at all — the regime is not applicable to that family. A row that EXISTS
    with a blank `auc_mean` is an incomplete recalibration cell: notebook 02 attempted it,
    could not estimate every seed, and blanked the across-seed values. It is emitted with
    `estimability_status='incomplete'` rather than dropped, because dropping it would make an
    attempted-and-unestimable cell indistinguishable from a regime that does not apply.
    """
    fams = list(families) if families is not None else list(GRID_FAMS)

    def sm(fam, arm, thr, reg, col):
        r = summary[(summary.family == fam) & (summary.arm == arm)
                    & (summary.threshold == thr) & (summary.regime == reg)]
        return (float(r[col].iloc[0]) if len(r) and col in r.columns
                and pd.notna(r[col].iloc[0]) else np.nan)

    def has_row(fam, arm, thr, reg):
        return bool(len(summary[(summary.family == fam) & (summary.arm == arm)
                                & (summary.threshold == thr) & (summary.regime == reg)]))

    rows = []
    for arm in arms:
        for reg in regimes:
            for thr in thresholds:
                block = []
                for fam in fams:
                    if not has_row(fam, arm, thr, reg):
                        continue                      # not applicable to this family
                    au = sm(fam, arm, thr, reg, "auc_mean")
                    pr = sm(fam, arm, thr, reg, "prauc_mean")
                    nl = sm(fam, arm, thr, reg, "prevalence")
                    if reg in GRID_REFERENCE:
                        dna = dnp = np.nan
                    else:
                        dna = au - sm(fam, arm, thr, "unadapted", "auc_mean")
                        dnp = pr - sm(fam, arm, thr, "unadapted", "prauc_mean")
                    block.append(dict(
                        arm=arm, threshold=thr,
                        label_use_class=GRID_CLASS.get(reg, GROUP_OF_REGIME.get(reg, "OTHER")),
                        reporting_group=GROUP_OF_REGIME.get(reg, "unspecified"),
                        family=fam, regime=reg,
                        auc=(np.nan if au != au else round(au, 4)),
                        prauc=(np.nan if pr != pr else round(pr, 4)),
                        null=(np.nan if nl != nl else round(nl, 4)),
                        lift=(np.nan if (pr != pr or nl != nl or not nl) else round(pr / nl, 4)),
                        delta_unadapted_auc=(np.nan if dna != dna else round(dna, 4)),
                        delta_unadapted_prauc=(np.nan if dnp != dnp else round(dnp, 4)),
                        estimability_status=("estimable" if au == au else "incomplete")))
                # NaN sorts last, so an incomplete cell falls to the foot of its block.
                block.sort(key=lambda d: (d["lift"] != d["lift"], -(d["lift"] or 0)))
                rows.extend(block)
    return pd.DataFrame(rows, columns=["arm", "threshold", "label_use_class", "reporting_group",
                                       "family", "regime", "auc", "prauc", "null", "lift",
                                       "delta_unadapted_auc", "delta_unadapted_prauc",
                                       "estimability_status"])


def regime_grid_coverage(grid: pd.DataFrame, *, thresholds: Sequence[str],
                         regimes: Sequence[str],
                         families: Sequence[str] | None = None) -> pd.DataFrame:
    """What each requested regime actually produced: families, thresholds and estimability.

    An empty requested block is a row saying so, not an absence. Nothing is manufactured to
    make the table rectangular — a (family, regime) the battery never ran simply reports a
    smaller family count, and the reader can see which.
    """
    fams = list(families) if families is not None else list(GRID_FAMS)
    rows = []
    for reg in regimes:
        g = grid[grid["regime"] == reg]
        rows.append(dict(
            regime=reg, reporting_group=GROUP_OF_REGIME.get(reg, "unspecified"),
            families_expected=len(fams), families_present=int(g["family"].nunique()),
            thresholds_requested=len(thresholds),
            thresholds_present=int(g["threshold"].nunique()),
            thresholds_missing="; ".join(t for t in thresholds
                                         if t not in set(g["threshold"])),
            cells=int(len(g)),
            estimable=int((g["estimability_status"] == "estimable").sum()),
            incomplete=int((g["estimability_status"] == "incomplete").sum())))
    return pd.DataFrame(rows)


# The flag rates are summarised alongside the rest because notebook 03 reconciles them against
# its own capacity tables, and a quantity that is checked has to be present to check.
SUMMARY_METRICS: Sequence[str] = (
    "auc", "prauc", "prauc_lift", "brier", "ece", "cal_slope", "cal_intercept",
    "dec_precision", "dec_recall", "dec_specificity", "dec_f1", "dec_mcc", "dec_flag_rate",
    "qui_precision", "qui_recall", "qui_specificity", "qui_f1", "qui_mcc", "qui_flag_rate",
)
PCT_METRICS: Sequence[str] = ("auc", "prauc_lift", "brier", "ece", "dec_precision", "dec_recall")
SUMMARY_GROUP: Sequence[str] = (
    "family", "label", "arm", "hyperparameter_source", "regime", "threshold", "lineage", "k",
)


# The percentile columns are `.round(4)` BEFORE the merge, matching how the stored summary was
# written; rounding after would disagree with it in the fourth decimal place.
def summarise_seeds(per_seed: pd.DataFrame) -> pd.DataFrame:
    """20-seed mean, sd and 2.5/97.5 percentiles, in the s4_regime_battery_summary shape.

    `*_sd` and the `*_plo` / `*_phi` percentile columns describe SPLIT STABILITY — how the
    metric moves across the twenty overlapping splits. They are not confidence intervals for
    the mean beside them, and the splits share too many rows for them to be read as standard
    errors. The percentile columns are `.round(4)` BEFORE the merge, not after.

    S4's `auc_lo` / `auc_hi` are no longer produced. They were the max of the per-seed
    bootstrap bounds, which existed on seed 0 alone, so they placed one split's conditional
    interval beside a twenty-split mean under names that read as its interval.

    The reference-relative columns are NOT computed here: they read `mcs_internal` and
    `yrbs_local` rows in the same frame, which only makes sense for a complete battery. Use
    `add_reference_gaps` for those.

    THE OTHER SUMMARISER IS `scores.method_table`, and the two are not interchangeable. This
    one takes the whole battery at once and reproduces the published summary's shape,
    percentile columns included. That one takes one regime and carries its context columns,
    which is what the per-method grids read.
    """
    grp = list(SUMMARY_GROUP)
    agg = {}
    for m in SUMMARY_METRICS:
        agg[f"{m}_mean"] = (m, "mean"); agg[f"{m}_sd"] = (m, "std")
    # THE TARGET SEARCH IS COUNTED, NOT SAMPLED. `first` is right for `ft_mechanism`, which is
    # a property of the family; it would be wrong here, because the whole question is whether
    # the search succeeded on every seed. Three counts say it: how many seeds the cell has, how
    # many selected a configuration, and how many could not. They sum, so a shortfall is visible
    # rather than inferred from a blank mean.
    if "target_selection_status" in per_seed.columns:
        status = per_seed["target_selection_status"].astype(str)
        agg["target_selection_selected"] = ("_selected", "sum")
        agg["target_selection_non_estimable"] = ("_non_estimable", "sum")
        per_seed = per_seed.assign(
            _selected=(status == "selected").astype(int),
            _non_estimable=status.str.startswith("non_estimable").astype(int))
    summary = per_seed.groupby(grp, dropna=False).agg(
        n_seeds=("seed", "nunique"), prevalence=("prevalence", "mean"), n_test=("n_test", "mean"),
        degenerate=("degenerate", "any"), ft_mechanism=("ft_mechanism", "first"), **agg).reset_index()
    if "target_selection_selected" in summary.columns:
        # The expected count is the cell's own seed count, named so a reader does not have to
        # know that `n_seeds` doubles as the denominator of the two columns beside it.
        summary["target_selection_expected"] = summary["n_seeds"]
    for m in PCT_METRICS:
        for q, lab in ((2.5, "lo"), (97.5, "hi")):
            summary = summary.merge(per_seed.groupby(grp, dropna=False)[m].quantile(q / 100)
                                    .round(4).rename(f"{m}_p{lab}").reset_index(), on=grp, how="left")
    # BEFORE ANY READER SEES IT, and therefore before `add_reference_gaps` can read the
    # benchmark as an anchor: a cell whose target-side search failed on any seed carries no
    # across-seed benchmark, and the mean above averaged whichever seeds succeeded.
    #
    # THE DENOMINATOR IS THE RUN'S SEED COUNT, not the benchmark cell's own row count, so a
    # cell that is short a seed entirely is short rather than complete.
    return blank_incomplete_benchmark(
        summary, benchmark_complete_cells(
            per_seed, seeds_expected=per_seed["seed"].nunique()))


# The two anchors every reference-relative column is read against, and what each one is. Both are
# LOCAL references — a model developed and evaluated inside one cohort — and neither is a ceiling.
BASELINE_REGIME = "unadapted"
SOURCE_REFERENCE = "mcs_internal"
TARGET_REFERENCE = "yrbs_local"

# Why a reference-relative quantity is blank. Carried on the row rather than left as a bare NaN,
# so "the reference could not be estimated here" and "the gap is exactly zero" are not the same
# empty cell.
GAP_AVAILABLE = ""
GAP_NO_TARGET = "yrbs_local_reference_unavailable"
GAP_NO_BASELINE = "unadapted_baseline_unavailable"
GAP_ZERO = "target_resource_gap_is_zero"

# ---- the nested sensitivity's completeness rule -----------------------------
#
# NOT A HEADLINE RULE. It applies to `yrbs_resource_rich` — the off-by-default nested per-split
# target redevelopment — and to nothing else. A cell of that sensitivity is estimable only where
# its per-seed search selected a configuration on every expected seed; anywhere else, `mean`
# would average the seeds that happened to succeed and report a survivor-only figure.
#
# THE HEADLINE REFERENCES DO NOT NEED IT AND MUST NOT BE SUBJECT TO IT. `mcs_internal` and
# `yrbs_local` take one fixed configuration per cell, so every seed is configured by
# construction; a blanking rule wired to them could only ever remove a row that was complete.
# `summarise_seeds` and `scores.method_table` therefore apply this only when the frame actually
# carries the sensitivity's rows.

TARGET_SELECTION_SELECTED = "selected"

# The sensitivity regime the rule below applies to. Named separately from the anchors above so
# that a reader can see at a glance that no anchor is involved.
NESTED_TARGET_SENSITIVITY = "yrbs_resource_rich"

BENCHMARK_STATUS_COLUMN = "resource_rich_status"
BENCHMARK_COMPLETE = "complete"
BENCHMARK_INCOMPLETE = "incomplete_target_selection"
BENCHMARK_NOT_APPLICABLE = "not applicable"

# What "performance" means for the blanking: the across-seed quantities. `prevalence` and
# `n_test` are cell properties rather than results and are left alone.
BENCHMARK_BLANKED_SUFFIXES = ("_mean", "_sd", "_plo", "_phi")


def _cell_keys(frame: pd.DataFrame, keys: Sequence[str]) -> pd.Series:
    """Each row's cell as a tuple, for membership tests against a set of cells."""
    keys = list(keys)
    if frame.empty:
        return pd.Series([], dtype=object, index=frame.index)
    return pd.Series(list(map(tuple, frame[keys].to_numpy())), index=frame.index)


def benchmark_complete_cells(per_seed: pd.DataFrame, *,
                             keys: Sequence[str] = ("family", "arm", "threshold"),
                             seeds_expected: int | None = None) -> set | None:
    """The benchmark cells whose target-side search selected on EVERY expected seed.

    Returns a set of key tuples, or None when the frame carries no benchmark rows to judge —
    which the blanking function reads as "leave this frame alone" rather than "nothing is
    complete", so a frame that never held the benchmark is not silently emptied.

    `seeds_expected` defaults to the cell's own seed count. THAT DEFAULT IS NOT WHAT A CALLER
    HOLDING THE WHOLE RUN WANTS: a cell with nineteen rows that all say "selected" is short a
    seed, and judged against its own nineteen it would come out complete. Every library caller
    therefore passes the run's distinct-seed count, and the default is left for a caller that
    genuinely has only the cell.
    """
    keys = list(keys)
    needed = {"regime", "target_selection_status", "seed", *keys}
    if not needed <= set(per_seed.columns):
        return None
    rows = per_seed[per_seed["regime"] == NESTED_TARGET_SENSITIVITY]
    if rows.empty:
        return None
    counts = (
        rows.assign(
            _selected=(rows["target_selection_status"].astype(str)
                       == TARGET_SELECTION_SELECTED).astype(int))
        .groupby(keys, dropna=False)
        .agg(_selected=("_selected", "sum"), _seeds=("seed", "nunique"))
    )
    expected = counts["_seeds"] if seeds_expected is None else seeds_expected
    # `_seeds > 0` matters: a categorical key produces empty groups for combinations that never
    # occurred, and 0 selected of 0 expected would otherwise read as complete.
    complete = counts.index[counts["_selected"].eq(expected) & counts["_seeds"].gt(0)]
    return {c if isinstance(c, tuple) else (c,) for c in complete}


def blank_incomplete_benchmark(summary: pd.DataFrame, complete: set | None, *,
                               keys: Sequence[str] = ("family", "arm", "threshold")
                               ) -> pd.DataFrame:
    """Blank the benchmark's across-seed quantities wherever its cell is not complete.

    THE ROW STAYS. An incomplete family is visible with blank benchmark metrics rather than
    absent from the table, and `resource_rich_status` carries the reason. No seed count is
    added: the count is what a public output must not carry, and the status says the same thing
    without it.
    """
    if complete is None or summary.empty or "regime" not in summary.columns:
        return summary
    out = summary.copy()
    is_benchmark = out["regime"] == NESTED_TARGET_SENSITIVITY
    incomplete = is_benchmark & ~_cell_keys(out, keys).isin(complete)
    columns = [c for c in out.columns if c.endswith(BENCHMARK_BLANKED_SUFFIXES)]
    if columns:
        out.loc[incomplete, columns] = np.nan
    out[BENCHMARK_STATUS_COLUMN] = np.where(
        ~is_benchmark, BENCHMARK_NOT_APPLICABLE,
        np.where(incomplete, BENCHMARK_INCOMPLETE, BENCHMARK_COMPLETE))
    return out


def add_reference_gaps(summary: pd.DataFrame) -> pd.DataFrame:
    """Append the reference-relative AUC columns, read against the two local references.

    TWO ANCHORS, ANSWERING TWO QUESTIONS, AND THEY ARE SYMMETRIC. `mcs_internal` is what a model
    developed and evaluated inside MCS achieves without crossing a border. `yrbs_local` is what a
    model developed and evaluated inside YRBS achieves. Both configurations come from the SAME
    consensus procedure run inside each cohort's own outer-training partitions, so the distance
    between them is not confounded by one side having had a different development budget.
    NEITHER IS A CEILING: a transfer procedure exceeding one is a result, not a contradiction.

        transfer_loss         mcs_internal - this row
                              (on the `unadapted` row, the headline transfer loss)
        target_resource_gap   yrbs_local   - unadapted        (cell-level)
        adaptation_gain       this row     - unadapted
        target_gap_recovered  adaptation_gain / target_resource_gap
        target_gap_reason     why `target_gap_recovered` is blank, where it is

    RATIO OF MEANS, NOT MEAN OF RATIOS. `target_gap_recovered` divides one twenty-seed mean gain
    by one twenty-seed mean gap. Because the mean is linear, the numerator and denominator equal
    the means of the per-seed paired differences, so the seed-level and summary formulations
    agree — the estimand choice is only where the division happens. Dividing after averaging
    leaves one denominator to be near zero instead of twenty, and it is what `bootstrap_intervals`
    computes, so the two reconcile.

    NOT CLAMPED. A value above one means the procedure exceeded the target local reference; a
    negative denominator means unadapted transfer already matched or beat it. Both are results.
    Where the gap is exactly zero, or an anchor could not be estimated, the dependent columns are
    blank and `target_gap_reason` says which.

    RETIRED COLUMNS. `gap_b`, `target_attainment` and `fixed_configuration_gap` were all read
    against a YRBS-trained model under the MCS-selected configuration, which this analysis no
    longer fits; `remaining_resource_gap` and the `resource_*` family were read against the
    per-split nested search, which is now an off-by-default sensitivity and anchors nothing.
    `regime_names.RETIRED_COLUMNS` records each and what replaced it, so a frozen table's column
    is not silently read as a live one.

    Two identities hold WHENEVER THE TARGET RESOURCE GAP IS NON-ZERO: `target_gap_recovered` is 0
    on the `unadapted` row and 1 on the `yrbs_local` row.
    """
    amean = summary.set_index(["family", "arm", "threshold", "regime"])["auc_mean"]
    if amean.index.has_duplicates:
        clashing = amean.index[amean.index.duplicated()].unique().tolist()[:4]
        raise ValueError(
            f"the summary has more than one row per (family, arm, threshold, regime), e.g. "
            f"{clashing}. Every anchor lookup below would return a Series rather than a value, "
            f"so the grouping upstream has changed and the gaps cannot be formed.")

    def _anchor(r, ref):
        """One reference's AUC in this row's cell. NaN when the row is absent OR blank.

        A non-estimable reference keeps its row with a blank `auc_mean`, so both cases have to
        land in the same place — and neither may raise. A KeyError here would name a tuple and
        nothing else.
        """
        try:
            return float(amean[(r.family, r.arm, r.threshold, ref)])
        except KeyError:
            return np.nan

    def _round(value, places=4):
        return np.nan if value != value else round(float(value), places)

    def _gap(r, ref):
        return _round(_anchor(r, ref) - float(r.auc_mean))

    def _reason(r):
        if _anchor(r, BASELINE_REGIME) != _anchor(r, BASELINE_REGIME):
            return GAP_NO_BASELINE
        gap = _anchor(r, TARGET_REFERENCE) - _anchor(r, BASELINE_REGIME)
        if gap != gap:
            return GAP_NO_TARGET
        if gap == 0.0:
            return GAP_ZERO
        return GAP_AVAILABLE

    def _target_resource_gap(r):
        return _round(_anchor(r, TARGET_REFERENCE) - _anchor(r, BASELINE_REGIME))

    def _recovered(r):
        gap = _anchor(r, TARGET_REFERENCE) - _anchor(r, BASELINE_REGIME)
        if gap != gap or gap == 0.0:
            return np.nan
        return _round((float(r.auc_mean) - _anchor(r, BASELINE_REGIME)) / gap, 4)

    summary = summary.copy()
    summary["transfer_loss"] = summary.apply(lambda r: _gap(r, SOURCE_REFERENCE), axis=1)
    summary["target_resource_gap"] = summary.apply(_target_resource_gap, axis=1)
    summary["adaptation_gain"] = summary.apply(
        lambda r: _round(float(r.auc_mean) - _anchor(r, BASELINE_REGIME)), axis=1)
    summary["target_gap_recovered"] = summary.apply(_recovered, axis=1)
    summary["target_gap_reason"] = summary.apply(_reason, axis=1)
    return summary


# 1b. The reverse direction's summary
#
# THE SAME TWO STEPS AS THE FORWARD SIDE, in the same order: `summarise_reverse_transfer` is
# `summarise_seeds` for the three reverse roles, and `reverse_transfer_comparison` is
# `add_reference_gaps` — one row per cell, the reference-relative quantities appended.
#
# WHY THEY ARE SEPARATE FUNCTIONS RATHER THAN THE FORWARD PAIR REUSED. `summarise_seeds`
# aggregates `SUMMARY_METRICS`, which carries the two capacity blocks, and it puts `n_test` on
# every row. On an MCS evaluation slice `n_test` is a denominator about restricted records, and
# the capacity metrics are not part of what the reverse reading asks. What the two pairs DO
# share is the definition of every quantity: the mean and standard deviation across the twenty
# splits, and the transfer loss as a DIFFERENCE from the evaluating cohort's local reference.

REVERSE_SUMMARY_METRICS: Sequence[str] = (
    "auc", "prauc", "brier", "ece", "cal_intercept", "cal_slope",
)
REVERSE_SUMMARY_GROUP: Sequence[str] = ("family", "label", "threshold", "role")

# Why a reverse transfer loss is blank. Carried on the row rather than left as a bare NaN, so
# "the local reference could not be estimated here" is distinguishable from a loss of zero.
ATTAINMENT_AVAILABLE = ""
ATTAINMENT_NO_REFERENCE = "mcs_local_reference_not_estimated"


def summarise_reverse_transfer(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Across-seed mean and standard deviation per family, threshold and role, with counts.

    Three counts, and they sum: `seeds_expected` is how many seeds the cell was attempted on,
    `seeds_estimated` how many produced a metric, and `seeds_non_estimable` how many were
    recorded as attempted and could not be estimated. A shortfall is then visible on the row
    instead of being inferred from a blank mean, which is the treatment `summarise_seeds` gives
    the forward side's target search.

    THE MEANS ARE OVER ESTIMATED SEEDS ONLY, which is what a blank metric means and why the
    counts have to travel beside them. A cell that is complete on eight seeds and blank on
    twelve carries an eight-seed mean and says so.

    **THE STANDARD DEVIATION IS NOT A CONFIDENCE INTERVAL.** The twenty splits are overlapping
    75% draws from one cohort and share most of their rows, so the spread describes sensitivity
    to partitioning and nothing else. `summarise_seeds` states the same limitation for the
    forward battery.
    """
    missing = [c for c in (*REVERSE_SUMMARY_GROUP, "seed", "degenerate", "prevalence_null")
               if c not in per_seed.columns]
    if missing:
        raise ValueError(
            f"the per-seed reverse frame is missing column(s) {missing}; it is built by "
            f"`transfer.reverse_metric_rows` and carries all of them.")
    absent = [m for m in REVERSE_SUMMARY_METRICS if m not in per_seed.columns]
    if absent:
        raise ValueError(
            f"the per-seed reverse frame is missing metric(s) {absent}. Every reverse row "
            f"carries the whole battery, so a partial frame would summarise a different set of "
            f"quantities under the same column names.")

    grp = list(REVERSE_SUMMARY_GROUP)
    counted = per_seed.assign(
        _estimated=(~per_seed["degenerate"].astype(bool)).astype(int),
        _non_estimable=per_seed["degenerate"].astype(bool).astype(int))
    agg = {}
    for metric in REVERSE_SUMMARY_METRICS:
        agg[f"{metric}_mean"] = (metric, "mean")
        agg[f"{metric}_sd"] = (metric, "std")
    summary = counted.groupby(grp, dropna=False).agg(
        seeds_expected=("seed", "nunique"),
        seeds_estimated=("_estimated", "sum"),
        seeds_non_estimable=("_non_estimable", "sum"),
        prevalence_null=("prevalence_null", "mean"),
        **agg).reset_index()
    if not (summary["seeds_estimated"] + summary["seeds_non_estimable"]
            == summary["seeds_expected"]).all():
        raise ValueError(
            "a reverse cell's estimated and non-estimable seed counts do not add up to the "
            "seeds it was attempted on, so one seed has been counted twice or not at all.")
    return summary


def reverse_transfer_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per (family, threshold), with the two roles side by side and their distance.

        reverse_transfer_loss        mcs_local_reference - reverse_transfer        (AUC)
        reverse_transfer_prauc_loss  mcs_local_reference - reverse_transfer        (PR-AUC)

    THE EXACT MIRROR OF THE FORWARD `transfer_loss`. Forward, a model developed in MCS is read
    against the MCS local reference; here a model developed in YRBS is read against the MCS local
    reference, on the same held-out MCS records. Both cohorts' configurations come from the same
    consensus procedure run inside their own outer-training partitions, so the two directions are
    comparable without a caveat about which side had a per-split search.

    **THE LOSS IS A DIFFERENCE AND NOT A RATIO.** `reverse / reference` reads as a proportion of
    something retained and is not one: neither AUC is anchored at zero, so the quotient of two
    AUCs has no interpretation as a fraction of attainable performance.

    NO CHANCE-ANCHORED ATTAINMENT IS REPORTED. The forward side's `target_attainment` is retired
    — its numerator was anchored at chance rather than at the transferred model, so a model that
    adapted nothing still scored highly on it and it was read as a recovery when it was not.
    The reverse reading does not reintroduce it.

    THE PR-AUC COMPARISON IS THE DIFFERENCE ALONE, for the same reason it always was: PR-AUC's
    chance level is the prevalence rather than a constant, so a chance-corrected version would be
    an invention rather than a translation.

    NEITHER ROLE IS A CEILING. A reverse-transfer AUC above the local reference is a result.
    """
    import regime_names as RN

    role_column = summary.set_index(["family", "threshold", "role"])
    if role_column.index.has_duplicates:
        clashing = role_column.index[role_column.index.duplicated()].unique().tolist()[:4]
        raise ValueError(
            f"the reverse summary has more than one row per (family, threshold, role), e.g. "
            f"{clashing}, so a role lookup would return a Series rather than a value.")

    def _value(family, threshold, role, column):
        """One role's summarised value in this cell, or NaN where the role has none."""
        try:
            return float(role_column.loc[(family, threshold, role), column])
        except KeyError:
            return np.nan

    def _round(value, places=4):
        return np.nan if value != value else round(float(value), places)

    wide, cells = [], summary[["family", "label", "threshold"]].drop_duplicates()
    for cell in cells.itertuples(index=False):
        row = {"family": cell.family, "label": cell.label, "threshold": cell.threshold}
        for role in RN.REVERSE_ROLES:
            for column in ("auc_mean", "auc_sd", "prauc_mean", "prauc_sd", "brier_mean",
                           "brier_sd", "ece_mean", "ece_sd", "cal_intercept_mean",
                           "cal_intercept_sd", "cal_slope_mean", "cal_slope_sd",
                           "seeds_expected", "seeds_estimated", "seeds_non_estimable"):
                row[f"{role}_{column}"] = _value(cell.family, cell.threshold, role, column)

        reverse = row["reverse_transfer_auc_mean"]
        local = row["mcs_local_reference_auc_mean"]
        row["reverse_transfer_loss"] = _round(local - reverse)
        row["reverse_transfer_prauc_loss"] = _round(
            row["mcs_local_reference_prauc_mean"] - row["reverse_transfer_prauc_mean"])
        row["reverse_transfer_loss_reason"] = (
            ATTAINMENT_NO_REFERENCE if (local != local or reverse != reverse)
            else ATTAINMENT_AVAILABLE)
        wide.append(row)

    out = pd.DataFrame(wide)
    order = {family: position for position, family in enumerate(RN.FAMILIES)}
    return (out.assign(_family=out["family"].map(order))
            .sort_values(["threshold", "_family"]).drop(columns="_family")
            .reset_index(drop=True))


# 2. Per-family calibration
# 2a. Lineage — decision 2
# THE TWO LINEAGES, and why every function below has to say which one it means.
#
# This pipeline has TWO result stores and neither contains the other:
#
#   "s4"      CANONICAL.csv / regime_battery_summary.csv — one row per
#             family x regime x arm x threshold, produced by the transfer battery.
#             A CELL IS FIXED AT k = 500.
#   "budget"  all_results.csv — one row per family x regime x threshold x k x metric,
#             produced by the nine analysis/*.py scripts. A CELL IS A POINT ON A k-CURVE,
#             so the frame has a `k` column that the s4 frame does not.
#
# `all_results.csv` holds NO s4 rows at all. The paper puts Table II
# and Table III side by side and they come from tables that share no rows — the only join is
# 72 hand-typed values. Making the lineage visible at the call site is the fix for the thing
# the paper currently gets wrong.
#
# DECISION 2: `lineage` is a REQUIRED keyword with NO DEFAULT wherever the
# INPUTS differ, and a pair of COLUMNS (`lineage`, `source`) wherever only the ROWS differ.
#
# No default, deliberately. A default would make `per_family_calibration()` silently return
# the s4 frame while Table III came from the other one — which is exactly how the paper ended
# up with two evidence bases and no join between them. Being made to type `lineage="budget"`
# costs one keyword and buys a call site that says what it means.
Lineage = Literal["s4", "budget"]

LINEAGE_SOURCE: Mapping[str, str] = {
    "s4": "$THESIS_WORK_DIR/tables/regime_battery_summary.csv",
    "budget": "outputs/<analysis>.csv",
}


def _threshold_labels(thresholds) -> set:
    """`1`, `(1, 2)` and `">=2"` all name thresholds. Return the stored labels for any of them.

    The result files spell a threshold `">=1"`. Every function here takes the integer, because
    that is what a caller has, and converts once — here rather than in nine places, so a
    caller that passes a bare int and one that passes a sequence cannot diverge.
    """
    if isinstance(thresholds, (int, np.integer, str)):
        thresholds = (thresholds,)
    out = set()
    for t in thresholds:
        s = str(t)
        out.add(s if s.startswith(">=") else f">={int(t)}")
    return out


def _require_lineage(lineage) -> str:
    """Reject a missing or unknown lineage with a message that says what to pass."""
    if lineage not in ("s4", "budget"):
        raise ValueError(
            f"lineage must be 's4' or 'budget', got {lineage!r}. There is no default: the two "
            f"result stores share no rows and differ in shape (the budget frame has a `k` "
            f"column, the s4 frame does not), so guessing one would silently answer a "
            f"different question. 's4' = the regime battery at k=500 "
            f"({LINEAGE_SOURCE['s4']}); 'budget' = the analysis/ k-curve lineage "
            f"({LINEAGE_SOURCE['budget']}).")
    return lineage


# The two lineages read different files: $THESIS_WORK_DIR/per_family_calibration.csv for the
# budget lineage, the calibration columns of s4_regime_battery_summary.csv for s4.
# THE TWO LINEAGES ARE NOT INTERCHANGEABLE, and the column sets say so: the budget lineage
# carries `k` and covers three families after alias normalisation; the s4 lineage carries no
# `k` and covers all nine. Asking for one and reading the other is the mistake `_require_lineage`
# exists to prevent.
def per_family_calibration(*, lineage: Lineage, thresholds, outputs_dir=None, families=None,
                           regimes=("unadapted", "target_only", "full_revision"),
                           frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Calibration and operational metrics per family. Both lineages; say which.

    `families=None` means all nine. The battery lineage has nine; the budget lineage has three,
    and asking it for nine returns three rather than padding six families of empty rows.

    THE TWO ARE NOT INTERCHANGEABLE, and the differences are larger than they look:

      `lineage="budget"`  reads `$THESIS_WORK_DIR/per_family_calibration.csv` — 198 rows of
                          `all_results.csv`, THREE families (L1_LR, RF, XGB), the tuned arm,
                          at k=500. Its metric definitions are the reference the label-budget
                          curve's k=500 column reconciles against, cell for cell.
      `lineage="s4"`      reads the calibration columns of
                          `$THESIS_WORK_DIR/tables/regime_battery_summary.csv` — NINE families,
                          BOTH arms, every regime. Different families, different arms, and no
                          `k` column.

    AND TABLE III USES NEITHER. The paper's Table III comes from
    `$THESIS_WORK_DIR/label_budget_curve.csv`. So "the per-family calibration numbers" names three
    different frames depending on who is asking, which is the whole reason `lineage` is
    required here.

    The returned frames DIFFER IN SHAPE — the budget frame has no `arm` column (it is tuned
    throughout) and the s4 frame has no `k` column. Both carry `lineage` and `source` columns
    so a concatenation of the two is still readable, but the caller has to branch. The
    signature cannot hide that and does not try to.
    """
    from pathlib import Path
    import regime_names as RN
    lineage = _require_lineage(lineage)
    outputs_dir = Path(outputs_dir) if outputs_dir is not None else _work_root()
    thr = _threshold_labels(thresholds)
    families = RN.resolve_families(families)
    if lineage == "budget":
        df = frame if frame is not None else pd.read_csv(outputs_dir / "per_family_calibration.csv")
        # THE BUDGET LINEAGE SPELLS FAMILIES DIFFERENTLY. `analysis/*.csv` carries
        # `l1_lr` / `random_forest` / `xgboost` / `catboost`; the S4 lineage carries
        # `L1_LR` / `RF` / `XGB` / `CatBoost`. That is the reason consolidate_results.py has
        # a FAM_ALIAS table at all. Normalising here means the caller passes ONE vocabulary
        # (S4's) whichever lineage it asks for, and the returned `family` column is canonical.
        df = df.copy()
        df["family"] = df["family"].map(_famkey)
        out = df[df["family"].isin(families) & df["regime"].isin(regimes)
                 & df["threshold"].isin(thr)].copy()
        out["arm"] = "tuned"
        out["k"] = 500
        out["lineage"] = "budget"
        out["source"] = "per_family_calibration.csv"
        return out.reset_index(drop=True)
    df = frame if frame is not None else pd.read_csv(
        outputs_dir / "tables" / "regime_battery_summary.csv")
    keep = df[df["family"].isin(families) & df["threshold"].isin(thr)].copy()
    metrics_ = ("brier", "ece", "cal_slope", "cal_intercept", "auc", "prauc")
    rows = []
    for _, r in keep.iterrows():
        for m in metrics_:
            if f"{m}_mean" not in r:
                continue
            rows.append(dict(family=r["family"], regime=r["regime"], arm=r["arm"],
                             threshold=r["threshold"], metric=m,
                             mean=r.get(f"{m}_mean"), sd=r.get(f"{m}_sd"),
                             n_seeds=r.get("n_seeds"), lineage="s4",
                             source="regime_battery_summary.csv"))
    return pd.DataFrame(rows)


# 3. Conformal prediction
def conformal_cell_audit(threshold: Threshold, *, tables_dir=None) -> pd.DataFrame:
    """Mondrian cell sizes BEFORE any coverage number is computed. No model fitting.

    Flags any calibration cell with n < 50 (and n < 20 positives). Run this first: a
    coverage guarantee on a starved cell is not a guarantee, and at >=2 the lower
    prevalence shrinks the calibration cells further.

    THE TWO THRESHOLDS COME FROM DIFFERENT SCRIPTS AND THE FRAMES DIFFER IN SHAPE. That is a
    property of the repository, not something this function can paper over:

      >=1  `$THESIS_WORK_DIR/tables/conformal_prereq_cells.csv` from `cell_eligibility.py` —
           sex and ethnicity as separate columns, three slices
           (calibration / full_analytic / test), `flag_lt50`. NO positives column, so the
           "n < 20 positives" half of the audit cannot be answered at >=1 from this file.
      >=2  `$THESIS_WORK_DIR/tables/e7e_cell_counts_geq2.csv` from `cell_audit_geq2.py` —
           one combined `cell`, two slices (cal / test), and `pos_mean` / `pos_min`, with a
           three-level `flag` (ok / SUGGESTIVE / UNINTERPRETABLE) rather than a boolean.

    A `cell` column and a `threshold` column are added to both so the two can be
    concatenated and still read, and the columns each file uniquely carries are left alone.
    Nothing is invented to make the shapes match.
    """
    tables_dir = Path(tables_dir) if tables_dir is not None else _work_tables()
    thr = int(threshold)
    if thr not in (1, 2):
        raise ValueError(f"conformal_cell_audit: threshold must be 1 or 2, not {threshold!r}.")
    name = "conformal_prereq_cells.csv" if thr == 1 else "e7e_cell_counts_geq2.csv"
    path = tables_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is absent. The conformal cell audit reads an archived input that this "
            f"pipeline does not produce, so it cannot be rebuilt here.")
    df = pd.read_csv(path)
    if "cell" not in df.columns:
        df.insert(1, "cell", df["sex"].astype(str) + "|" + df["ethnicity"].astype(str))
    if "flag" not in df.columns:
        # normalise the >=1 boolean onto the >=2 file's three-level vocabulary, so a caller
        # can filter on one column; the original flag_lt50 column is left in place.
        df["flag"] = np.where(df["flag_lt50"].astype(bool), "below_floor", "ok")
    df.insert(0, "threshold", f">={thr}")
    return df.reset_index(drop=True)


GRID_CONFIGS = ("1_MCS_internal", "2_naive_transfer", "2b_source_calibrated", "3_finetuned_k500")
CONFORMAL_MIN_CAL_N = 50


def _conformal_grid(tables_dir=None, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """The long-format nine-family conformal grid, read from its archived file.

    Pass `frame=` to supply it directly. There is no live producer: the grid is an archived
    input this pipeline does not rebuild.
    """
    if frame is not None:
        return frame.copy()
    tables_dir = Path(tables_dir) if tables_dir is not None else _work_tables()
    path = tables_dir / "conformal_full_grid.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is absent. It is an archived input that this pipeline does not produce, "
            f"so the nine-family conformal grid cannot be rebuilt here.")
    return pd.read_csv(path)


def _conformal_grid_cells(*, config, family, threshold, alpha=0.10, tables_dir=None,
                          frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """One (config, family, threshold) slice of the grid, averaged over seeds, per cell."""
    if abs(float(alpha) - 0.10) > 1e-9:
        raise ValueError(
            f"alpha={alpha} but the grid was computed at a 90% target (alpha=0.10) — the "
            f"target is fixed when the conformal quantile is taken, so it cannot be changed "
            f"after the fact. A different alpha needs the grid recomputed, which this "
            f"pipeline does not do.")
    if config == "2c_weighted":
        raise ValueError(
            "2c_weighted is not in conformal_full_grid.csv, which carries "
            f"{list(GRID_CONFIGS)}. Density-ratio weighted conformal is CatBoost-only and "
            "lives in $THESIS_WORK_DIR/tables/conformal_weighted_summary.csv, which is an "
            "archived input rather than something this pipeline produces.")
    if config not in GRID_CONFIGS:
        raise ValueError(f"config must be one of {list(GRID_CONFIGS)}, not {config!r}.")
    g = _conformal_grid(tables_dir, frame)
    thr = f">={int(threshold)}"
    sel = g[(g.regime == config) & (g.family == family) & (g.threshold == thr)
            & (g.cell != "__meta__")]
    if sel.empty:
        raise ValueError(
            f"no rows for config={config!r} family={family!r} threshold={thr}. The grid "
            f"holds families {sorted(g.family.unique())} at thresholds "
            f"{sorted(g.threshold.unique())}.")
    wide = (sel.pivot_table(index=["family", "threshold", "regime", "arm", "lineage", "cell"],
                            columns="metric", values="value", aggfunc="mean")
               .reset_index())
    wide.columns.name = None
    if "cal_n" in wide.columns:
        wide["suppressed"] = wide["cal_n"] < CONFORMAL_MIN_CAL_N
    return wide.sort_values("cell").reset_index(drop=True)


def mondrian_conformal(config: str, family: str = "CatBoost", threshold: Threshold = 1,
                       alpha: float = 0.10, *, tables_dir=None,
                       frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mondrian split-conformal cells, read from the conformal grid.

    Per sex x ethnicity cell, per configuration: empirical coverage at the 90% target,
    abstention rate and singleton precision. Isotonic recalibration is applied before the
    conformal scores, and the finite-sample quantile is ceil((n_g + 1)(1 - alpha)) within
    each cell. A cell calibrated on fewer than `CONFORMAL_MIN_CAL_N` rows is flagged
    suppressed rather than reported with an unstable guarantee.

    The configurations are `1_MCS_internal`, `2_naive_transfer`, `2b_source_calibrated`,
    `2c_weighted` and `3_fine_tuned`.

    The frozen conformal CatBoost was fitted with `thread_count=-1` and is therefore not
    bit-reproducible; comparisons against it need a wider tolerance than the rest of the
    battery.
    """
    return _conformal_grid_cells(config=config, family=family, threshold=threshold,
                                 alpha=alpha, tables_dir=tables_dir, frame=frame)


def conformal_per_family(families=("CatBoost", "L1_LR", "RF"), threshold: Threshold = 1, *,
                         scope: Literal["marginal", "cells", "both"] = "both",
                         tables_dir=None, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """The conformal battery across families, from the nine-family grid.

    Uses the conformal protocol's own split structure rather than `build_splits` — the two are
    different splits, and using the battery's would change the test set and break the
    reproduction gate. `CONF_GRID_SPLITS=conformal` is the mode that does this, and it is the
    default. The choice is settled, not open.

    THE DEFAULT IS THE ORIGINAL THREE FAMILIES, AND THAT IS NOW A CHOICE RATHER THAN A LIMIT.
    `$THESIS_WORK_DIR/conformal_per_family.csv` covered catboost / l1_lr / random_forest at >=1 only.
    The grid covers all nine at both thresholds, so `families=None` returns nine. The default
    stays at three because that is the scope the published numbers were computed at, and a
    caller asking for "the conformal battery" should get what the paper reports unless it
    asks otherwise.

    The grid reproduces the frozen three-family file exactly — CatBoost >=1 marginal
    coverage 0.9089 against a frozen 0.9089, and likewise across every config at both
    thresholds — so the two are interchangeable where they overlap.
    """
    import regime_names as RN
    g = _conformal_grid(tables_dir, frame)
    fams = sorted(g.family.unique()) if families is None else list(RN.resolve_families(families))
    thr = f">={int(threshold)}"
    sel = g[(g.family.isin(fams)) & (g.threshold == thr) & (g.cell != "__meta__")]
    if scope == "marginal":
        sel = sel[sel.cell == "marginal"]
    elif scope == "cells":
        sel = sel[sel.cell != "marginal"]
    wide = (sel.pivot_table(index=["family", "threshold", "regime", "arm", "lineage", "cell"],
                            columns="metric", values="value", aggfunc="mean").reset_index())
    wide.columns.name = None
    if "cal_n" in wide.columns:
        wide["suppressed"] = wide["cal_n"] < CONFORMAL_MIN_CAL_N
    return wide.sort_values(["family", "regime", "cell"]).reset_index(drop=True)


def conformal_threshold2(family: str = "CatBoost", *, tables_dir=None,
                         frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """The conformal battery at both thresholds side by side, so the comparison is one frame.

    Both thresholds live in the one frame so the comparison is not split across artefacts.
    The lower prevalence at >=2 starves the calibration cells, which is why the `suppressed`
    flag matters more here than at >=1 — read it, do not average over it.
    """
    frames = [conformal_per_family(families=(family,), threshold=t, tables_dir=tables_dir,
                                   frame=frame) for t in (1, 2)]
    return pd.concat(frames, ignore_index=True)


def conformal_mcs_marginal(*, tables_dir=None, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """MCS config-1 coverage at the three MARGINAL cell definitions the panels use.

    Exists because MCS conformal scores are not persisted (Tier 1a) and marginal coverage
    cannot be reconstructed from intersectional aggregates — the conformal quantile is
    defined per cell. Resolves the inconsistency where sex x ethnicity Black cells are
    suppressed in the panels but reported in conformal_coverage.csv.

    Three panels: `1_sex`, `2_binary` (White / non-White) and `3_ethnicity`. Counts are
    SDC-rounded to the nearest ten at source, which is why `cal_n_mean` reads 1020 rather
    than an exact figure.
    """
    if frame is not None:
        return frame.copy()
    tables_dir = Path(tables_dir) if tables_dir is not None else _work_tables()
    path = tables_dir / "conformal_mcs_marginal.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is absent. It is an archived input that this pipeline does not produce.")
    return pd.read_csv(path)


# 4. Subgroup fairness
# Reads $THESIS_WORK_DIR/tables/subgroup_discrimination_summary.csv, an archived input.
# `calibration_valid` is False on some rows at >=1 and none at >=2, which is not an anomaly:
# it follows from the scope note below, that conformal configs 2 and 3 exist at >=1 only.
def subgroup_discrimination(families=None, thresholds=(1, 2), *, tables_dir=None,
                            frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per sex x ethnicity cell discrimination on the persisted YRBS scores. No refits.

    `families=None` means all nine and `thresholds=(1, 2)` means both. The battery ran the
    nine, so this returns the nine; a family it did not run would be absent rather than
    fabricated, and `subgroup_coverage()` is the record of which is which.

    DECISION 2: this is an S4-LINEAGE-ONLY function, so it takes NO `lineage`
    keyword. What it gets instead is a pair of COLUMNS — `lineage` and `source` — on the
    returned frame, because only the ROWS differ between what this produces and what a
    caller might concatenate it with, not the inputs. A keyword here would offer a choice
    that does not exist.

    TWO CAVEATS THAT MUST SURVIVE CONSOLIDATION, both carried on the frame rather than in a
    docstring a caller will not read:

      * **Conformal configs 2 and 3 store RAW PRE-ISOTONIC scores.** AUC, PR-AUC and lift are
        rank metrics and stay valid, because isotonic regression is monotone. Brier, ECE and
        calibration intercept do NOT — they describe the raw model, not the
        conformal-calibrated one that is actually deployed. The source computes them into the
        per-seed CSV and then marks them `calibration_valid=False` in the summary. That flag
        is passed through here untouched. A summary that averaged over it would report the
        calibration of a model nobody uses.
      * **Conformal exists at >=1 only.** The >=2 configs-2/3 cells are emitted as
        scope-present with `metric=NaN` and an explicit note — never silently blank. An
        absent row reads as "no disparity found"; a NaN row with a note reads as "not run".

    `source` distinguishes measured rows from anything a caller appends. Every row this
    function returns is measured, so it is constant here; it exists so that a concatenation
    with `subgroup_panels` (whose MCS side is REGENERATED, not measured) stays legible.
    """
    from pathlib import Path
    tables_dir = Path(tables_dir) if tables_dir is not None else _work_tables()
    from inputs import resolve as _resolve
    df = frame if frame is not None else pd.read_csv(
        _resolve(tables_dir / "subgroup_discrimination_summary.csv"))
    out = df[df["threshold"].isin(_threshold_labels(thresholds))].copy()
    if families is not None:
        import regime_names as RN
        out = out[out["family"].isin(RN.resolve_families(families))].copy()
    out["lineage"] = "s4"
    out["source"] = "measured:yrbs_scores.parquet"
    if "calibration_valid" in out.columns:
        invalid = int((~out["calibration_valid"].astype(bool)).sum())
        if invalid:
            out.loc[~out["calibration_valid"].astype(bool), "source"] = \
                "measured:yrbs_scores.parquet (RAW pre-isotonic — calibration columns "
            out.loc[~out["calibration_valid"].astype(bool), "source"] += \
                "describe the raw model, not the conformal one)"
    return out.reset_index(drop=True)


# Reads $THESIS_WORK_DIR/tables/subgroup_panels_summary.csv, an archived input.
# THE DERIVED `source` COLUMN SPLITS ON COHORT, and that is the point of it: every MCS row is
# marked regenerated and no YRBS row is, so a reader can tell a measured panel from a
# reconstructed one without knowing which file it came from.
def subgroup_panels(families=None, thresholds=(1, 2), *, tables_dir=None,
                    frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Marginal subgroup panels for BOTH cohorts.

    `families=None` means all nine, `thresholds=(1, 2)` means both.

    Panels: 1 sex; 2 White vs non-White; 3 ethnicity-only (MCS 4-way ONS
    White/Asian/Black/Mixed-or-Other; YRBS 4-way White/Black/Hispanic/Other). Panel 4
    (sex x ethnicity intersectional) is YRBS-only — the MCS cells suppress.

    WITHIN-COHORT ONLY. There is no ethnicity crosswalk between MCS and YRBS and no
    cross-cohort cell comparison is licensed. Sex is the only harmonised attribute, and even
    there the comparison is of PATTERN — gradient direction and magnitude — not of level.
    Reading "MCS Black 0.71 vs YRBS Black 0.68" as a difference is the specific error this
    warning exists to prevent: the two categories are not the same category.

    DECISION 2: S4 lineage only, so no `lineage` keyword — but the frame carries
    `lineage` and `source` COLUMNS, and here the `source` column is doing real work rather
    than being bookkeeping.

    **THE MCS ROWS ARE REGENERATED, THE YRBS ROWS ARE MEASURED.** MCS scores are not persisted
    under the Tier 1a licence, so the MCS side is reproduced by DETERMINISTIC REFIT of the
    frozen configs while the YRBS side is read from the score parquet. Both are legitimate
    and they are not the same kind of number: a refit reproduces the model exactly but is
    contingent on the library versions still behaving identically, and CatBoost with
    `thread_count=-1` is documented as not bit-reproducible. The `source` column keeps the two
    apart, because a frame that mixed them with no column saying which is unreadable:

        measured:subgroup_scores.parquet        YRBS, read from persisted scores
        regenerated:deterministic-refit         MCS, refitted because scores cannot be kept
    """
    from pathlib import Path
    tables_dir = Path(tables_dir) if tables_dir is not None else _work_tables()
    df = frame if frame is not None else pd.read_csv(tables_dir / "subgroup_panels_summary.csv")
    out = df[df["threshold"].isin(_threshold_labels(thresholds))].copy()
    if families is not None:
        import regime_names as RN
        keep = RN.resolve_families(families)
        out = out[out["family"].isin(keep) | out["family"].isna()].copy()
    out["lineage"] = "s4"
    out["source"] = np.where(out["cohort"].astype(str).str.upper() == "MCS",
                             "regenerated:deterministic-refit",
                             "measured:subgroup_scores.parquet")
    return out.reset_index(drop=True)


# Two archived inputs: $THESIS_WORK_DIR/subgroup_calibration.csv (budget lineage) and
# $THESIS_WORK_DIR/tables/subgroup_panels_summary.csv (s4).
# The s4 lineage projects the panels summary into long form across the panels
# {1_sex, 2_binary, 3_ethnicity, marginal}; it has no stored long-form counterpart, so only its
# shape can be checked, not its values.
def subgroup_calibration(*, lineage: Lineage, outputs_dir=None,
                         families=None, thresholds=(1, 2),
                         frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Calibration stratified by cell. Both lineages; say which.

    `families=None` means all nine — but ONLY THE S4 LINEAGE HAS NINE. The budget lineage's
    `$THESIS_WORK_DIR/subgroup_calibration.csv` covers catboost, l1_lr and random_forest and nothing
    else, so asking it for all nine returns three. It does not raise and it does not pad:
    the three that exist come back and `subgroup_coverage()` names the six that do not. A
    padded frame of NaN rows would be indistinguishable from six cells that calibrated
    perfectly badly.

      `lineage="budget"`  `$THESIS_WORK_DIR/subgroup_calibration.csv` — 864 rows of
                          `all_results.csv`, the second largest single source in that file.
      `lineage="s4"`      `$THESIS_WORK_DIR/tables/subgroup_panels_summary.csv` — the S4-lineage panels,
                          which cover BOTH cohorts and a different cell definition.

    THE CELL DEFINITIONS ARE NOT THE SAME OBJECT. The budget frame's `cell` is the YRBS
    sex x ethnicity intersection; the panels frame carries `panel` as well as `cell`, because
    it spans four different stratifications (sex; White vs non-White; ethnicity-only;
    intersectional). Joining the two on `cell` alone would silently match a marginal cell to
    an intersectional one. The returned frame keeps `panel` where the source has it.
    """
    from pathlib import Path
    import regime_names as RN
    lineage = _require_lineage(lineage)
    outputs_dir = Path(outputs_dir) if outputs_dir is not None else _work_root()
    thr = _threshold_labels(thresholds)
    families = RN.resolve_families(families)
    if lineage == "budget":
        df = frame if frame is not None else pd.read_csv(outputs_dir / "subgroup_calibration.csv")
        df = df.copy()
        df["family"] = df["family"].map(_famkey)   # see per_family_calibration on the aliases
        out = df[df["family"].isin(families) & df["threshold"].isin(thr)].copy()
        out["panel"] = "4_intersectional"
        out["lineage"] = "budget"
        out["source"] = "subgroup_calibration.csv"
        return out.reset_index(drop=True)
    df = frame if frame is not None else pd.read_csv(
        outputs_dir / "tables" / "subgroup_panels_summary.csv")
    keep = df[df["threshold"].isin(thr)].copy()
    if "family" in keep.columns:
        keep = keep[keep["family"].isin(families) | keep["family"].isna()]
    rows = []
    for _, r in keep.iterrows():
        for m in ("brier", "ece", "cal_intercept", "auc", "prauc", "lift"):
            if f"{m}_mean" not in r:
                continue
            rows.append(dict(cohort=r.get("cohort"), family=r.get("family"),
                             arm=r.get("arm"), threshold=r["threshold"],
                             panel=r.get("panel"), cell=r.get("cell"), metric=m,
                             mean=r.get(f"{m}_mean"), sd=r.get(f"{m}_sd"),
                             n_seeds=r.get("n_seeds"),
                             suppressed=bool(r.get("suppressed", False)),
                             lineage="s4", source="subgroup_panels_summary.csv"))
    return pd.DataFrame(rows)


# Two archived inputs: $THESIS_WORK_DIR/subgroup_by_family.csv (budget lineage) and
# $THESIS_WORK_DIR/tables/subgroup_discrimination_summary.csv (s4).
def subgroup_by_family(*, lineage: Lineage, outputs_dir=None, families=None,
                       thresholds=(1, 2),
                       frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Is the subgroup ordering stable across families and arms? A verification of that claim.

    All nine families x both arms x both thresholds, naive transfer, read from the persisted
    scores rather than refitted. Per-cell AUC, the best-minus-worst gap, and the rank order.

    IF THE CLAIM DOES NOT HOLD, THIS MUST SAY SO. The function returns the ranks and gaps; it
    does not assert stability. `subgroup_rank_concordance` below computes the mean pairwise
    Kendall's tau between family orderings and the concordance count on the worst-served
    cell, which is the evidence for or against. A low tau is a finding, not a bug.

      `lineage="budget"`  `$THESIS_WORK_DIR/subgroup_by_family.csv` — 288 rows of `all_results.csv`.
      `lineage="s4"`      `$THESIS_WORK_DIR/tables/subgroup_discrimination_summary.csv`.
    """
    from pathlib import Path
    import regime_names as RN
    lineage = _require_lineage(lineage)
    outputs_dir = Path(outputs_dir) if outputs_dir is not None else _work_root()
    thr = _threshold_labels(thresholds)
    keep = RN.resolve_families(families)
    if lineage == "budget":
        df = frame if frame is not None else pd.read_csv(outputs_dir / "subgroup_by_family.csv")
        out = df[df["threshold"].isin(thr) & df["family"].isin(keep)].copy()
        out["lineage"] = "budget"
        out["source"] = "subgroup_by_family.csv"
        return out.reset_index(drop=True)
    from inputs import resolve as _resolve
    df = frame if frame is not None else pd.read_csv(
        _resolve(outputs_dir / "tables" / "subgroup_discrimination_summary.csv"))
    out = df[df["threshold"].isin(thr) & (df["scope"] == "cell")
             & df["family"].isin(keep)].copy()
    out["lineage"] = "s4"
    out["source"] = "subgroup_discrimination_summary.csv"
    return out.reset_index(drop=True)


def subgroup_rank_concordance(frame: pd.DataFrame, *, value_col: str = "auc_mean"
                              ) -> pd.DataFrame:
    """Mean pairwise Kendall's tau between family cell-orderings, per (arm, threshold).

    The evidence for or against "the subgroup ordering is stable across families". Reported
    with the number of family pairs it averages over, because a tau computed from two
    families is not the same claim as one computed from nine.
    """
    from itertools import combinations
    from scipy.stats import kendalltau
    out = []
    keys = [c for c in ("arm", "threshold") if c in frame.columns]
    for keyvals, g in frame.groupby(keys) if keys else [((), frame)]:
        by_fam = {fam: sub.set_index("cell")[value_col]
                  for fam, sub in g.groupby("family") if sub["cell"].is_unique}
        taus = []
        for a, b in combinations(sorted(by_fam), 2):
            common = by_fam[a].index.intersection(by_fam[b].index)
            if len(common) >= 3:
                t, _ = kendalltau(by_fam[a].loc[common], by_fam[b].loc[common])
                if t == t:
                    taus.append(t)
        worst = {fam: s.idxmin() for fam, s in by_fam.items() if len(s)}
        from collections import Counter
        wc = Counter(worst.values())
        row = dict(zip(keys, keyvals if isinstance(keyvals, tuple) else (keyvals,)))
        row.update(n_families=len(by_fam), n_pairs=len(taus),
                   mean_kendall_tau=(float(np.mean(taus)) if taus else np.nan),
                   worst_cell_modal=(wc.most_common(1)[0][0] if wc else None),
                   worst_cell_concordance=(wc.most_common(1)[0][1] if wc else 0))
        out.append(row)
    return pd.DataFrame(out)


# Reads $THESIS_WORK_DIR/tables/subgroup_operational.csv, an archived input, and it is what the
# paper's SS VI-F reports from.
# The `__marginal__` rows are excluded by design and counted separately: a marginal panel is
# the whole cohort, not a cell, and averaging it in with the cells would double-count.
CAPACITY_METRICS: Sequence[str] = ("precision15", "recall15", "fpr15", "flagrate15")


def subgroup_precision_at_capacity(families=None, thresholds=(1, 2), *, tables_dir=None,
                                   capacities=(0.15,),
                                   frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Operational fairness at a GLOBAL screening capacity, stratified by cell.

    `families=None` means all nine and `thresholds=(1, 2)` means both — but this reads the
    file the paper read, and THAT FILE IS CATBOOST ONLY. Ask for nine and one comes back.
    The eight absent families are not a bug in this function and they are not empty rows:
    they are in `subgroup_scores()`, which reads the nine-family discrimination battery, and
    the two agree exactly where they overlap (80 comparisons, max difference 0.0). Use this
    when the question is "what does §VI-F say"; use `subgroup_scores` when the question is
    "what happens for the other eight".

    The cut-off is applied ONCE across the whole cohort, not per cell, and the flagged set is
    then broken down by cell. That single global cut is the point of the analysis — per-cell
    thresholds would equalise the very disparity being measured, and would answer a different
    question.

    DECISION: FOLLOW THE PAPER. This reads `subgroup_operational.csv`, which
    is where SS VI-F's operational-fairness numbers actually come from — the FPR spread of
    0.019-0.045 and the 21% / 10% flag rates all trace there. A second file,
    `$THESIS_WORK_DIR/subgroup_precision_at_capacity.csv`, answered the same question and
    NOTHING read it: not the paper, not `all_results.csv`, not `CANONICAL.csv`. This function
    reads the one that was actually used.

    WHAT IS LOST BY THAT CHOICE, stated rather than glossed: the unused file covered
    9 families x 3 regimes x 2 thresholds at BOTH 5% and 15% capacity (1,728 rows).
    `subgroup_operational.csv` covers 15% only, CatBoost only, across its configs (54 rows,
    48 of them cells). So the 5% capacity and the eight other families are not available from
    this function, and this pipeline does not produce them.

    `capacities` therefore accepts only 0.15 today, and raises rather than silently returning
    an empty frame for 0.05. A silent empty frame is how "we did not measure this" becomes
    "there was no disparity".

    Returns one row per (config, threshold, cell, metric) with mean/sd/lo/hi and the
    `unstable` flag carried through. The `__marginal__` rows of the source are DROPPED — they
    are the whole-cohort reference, not a cell, and averaging them in with the cells would
    dilute the spread that is the finding.
    """
    from pathlib import Path
    bad = [c for c in capacities if abs(float(c) - 0.15) > 1e-9]
    if bad:
        raise ValueError(
            f"subgroup_precision_at_capacity: capacities {bad} are not available from "
            f"$THESIS_WORK_DIR/tables/subgroup_operational.csv, which carries the 15% screening "
            f"capacity only. The 5% capacity existed only in a file the paper never used, and "
            f"this pipeline does not produce it. Do not infer it.")
    import regime_names as RN
    if frame is None:
        tables_dir = Path(tables_dir) if tables_dir is not None else _work_tables()
        frame = pd.read_csv(tables_dir / "subgroup_operational.csv")
    keep = set(RN.resolve_families(families))
    cells = frame[(frame["cell"] != "__marginal__")
                  & frame["threshold"].isin(_threshold_labels(thresholds))]
    rows = []
    for _, r in cells.iterrows():
        # `config` is `naive:CatBoost:tuned` or `conformal:2_naive_transfer`. The conformal
        # configs are CatBoost by construction and carry no family in the string.
        parts = str(r["config"]).split(":")
        fam = "CatBoost"
        if len(parts) >= 2:
            try:
                fam = RN.canonical_family(parts[1])
            except ValueError:
                fam = "CatBoost"
        if fam not in keep:
            continue
        for m in CAPACITY_METRICS:
            rows.append(dict(
                family=fam, config=r["config"], threshold=r["threshold"], cell=r["cell"],
                capacity=0.15, metric=m,
                mean=r.get(f"{m}_mean"), sd=r.get(f"{m}_sd"),
                lo=r.get(f"{m}_lo"), hi=r.get(f"{m}_hi"),
                n=r.get("n_mean"), n_min=r.get("n_min"),
                prevalence=r.get("prevalence_mean"),
                unstable=bool(r.get("unstable", False)),
                lineage="s4", source="subgroup_operational.csv"))
    if not rows:
        raise ValueError(
            f"subgroup_operational.csv has no rows for {sorted(keep)}. It covers CatBoost "
            f"only — it is the CatBoost-shaped file the paper's §VI-F was read from. The "
            f"same quantities for all nine families are in "
            f"`subgroup_scores(metrics=('fpr15', 'flagrate15'))`, which reads the "
            f"discrimination battery and agrees with this file exactly on CatBoost (80 "
            f"comparisons, max difference 0.0). Raising rather than returning an empty "
            f"frame, because an empty fairness result reads as 'no disparity found'.")
    out = pd.DataFrame(rows)
    absent = sorted(keep - set(out["family"]))
    out["families_absent_from_this_file"] = ";".join(absent)
    return out

# 4a. Subgroup analysis for any family, persisted
# The paper reports subgroup fairness for CatBoost. The battery measured nine families. The
# functions below make the other eight reachable and write every score they compute, not
# only the summary — a fairness number that exists solely inside a printed summary cannot be
# rechecked, and the summary is the part most likely to be quoted.
#
# WHAT IS ACTUALLY ON DISK, per family. This is the whole reason these functions refuse to
# take a `families` list at face value:
#
#   nine families   $THESIS_WORK_DIR/tables/subgroup_discrimination_summary.csv
#                   `naive:<family>:<arm>`, 8 cells + marginal, both arms, both thresholds.
#                   Carries discrimination (auc, prauc, lift), calibration (brier, ece,
#                   cal_intercept) AND the 15 %-capacity operational columns.
#   nine families   $THESIS_WORK_DIR/tables/subgroup_panels_summary.csv — the marginal panels, both
#                   cohorts, four stratifications.
#   THREE families  $THESIS_WORK_DIR/subgroup_calibration.csv (budget lineage) — catboost, l1_lr,
#                   random_forest only.
#   THREE families  $THESIS_WORK_DIR/subgroup_precision_at_capacity.csv — same three, archived producer.
#   ONE family      $THESIS_WORK_DIR/tables/subgroup_operational.csv — CatBoost configs only. This is
#                   the file §VI-F was read from.
#
# The nine-family operational columns and the one-family file agree EXACTLY where they
# overlap — 80 cell x metric comparisons, max absolute difference 0.0
# (`diagnostics.subgroup_operational_crosscheck`). That check is what licenses drawing
# the other eight families from the discrimination battery: without it, the eight would be a
# different quantity wearing the same name.

# metric name -> the column stem it lives under in subgroup_discrimination_summary.csv
SUBGROUP_METRICS: Mapping[str, str] = {
    "auc": "auc", "prauc": "prauc", "lift": "lift",
    "brier": "brier", "ece": "ece", "cal_intercept": "cal_intercept",
    "fpr15": "fpr_op15", "tpr15": "tpr_op15", "flagrate15": "flagrate_op15",
}

# Metrics that describe CALIBRATION rather than ranking. On the conformal configs these
# describe the raw pre-isotonic scores, not the deployed model, and the source marks the row
# `calibration_valid=False`. Carried through, never averaged over.
SUBGROUP_CALIBRATION_METRICS: Sequence[str] = ("brier", "ece", "cal_intercept")

# The small-cell rule. n < 50 in the evaluation split is suppressed, and the flag survives
# into every summary and figure.
SUBGROUP_MIN_CELL_N = 50


def subgroup_scores(families=None, thresholds=(1, 2), *, metrics=None, arms=("tuned",),
                    tables_dir=None, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every subgroup score, long-format, for any family the battery actually ran.

    `families=None` means all nine; `thresholds=(1, 2)` means both. A family the battery did
    not run is NOT computed and NOT invented — it is simply absent from the returned frame,
    and `subgroup_coverage()` says so by name.

    One row per (family, arm, threshold, cell, metric) with:

        mean, sd, lo, hi        the value and its across-seed interval, or NaN if suppressed
        n_mean, n_min           the cell size, so a reader can see why something suppressed
        prevalence              the cell's own outcome rate — a PR-AUC or lift without it is
                                not interpretable, because the null differs per cell
        suppressed              True if n < 50 or the source flagged the cell unstable
        calibration_valid       False where the metric describes raw pre-isotonic scores
        lineage, source         which store the row came from and whether it was measured

    SUPPRESSED CELLS CARRY NO VALUE. `mean`, `sd`, `lo` and `hi` are NaN on a suppressed row
    and `n_mean` / `n_min` / `suppressed` are kept, so the row says "this cell exists, it is
    too small to report" rather than either disappearing or reporting an unstable number. A
    disappeared row reads as an absent group, which is the specific misreading the whole
    subgroup section exists to avoid.
    """
    from pathlib import Path
    import regime_names as RN
    fams = RN.resolve_families(families)
    metrics = tuple(SUBGROUP_METRICS) if metrics is None else tuple(metrics)
    bad = [m for m in metrics if m not in SUBGROUP_METRICS]
    if bad:
        raise ValueError(
            f"unknown subgroup metric(s) {bad}. Available: {list(SUBGROUP_METRICS)}. These "
            f"are the columns subgroup_discrimination_summary.csv actually carries; asking "
            f"for anything else would mean computing it from scores this function does not "
            f"read.")
    tables_dir = Path(tables_dir) if tables_dir is not None else _work_tables()
    from inputs import resolve as _resolve
    df = frame if frame is not None else pd.read_csv(
        _resolve(tables_dir / "subgroup_discrimination_summary.csv"))
    thr = [f">={int(t)}" for t in thresholds]
    rows = []
    for family in fams:
        configs = [f"naive:{family}:{a}" for a in arms]
        sub = df[df["config"].isin(configs) & df["threshold"].isin(thr)]
        for _, r in sub.iterrows():
            n_min = r.get("n_min")
            suppressed = bool(r.get("unstable", False)) or (
                pd.notna(n_min) and float(n_min) < SUBGROUP_MIN_CELL_N)
            cal_valid = bool(r.get("calibration_valid", True))
            for m in metrics:
                stem = SUBGROUP_METRICS[m]
                if f"{stem}_mean" not in r.index:
                    continue
                is_cal = m in SUBGROUP_CALIBRATION_METRICS
                rows.append(dict(
                    family=family, config=r["config"], arm=r.get("arm"),
                    regime=r.get("regime"), threshold=r["threshold"],
                    scope=r.get("scope"), cell=r.get("cell"),
                    sex=r.get("sex"), ethnicity=r.get("ethnicity"),
                    metric=m,
                    mean=(np.nan if suppressed else _num(r.get(f"{stem}_mean"))),
                    sd=(np.nan if suppressed else _num(r.get(f"{stem}_sd"))),
                    lo=(np.nan if suppressed else _num(r.get(f"{stem}_plo"))),
                    hi=(np.nan if suppressed else _num(r.get(f"{stem}_phi"))),
                    n_mean=_num(r.get("n_mean")), n_min=_num(n_min),
                    prevalence=_num(r.get("prevalence_mean")),
                    n_seeds=_num(r.get("n_seeds")),
                    suppressed=suppressed,
                    calibration_valid=(cal_valid if is_cal else True),
                    note=("" if cal_valid or not is_cal else
                          "raw pre-isotonic scores: this describes the uncalibrated model"),
                    lineage="s4",
                    source="measured:yrbs_scores.parquet"))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["family", "threshold", "arm", "metric", "cell"]).reset_index(drop=True)


def _num(v):
    """Float or NaN. Keeps a missing column and a missing value indistinguishable downstream,
    which is correct here — both mean "no number", and neither is a zero."""
    try:
        return float(v) if pd.notna(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def subgroup_gap_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Best-minus-worst spread per (family, arm, threshold, metric) — the §VI-F quantity.

    The paper reports this for CatBoost: AUC varies by 0.112 across cells at >=2. This
    computes the same thing for whichever families are in `scores`, so the claim can be read
    as "CatBoost is typical" or "CatBoost is the best case" rather than left as one number.

    SUPPRESSED CELLS ARE COUNTED AND EXCLUDED FROM THE SPREAD, not dropped quietly. The
    `n_suppressed` column says how many cells the spread could not see. A spread computed
    over six of eight cells is not the same claim as one computed over eight, and the small
    cells are usually the ones carrying the disparity.
    """
    if scores.empty:
        return pd.DataFrame()
    cells = scores[scores["scope"] == "cell"]
    marg = (scores[scores["scope"] == "marginal"]
            .set_index(["family", "arm", "threshold", "metric"])["mean"])
    out = []
    for key, g in cells.groupby(["family", "arm", "threshold", "metric"]):
        usable = g[g["mean"].notna()]
        row = dict(zip(("family", "arm", "threshold", "metric"), key))
        row.update(n_cells=int(len(g)), n_usable=int(len(usable)),
                   n_suppressed=int(g["suppressed"].sum()))
        if len(usable) >= 2:
            hi = usable.loc[usable["mean"].idxmax()]
            lo = usable.loc[usable["mean"].idxmin()]
            row.update(max_cell=hi["cell"], max_value=float(hi["mean"]),
                       min_cell=lo["cell"], min_value=float(lo["mean"]),
                       gap=float(hi["mean"]) - float(lo["mean"]))
        else:
            row.update(max_cell=None, max_value=np.nan, min_cell=None,
                       min_value=np.nan, gap=np.nan)
        row["marginal"] = float(marg.get(key, np.nan))
        row["lineage"] = "s4"
        row["source"] = "subgroup_discrimination_summary.csv"
        out.append(row)
    return (pd.DataFrame(out)
            .sort_values(["metric", "threshold", "family", "arm"]).reset_index(drop=True))


def subgroup_worst_cell(scores: pd.DataFrame, *, metric: str = "auc") -> pd.DataFrame:
    """Which cell is worst served, per family x arm x threshold, and how often it is the same.

    The paper's claim is that the Black male cell is worst in 29 of 36 family x arm x
    threshold combinations. This returns the 36 rows behind that count so it can be checked
    rather than taken.
    """
    g = scores[(scores["scope"] == "cell") & (scores["metric"] == metric)
               & scores["mean"].notna()]
    out = []
    for key, sub in g.groupby(["family", "arm", "threshold"]):
        w = sub.loc[sub["mean"].idxmin()]
        b = sub.loc[sub["mean"].idxmax()]
        out.append(dict(zip(("family", "arm", "threshold"), key)) | dict(
            metric=metric, worst_cell=w["cell"], worst_value=float(w["mean"]),
            best_cell=b["cell"], best_value=float(b["mean"]),
            n_cells_compared=int(len(sub)),
            lineage="s4", source="subgroup_discrimination_summary.csv"))
    return pd.DataFrame(out).sort_values(["family", "threshold", "arm"]).reset_index(drop=True)


def subgroup_coverage(*, tables_dir=None, outputs_dir=None) -> pd.DataFrame:
    """Which subgroup analysis exists for which family, and where it does not, why.

    One row per (analysis, family). A function that returns what exists must also be able to
    say what does not, or a caller cannot tell an empty result from an unmeasured one.
    """
    from pathlib import Path
    import regime_names as RN
    tables_dir = Path(tables_dir) if tables_dir is not None else _work_tables()
    outputs_dir = Path(outputs_dir) if outputs_dir is not None else _work_root()

    def _fams(path, col, transform=lambda s: s):
        if not path.exists():
            return set()
        d = pd.read_csv(path)
        if col not in d.columns:
            return set()
        vals = set()
        for v in d[col].dropna().unique():
            try:
                vals.add(RN.canonical_family(transform(str(v))))
            except ValueError:
                continue
        return vals

    disc = tables_dir / "subgroup_discrimination_summary.csv"
    have_disc = _fams(disc, "family")
    have_panels = _fams(tables_dir / "subgroup_panels_summary.csv", "family")
    have_cal = _fams(outputs_dir / "subgroup_calibration.csv", "family")
    have_cap = _fams(outputs_dir / "subgroup_precision_at_capacity.csv", "family")
    have_byfam = _fams(outputs_dir / "subgroup_by_family.csv", "family")
    # subgroup_operational.csv is keyed by config, not family: `naive:CatBoost:tuned`.
    have_oper = set()
    op = tables_dir / "subgroup_operational.csv"
    if op.exists():
        for cfg in pd.read_csv(op)["config"].dropna().unique():
            parts = str(cfg).split(":")
            if len(parts) >= 2:
                try:
                    have_oper.add(RN.canonical_family(parts[1]))
                except ValueError:
                    continue

    analyses = (
        ("discrimination", have_disc, "$THESIS_WORK_DIR/tables/subgroup_discrimination_summary.csv",
         "per-cell AUC, PR-AUC, lift, calibration and 15 %-capacity operational columns"),
        ("marginal_panels", have_panels, "$THESIS_WORK_DIR/tables/subgroup_panels_summary.csv",
         "sex / White-vs-non-White / ethnicity panels, both cohorts"),
        ("calibration_budget_lineage", have_cal, "$THESIS_WORK_DIR/subgroup_calibration.csv",
         "per-cell calibration in the analysis/ lineage"),
        ("precision_at_capacity", have_cap, "$THESIS_WORK_DIR/subgroup_precision_at_capacity.csv",
         "5 % and 15 % capacity, from the archived producer"),
        ("operational_paper_source", have_oper, "$THESIS_WORK_DIR/tables/subgroup_operational.csv",
         "the file §VI-F's flag-rate and FPR numbers were read from"),
        ("by_family_budget_lineage", have_byfam, "$THESIS_WORK_DIR/subgroup_by_family.csv",
         "per-cell AUC and rank in the analysis/ lineage"),
    )
    rows = []
    for name, have, path, what in analyses:
        for family in RN.FAMILIES:
            ok = family in have
            rows.append(dict(
                analysis=name, family=family, available=ok, source=path, covers=what,
                gap=("" if ok else
                     f"{path} has no rows for {family}; it covers "
                     f"{sorted(have) if have else 'nothing on disk'}")))
    return pd.DataFrame(rows)


# SIDE EFFECTS: READS the frozen subgroup tables and WRITES one CSV per family
# under `out_root`. The only writer in this module.
def write_subgroup_by_model(families=None, thresholds=(1, 2), *, tables_dir=None,
                            out_root=None, metrics=None) -> pd.DataFrame:
    """Persist every subgroup score per family, one file per metric per threshold.

        <work>/diagnostics/tables/<family>/subgroup_<metric>_ge<threshold>.csv
        <work>/diagnostics/tables/<family>/subgroup_gap_summary.csv
        <work>/diagnostics/tables/<family>/subgroup_worst_cell.csv
        <work>/diagnostics/tables/subgroup_coverage.csv

    Every file carries `lineage` and `source`. Every suppressed cell is written as a row with
    its size and no value.

    Returns a manifest of what was written, and the coverage frame is written whether or not
    any family was asked for — the gap record is the part that must not depend on the call.
    """
    from pathlib import Path
    import regime_names as RN
    # per-family subgroup tables are diagnostics; they go to the working root
    import config as _C
    root = Path(out_root) if out_root is not None else Path(_C.work_path("diagnostics", "tables"))
    scores = subgroup_scores(families, thresholds, metrics=metrics, tables_dir=tables_dir)
    coverage = subgroup_coverage(tables_dir=tables_dir)
    root.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(root / "subgroup_coverage.csv", index=False)
    manifest = []
    for family in RN.resolve_families(families):
        fam_rows = scores[scores["family"] == family]
        d = root / family
        if fam_rows.empty:
            manifest.append(dict(family=family, file="", n_rows=0, written=False,
                                 reason="the discrimination battery has no rows for this "
                                        "family; see subgroup_coverage.csv"))
            continue
        d.mkdir(parents=True, exist_ok=True)
        for (metric, threshold), g in fam_rows.groupby(["metric", "threshold"]):
            slug = f'ge{int(str(threshold).replace(">=", ""))}'   # ">=1" is not a filename
            path = d / f"subgroup_{metric}_{slug}.csv"
            g.to_csv(path, index=False)
            manifest.append(dict(
                family=family, file=str(path), n_rows=int(len(g)), written=True,
                n_suppressed=int(g["suppressed"].sum()), reason=""))
        gaps = subgroup_gap_summary(fam_rows)
        gaps.to_csv(d / "subgroup_gap_summary.csv", index=False)
        worst = subgroup_worst_cell(fam_rows)
        worst.to_csv(d / "subgroup_worst_cell.csv", index=False)
        manifest.append(dict(family=family, file=str(d / "subgroup_gap_summary.csv"),
                             n_rows=int(len(gaps)), written=True, n_suppressed=0, reason=""))
        manifest.append(dict(family=family, file=str(d / "subgroup_worst_cell.csv"),
                             n_rows=int(len(worst)), written=True, n_suppressed=0, reason=""))
    return pd.DataFrame(manifest)


# 5. Operational layer
def _operational_frame(name: str, tables_dir=None, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Read one of the archived operational result files. Pass `frame=` to supply it directly."""
    if frame is not None:
        return frame.copy()
    tables_dir = Path(tables_dir) if tables_dir is not None else _work_tables()
    path = tables_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is absent. It is an archived input that this pipeline does not produce.")
    return pd.read_csv(path, low_memory=False)


# The operational layer is computed from the model scores, not read back from a summary.
#
# THESE READ THE PER-SEED SCORE PARQUETS notebook 02 writes: one row per person per seed, the
# actual predictions of the fitted models. Nothing is refitted — the fitting is notebook 02's
# job and this is arithmetic over what it produced — but nothing is taken from a pre-aggregated
# CSV either. A metric read back out of a summary file is not a metric this notebook computed.
#
# TIER 1a: the frames below are row-level and are never printed, sampled or returned. Only the
# per-config aggregates leave these functions.
_OPS_META = ("source", "config", "label", "model", "arm", "lineage", "k", "threshold")


# The one score handoff, and the live YRBS subgroup layer
# ---------------------------------------------------------------------------
# Notebook 02 writes ONE person-level file. `subgroup_scores.parquet` and
# `yrbs_scores_by_budget.parquet` were preconditions of the retired pipeline and have no
# producer in any notebook; requiring them blocked every score-reading function here.

SCORE_COLUMNS: Sequence[str] = ("threshold", "family", "regime", "seed", "row_id",
                                "y_true", "score")
SCORE_KEY: Sequence[str] = ("threshold", "family", "regime", "seed", "row_id")

# The two capacities the manuscript reports. 5% and 15% belong to the label-budget curve and
# are that experiment's operating points, not this one's.
CAPACITIES: Sequence[float] = (0.10, 0.20)

# The eight YRBS sex x coarse-ethnicity cells. The categories are the ones notebook 01 built —
# `attr_sex` as 0/1 and `attr_ethnicity_coarse` as the CDC four-way collapse. No new crosswalk
# is invented here and no category is merged.
SUBGROUP_SEX = {0.0: "male", 1.0: "female"}
SUBGROUP_ETHNICITIES: Sequence[str] = ("White", "Black", "Hispanic", "Other")
# The eight classified cells, declared rather than discovered. A cell that no seed happens to
# populate must still appear as a visible non-estimable row instead of vanishing from the grid.
SUBGROUP_CELLS: Sequence[str] = tuple(
    f"{sex} {eth}" for sex in SUBGROUP_SEX.values() for eth in SUBGROUP_ETHNICITIES)
UNCLASSIFIED = "unclassified"


def load_yrbs_scores(path=None) -> pd.DataFrame:
    """The person-level YRBS predictions notebook 02 wrote, validated on the way in.

    ONE FILE. The schema and the scientific key are checked here so that every consumer can
    take the frame at face value, and so a handoff that has drifted fails at the read rather
    than inside an analysis.

    The frame is returned whole. Which regimes a comparison wants is a scientific question and
    is decided at the call site, not here.
    """
    import config as _C
    path = Path(path) if path is not None else Path(_C.MCS_SCORES) / "yrbs_scores.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} is absent from {path.parent}. It is written once by notebook 02's "
            f"final section; run that notebook.")
    frame = pd.read_parquet(path)
    missing = [c for c in SCORE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing column(s) {missing}; expected "
                         f"{list(SCORE_COLUMNS)}")
    extra = [c for c in frame.columns if c not in SCORE_COLUMNS]
    if extra:
        raise ValueError(f"{path.name} carries unexpected column(s) {extra}. The handoff "
                         f"schema is {list(SCORE_COLUMNS)}.")
    dup = int(frame.duplicated(list(SCORE_KEY)).sum())
    if dup:
        raise ValueError(f"{path.name}: {dup} row(s) share the scientific key "
                         f"{list(SCORE_KEY)}, so a respondent appears twice in one cell")
    if "mcs_internal" in set(frame["regime"]):
        raise ValueError(f"{path.name} carries an MCS-evaluated regime; MCS person-level "
                         f"scores are excluded at source and must not appear here")
    return frame[list(SCORE_COLUMNS)]


def _score_sources(tables_dir=None):
    """Yield `(meta, frame)` per (regime, family, threshold) from the one score handoff.

    Rewritten to read `yrbs_scores.parquet` and nothing else. The canonical fields are used
    directly — no `label`, `config` or filename convention is reconstructed from the retired
    pipeline's vocabulary.

    `operating_metrics` and `ventile_stratification` consume this. Their capacities and
    threshold coverage are corrected in a later phase; this change removes the two absent-file
    preconditions that blocked them, and nothing more.
    """
    frame = load_yrbs_scores(None if tables_dir is None
                             else Path(tables_dir) / "yrbs_scores.parquet")
    for (reg, fam, thr), d in frame.groupby(["regime", "family", "threshold"], sort=False):
        yield (dict(source="yrbs_scores", config=reg, model=fam, arm="", lineage="cohort-std",
                    k="", threshold=thr, label=f"{reg}:{fam}"), d)



def attach_subgroup_cells(scores: pd.DataFrame, attributes: pd.DataFrame | None = None,
                          *, attributes_path=None) -> pd.DataFrame:
    """Join the YRBS attribute frame onto the score handoff on `row_id`, and label the cell.

    THE JOIN IS ONE-TO-ONE OR IT RAISES. A duplicated attribute index would attach one
    respondent's sex to another's prediction and inflate the row count silently, which is the
    specific failure this checks for rather than trusts.

    Returns the score frame with `sex`, `ethnicity` and `cell` added and the row count
    unchanged. Rows whose attributes are missing keep a `cell` of `unclassified` rather than
    being dropped: a disappeared row reads as an absent group.
    """
    import config as _C
    if attributes is None:
        path = (Path(attributes_path) if attributes_path is not None
                else Path(_C.YRBS_ATTRIBUTES))
        attributes = pd.read_parquet(path)
    for col in ("attr_sex", "attr_ethnicity_coarse"):
        if col not in attributes.columns:
            raise ValueError(f"the attribute frame has no {col!r}; the subgroup cells are "
                             f"sex x coarse ethnicity and cannot be built without it")
    if not attributes.index.is_unique:
        raise ValueError("the YRBS attribute index is not unique, so a join on row_id would "
                         "attach one respondent's attributes to another's prediction")

    unknown = set(scores["row_id"]) - set(attributes.index)
    if unknown:
        raise ValueError(f"{len(unknown)} score row_id(s) do not resolve in the attribute "
                         f"frame; the two frames are not on the same index")

    attrs = attributes.loc[:, ["attr_sex", "attr_ethnicity_coarse"]]
    n_before = len(scores)
    out = scores.merge(attrs, left_on="row_id", right_index=True, how="left",
                       validate="many_to_one")
    if len(out) != n_before:
        raise ValueError(f"the attribute join changed the row count from {n_before} to "
                         f"{len(out)}")

    out["sex"] = out["attr_sex"].map(SUBGROUP_SEX)
    out["ethnicity"] = out["attr_ethnicity_coarse"].where(
        out["attr_ethnicity_coarse"].isin(SUBGROUP_ETHNICITIES))
    out["cell"] = np.where(out["sex"].isna() | out["ethnicity"].isna(), "unclassified",
                           out["sex"].astype(str) + " " + out["ethnicity"].astype(str))
    return out.drop(columns=["attr_sex", "attr_ethnicity_coarse"])


def cohort_capacity_flags(p, capacities: Sequence[float] = CAPACITIES) -> dict:
    """The cohort-wide top-k flag per capacity, over the whole evaluable slice.

    ONE RANKING FOR THE WHOLE COHORT. A fixed review capacity is a property of the system, not
    of a subgroup: at 10% capacity the service reviews the highest-scoring tenth of everyone it
    screens, and the question is which groups end up inside that set. Ranking within each
    subgroup instead would hand every group its own tenth and force every flag rate to the
    capacity, which answers nothing.

    THE ONE CAPACITY SELECTION. `operating_point_confusion` calls this too, so notebook 02's
    decile and quintile columns and notebook 03's capacity tables flag the same respondents.

    EXACTLY `ceil(capacity x n)` ROWS ARE FLAGGED, whatever happens at the boundary. The
    ordering is `np.argsort(-p, kind="stable")`, so where several rows share the boundary score
    the tie is resolved by the order the rows arrive in — the order of that seed's held-out set,
    which is what `data.build_splits` emits and is identical across every model compared at that
    threshold and seed. That is deterministic and reproducible, and it remains an arbitrary
    allocation among equal scores: it produces exactly the stated capacity, and it does not
    follow from anything about the respondents it separates. `capacity_boundary_ties` reports
    how often it decides anything, and that diagnostic is the evidence about how much the rule
    matters.
    """
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(-p, kind="stable")
    out = {}
    for cap in capacities:
        k = int(np.ceil(cap * n))
        flag = np.zeros(n, bool)
        flag[order[:k]] = True
        out[cap] = flag
    return out


def subgroup_own_metrics(y, p) -> tuple:
    """Discrimination and calibration for one cell, from that cell's own rows. `(values, reasons)`.

    EACH METRIC STANDS OR FALLS ON ITS OWN. A cell with one outcome class leaves ROC-AUC and
    PR-AUC undefined while the Brier score and the expected calibration error remain perfectly
    well defined, so one undefined metric does not blank the row.

    The definitions are the module's own — `roc_auc_score`, `average_precision_score`, `ece`
    and `cal_slope_intercept` — called here rather than reimplemented. Capacity metrics are NOT
    here: they depend on a cohort-wide cut and are computed by `subgroup_capacity_metrics`.
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    values: dict = {}
    reasons: dict = {}

    def record(name, value, reason=""):
        values[name] = value
        reasons[name] = reason

    n = int(y.size)
    if n == 0:
        for m in ("prevalence", "auc", "prauc", "brier", "ece", "cal_intercept", "cal_slope"):
            record(m, np.nan, "no evaluable row in this cell")
        return values, reasons

    record("prevalence", float(y.mean()))
    both = np.unique(y).size == 2
    single = "single outcome class in the cell"
    record("auc", float(roc_auc_score(y, p)) if both else np.nan, "" if both else single)
    # PR-AUC's no-skill value IS the prevalence, so at 0 or 1 it describes nothing.
    record("prauc", float(average_precision_score(y, p)) if both else np.nan,
           "" if both else single)
    record("brier", float(np.mean((p - y) ** 2)))
    record("ece", ece(y, p))
    if both:
        cs, ci = cal_slope_intercept(y, p)
        record("cal_slope", cs, "" if cs == cs else "calibration fit did not converge")
        record("cal_intercept", ci, "" if ci == ci else "no sign change on the intercept")
    else:
        record("cal_slope", np.nan, single)
        record("cal_intercept", np.nan, single)
    return values, reasons


def _confusion(y, flagged) -> tuple:
    """`(tp, fp, fn, tn)` for one group under a given flag vector. Plain integer counts.

    THESE ARE ACCOUNTING QUANTITIES, NOT REPORTABLE METRICS. They exist so the cohort-wide cut
    and the per-cell tables can be reconciled exactly, and they stay available for a cell whose
    public rates the stability rule blanks — a small cell still contains flagged respondents,
    and dropping them from the accounting would lose them from the cohort total.
    """
    y = np.asarray(y, float)
    flagged = np.asarray(flagged, bool)
    return (int(np.sum(flagged & (y == 1))), int(np.sum(flagged & (y == 0))),
            int(np.sum(~flagged & (y == 1))), int(np.sum(~flagged & (y == 0))))


def subgroup_capacity_metrics(y, flagged, *, tag: int) -> tuple:
    """One cell's operating metrics, given the COHORT-WIDE flagged decision. `(values, reasons)`.

    `flagged` is the boolean the cohort-wide cut assigned to this cell's respondents. It is
    never recomputed inside the cell: every subgroup at a given model, seed and capacity
    inherits the same selection boundary, which is what makes the flag rates comparable.

    `flagrate` is therefore the share of THIS cell that fell inside the system-wide set, and it
    is free to differ from the capacity — that difference is the finding the analysis exists to
    surface.
    """
    y = np.asarray(y, float)
    flagged = np.asarray(flagged, bool)
    values: dict = {}
    reasons: dict = {}

    def record(stem, value, reason=""):
        values[f"{stem}_at_{tag}"] = value
        reasons[f"{stem}_at_{tag}"] = reason

    n = int(y.size)
    if n == 0:
        for stem in ("precision", "recall", "flagrate", "fpr"):
            record(stem, np.nan, "no evaluable row in this cell")
        return values, reasons

    tp, fp, fn, tn = _confusion(y, flagged)

    record("flagrate", (tp + fp) / n)
    record("precision", tp / (tp + fp) if (tp + fp) else np.nan,
           "" if (tp + fp) else "no respondent in this cell was flagged")
    record("recall", tp / (tp + fn) if (tp + fn) else np.nan,
           "" if (tp + fn) else "no positive outcome in this cell")
    record("fpr", fp / (fp + tn) if (fp + tn) else np.nan,
           "" if (fp + tn) else "no negative outcome in this cell")
    return values, reasons


def subgroup_metrics_per_seed(scores: pd.DataFrame, *,
                              capacities: Sequence[float] = CAPACITIES,
                              cells: Sequence[str] = SUBGROUP_CELLS,
                              grid: Sequence[tuple] | None = None) -> pd.DataFrame:
    """One row per (regime, family, threshold, seed, cell, metric), computed from the scores.

    PER SEED, BEFORE ANY AGGREGATION. Pooling a respondent's twenty predictions would treat
    twenty correlated predictions as twenty observations.

    THE CAPACITY CUT IS COHORT-WIDE AND IS TAKEN FIRST. For each (regime, family, threshold,
    seed) the whole evaluable slice is ranked together — **including respondents whose subgroup
    is unclassified**, because they are part of the population the service screens — the top-k
    is selected, and only then are the flags grouped by cell. Dropping the unclassified before
    ranking would change the system's capacity; the classified cells are what is *reported*, not
    what is screened.

    THE GRID IS DECLARED, NOT DISCOVERED. Every cell in `cells` gets a row for every group,
    whether or not the data populated it, and `grid` may name (regime, family, threshold, seed)
    combinations that produced no scores at all — a recalibration seed that could not be
    estimated, say. Those arrive as visible non-estimable rows rather than absent ones.

    `stability_status` records the prespecified YRBS minimum-cell-size rule. **That is an
    analytical rule about estimating from a small cell, not disclosure suppression** — YRBS is
    open CDC data — and it is applied AFTER the cohort-wide selection, so it never moves the cut.
    """
    tags = [int(round(c * 100)) for c in capacities]
    present = {}
    for keyvals, d in scores.groupby(["regime", "family", "threshold", "seed"], sort=False):
        present[keyvals] = d
    keys = list(grid) if grid is not None else list(present)

    rows = []
    for key in keys:
        d = present.get(tuple(key))
        reg, fam, thr, seed = key
        if d is None:
            for cell in cells:
                for stem in ("prevalence", "auc", "prauc", "brier", "ece",
                             "cal_intercept", "cal_slope"):
                    rows.append(dict(regime=reg, family=fam, threshold=thr, seed=seed,
                                     cell=cell, metric=stem, value=np.nan, n=0,
                                     estimability_status="non_estimable",
                                     estimability_reason="no scores for this seed",
                                     stability_status="below_minimum"))
                for tag in tags:
                    for stem in ("precision", "recall", "flagrate", "fpr"):
                        rows.append(dict(regime=reg, family=fam, threshold=thr, seed=seed,
                                         cell=cell, metric=f"{stem}_at_{tag}", value=np.nan,
                                         n=0, estimability_status="non_estimable",
                                         estimability_reason="no scores for this seed",
                                         stability_status="below_minimum"))
            continue

        y_all = np.asarray(d["y_true"], float)
        keep = ~np.isnan(y_all)
        d = d[keep]
        y_all = y_all[keep]
        p_all = np.asarray(d["score"], float)
        # COHORT-WIDE, over every evaluable respondent including the unclassified.
        flags = cohort_capacity_flags(p_all, capacities)
        cell_arr = d["cell"].to_numpy()

        for cell in cells:
            m = cell_arr == cell
            y_c, p_c = y_all[m], p_all[m]
            n_c = int(y_c.size)
            stability = ("below_minimum" if n_c < SUBGROUP_MIN_CELL_N
                         else "at_or_above_minimum")
            values, reasons = subgroup_own_metrics(y_c, p_c)
            for cap, tag in zip(capacities, tags):
                v, r = subgroup_capacity_metrics(y_c, flags[cap][m], tag=tag)
                values.update(v)
                reasons.update(r)
            for metric, value in values.items():
                rows.append(dict(
                    regime=reg, family=fam, threshold=thr, seed=seed, cell=cell,
                    metric=metric, value=value, n=n_c,
                    estimability_status=("estimable" if reasons[metric] == ""
                                         and value == value else "non_estimable"),
                    estimability_reason=reasons[metric], stability_status=stability))
    return pd.DataFrame(rows)


def summarise_metric_frame(per_seed: pd.DataFrame, *, keys: Sequence[str],
                           seeds_expected: int = 20,
                           seed_status: pd.DataFrame | None = None) -> pd.DataFrame:
    """Across-seed summary of a long per-seed metric frame, without survivor bias.

    THE DISPLAYED POINT ESTIMATE IS THE MEAN OF THE TWENTY SEED-SPECIFIC VALUES. If any of the
    twenty is missing or non-estimable, no mean is put in `mean`: a mean over whichever seeds
    happened to work is a different statistic from the one every other row reports, and the two
    must not share a column. The per-seed rows keep every value that was computed.

    `split_sd` is the spread across the twenty overlapping splits. It describes **stability**,
    not sampling uncertainty, and is not a confidence interval.

    `seed_status` optionally carries notebook 02's per-seed `recal_status`, keyed
    (regime, family, threshold, seed), so a recalibration cell that could not be estimated
    reports why rather than merely being short.
    """
    if per_seed.empty:
        return per_seed
    keys = list(keys)
    reasons_by_cell = {}
    if seed_status is not None and not seed_status.empty:
        bad = seed_status[seed_status["recal_status"] != "estimated"]
        for (reg, fam, thr), g in bad.groupby(["regime", "family", "threshold"]):
            reasons_by_cell[(reg, fam, thr)] = "; ".join(
                f"{r}={n}" for r, n in g["recal_status"].value_counts().items())

    has_n = "n" in per_seed.columns
    has_stability = "stability_status" in per_seed.columns
    rows = []
    for keyvals, g in per_seed.groupby(keys, sort=False):
        keyvals = keyvals if isinstance(keyvals, tuple) else (keyvals,)
        rec = dict(zip(keys, keyvals))
        est = g[g["estimability_status"] == "estimable"]
        if has_stability:
            est = est[est["stability_status"] == "at_or_above_minimum"]
        attempted = int(g["seed"].nunique())
        estimated = int(est["seed"].nunique())
        complete = estimated == seeds_expected
        why = "; ".join(sorted({r for r in g["estimability_reason"] if r}))
        if not complete and not why:
            why = reasons_by_cell.get(
                (rec.get("regime"), rec.get("family"), rec.get("threshold")), "")
            if not why and attempted < seeds_expected:
                why = f"{seeds_expected - attempted} seed(s) produced no scores"
        rec.update(seeds_expected=seeds_expected, seeds_attempted=attempted,
                   seeds_estimated=estimated, seeds_non_estimable=attempted - estimated,
                   mean=(float(est["value"].mean()) if complete else np.nan),
                   split_sd=(float(est["value"].std()) if complete else np.nan),
                   estimability_status=("complete" if complete else "incomplete"),
                   estimability_reason=("" if complete else why))
        if has_n:
            rec.update(n_mean=float(g["n"].mean()), n_min=int(g["n"].min()))
        if has_stability:
            below = g[g["stability_status"] == "below_minimum"]["seed"].nunique()
            rec["stability_status"] = (
                "below_minimum_in_every_seed" if below == attempted and attempted
                else "below_minimum_in_some_seeds" if below
                else "at_or_above_minimum_in_every_seed")
        rows.append(rec)
    return pd.DataFrame(rows)


def summarise_subgroup_metrics(per_seed: pd.DataFrame, *, seeds_expected: int = 20,
                               seed_status: pd.DataFrame | None = None) -> pd.DataFrame:
    """Across-seed subgroup summary. See `summarise_metric_frame` for the rules."""
    return summarise_metric_frame(
        per_seed, keys=["regime", "family", "threshold", "cell", "metric"],
        seeds_expected=seeds_expected, seed_status=seed_status)


COHORT_GROUP = "__cohort__"


def capacity_accounting(scores: pd.DataFrame, *, capacities: Sequence[float] = CAPACITIES,
                        cells: Sequence[str] = SUBGROUP_CELLS,
                        grid: Sequence[tuple] | None = None) -> pd.DataFrame:
    """Raw confusion counts per (regime, family, threshold, seed, capacity, group).

    `group` is one of the eight declared cells, `unclassified`, or `__cohort__` for the whole
    evaluable slice. One row per group; the columns are integers, not rates.

    THIS IS AN INTERNAL CONSISTENCY DIAGNOSTIC, NOT A MANUSCRIPT OUTPUT. It exists so the
    cohort-wide cut and the per-cell tables can be reconciled exactly. It is deliberately
    separate from the reportable metrics: the `n < 50` stability rule blanks a small cell's
    public rates, and reconstructing counts from a blanked rate would silently drop that cell's
    flagged respondents from the total. A small cell still contains people who were flagged.

    Score cells absent from `scores` produce no row. A recalibration seed that could not be
    estimated is reconciled through its estimability status, not here — it has no score vector
    and therefore no capacity to account for.
    """
    tags = {c: int(round(c * 100)) for c in capacities}
    present = {k: d for k, d in scores.groupby(
        ["regime", "family", "threshold", "seed"], sort=False)}
    keys = [tuple(k) for k in (grid if grid is not None else present)]
    rows = []
    for key in keys:
        d = present.get(key)
        if d is None:
            continue
        reg, fam, thr, seed = key
        y_all = np.asarray(d["y_true"], float)
        keep = ~np.isnan(y_all)
        d = d[keep]
        y_all = y_all[keep]
        p_all = np.asarray(d["score"], float)
        cell_arr = (d["cell"].to_numpy() if "cell" in d.columns
                    else np.full(len(d), UNCLASSIFIED))
        flags = cohort_capacity_flags(p_all, capacities)
        for cap, tag in tags.items():
            f = flags[cap]
            groups = [(COHORT_GROUP, np.ones(len(y_all), bool))]
            groups += [(c, cell_arr == c) for c in cells]
            groups.append((UNCLASSIFIED, cell_arr == UNCLASSIFIED))
            for name, m in groups:
                tp, fp, fn, tn = _confusion(y_all[m], f[m])
                rows.append(dict(regime=reg, family=fam, threshold=thr, seed=seed,
                                 capacity=cap, capacity_pct=tag, group=name,
                                 n=int(m.sum()), n_flagged=tp + fp,
                                 tp=tp, fp=fp, fn=fn, tn=tn))
    return pd.DataFrame(rows)


def reconcile_capacity_accounting(audit: pd.DataFrame) -> pd.DataFrame:
    """Every score cell's classified + unclassified counts against the cohort total.

    Returns one row per (regime, family, threshold, seed, capacity) with the cohort totals, the
    summed parts and a boolean per identity. `reconciles` false anywhere is a disagreement.

    `n_flagged` on the cohort row must also equal `ceil(capacity x n_evaluable)`; a cut that
    selected a different number is a different cut.
    """
    if audit.empty:
        return audit
    key = ["regime", "family", "threshold", "seed", "capacity", "capacity_pct"]
    counts = ["n", "n_flagged", "tp", "fp", "fn", "tn"]
    cohort = audit[audit["group"] == COHORT_GROUP].set_index(key)[counts]
    parts = audit[audit["group"] != COHORT_GROUP].groupby(key)[counts].sum()
    out = cohort.join(parts, lsuffix="_cohort", rsuffix="_parts").reset_index()
    for c in counts:
        out[f"{c}_agrees"] = out[f"{c}_cohort"] == out[f"{c}_parts"]
    out["capacity_exact"] = out["n_flagged_cohort"] == np.ceil(
        out["capacity"] * out["n_cohort"]).astype(int)
    out["reconciles"] = out[[f"{c}_agrees" for c in counts] + ["capacity_exact"]].all(axis=1)
    return out


def cohort_operating_metrics_per_seed(scores: pd.DataFrame, *,
                                      capacities: Sequence[float] = CAPACITIES,
                                      grid: Sequence[tuple] | None = None) -> pd.DataFrame:
    """Whole-cohort operating metrics, one row per (regime, family, threshold, seed, metric).

    THE SAME CUT THE SUBGROUP LAYER USES. `cohort_capacity_flags` is called on the same
    evaluable slice, so the flagged set behind this table and the one behind the subgroup table
    are the same respondents — the subgroup confusion counts sum to the classified portion of
    these. Nothing is recomputed with a different rule.

    `no_skill_precision` is the slice prevalence: the precision a random flag of the same size
    would achieve, and the only thing that makes a precision at a fixed capacity readable.

    `grid` declares which (regime, family, threshold, seed) must appear, so a combination that
    produced no scores arrives as a visible non-estimable row rather than an absent one.
    """
    tags = [int(round(c * 100)) for c in capacities]
    present = {k: d for k, d in scores.groupby(
        ["regime", "family", "threshold", "seed"], sort=False)}
    keys = list(grid) if grid is not None else list(present)
    metric_names = ["n", "prevalence", "no_skill_precision"] + [
        f"{stem}_at_{t}" for t in tags for stem in ("precision", "recall", "flagrate", "fpr")]

    rows = []
    for key in keys:
        reg, fam, thr, seed = key
        d = present.get(tuple(key))
        if d is None:
            for m in metric_names:
                rows.append(dict(regime=reg, family=fam, threshold=thr, seed=seed, metric=m,
                                 value=np.nan, n=0, estimability_status="non_estimable",
                                 estimability_reason="no scores for this seed"))
            continue
        y = np.asarray(d["y_true"], float)
        keep = ~np.isnan(y)
        y = y[keep]
        p_arr = np.asarray(d["score"], float)[keep]
        n = int(y.size)
        flags = cohort_capacity_flags(p_arr, capacities)
        values = {"n": float(n), "prevalence": float(y.mean()) if n else np.nan}
        values["no_skill_precision"] = values["prevalence"]
        reasons = {m: ("" if n else "no evaluable row") for m in ("n", "prevalence",
                                                                 "no_skill_precision")}
        for cap, tag in zip(capacities, tags):
            v, r = subgroup_capacity_metrics(y, flags[cap], tag=tag)
            values.update(v)
            reasons.update(r)
        for m in metric_names:
            val = values.get(m, np.nan)
            rows.append(dict(regime=reg, family=fam, threshold=thr, seed=seed, metric=m,
                             value=val, n=n,
                             estimability_status=("estimable" if reasons.get(m, "") == ""
                                                  and val == val else "non_estimable"),
                             estimability_reason=reasons.get(m, "")))
    return pd.DataFrame(rows)


def summarise_cohort_operating(per_seed: pd.DataFrame, *, seeds_expected: int = 20,
                               seed_status: pd.DataFrame | None = None) -> pd.DataFrame:
    """Across-seed whole-cohort operating summary. See `summarise_metric_frame` for the rules."""
    return summarise_metric_frame(per_seed, keys=["regime", "family", "threshold", "metric"],
                                  seeds_expected=seeds_expected, seed_status=seed_status)


# Ventile stratification, live
VENTILES = 20


def ventile_prevalence_per_seed(scores: pd.DataFrame, *, n_ventiles: int = VENTILES,
                                grid: Sequence[tuple] | None = None) -> pd.DataFrame:
    """Observed outcome prevalence per predicted-risk ventile, per seed.

    BINNING IS ON THE RANK WITHIN ONE SEED, so every ventile holds the same number of
    adolescents and the prevalences are comparable down the column. Ranking across seeds
    together would pool a respondent's twenty predictions.

    All `n_ventiles` bins are emitted for every group, whether or not the data populated them:
    an empty bin is a visible non-estimable row with `n=0`, not a gap in the grid.
    """
    present = {k: d for k, d in scores.groupby(
        ["regime", "family", "threshold", "seed"], sort=False)}
    keys = list(grid) if grid is not None else list(present)
    rows = []
    for key in keys:
        reg, fam, thr, seed = key
        d = present.get(tuple(key))
        if d is None:
            for v in range(1, n_ventiles + 1):
                rows.append(dict(regime=reg, family=fam, threshold=thr, seed=seed, ventile=v,
                                 metric="obs_prevalence", value=np.nan, n=0,
                                 estimability_status="non_estimable",
                                 estimability_reason="no scores for this seed"))
            continue
        y = np.asarray(d["y_true"], float)
        keep = ~np.isnan(y)
        y = y[keep]
        p_arr = np.asarray(d["score"], float)[keep]
        n = y.size
        if n:
            rank = np.argsort(np.argsort(p_arr))
            vb = np.minimum(n_ventiles - 1, rank * n_ventiles // n)
        for v in range(1, n_ventiles + 1):
            m = (vb == v - 1) if n else np.zeros(0, bool)
            n_v = int(m.sum()) if n else 0
            rows.append(dict(
                regime=reg, family=fam, threshold=thr, seed=seed, ventile=v,
                metric="obs_prevalence",
                value=(float(y[m].mean()) if n_v else np.nan), n=n_v,
                estimability_status=("estimable" if n_v else "non_estimable"),
                estimability_reason=("" if n_v else "no respondent in this ventile")))
    return pd.DataFrame(rows)


def summarise_ventiles(per_seed: pd.DataFrame, *, seeds_expected: int = 20) -> pd.DataFrame:
    """Across-seed ventile summary with an across-split stability band.

    `band_lo` / `band_hi` are `mean +/- 1.96 * split_sd`, clipped to [0, 1]. **That is a
    stability band, not a confidence interval**: the twenty splits overlap, so the spread
    describes how the estimate moves across them and not how it would move in a fresh sample.
    """
    out = summarise_metric_frame(per_seed,
                                 keys=["regime", "family", "threshold", "ventile", "metric"],
                                 seeds_expected=seeds_expected)
    if out.empty:
        return out
    out["band_lo"] = np.clip(out["mean"] - 1.96 * out["split_sd"], 0.0, 1.0)
    out["band_hi"] = np.clip(out["mean"] + 1.96 * out["split_sd"], 0.0, 1.0)
    return out


def ventile_extremes_per_seed(ventiles: pd.DataFrame, *,
                              n_ventiles: int = VENTILES) -> pd.DataFrame:
    """Top-versus-bottom ventile, per seed: the risk ratio and the risk difference.

    THE ZERO-DENOMINATOR CASE IS REPORTED, NOT SKIPPED. Where the bottom ventile's prevalence
    is exactly zero the ratio is undefined, and the historical implementation dropped that seed
    on a truthiness test — which turned a twenty-seed mean into a mean over whichever seeds
    happened to have a non-zero denominator. Here the seed is recorded as non-estimable with
    that reason, so the summary refuses to report an ordinary mean.

    The RISK DIFFERENCE stays defined when the bottom prevalence is zero and is emitted beside
    the ratio. **They are different quantities and are labelled separately**; the difference is
    not a substitute for the manuscript's ratio claim.
    """
    rows = []
    keys = ["regime", "family", "threshold", "seed"]
    for keyvals, g in ventiles.groupby(keys, sort=False):
        rec = dict(zip(keys, keyvals if isinstance(keyvals, tuple) else (keyvals,)))
        top = g[g["ventile"] == n_ventiles]
        bot = g[g["ventile"] == 1]
        t = float(top["value"].iloc[0]) if len(top) else np.nan
        b = float(bot["value"].iloc[0]) if len(bot) else np.nan

        if t != t or b != b:
            why = "the top or bottom ventile is not estimable"
            rr, rd, rr_why, rd_why = np.nan, np.nan, why, why
        elif b == 0.0:
            rr, rr_why = np.nan, "bottom-ventile prevalence is zero: the ratio is undefined"
            rd, rd_why = t - b, ""
        else:
            rr, rr_why = t / b, ""
            rd, rd_why = t - b, ""
        for metric, value, why in (("risk_ratio_top_bottom", rr, rr_why),
                                   ("risk_difference_top_bottom", rd, rd_why)):
            rows.append(dict(**rec, metric=metric, value=value,
                             estimability_status=("estimable" if why == "" and value == value
                                                  else "non_estimable"),
                             estimability_reason=why))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Uncertainty conditional on the fitted models
#
# WHAT IS RESAMPLED. Respondents, and nothing else. One Poisson(1) multiplicity is drawn per
# eligible YRBS respondent per replicate and that multiplicity follows the respondent into every
# seed whose test set contains them, and into every regime and family at that threshold. The
# twenty test sets overlap — a respondent sits in about five of them — and reusing one draw is
# what keeps that overlap in the interval instead of assuming it away.
#
# THIS IS A RESPONDENT-LEVEL POISSON (MULTIPLIER) BOOTSTRAP, NOT A FIXED-SIZE PAIRS BOOTSTRAP.
# The number of rows a seed contributes to a replicate is random, with mean equal to that seed's
# test-set size. The capacity cut is therefore recomputed on the realised expanded size rather
# than on a fixed k.
#
# WHAT IS HELD FIXED. The twenty fitted prediction functions, the training samples and random
# seeds behind them, preprocessing, adaptation, the cross-fitted recalibration mappings, the
# k=500 samples, every modelling and hyperparameter decision, and each respondent's train/test
# assignment. Only who was evaluated varies.
#
# WHAT IT DOES NOT COVER. Model development, fitting, seed selection, the split design itself,
# the survey design and clustering. Ignoring the design can misestimate uncertainty and commonly
# understates it under positive within-cluster correlation. These are not procedure-level
# confidence intervals and they satisfy no reporting guideline on their own.

BOOTSTRAP_METRICS: Sequence[str] = (
    "auc", "prauc", "brier", "ece", "cal_intercept", "cal_slope",
    "precision_at_10", "recall_at_10", "flagrate_at_10", "fpr_at_10",
    "precision_at_20", "recall_at_20", "flagrate_at_20", "fpr_at_20",
)

# Why a replicate produced no value. Every invalid replicate carries one of these and nothing
# else, so the accounting can be counted rather than read.
INVALID_REPLICATE_REASONS: Sequence[str] = (
    "no_respondent_drawn",
    "single_outcome_class",
    "calibration_did_not_converge",
    "calibration_no_sign_change",
    "no_flagged_respondent",
    "no_positive_outcome",
    "no_negative_outcome",
    "seed_component_undefined",
    "reference_at_or_below_chance",
    # the target resource gap is the denominator of target_gap_recovered; a replicate that draws
    # a local reference at or below unadapted transfer has no gap to recover a share of
    "target_resource_gap_not_positive",
    "target_selection_non_estimable",
)

# What a declared model cell can be. Nothing else is representable.
COVERAGE_STATES: Sequence[str] = ("complete", "non_estimable", "invalid_handoff")

# What is wrong with a cell that is neither complete nor recorded non-estimable.
HANDOFF_PROBLEMS: Sequence[str] = (
    "missing_expected_respondents",
    "unexpected_respondents",
    "duplicate_respondents",
    "outcome_conflicts_with_the_cohort",
    "recorded_estimable_but_carries_no_scores",
    "recorded_non_estimable_but_carries_scores",
    "scored_cell_was_never_declared",
)

# `subgroup_capacity_metrics` reports in prose, because a person reads it too. The bootstrap
# needs a code, and the mapping is explicit rather than inferred.
_CAPACITY_REASON_CODE = {
    "no evaluable row in this cell": "no_respondent_drawn",
    "no respondent in this cell was flagged": "no_flagged_respondent",
    "no positive outcome in this cell": "no_positive_outcome",
    "no negative outcome in this cell": "no_negative_outcome",
}


def _eligible_yrbs_rows(threshold: int, *, pillars: pd.DataFrame | None = None, pillars_path=None):
    """Every YRBS respondent whose outcome is defined at this threshold, with that outcome.

    The eligible cohort, from `data.make_outcome` and its eligibility rule — the population the
    seed-specific test sets are drawn from. It is not derived from the score handoff, and the
    outcome travels with it so a cell's `y_true` can be checked against the cohort's own value
    rather than against another cell.
    """
    import data as D
    import config as _C
    if pillars is None:
        pillars = pd.read_parquet(Path(pillars_path) if pillars_path is not None
                                  else Path(_C.YRBS_PILLARS))
    outcome = D.make_outcome(pillars, int(threshold))
    eligible = outcome[outcome.notna()].astype(float)
    eligible.index.name = "row_id"
    eligible.name = "y_true"
    return eligible


def _yrbs_evaluation_sets(thresholds: Sequence[int], seeds: Sequence[int], *,
                         pillars: pd.DataFrame | None = None, pillars_path=None) -> dict:
    """`{(threshold, seed): the eligible respondents in that seed's held-out quarter}`.

    NOTEBOOK 02'S OWN SPLIT, ASKED FOR DIRECTLY. Each seed cuts the YRBS cohort 75/25, takes the
    k=500 adaptation sample from the training portion and leaves the test portion untouched
    until scoring, so the test portion is what every score-bearing procedure predicted on.
    `data.yrbs_test_index` is the cut `data.build_splits` uses, not a second copy of it.

    Rows with an undefined outcome stay in the test frame and are dropped at metric time, so the
    evaluable set is the intersection with the eligible cohort. The cut is stratified on the
    outcome, so a different threshold at the same seed holds out different respondents.

    The cut was stratified on the FULL cohort outcome, undefined rows included, so that is what
    is reconstructed here before the eligible rows are selected out of it.
    """
    import data as D
    import config as _C
    if pillars is None:
        pillars = pd.read_parquet(Path(pillars_path) if pillars_path is not None
                                  else Path(_C.YRBS_PILLARS))
    sets = {}
    for threshold in thresholds:
        outcome = D.make_outcome(pillars, int(threshold))
        label = f">={int(threshold)}"
        eligible = outcome.index[outcome.notna()]
        for seed in seeds:
            held_out = D.yrbs_test_index(outcome, int(seed))
            # Sorted, so this and a score cell's rows are compared in one canonical order and
            # the split's own shuffle order never enters a comparison.
            sets[(label, int(seed))] = held_out.intersection(eligible).sort_values()
    return sets


def _draw_respondent_multiplicities(eligible, *, replicates: int, seed: int) -> np.ndarray:
    """`(replicates, len(eligible))` independent Poisson(1) multiplicities.

    ONE DRAW PER RESPONDENT PER REPLICATE, REUSED EVERYWHERE AT THAT THRESHOLD. A respondent who
    is drawn twice appears twice in every seed whose test set contains them, in every regime and
    every family. That is the dependence between the twenty overlapping evaluations, carried
    rather than assumed away.

    POISSON, NOT MULTINOMIAL. The multipliers are independent, so restricting them to a seed's
    test set is exactly a bootstrap of that test set; a fixed total over the whole cohort would
    couple respondents who never share a test set and make the restriction only approximate. The
    number of rows a seed contributes is therefore random, with mean its test-set size.

    ONE DETERMINISTIC NESTED STREAM PER THRESHOLD. The first fifty rows of a two-hundred-replicate
    draw are the fifty-replicate draw, so comparing replicate counts compares nested prefixes and
    any movement is the effect of adding replicates. Thresholds draw separately: their eligible
    cohorts and their test sets are both threshold-specific.
    """
    generator = np.random.default_rng(seed)
    return generator.poisson(1.0, size=(int(replicates), len(eligible))).astype(np.int32)


def _cell_problems(realised_rows, realised_outcome, expected_rows, expected_outcome) -> dict:
    """What is wrong with one cell-seed's coverage, as counts. Empty when it is complete."""
    unique_rows, repeats = np.unique(realised_rows, return_counts=True)
    problems = {}
    n_duplicate = int((repeats - 1).sum())
    n_unexpected = int(np.isin(unique_rows, expected_rows, invert=True).sum())
    n_missing = int(np.isin(expected_rows, unique_rows, invert=True).sum())
    if n_missing:
        problems["missing_expected_respondents"] = n_missing
    if n_unexpected:
        problems["unexpected_respondents"] = n_unexpected
    if n_duplicate:
        problems["duplicate_respondents"] = n_duplicate
    if not problems:
        n_conflict = int((~np.isclose(realised_outcome, expected_outcome)).sum())
        if n_conflict:
            problems["outcome_conflicts_with_the_cohort"] = n_conflict
    return problems


def yrbs_evaluation_design(thresholds: Sequence[int], seeds: Sequence[int], *,
                           pillars: pd.DataFrame | None = None, pillars_path=None) -> tuple:
    """The population, the twenty held-out sets, and how often a respondent is evaluated.

    Notebook 02 cuts the YRBS cohort 75/25 for each threshold and seed, stratified on the
    outcome. The k=500 adaptation sample is drawn from the training portion, so the held-out
    quarter is untouched until scoring and every score-bearing procedure predicts on it.

    Returns `(eligible, evaluation_sets, appearances)`: the eligible cohort per threshold with
    its outcome; the held-out respondents per `(threshold, seed)`; and a small frame counting
    how many of the twenty held-out sets each respondent falls into, which is what makes the
    twenty per-seed results dependent on one another.
    """
    import config as _C
    if pillars is None:
        pillars = pd.read_parquet(Path(pillars_path) if pillars_path is not None
                                  else Path(_C.YRBS_PILLARS))
    eligible = {f">={int(t)}": _eligible_yrbs_rows(t, pillars=pillars) for t in thresholds}
    evaluation_sets = _yrbs_evaluation_sets(thresholds, seeds, pillars=pillars)

    rows = []
    for threshold, cohort in eligible.items():
        counted = pd.Series(0, index=cohort.index)
        sizes = []
        for seed in seeds:
            held_out = evaluation_sets[(threshold, int(seed))]
            counted.loc[held_out] += 1
            sizes.append(len(held_out))
        rows.append(dict(threshold=threshold, eligible=len(cohort),
                         held_out_min=min(sizes), held_out_max=max(sizes),
                         held_out_share=round(float(np.mean(sizes)) / len(cohort), 4),
                         mean_appearances=round(float(counted.mean()), 2),
                         never_evaluated=int((counted == 0).sum())))
    return eligible, evaluation_sets, pd.DataFrame(rows)


def validate_score_coverage(scores: pd.DataFrame, eligible: Mapping[str, pd.Series],
                            evaluation_sets: Mapping[tuple, pd.Index], *,
                            declared: Sequence[tuple], non_estimable: Sequence[tuple] = (),
                            reasons: Mapping[tuple, str] | None = None,
                            seeds: Sequence[int] = tuple(range(20))) -> dict:
    """Check every declared model cell against the evaluation set it should have covered.

    EACH CELL IS EXACTLY ONE OF THREE THINGS.

    | state | what it means |
    |---|---|
    | `complete` | every seed covers that seed's evaluation set once each, with the cohort's outcome |
    | `non_estimable` | at least one seed is recorded non-estimable and carries no score rows; the rest are complete |
    | `invalid_handoff` | anything else |

    **Only complete cells are resampled.** A non-estimable cell keeps its recorded reason and
    gets no interval. An invalid cell is reported with counts and stops the analysis: partial or
    inconsistent coverage is a malformed handoff, not an alternative evaluation population, and
    nothing here infers a population from what happens to be present.

    Comparable models within a threshold and seed are held to the same evaluation set, so two
    complete cells necessarily hold the same respondents in the same order — which is what makes
    a paired difference paired. `paired_rows_identical` in the summary reports that directly.

    Returns `{"cells", "summary", "problems", "states", "seeds"}`. `cells` maps each complete
    `(regime, family, threshold)` to a per-seed list of `(universe_positions, y_true, score)`.
    No respondent identifier appears in `summary`, `problems` or `states`.
    """
    def cell_key(key):
        return (str(key[0]), str(key[1]), str(key[2]), int(key[3]))

    declared = [(str(regime), str(family), str(threshold))
                for regime, family, threshold in declared]
    non_estimable = {cell_key(key) for key in non_estimable}
    reasons = {cell_key(key): str(why) for key, why in dict(reasons or {}).items()}
    seeds = [int(seed) for seed in seeds]

    scored = {key: frame for key, frame in
              scores.groupby(["regime", "family", "threshold", "seed"], sort=False)}
    undeclared = sorted({key[:3] for key in scored} - set(declared))

    positions_in_cohort = {threshold: pd.Series(np.arange(len(rows)), index=rows.index)
                           for threshold, rows in eligible.items()}

    cells, state_rows, problem_rows = {}, [], []
    for regime, family, threshold in declared:
        cohort = eligible[threshold]
        per_seed, seed_states = [], []

        for seed in seeds:
            expected = evaluation_sets[(threshold, seed)]
            expected_rows = expected.to_numpy()
            expected_outcome = cohort.loc[expected].to_numpy(float)
            frame = scored.get((regime, family, threshold, seed))
            n_rows = 0 if frame is None else int(len(frame))

            def record(problem, count):
                problem_rows.append(dict(regime=regime, family=family, threshold=threshold,
                                         seed=seed, problem=problem, count=count,
                                         expected_respondents=len(expected_rows),
                                         realised_rows=n_rows))

            if (regime, family, threshold, seed) in non_estimable:
                if n_rows:
                    seed_states.append("invalid_handoff")
                    record("recorded_non_estimable_but_carries_scores", n_rows)
                else:
                    seed_states.append("non_estimable")
                continue

            if not n_rows:
                seed_states.append("invalid_handoff")
                record("recorded_estimable_but_carries_no_scores", len(expected_rows))
                continue

            # THE ARRIVAL ORDER IS THE EVALUATION ORDER, and it survives this check. Rows come
            # in the order that seed's held-out set was built in, which is what notebook 02
            # scored and what the capacity cut breaks its ties on. Sorting them here would have
            # left the bootstrap taking a different tied selection from every other consumer.
            #
            # The identity check needs the canonical order, because `_cell_problems` compares
            # the outcome against the cohort's own value row by row and the expected set is
            # sorted. So a sorted VIEW is taken for the check, and row_id, outcome and score
            # move together through it; the arrays kept are the unsorted ones.
            realised_rows = frame["row_id"].to_numpy()
            realised_outcome = frame["y_true"].to_numpy(float)
            realised_score = frame["score"].to_numpy(float)
            canonical = np.argsort(realised_rows, kind="stable")
            problems = _cell_problems(realised_rows[canonical], realised_outcome[canonical],
                                      expected_rows, expected_outcome)
            if problems:
                seed_states.append("invalid_handoff")
                for problem, count in problems.items():
                    record(problem, count)
                continue

            seed_states.append("complete")
            per_seed.append((positions_in_cohort[threshold].loc[realised_rows].to_numpy(),
                             realised_outcome, realised_score))

        n_complete = seed_states.count("complete")
        n_non_estimable = seed_states.count("non_estimable")
        n_invalid = seed_states.count("invalid_handoff")
        state = ("invalid_handoff" if n_invalid
                 else "non_estimable" if n_non_estimable else "complete")
        recorded = sorted({reasons.get((regime, family, threshold, seed), "recorded_non_estimable")
                           for seed in seeds
                           if (regime, family, threshold, seed) in non_estimable})
        state_rows.append(dict(regime=regime, family=family, threshold=threshold, state=state,
                               seeds_complete=n_complete, seeds_non_estimable=n_non_estimable,
                               seeds_invalid=n_invalid,
                               reason=("; ".join(recorded) if state == "non_estimable" else "")))
        if state == "complete":
            cells[(regime, family, threshold)] = per_seed

    for regime, family, threshold in undeclared:
        problem_rows.append(dict(regime=regime, family=family, threshold=threshold, seed=-1,
                                 problem="scored_cell_was_never_declared", count=0,
                                 expected_respondents=0, realised_rows=0))
        state_rows.append(dict(regime=regime, family=family, threshold=threshold,
                               state="invalid_handoff", seeds_complete=0,
                               seeds_non_estimable=0, seeds_invalid=len(seeds), reason=""))

    states = pd.DataFrame(state_rows, columns=[
        "regime", "family", "threshold", "state", "seeds_complete", "seeds_non_estimable",
        "seeds_invalid", "reason"])
    problems = pd.DataFrame(problem_rows, columns=[
        "regime", "family", "threshold", "seed", "problem", "count", "expected_respondents",
        "realised_rows"])

    summary_rows = []
    for threshold in sorted(eligible):
        at_threshold = states[states["threshold"] == threshold]
        sizes = [len(evaluation_sets[(threshold, seed)]) for seed in seeds]
        n_eligible = len(eligible[threshold])
        complete_here = [key for key in cells if key[2] == threshold]
        identical = True
        for position, seed in enumerate(seeds):
            rows_per_cell = [cells[key][position][0] for key in complete_here]
            identical &= all(np.array_equal(rows, rows_per_cell[0]) for rows in rows_per_cell)
        summary_rows.append(dict(
            threshold=threshold, eligible_respondents=n_eligible,
            test_set_min=min(sizes), test_set_max=max(sizes),
            test_share_min=round(min(sizes) / n_eligible, 4),
            test_share_max=round(max(sizes) / n_eligible, 4),
            declared_cells=int((at_threshold["state"] != "").sum()),
            complete=int((at_threshold["state"] == "complete").sum()),
            non_estimable=int((at_threshold["state"] == "non_estimable").sum()),
            invalid=int((at_threshold["state"] == "invalid_handoff").sum()),
            paired_rows_identical=bool(identical)))

    return {"cells": cells, "seeds": seeds, "states": states, "problems": problems,
            "summary": pd.DataFrame(summary_rows)}


# ---- one seed's metrics, on the expanded resample ---------------------------

def _expand(y_true, score, multiplicities):
    """The resample itself: each respondent repeated as many times as they were drawn."""
    repeated = np.repeat(np.arange(y_true.size), np.asarray(multiplicities, np.int64))
    return y_true[repeated], score[repeated]


def _seed_metrics(y_true, score, *, capacities: Sequence[float] = CAPACITIES) -> tuple:
    """One seed's metrics on rows already expanded. `(values, reason codes)`.

    THE SAME FUNCTIONS THE POINT ESTIMATE USES — `roc_auc_score`, `average_precision_score`, the
    mean squared error, `ece`, `cal_slope_intercept`, `cohort_capacity_flags` and
    `subgroup_capacity_metrics`. At unit multiplicity this reproduces `metrics` and
    `cohort_operating_metrics_per_seed` exactly, which is what makes the reconciliation an
    identity rather than an approximation.

    EACH METRIC STANDS OR FALLS ON ITS OWN. A replicate with one outcome class leaves ROC-AUC,
    PR-AUC and the calibration pair undefined while the Brier score and the ECE remain well
    defined.

    THE CAPACITY CUT IS REDERIVED HERE, on the expanded rows: `k = ceil(capacity x expanded n)`,
    the ranking retaken, the flags recomputed. The realised-sample boundary is never reused, and
    a respondent drawn three times occupies three ranking positions.
    """
    values, reasons = {}, {}

    def record(name, value, reason=""):
        values[name] = value
        reasons[name] = reason

    y_true = np.asarray(y_true, float)
    score = np.asarray(score, float)
    if y_true.size == 0:
        for metric in BOOTSTRAP_METRICS:
            record(metric, np.nan, "no_respondent_drawn")
        return values, reasons

    both_classes = np.unique(y_true).size == 2
    single = "single_outcome_class"
    record("auc", float(roc_auc_score(y_true, score)) if both_classes else np.nan,
           "" if both_classes else single)
    record("prauc", float(average_precision_score(y_true, score)) if both_classes else np.nan,
           "" if both_classes else single)
    record("brier", float(np.mean((score - y_true) ** 2)))
    record("ece", ece(y_true, score))
    if both_classes:
        slope, intercept = cal_slope_intercept(y_true, score)
        record("cal_slope", slope, "" if slope == slope else "calibration_did_not_converge")
        record("cal_intercept", intercept,
               "" if intercept == intercept else "calibration_no_sign_change")
    else:
        record("cal_slope", np.nan, single)
        record("cal_intercept", np.nan, single)

    flags = cohort_capacity_flags(score, capacities)
    for capacity in capacities:
        capacity_values, capacity_reasons = subgroup_capacity_metrics(
            y_true, flags[capacity], tag=int(round(capacity * 100)))
        values.update(capacity_values)
        for name, prose in capacity_reasons.items():
            reasons[name] = _CAPACITY_REASON_CODE.get(prose, "undefined" if prose else "")
    return values, reasons


def capacity_boundary_ties(score, capacities: Sequence[float] = CAPACITIES) -> list:
    """Whether the capacity cut had to break a tie, and how large the tied group was.

    One entry per capacity. The boundary score is the lowest score inside the flagged set, and a
    tie exists when more rows carry that score than there are places left for them. Counts only
    — no respondent is identified.
    """
    p = np.asarray(score, float)
    n = p.size
    out = []
    for capacity in capacities:
        k = int(np.ceil(capacity * n))
        if not n or not k:
            out.append(dict(capacity=capacity, boundary_tied=False, rows_at_boundary=0,
                            places_at_boundary=0))
            continue
        order = np.argsort(-p, kind="stable")
        boundary = p[order[k - 1]]
        rows_at_boundary = int((p == boundary).sum())
        places = k - int((p > boundary).sum())
        out.append(dict(capacity=capacity, boundary_tied=bool(rows_at_boundary > places),
                        rows_at_boundary=rows_at_boundary, places_at_boundary=int(places)))
    return out


def _metrics_for_each_seed(cell, multiplicities, metrics, capacities) -> tuple:
    """`(seeds, metrics)` values and reason codes for one cell under one replicate.

    `multiplicities` is that replicate's draw over the whole eligible cohort, or `None` for the
    unit-multiplicity pass. Each seed takes the multiplicities of the respondents in its own
    evaluation set, so a respondent in no test set contributes nothing.
    """
    values = np.full((len(cell), len(metrics)), np.nan)
    why = np.empty((len(cell), len(metrics)), dtype=object)
    why[:] = ""
    for seed_position, (universe_positions, y_true, score) in enumerate(cell):
        weights = (np.ones(y_true.size, np.int64) if multiplicities is None
                   else multiplicities[universe_positions])
        seed_values, seed_reasons = _seed_metrics(*_expand(y_true, score, weights),
                                                  capacities=capacities)
        for metric_position, metric in enumerate(metrics):
            values[seed_position, metric_position] = seed_values[metric]
            why[seed_position, metric_position] = seed_reasons.get(metric, "")
    return values, why


def _combine_seeds(values: np.ndarray, why: np.ndarray) -> tuple:
    """The displayed combination — the mean over ALL seeds — or a reason code.

    ALL SEEDS ARE REQUIRED. One undefined seed component invalidates the whole statistic for
    that replicate: a nineteen-seed mean is a different quantity from the twenty-seed one it
    would sit beside, so there is no survivor average.
    """
    undefined = ~np.isfinite(values)
    combined = np.where(undefined.any(axis=0), np.nan, values.mean(axis=0))
    codes = np.empty(values.shape[1], dtype=object)
    for metric_position in range(values.shape[1]):
        if not undefined[:, metric_position].any():
            codes[metric_position] = ""
            continue
        named = [why[seed, metric_position] for seed in range(values.shape[0])
                 if undefined[seed, metric_position] and why[seed, metric_position]]
        codes[metric_position] = named[0] if named else "seed_component_undefined"
    return combined, codes


# ---- the bootstrap ---------------------------------------------------------

def _bootstrap_from_multiplicities(coverage: Mapping,
                                   multiplicities: Mapping[str, np.ndarray], *,
                      metrics: Sequence[str] = BOOTSTRAP_METRICS,
                      capacities: Sequence[float] = CAPACITIES,
                      cells: Sequence[tuple] | None = None) -> dict:
    """Every complete cell's twenty-seed statistic under every replicate, and at unit weight.

    THE ONE EXPENSIVE STEP, RUN ONCE. Every interval, every transfer quantity, every paired
    change and both stability diagnostics are functions of what this returns, so the displays
    cannot disagree about what a replicate produced.

    `cells` restricts the work to named cells, which is how the notebook's cost probe runs a
    bounded piece of THE SAME calculation rather than a cheaper stand-in.
    """
    metrics = tuple(metrics)
    all_cells = coverage["cells"]
    wanted = list(all_cells) if cells is None else [tuple(key) for key in cells]
    replicate_counts = {len(multiplicities[key[2]]) for key in wanted}
    if len(replicate_counts) != 1:
        raise ValueError(
            f"the thresholds in scope were drawn with different replicate counts "
            f"({sorted(replicate_counts)}); a quantity combined across them would rest on "
            f"unequal evidence")
    replicates = replicate_counts.pop()

    seed_values, combined, combined_reasons = {}, {}, {}
    observed_seed_values, observed_rows, tie_rows = {}, [], []
    for key in wanted:
        cell = all_cells[key]
        unit_values, unit_why = _metrics_for_each_seed(cell, None, metrics, capacities)
        observed_seed_values[key] = unit_values
        for seed_position, (_, _, seed_score) in enumerate(cell):
            for entry in capacity_boundary_ties(seed_score, capacities):
                tie_rows.append(dict(regime=key[0], family=key[1], threshold=key[2],
                                     seed=seed_position, **entry))
        unit_combined, _ = _combine_seeds(unit_values, unit_why)
        for metric_position, metric in enumerate(metrics):
            observed_rows.append(dict(regime=key[0], family=key[1], threshold=key[2],
                                      metric=metric, observed=unit_combined[metric_position]))

        per_replicate = np.full((replicates, len(cell), len(metrics)), np.nan)
        per_replicate_combined = np.full((replicates, len(metrics)), np.nan)
        why_by_replicate = {}
        draw = multiplicities[key[2]]
        for replicate in range(replicates):
            values, why = _metrics_for_each_seed(cell, draw[replicate], metrics, capacities)
            per_replicate[replicate] = values
            per_replicate_combined[replicate], codes = _combine_seeds(values, why)
            for metric_position, code in enumerate(codes):
                if code:
                    why_by_replicate[(replicate, metrics[metric_position])] = code
        seed_values[key] = per_replicate
        combined[key] = per_replicate_combined
        combined_reasons[key] = why_by_replicate

    return {"metrics": metrics, "seeds": coverage["seeds"], "replicates": replicates,
            "capacities": tuple(capacities), "multiplicities": dict(multiplicities),
            "seed_values": seed_values,
            "combined": combined, "combined_reasons": combined_reasons,
            "observed_seed_values": observed_seed_values,
            "observed": pd.DataFrame(observed_rows),
            "capacity_ties": pd.DataFrame(tie_rows)}


def run_respondent_bootstrap(coverage: Mapping, eligible: Mapping[str, pd.Series], *,
                             replicates: int, seed: int,
                             metrics: Sequence[str] = BOOTSTRAP_METRICS,
                             capacities: Sequence[float] = CAPACITIES,
                             cells: Sequence[tuple] | None = None) -> dict:
    """Draw the respondent multiplicities and compute every cell's statistic under each one.

    One independent Poisson(1) multiplicity per eligible respondent per replicate. The same
    multiplicity follows that respondent into every seed whose held-out set contains them, and
    into every regime and family at that threshold, so the overlap between the twenty
    evaluations is carried rather than assumed away.

    Each threshold draws from its own stream, `seed + its position in `eligible``, because the
    eligible cohort and the held-out sets are both threshold-specific. The streams are nested:
    fifty replicates are the first fifty of two hundred, so comparing replicate counts compares
    prefixes of one draw.

    `cells` restricts the work to named cells, which is how a bounded cost probe runs the same
    calculation on a small piece of it.

    `replicates=0` IS THE OBSERVED PASS AND NOTHING ELSE. Every unit-multiplicity estimate is
    computed and no replicate is drawn, so the numbers the notebook reconciles against can be
    produced — and disagree — before the expensive run begins. The estimates are the same ones
    the full run reports, because the unit pass does not depend on how many replicates follow it.
    """
    multiplicities = {threshold: _draw_respondent_multiplicities(
                          cohort, replicates=replicates, seed=seed + position)
                      for position, (threshold, cohort) in enumerate(eligible.items())}
    return _bootstrap_from_multiplicities(coverage, multiplicities, metrics=metrics,
                                          capacities=capacities, cells=cells)


def _percentile_interval(values: np.ndarray, reasons: Sequence[str], *, alpha: float,
                         min_valid_fraction: float) -> dict:
    """The percentile interval, with the replicate count it rests on.

    A replicate is invalid for a quantity when any of the twenty seeds leaves that quantity
    undefined, and an invalid replicate is dropped rather than repaired — nothing is imputed,
    replaced with zero, or rebuilt from the seeds that survived.

    Because undefinedness is not random — it happens where the quantity would have been extreme
    — an interval built from what remains describes the distribution conditional on the
    statistic being defined. The interval is therefore reported only when at least
    `min_valid_fraction` of replicates are valid, and carries a short note whenever any were
    dropped.
    """
    values = np.asarray(values, float)
    usable = np.isfinite(values)
    good = values[usable]
    named = [reason or "seed_component_undefined"
             for reason, keep in zip(reasons, usable) if not keep]
    counts = pd.Series(named).value_counts() if named else pd.Series(dtype=int)
    requested = int(values.size)
    valid = int(good.size)
    fraction = valid / requested if requested else np.nan

    if not requested:
        return dict(requested_replicates=0, valid_replicates=0, valid_fraction=np.nan,
                    interval_lo=np.nan, interval_hi=np.nan, invalid_reasons="",
                    interval_note="no interval: this is the observed-estimate pass")

    if valid and fraction >= min_valid_fraction:
        low = float(np.percentile(good, 100 * alpha / 2))
        high = float(np.percentile(good, 100 * (1 - alpha / 2)))
        note = ("" if valid == requested else
                f"conditional on the statistic being defined; "
                f"{requested - valid} of {requested} replicates dropped")
    else:
        low = high = np.nan
        note = (f"no interval: only {valid} of {requested} replicates were defined"
                if valid else "no interval: the statistic was undefined in every replicate")

    return dict(requested_replicates=requested, valid_replicates=valid,
                valid_fraction=fraction, interval_lo=low, interval_hi=high,
                invalid_reasons="; ".join(f"{reason}={n}" for reason, n in counts.items()),
                interval_note=note)


def _replicate_reasons(cell_reasons: Mapping, metric: str, replicates: int) -> list:
    return [cell_reasons.get((replicate, metric), "") for replicate in range(replicates)]


def _performance_intervals(bootstrap, alpha, min_valid_fraction) -> pd.DataFrame:
    metrics = bootstrap["metrics"]
    observed = bootstrap["observed"].set_index(
        ["regime", "family", "threshold", "metric"])["observed"]
    replicates = bootstrap["replicates"]
    rows = []
    for key, combined in bootstrap["combined"].items():
        why = bootstrap["combined_reasons"][key]
        for position, metric in enumerate(metrics):
            rows.append(dict(
                quantity="performance", regime=key[0], family=key[1], threshold=key[2],
                metric=metric, baseline="", observed=float(observed.loc[(*key, metric)]),
                **_percentile_interval(combined[:, position],
                                       _replicate_reasons(why, metric, replicates),
                                       alpha=alpha, min_valid_fraction=min_valid_fraction)))
    return pd.DataFrame(rows)


def _transfer_intervals(bootstrap, alpha, min_valid_fraction, reference, metric,
                        transfer_loss_points) -> pd.DataFrame:
    """The distance to the YRBS local reference, formed inside each replicate from its means.

        distance_to_yrbs_local = AUC(yrbs_local) - AUC(method)

    On the `unadapted` row that IS the target resource gap; on an adapted row it is what the
    procedure still has to make up. One quantity rather than two, because with a single target
    anchor there is only one distance to report.

    Both procedures are evaluated on the same adolescents drawn the same number of times, so
    what the two share stays shared; resampling them apart would give an interval for a
    difference of independent quantities, which this is not.

    `target_attainment` — the chance-anchored share, formerly emitted here — is retired. Its
    numerator was anchored at chance rather than at unadapted transfer, so unadapted transfer
    scored highly on it before adapting anything, and it was read as a recovery when it was not.
    The recovery quantity is `target_gap_recovered`, in `_target_gap_intervals`.

    `transfer_loss` is appended as a POINT ESTIMATE only. It compares against the MCS local
    reference, and this notebook holds no MCS person-level scores, so the two cohorts cannot be
    resampled together and there is no interval to report.
    """
    metrics = bootstrap["metrics"]
    if metric not in metrics:
        raise ValueError(f"{metric} was not resampled; the metrics were {list(metrics)}")
    position = metrics.index(metric)
    observed = bootstrap["observed"].set_index(
        ["regime", "family", "threshold", "metric"])["observed"]
    combined = bootstrap["combined"]
    replicates = bootstrap["replicates"]

    rows = []
    for key in combined:
        regime, family, threshold = key
        reference_key = (reference, family, threshold)
        if regime == reference or reference_key not in combined:
            continue
        method = combined[key][:, position]
        anchor = combined[reference_key][:, position]
        why = [own or ref for own, ref in zip(
            _replicate_reasons(bootstrap["combined_reasons"][key], metric, replicates),
            _replicate_reasons(bootstrap["combined_reasons"][reference_key], metric,
                               replicates))]
        observed_method = float(observed.loc[(*key, metric)])
        observed_anchor = float(observed.loc[(*reference_key, metric)])

        rows.append(dict(
            quantity="transfer", regime=regime, family=family, threshold=threshold,
            metric="distance_to_yrbs_local", baseline=reference,
            observed=observed_anchor - observed_method,
            **_percentile_interval(anchor - method, why, alpha=alpha,
                                   min_valid_fraction=min_valid_fraction)))

    frame = pd.DataFrame(rows)
    if transfer_loss_points is not None and len(transfer_loss_points):
        # `transfer_loss` compares against the MCS local reference. Its point estimate comes from
        # the live notebook 02 summary; this notebook holds no MCS person-level scores, so the
        # two cohorts cannot be resampled together and there is no interval to report.
        points = transfer_loss_points.copy()
        points["quantity"] = "transfer"
        points["metric"] = "transfer_loss"
        points["baseline"] = "mcs_internal"
        points["interval_lo"] = np.nan
        points["interval_hi"] = np.nan
        points["interval_note"] = np.where(
            points["observed"].notna(),
            "point estimate only: no MCS person-level scores in this notebook",
            "no point estimate in the live summary, and no interval either")
        frame = pd.concat([frame, points], ignore_index=True)
    return frame


def _target_gap_intervals(bootstrap, alpha, min_valid_fraction, *, benchmark,
                          baseline, metric) -> pd.DataFrame:
    """The target-gap family, formed inside each replicate from its own twenty-seed means.

        target_resource_gap  = AUC(yrbs_local) - AUC(baseline)      cell-level
        adaptation_gain      = AUC(method)     - AUC(baseline)
        target_gap_recovered = adaptation_gain / target_resource_gap

    ONE TARGET ANCHOR. `benchmark` is the YRBS local reference. The `fixed_configuration_gap` and
    `remaining_resource_gap` rows this once emitted are retired with the arms they were read
    against — a YRBS-trained model under the MCS-selected configuration, and the per-split nested
    search. `_transfer_intervals` reports the distance from a method to the local reference.

    RATIO OF MEANS. Each difference is formed from twenty-seed means, so the numerator and
    denominator of the ratio are `mean(gain_s)` and `mean(gap_s)` — the same quantities the
    summary table carries, because the mean is linear. Dividing after averaging leaves ONE
    denominator that can be near zero rather than twenty, and it is what `add_reference_gaps`
    computes, so the two reconcile at the notebook's gate.

    `adaptation_gain` IS THE SAME NUMBER as the `paired_change` row for `metric` against the same
    baseline. It is emitted here as well because a reader needs the ratio's numerator beside its
    denominator; a test pins the two together rather than trusting the identity.

    THE DENOMINATOR RULE, and it is not a clamp. A replicate whose target resource gap is not
    strictly positive has no gap for a procedure to recover a share of, so its ratio is undefined
    and is recorded as such — the numerator and denominator rows for that replicate are
    unaffected and are always reported. The ratio's interval then follows the ordinary
    valid-fraction rule, so it appears when at least `min_valid_fraction` of replicates had a
    positive denominator and is withheld with a reason otherwise. Nothing is clipped, repaired or
    imputed.

    `min_target_resource_gap` reports the smallest gap any replicate produced, so a
    barely-positive denominator is visible rather than inferred from a wide interval.
    """
    metrics = bootstrap["metrics"]
    if metric not in metrics:
        raise ValueError(f"{metric} was not resampled; the metrics were {list(metrics)}")
    position = metrics.index(metric)
    observed = bootstrap["observed"].set_index(
        ["regime", "family", "threshold", "metric"])["observed"]
    combined = bootstrap["combined"]
    replicates = bootstrap["replicates"]

    def reasons_for(key):
        return _replicate_reasons(bootstrap["combined_reasons"][key], metric, replicates)

    def observed_for(key):
        try:
            return float(observed.loc[(*key, metric)])
        except KeyError:
            return np.nan

    cells = sorted({(family, threshold) for _, family, threshold in combined})
    rows = []
    for family, threshold in cells:
        benchmark_key = (benchmark, family, threshold)
        baseline_key = (baseline, family, threshold)
        if benchmark_key not in combined or baseline_key not in combined:
            continue

        gap = combined[benchmark_key][:, position] - combined[baseline_key][:, position]
        gap_why = [own or ref for own, ref in
                   zip(reasons_for(benchmark_key), reasons_for(baseline_key))]
        finite_gap = gap[np.isfinite(gap)]
        smallest_gap = float(finite_gap.min()) if finite_gap.size else np.nan

        rows.append(dict(
            quantity="target_gap", regime="", family=family, threshold=threshold,
            metric="target_resource_gap", baseline=baseline,
            observed=observed_for(benchmark_key) - observed_for(baseline_key),
            min_target_resource_gap=smallest_gap,
            **_percentile_interval(gap, gap_why, alpha=alpha,
                                   min_valid_fraction=min_valid_fraction)))

        positive = np.isfinite(gap) & (gap > 0)
        for regime, key_family, key_threshold in combined:
            if (key_family, key_threshold) != (family, threshold):
                continue
            if regime in (benchmark, baseline):
                continue
            key = (regime, family, threshold)
            method = combined[key][:, position]
            method_why = reasons_for(key)

            gain = method - combined[baseline_key][:, position]
            gain_why = [own or ref for own, ref in
                        zip(method_why, reasons_for(baseline_key))]
            rows.append(dict(
                quantity="target_gap", regime=regime, family=family, threshold=threshold,
                metric="adaptation_gain", baseline=baseline,
                observed=observed_for(key) - observed_for(baseline_key),
                min_target_resource_gap=np.nan,
                **_percentile_interval(gain, gain_why, alpha=alpha,
                                       min_valid_fraction=min_valid_fraction)))

            # UNDEFINED, NOT CLAMPED. Where the replicate's gap is not strictly positive the
            # ratio is NaN and carries its own reason, so `_percentile_interval` counts it out
            # under the ordinary valid-fraction rule instead of a value being invented.
            ratio = np.where(positive, gain / np.where(positive, gap, 1.0), np.nan)
            ratio_why = [reason or ("target_resource_gap_not_positive" if not ok else "")
                         for reason, ok in zip(gain_why, positive)]
            observed_gap = observed_for(benchmark_key) - observed_for(baseline_key)
            observed_gain = observed_for(key) - observed_for(baseline_key)
            observed_ratio = (observed_gain / observed_gap
                              if observed_gap == observed_gap and observed_gap != 0
                              else np.nan)
            rows.append(dict(
                quantity="target_gap", regime=regime, family=family, threshold=threshold,
                metric="target_gap_recovered", baseline=baseline,
                observed=observed_ratio, min_target_resource_gap=smallest_gap,
                **_percentile_interval(ratio, ratio_why, alpha=alpha,
                                       min_valid_fraction=min_valid_fraction)))

    return pd.DataFrame(rows)


def _paired_change_intervals(bootstrap, alpha, min_valid_fraction, baseline) -> pd.DataFrame:
    """The mean of the twenty within-seed differences from `baseline`, with its interval.

    Descriptive. Twenty overlapping held-out sets are not independent replicates, so there is no
    p-value, no significance marker and no multiplicity adjustment here.
    """
    metrics = bootstrap["metrics"]
    per_replicate = bootstrap["seed_values"]
    unit = bootstrap["observed_seed_values"]
    replicates = bootstrap["replicates"]
    rows = []
    for key in per_replicate:
        regime, family, threshold = key
        baseline_key = (baseline, family, threshold)
        if regime == baseline or baseline_key not in per_replicate:
            continue
        mean_change = (per_replicate[key] - per_replicate[baseline_key]).mean(axis=1)
        observed_change = (unit[key] - unit[baseline_key]).mean(axis=0)
        own_reasons = bootstrap["combined_reasons"][key]
        baseline_reasons = bootstrap["combined_reasons"][baseline_key]
        for position, metric in enumerate(metrics):
            why = [own_reasons.get((replicate, metric), "")
                   or baseline_reasons.get((replicate, metric), "")
                   for replicate in range(replicates)]
            rows.append(dict(
                quantity="paired_change", regime=regime, family=family, threshold=threshold,
                metric=metric, baseline=baseline,
                observed=float(observed_change[position]),
                **_percentile_interval(mean_change[:, position], why, alpha=alpha,
                                       min_valid_fraction=min_valid_fraction)))
    return pd.DataFrame(rows)


def _reconcile_observed(intervals: pd.DataFrame, reported: pd.DataFrame,
                        tolerance: float) -> pd.DataFrame:
    """Does the bootstrap recompute the numbers already reported?

    Counting every adolescent exactly once reproduces each seed's own held-out set, so the
    unit-multiplicity pass should return precisely the figures the rest of the notebook shows.
    A value blank on one side only counts as a disagreement, since one side judging a quantity
    estimable and the other not is itself a difference.
    """
    keys = ["quantity", "regime", "family", "threshold", "metric"]
    both = (intervals[keys + ["observed"]]
            .merge(reported[keys + ["mean"]], on=keys, how="inner"))
    usable = both[both["observed"].notna() & both["mean"].notna()]
    blank_one_side = int((both["observed"].isna() != both["mean"].isna()).sum())
    differences = (usable["observed"] - usable["mean"]).abs()
    worst = float(differences.max()) if len(differences) else np.nan
    status = ("mismatch" if blank_one_side or (len(differences) and worst > tolerance)
              else "not checked" if not len(usable) else "ok")
    return pd.DataFrame([dict(quantities_checked=int(len(usable)),
                              blank_on_one_side=blank_one_side,
                              max_abs_difference=worst, tolerance=tolerance, status=status)])


def bootstrap_intervals(bootstrap: Mapping, *, alpha: float = 0.05,
                        min_valid_fraction: float = 0.99,
                        reference: str = TARGET_REFERENCE, baseline: str = BASELINE_REGIME,
                        transfer_metric: str = "auc",
                        transfer_loss_points: pd.DataFrame | None = None,
                        reported: pd.DataFrame | None = None,
                        tolerance: float = 1e-9) -> tuple:
    """Every interval this analysis reports, and a one-line check that it reproduces the
    estimates already on display.

    Four groups, in the `quantity` column: `performance` (each metric for each model),
    `transfer` (the distance to the YRBS local reference, and `transfer_loss` as a point
    estimate), `target_gap` (the gap, the gain and the recovered share) and `paired_change` (the
    mean of the twenty within-seed differences from `baseline`).

    ONE TARGET ANCHOR. `reference` is the YRBS local reference, and every target-relative
    quantity is read against it. There was formerly a second — a YRBS-trained model under the
    MCS-selected configuration — and the `resource` group read quantities against a third, the
    per-split nested search. Both are retired: a model uses the settings selected in the cohort
    it is trained on, so the first no longer exists, and the second is an off-by-default
    sensitivity that anchors nothing.

    Returns `(intervals, reconciliation)`. Pass `reported` — a long frame of the values shown
    elsewhere in the notebook, keyed the same way with a `mean` column — to have the
    reconciliation computed; the caller stops when its status is not `ok`.
    """
    intervals = pd.concat([
        _performance_intervals(bootstrap, alpha, min_valid_fraction),
        _transfer_intervals(bootstrap, alpha, min_valid_fraction, reference, transfer_metric,
                            transfer_loss_points),
        _target_gap_intervals(bootstrap, alpha, min_valid_fraction, benchmark=reference,
                              baseline=baseline, metric=transfer_metric),
        _paired_change_intervals(bootstrap, alpha, min_valid_fraction, baseline),
    ], ignore_index=True)
    reconciliation = (_reconcile_observed(intervals, reported, tolerance)
                      if reported is not None else pd.DataFrame())
    return intervals, reconciliation


# ---- split conformal prediction --------------------------------------------
#
# A DIFFERENT OBJECT FROM THE BOOTSTRAP ABOVE, and it is kept in its own section for that
# reason. The bootstrap resamples respondents to describe how far a PERFORMANCE METRIC would
# move. Conformal prediction issues a PREDICTION SET for each respondent and reports how often
# that set contained the outcome. Neither supplies the other's quantity, and a conformal
# coverage is not an interval for an AUC.

CONFORMAL_ALPHA = 0.10
CONFORMAL_SPLIT_SEED = 20250902

# The models the conformal section covers. BOTH target references and unadapted transfer at the
# four focal families, the four declared focal pipelines, and their recalibrated counterparts —
# twenty cells, which is the main comparison rather than the whole battery.
CONFORMAL_FAMILIES: Sequence[str] = ("L1_LR", "RF", "HistGB", "CatBoost")


def conformal_models(focal_pipelines: Sequence[tuple]) -> list:
    """The `(regime, family)` cells the conformal section covers.

    Twenty at the declared scope: four families under each target reference, the four focal
    adapted pipelines, and those four recalibrated.

    `focal_pipelines` is `transfer.FOCAL_PIPELINES`, passed in rather than imported so this
    module does not pull in the modelling layer to answer a reporting question.

    The target local reference is covered alongside unadapted transfer. Coverage is a property of
    a fitted model's scores, so each fitted model that the analysis reports needs its own row.
    The nested target sensitivity is not covered here: it is off by default, so requiring its
    scores would make a headline coverage table depend on a sensitivity analysis having been run.
    """
    models = [("yrbs_local", family) for family in CONFORMAL_FAMILIES]
    models += [("unadapted", family) for family in CONFORMAL_FAMILIES]
    models += [(regime, family) for regime, family in focal_pipelines]
    models += [(f"{regime}_logistic_recal", family) for regime, family in focal_pipelines]
    return models


def conformal_split(held_out, *, threshold: int, seed: int,
                    split_seed: int = CONFORMAL_SPLIT_SEED) -> tuple:
    """One deterministic half-and-half division of a seed's held-out respondents.

    ONE DIVISION PER (THRESHOLD, SEED), SHARED BY EVERY MODEL COMPARED THERE. It is derived
    from the held-out respondents alone — no score, no outcome, no model — so two models'
    coverages differ because the models differ and not because they were calibrated on
    different people.

    UNSTRATIFIED. Balancing the halves on the outcome would break the exchangeability the
    coverage guarantee rests on, which is the one thing the division has to preserve.

    Returns `(calibration_rows, test_rows)`.
    """
    rows = pd.Index(held_out)
    generator = np.random.default_rng([int(split_seed), int(threshold), int(seed)])
    order = generator.permutation(len(rows))
    n_calibration = len(rows) // 2
    return rows[order[:n_calibration]], rows[order[n_calibration:]]


def conformal_quantile_index(n_calibration: int, alpha: float) -> tuple:
    """The finite-sample split-conformal order statistic, and whether it fits.

    `ceil((n + 1)(1 - alpha))` is the rank whose calibration score becomes the threshold. When
    that rank exceeds the calibration set it cannot be taken, and the honest reading is that
    this calibration set is too small to certify `1 - alpha`: the index is held at `n`, every
    label enters every set, and the cell is reported as `quantile_clipped` rather than as a
    coverage that looks perfect.

    Returns `(index, clipped)` with `index` one-based.
    """
    if n_calibration <= 0:
        return 0, True
    index = int(np.ceil((n_calibration + 1) * (1.0 - float(alpha))))
    if index > n_calibration:
        return n_calibration, True
    return index, False


def conformal_sets_per_seed(scores: pd.DataFrame, evaluation_sets: Mapping[tuple, pd.Index], *,
                            models: Sequence[tuple], thresholds: Sequence[int],
                            seeds: Sequence[int], alpha: float = CONFORMAL_ALPHA,
                            split_seed: int = CONFORMAL_SPLIT_SEED,
                            non_estimable: Sequence[tuple] = (),
                            reasons: Mapping[tuple, str] | None = None) -> pd.DataFrame:
    """Split-conformal prediction sets for each model, threshold and seed. One row per cell.

    THE NONCONFORMITY IS `1 - p(true label)`: `1 - score` for a positive respondent and `score`
    for a negative one. The threshold is the calibration half's order statistic; a label enters
    a test respondent's set when its own nonconformity does not exceed it. The set is therefore
    one of four things — empty, `{0}`, `{1}`, or `{0, 1}` — and those four are exhaustive.

    NOTHING IS REFITTED. Every score already comes from a model that never saw the held-out
    quarter, so dividing that quarter into a calibration and a test half needs no new fit and no
    new split from notebook 02.

    EVERY DECLARED MODEL-SEED IS ONE OF THREE THINGS, and none of them is silence.

    | state | what it means |
    |---|---|
    | scored | the cell covers that seed's held-out set exactly, and its sets are computed |
    | `non_estimable` | the cell was recorded non-estimable and carries no scores; the recorded reason travels with the row |
    | invalid handoff | anything else — missing without a record, or covering only part of the set — and it stops the analysis |

    A recalibration seed whose mapping could not be fitted is the second case. It is NEVER
    filled in from the raw model's scores: `target_only` and `fine_tune` are different models
    from their recalibrated versions, and substituting one for the other would report a coverage
    for a procedure that did not run.

    `non_estimable` is the same `(regime, family, threshold, seed)` vocabulary
    `validate_score_coverage` takes, and `reasons` maps those keys to what notebook 02 recorded.
    """
    def cell_key(key):
        return (str(key[0]), str(key[1]), str(key[2]), int(key[3]))

    recorded = {cell_key(key) for key in non_estimable}
    recorded_reasons = {cell_key(key): str(why) for key, why in dict(reasons or {}).items()}
    present = {key: frame for key, frame in
               scores.groupby(["regime", "family", "threshold", "seed"], sort=False)}

    rows, invalid = [], []
    for threshold in thresholds:
        label = f">={int(threshold)}"
        for seed in seeds:
            held_out = pd.Index(evaluation_sets[(label, int(seed))])
            calibration_rows, test_rows = conformal_split(
                held_out, threshold=threshold, seed=seed, split_seed=split_seed)
            for regime, family in models:
                key = (regime, family, label, int(seed))
                frame = present.get(key)
                record = dict(regime=regime, family=family, threshold=label, seed=int(seed),
                              alpha=float(alpha))
                if key in recorded:
                    if frame is not None:
                        invalid.append(dict(**record, problem="recorded_non_estimable_but_"
                                            "carries_scores", count=int(len(frame))))
                        continue
                    rows.append({**record, "status": "non_estimable",
                                 "reason": recorded_reasons.get(key,
                                                                "recorded_non_estimable")})
                    continue
                if frame is None:
                    invalid.append(dict(**record, problem="no_scores_and_no_non_estimable_"
                                        "record", count=len(held_out)))
                    continue
                covered = pd.Index(frame["row_id"])
                if len(covered) != len(held_out) or not covered.sort_values().equals(
                        held_out.sort_values()):
                    invalid.append(dict(**record, problem="does_not_cover_the_held_out_set",
                                        count=abs(len(held_out) - len(covered))))
                    continue
                rows.append({**record, **_conformal_cell(
                    frame, calibration_rows, test_rows, alpha=alpha)})

    if invalid:
        report = pd.DataFrame(invalid).groupby(
            ["regime", "family", "threshold", "problem"]).agg(
            seeds=("seed", "size"), worst_count=("count", "max")).reset_index()
        raise ValueError(
            f"{len(invalid)} model-seed(s) are neither scored on that seed's held-out set nor "
            f"recorded non-estimable, so a conformal result for them would rest on a partial or "
            f"substituted population:\n{report.to_string(index=False)}")
    return pd.DataFrame(rows)


def _conformal_cell(frame: pd.DataFrame, calibration_rows, test_rows, *, alpha: float) -> dict:
    """One model-seed's calibration threshold, prediction sets and coverage."""
    # `.loc`, not `reindex`: coverage was checked before this was called, so a respondent that
    # does not resolve is a defect and should say so rather than quietly shrink the half.
    indexed = frame.set_index("row_id")
    calibration = indexed.loc[calibration_rows]
    test = indexed.loc[test_rows]

    calibration_y = calibration["y_true"].to_numpy(float)
    calibration_p = calibration["score"].to_numpy(float)
    n_calibration = int(calibration_y.size)
    index, clipped = conformal_quantile_index(n_calibration, alpha)
    counts = dict(calibration_n=n_calibration,
                  calibration_positives=int(calibration_y.sum()),
                  test_n=int(len(test)),
                  test_positives=int(test["y_true"].to_numpy(float).sum()),
                  quantile_index=index, quantile_clipped=bool(clipped))
    if not n_calibration or not len(test):
        return {**counts, "status": "non_estimable",
                "reason": ("no calibration respondent" if not n_calibration
                           else "no test respondent")}

    calibration_nonconformity = np.where(calibration_y == 1, 1.0 - calibration_p, calibration_p)
    threshold_score = float(np.sort(calibration_nonconformity)[index - 1])

    test_y = test["y_true"].to_numpy(float)
    test_p = test["score"].to_numpy(float)
    holds_one = (1.0 - test_p) <= threshold_score
    holds_zero = test_p <= threshold_score
    covered = np.where(test_y == 1, holds_one, holds_zero)
    set_size = holds_one.astype(int) + holds_zero.astype(int)

    return {**counts, "status": "estimated", "reason": "",
            "nonconformity_threshold": threshold_score,
            "coverage": float(covered.mean()),
            "empty_share": float((set_size == 0).mean()),
            "singleton_share": float((set_size == 1).mean()),
            "both_labels_share": float((set_size == 2).mean()),
            "mean_set_size": float(set_size.mean())}


CONFORMAL_REPORTED: Sequence[str] = (
    "coverage", "singleton_share", "empty_share", "both_labels_share", "mean_set_size",
    "calibration_n", "calibration_positives", "test_n", "test_positives")


def summarise_conformal(per_seed: pd.DataFrame, *, seeds_expected: int = 20) -> pd.DataFrame:
    """Across-seed conformal summary. **The spread is split stability, not a coverage interval.**

    `split_sd` describes how the quantity moves across the twenty overlapping held-out sets. The
    coverage guarantee is marginal within each seed's own construction; averaging twenty of them
    describes what happened across those splits and is not a second guarantee about the mean.

    A cell is summarised only when every seed was estimated, for the same reason the rest of the
    module requires all twenty: a mean over whichever seeds happened to work is a different
    quantity from the one beside it.
    """
    if per_seed.empty:
        return per_seed
    rows = []
    for (regime, family, threshold), group in per_seed.groupby(
            ["regime", "family", "threshold"], sort=False):
        estimated = group[group["status"] == "estimated"]
        not_estimable = group[group["status"] == "non_estimable"]
        complete = len(estimated) == seeds_expected
        clipped = int(estimated["quantile_clipped"].sum()) if len(estimated) else 0
        why = "; ".join(f"{reason}={n}" for reason, n in
                        not_estimable["reason"].fillna("").value_counts().items())
        for metric in CONFORMAL_REPORTED:
            values = (estimated[metric].to_numpy(float) if metric in estimated.columns
                      else np.array([]))
            rows.append(dict(
                regime=regime, family=family, threshold=threshold, metric=metric,
                seeds_expected=seeds_expected, seeds_estimated=int(len(estimated)),
                seeds_non_estimable=int(len(not_estimable)),
                mean=float(values.mean()) if complete and values.size else np.nan,
                split_sd=float(values.std(ddof=1)) if complete and values.size > 1 else np.nan,
                seeds_with_clipped_quantile=clipped,
                status="complete" if complete else "incomplete",
                non_estimable_reasons=why))
    return pd.DataFrame(rows)


def yrbs_analytic_prevalence(thresholds: Sequence[int], *, pillars: pd.DataFrame | None = None,
                             pillars_path=None) -> pd.DataFrame:
    """Outcome prevalence in the threshold-specific evaluable YRBS analytic cohort.

    THE REFERENCE LINE ON THE VENTILE FIGURE. One row per threshold, computed from the
    canonical pillar frame through the same strict outcome construction notebook 02 uses — so
    the definition and the evaluable universe match the score handoff by construction rather
    than by a literal copied into a figure.

    It is NOT an average over the score rows: those repeat every respondent once per regime,
    threshold and seed, and averaging them would weight by how many procedures happened to run.
    """
    import data as D
    import config as _C
    if pillars is None:
        pillars = pd.read_parquet(Path(pillars_path) if pillars_path is not None
                                  else Path(_C.YRBS_PILLARS))
    rows = []
    for t in thresholds:
        y = D.make_outcome(pillars, int(t))
        rows.append(dict(threshold=f">={int(t)}", n_analytic=int(y.notna().sum()),
                         prevalence=float(y.mean())))
    return pd.DataFrame(rows)


def subgroup_metric_gaps(summary: pd.DataFrame) -> pd.DataFrame:
    """Highest minus lowest eligible cell, per (regime, family, threshold, metric).

    A CELL IS ELIGIBLE when its across-seed mean is complete and it met the minimum-size rule
    in every seed. **If any cell of the group is not eligible the gap is reported as
    incomplete and no value is given**, because a spread computed over the surviving cells is
    systematically smaller than the one the question asks about — and the excluded cells are
    usually the small ones carrying the disparity.
    """
    if summary.empty:
        return summary
    rows = []
    for (reg, fam, thr, metric), g in summary.groupby(
            ["regime", "family", "threshold", "metric"], sort=False):
        eligible = g[(g["estimability_status"] == "complete")
                     & (g["stability_status"] == "at_or_above_minimum_in_every_seed")
                     & g["mean"].notna()]
        excluded = g[~g.index.isin(eligible.index)]
        complete = len(excluded) == 0 and len(eligible) >= 2
        row = dict(regime=reg, family=fam, threshold=thr, metric=metric,
                   n_cells=int(len(g)), n_eligible=int(len(eligible)),
                   n_excluded=int(len(excluded)),
                   excluded_cells="; ".join(sorted(excluded["cell"])),
                   gap_status="complete" if complete else "incomplete")
        if complete:
            hi = eligible.loc[eligible["mean"].idxmax()]
            lo = eligible.loc[eligible["mean"].idxmin()]
            row.update(max_cell=hi["cell"], max_value=float(hi["mean"]),
                       min_cell=lo["cell"], min_value=float(lo["mean"]),
                       gap=float(hi["mean"]) - float(lo["mean"]))
        else:
            row.update(max_cell=None, max_value=np.nan, min_cell=None,
                       min_value=np.nan, gap=np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _seed_ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if not v.size:
        return np.nan, np.nan, np.nan, np.nan
    return (float(v.mean()), float(v.std()),
            float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


OPERATING_POINTS = tuple(round(0.05 * i, 2) for i in range(1, 21))


def operating_metrics(capacities=(0.05, 0.10, 0.15), *, thresholds=(1, 2),
                      tables_dir=None, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Top-q% precision and recall across every score-bearing config. Arithmetic only.

    TARGET-SIDE ONLY, and that is a licence constraint rather than an oversight: producing
    MCS operating metrics would require persisting row-level MCS scores.

    `capacities` selects from the twenty ventile-boundary operating points, 5% to 100%. The
    default is the three literature-anchored points — 5% is the Hello Baby PRM convention,
    15% is AFST mandatory screen-in, 10% is the intermediate — because those are the ones
    chosen for a reason rather than for completeness. The manuscript reports 10% and 20%;
    pass `capacities=(0.10, 0.20)` for exactly those, or `None` for the whole curve.

    A capacity that is not on the ventile grid RAISES rather than returning an empty frame.
    An empty frame is how "we did not measure this" becomes "there was no effect".

    COMPUTED FROM THE SCORE PARQUETS, not read back from a summary CSV. Each capacity is a cut
    on the model's own ranking of the YRBS test set, per seed, and the returned frame is the
    across-seed aggregate. `no_skill_precision` is the prevalence — print it beside every
    precision or the lift is illegible.

    THE `source` AND `lineage` COLUMNS HERE ARE THE PRODUCER'S, NOT THIS MODULE'S. Elsewhere
    in `evaluation` a `source` column carries provenance (`measured:...parquet`) and
    `lineage` carries `s4` / `budget`. In this frame `source` is the score family
    (`s4` / `e4` / `conformal`) and `lineage` is the preprocessing path
    (`cohort-std` / `raw-complete-case`). Both are more specific than the module convention
    and overwriting them would lose information, so they are carried through untouched.
    """
    caps = list(OPERATING_POINTS) if capacities is None else [round(float(c), 2)
                                                             for c in capacities]
    bad = [c for c in caps if c not in OPERATING_POINTS]
    if bad:
        raise ValueError(
            f"operating_metrics: capacities {bad} are not on the ventile grid. Available: "
            f"{', '.join(f'{c:.2f}' for c in OPERATING_POINTS)}. Cuts off the grid are a "
            f"different analysis, not an interpolation of this one.")
    want_thr = set(_threshold_labels(thresholds))

    rows = []
    for meta, fr in _score_sources(tables_dir):
        if meta["threshold"] not in want_thr:
            continue
        for seed, d in fr.groupby("seed"):
            y = d["y_true"].to_numpy(float)
            p = d["score"].to_numpy(float)
            keep = ~np.isnan(y)
            y, p = y[keep], p[keep]
            order = np.argsort(p)[::-1]
            n, npos = len(y), float(y.sum())
            prev = float(y.mean()) if n else np.nan
            for frac in caps:
                k = int(np.ceil(frac * n))
                tp = float(y[order[:k]].sum())
                prec = tp / k if k else np.nan
                rows.append({**meta, "seed": int(seed),
                             "operating_point": f"top_{int(round(frac * 100))}pct",
                             "precision": prec, "recall": (tp / npos if npos else np.nan),
                             "n_flagged": k, "no_skill_precision": prev,
                             "lift": (prec / prev if prev else np.nan), "n_test": n})
    per_seed = pd.DataFrame(rows)

    out = []
    for keys, g in per_seed.groupby(list(_OPS_META) + ["operating_point"], sort=False):
        r = dict(zip(list(_OPS_META) + ["operating_point"], keys))
        for m in ("precision", "recall", "lift"):
            mm, ss, lo, hi = _seed_ci(g[m].values)
            r[f"{m}_mean"], r[f"{m}_sd"] = round(mm, 4), round(ss, 4)
            r[f"{m}_lo"], r[f"{m}_hi"] = round(lo, 4), round(hi, 4)
        r["no_skill_precision"] = round(float(g["no_skill_precision"].mean()), 4)
        r["n_flagged_mean"] = round(float(g["n_flagged"].mean()), 1)
        r["n_test_mean"] = round(float(g["n_test"].mean()), 1)
        r["n_seeds"] = int(g["seed"].nunique())
        out.append(r)
    return pd.DataFrame(out)


def ventile_stratification(threshold: Threshold = 1, *, n_ventiles: int = 20,
                           tables_dir=None) -> pd.DataFrame:
    """Observed positive prevalence across the 20 predicted-risk ventiles, with across-seed CIs.

    COMPUTED FROM THE SCORE PARQUETS. One row per (config, threshold, ventile). Ventile 1 is
    the lowest-scoring five per cent and ventile 20 the highest, so a clean stratification
    reads as a rising column. `rr_top_bottom_mean` is the ratio of the two ends with its
    across-seed interval.

    Binning is on the RANK, not on the score value, so every ventile holds the same number of
    adolescents and the prevalences are comparable down the column.

    The monotonicity columns are a separate analysis — see `ventile_monotonicity`.
    """
    want = set(_threshold_labels([threshold]))
    rows = []
    for meta, fr in _score_sources(tables_dir):
        if meta["threshold"] not in want:
            continue
        acc = {v: [] for v in range(n_ventiles)}
        ratios = []
        for _, d in fr.groupby("seed"):
            y = d["y_true"].to_numpy(float)
            p = d["score"].to_numpy(float)
            keep = ~np.isnan(y)
            y, p = y[keep], p[keep]
            n = len(y)
            if not n:
                continue
            rank = np.argsort(np.argsort(p))
            vb = np.minimum(n_ventiles - 1, rank * n_ventiles // n)
            seen = {}
            for v in range(n_ventiles):
                m = vb == v
                if m.any():
                    seen[v] = float(y[m].mean())
                    acc[v].append(seen[v])
            if seen.get(0):
                ratios.append(seen[n_ventiles - 1] / seen[0])
        means = np.array([np.mean(acc[v]) if acc[v] else np.nan for v in range(n_ventiles)])
        rr_m, _, rr_lo, rr_hi = _seed_ci(ratios)
        for v in range(n_ventiles):
            rows.append({**meta, "ventile": v + 1,
                         "obs_prevalence_mean": means[v],
                         "obs_prevalence_sd": float(np.std(acc[v])) if acc[v] else np.nan,
                         "rr_top_bottom_mean": rr_m, "rr_lo": rr_lo, "rr_hi": rr_hi})
    return pd.DataFrame(rows)


def ventile_monotonicity(criterion: Literal["exact", "sd", "ci"] = "ci", *, tables_dir=None,
                         frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Noise-aware monotonicity test on the ventile curve.

    Three criteria, and the choice changes the headline count: `exact` (adjacent drop >
    0.005) yields 51 inversions of 111; `sd`-tolerant yields 6; `ci`-based yields 0. The
    CI-based criterion is the canonical one, with exact and sd kept as supplementary, plus a
    criterion-free Spearman (~0.98) alongside.

    THE NINE MONOTONICITY COLUMNS WERE WRITTEN IN PLACE. `spearman_*`, `n_inversions_*` and
    `monotone_*` were appended to `ventile_stratification_full.csv` by a second pass over the
    same file rather than written to a new one, so a copy taken before that pass does not
    carry them. This function does not rewrite anything; it reads, and it says so plainly when
    the columns are missing rather than returning a frame with holes in it.
    """
    df = _operational_frame("ventile_stratification_full.csv", tables_dir, frame)
    criteria = {"exact": "n_inversions_exact", "sd": "n_inversions_sd", "ci": "n_inversions_ci"}
    if criterion not in criteria:
        raise ValueError(f"ventile_monotonicity: criterion must be one of {sorted(criteria)}, "
                         f"not {criterion!r}.")
    needed = list(criteria.values()) + ["monotone_exact", "monotone_sd", "monotone_ci",
                                        "spearman_mean", "spearman_ci_lo", "spearman_ci_hi"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"ventile_stratification_full.csv is missing the monotonicity columns {missing}. "
            f"They were written IN PLACE by a second pass over the same file, so a copy made "
            f"before that pass does not carry them. This pipeline does not recompute them.")
    keys = ["source", "config", "label", "model", "arm", "lineage", "k", "threshold"]
    out = df.drop_duplicates(keys)[keys + needed].copy()
    out["criterion"] = criterion
    out["n_inversions"] = out[criteria[criterion]]
    out["monotone"] = out[{"exact": "monotone_exact", "sd": "monotone_sd",
                           "ci": "monotone_ci"}[criterion]]
    return out.reset_index(drop=True)


# 6. Consolidation of results
CONSOL_MODEL_CLASS = {**{f: "linear" for f in ("L1_LR", "L2_LR", "EN_LR")},
                      **{f: "bagged" for f in ("RF", "ET")},
                      **{f: "boosted" for f in ("HistGB", "LightGBM", "XGB", "CatBoost")}}
CONSOL_FAM_ALIAS = {"l1lr": "L1_LR", "l2lr": "L2_LR", "enlr": "EN_LR", "randomforest": "RF",
                    "rf": "RF", "et": "ET", "extratrees": "ET", "histgb": "HistGB",
                    "lightgbm": "LightGBM", "xgboost": "XGB", "xgb": "XGB",
                    "catboost": "CatBoost"}

# VERIFIED: all 72 values match the draft's Tables VI and VII.
# These are hand-typed transcriptions of the published tables, not derived quantities, which
# is why they are declared here. The `notes` column still carries the source script's own
# "VERIFY vs paper" wording; rewording it would change 72 rows of the frozen all_results.csv.
PAPER_IV: Mapping[str, Mapping[str, tuple]] = {   # naive, tuned
    "L1_LR": {">=1": (0.743, 0.879), ">=2": (0.723, 0.705)},
    "L2_LR": {">=1": (0.744, 0.880), ">=2": (0.724, 0.706)},
    "EN_LR": {">=1": (0.743, 0.878), ">=2": (0.726, 0.707)},
    "RF": {">=1": (0.755, 0.886), ">=2": (0.739, 0.721)},
    "ET": {">=1": (0.747, 0.881), ">=2": (0.733, 0.716)},
    "HistGB": {">=1": (0.744, 0.880), ">=2": (0.730, 0.708)},
    "LightGBM": {">=1": (0.741, 0.879), ">=2": (0.732, 0.710)},
    "XGB": {">=1": (0.716, 0.866), ">=2": (0.702, 0.687)},
    "CatBoost": {">=1": (0.751, 0.885), ">=2": (0.734, 0.712)},
}
PAPER_V: Mapping[str, Mapping[str, tuple]] = {    # target_only, tuned, k=500
    "L1_LR": {">=1": (0.764, 0.893), ">=2": (0.756, 0.747)},
    "L2_LR": {">=1": (0.766, 0.895), ">=2": (0.758, 0.750)},
    "EN_LR": {">=1": (0.704, 0.833), ">=2": (0.759, 0.751)},
    "RF": {">=1": (0.769, 0.896), ">=2": (0.758, 0.748)},
    "ET": {">=1": (0.770, 0.896), ">=2": (0.758, 0.749)},
    "HistGB": {">=1": (0.755, 0.888), ">=2": (0.752, 0.740)},
    "LightGBM": {">=1": (0.751, 0.886), ">=2": (0.742, 0.729)},
    "XGB": {">=1": (0.719, 0.869), ">=2": (0.690, 0.680)},
    "CatBoost": {">=1": (0.756, 0.889), ">=2": (0.749, 0.735)},
}
# The seven ingested files, in load order.
#
# TWO SOURCES WERE RETIRED WITH THE STABILITY-SELECTION EXPERIMENT: `stability_tuning_arm.csv`
# and `stability_tuning_configs.csv`. The variance-penalised selection criterion they described
# was withdrawn along with `models.stability_selected_params`, so no notebook, script or module
# produces either file. Keeping them here would have been a tolerant reader of an experiment that
# no longer exists, and an absent source is skipped silently — so their names are gone rather
# than merely unproduced. This is unrelated to the subgroup minimum-cell rule and to the
# split-stability summaries elsewhere in this module, which are live and are named "stability"
# for a different reason.
CONSOL_SOURCES: Sequence[str] = (
    "label_budget_curve.csv", "label_budget_summary.csv", "per_family_calibration.csv",
    "subgroup_calibration.csv", "subgroup_by_family.csv", "conformal_per_family.csv",
    "conformal_threshold2.csv",
)


def _famkey(x):
    import re
    n = re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())
    return CONSOL_FAM_ALIAS.get(n, x)


# THE CONSOLIDATION AND CANONICAL LAYERS READ FROZEN TABLES, so they keep the frozen spelling
# of the two renamed anchors: `naive` and `yrbs_ceiling`, not `unadapted` and `yrbs_internal`.
# The frozen files under outputs/ were published under the old keys and still carry them.
# regime_names.RENAMED_KEYS records the pair; the two vocabularies meet at a republication.
def _consol_row(source_file, family, metric, mean, *, arm="tuned", regime="naive",
                threshold="", k="NA", sd="", n_seeds="", subgroup="marginal", notes=""):
    return dict(source_file=source_file, family=family,
                model_class=CONSOL_MODEL_CLASS.get(family, ""), arm=arm, regime=regime,
                threshold=threshold, k=k, metric=metric, mean=mean, sd=sd, n_seeds=n_seeds,
                subgroup=subgroup, notes=notes)


# COMPARING THIS AGAINST A STORED FRAME NEEDS A CSV ROUND TRIP, and that changes the answer.
# The stored file was written with `to_csv(index=False)` and is read back the same way, so a
# rebuilt frame must be round-tripped before the diff. Comparing the in-memory frame against
# the re-read CSV compares pandas' type coercion instead of the transcription: `read_csv` turns
# `k`'s "NA" into NaN, making the column float64, and `sd`/`n_seeds`' "" into NaN. A correct
# transcription then reports a total key-match failure.
#
# The full key is (source_file, family, arm, regime, threshold, k, metric, subgroup).
def consolidate_all_results(outputs_dir=None, *, frames: Mapping[str, pd.DataFrame] | None = None
                            ) -> pd.DataFrame:
    """Assemble $THESIS_WORK_DIR/all_results.csv from the seven analysis CSVs. Computes NOTHING.

    Pure consolidation: every value is copied from an existing aggregate CSV and relabelled
    onto one 13-column schema. No metric is recomputed, so this cannot disagree with its
    sources — it can only fail to find them.

    ABSENT SOURCES ARE SKIPPED, NOT FATAL, which is the source's behaviour and worth keeping:
    the point of the consolidation is to show what exists, so a missing file should shrink
    the table and be reported, not stop the run.

    THE TWO KNOWN ISSUES, both carried across with their current status:

      1. **The 72 hand-typed paper rows are now VERIFIED.** `PAPER_IV` / `PAPER_V` are typed
         from the paper rather than read from a file, and the script self-flagged them
         "VERIFY vs paper". All 72 have since been compared against the manuscript's Tables VI
         and VII — 72 match, 0 mismatch. They manufacture no false
         disagreements in the reconciliation. They are still hand-typed, so they are still the
         thing to re-check if the paper's tables move.
      2. **Two written files are still not ingested.** `calibration_correction_sweep.csv` and
         `subgroup_precision_at_capacity.csv` are produced by live scripts and loaded by
         nothing, so neither reaches `all_results.csv`. The first is the SOLE source of five
         claims in §VI-D. Adding them would change the row count, so this transcription does
         NOT add them — the frozen file is what the pipeline currently reproduces, and
         CLAUDE.md's rule is that the source file wins. `INGEST_GAPS` below names them.

    NOT A SUPERSET OF `CANONICAL.csv`, AND NOT A SUBSET. This table carries the `analysis/`
    lineage — one row per family x regime x threshold x k x metric. It holds NO S4 battery
    rows at all, which is exactly why the 72 hand-typed rows exist:
    they are the only way the paper's transfer grid appears here. `build_canonical_table`
    carries the other lineage. The two share no rows.

    Returns the frame. Does NOT write — `$THESIS_WORK_DIR/all_results.csv` is read-only.
    """
    from pathlib import Path
    outputs_dir = Path(outputs_dir) if outputs_dir is not None else _work_root()
    loaded: dict = {}
    for name in CONSOL_SOURCES:
        if frames is not None and name in frames:
            loaded[name] = frames[name]
            continue
        p = outputs_dir / name
        loaded[name] = pd.read_csv(p) if p.exists() else None

    lbc = loaded["label_budget_curve.csv"]
    lbs = loaded["label_budget_summary.csv"]
    pfc = loaded["per_family_calibration.csv"]
    sgc = loaded["subgroup_calibration.csv"]
    sbf = loaded["subgroup_by_family.csv"]
    cpf = loaded["conformal_per_family.csv"]
    ct2 = loaded["conformal_threshold2.csv"]

    R = _consol_row
    rows = []
    if lbc is not None:
        for _, r in lbc.iterrows():
            if str(r["family"]).startswith("("):     # (pool) n_pos rows -> context only
                continue
            rows.append(R("label_budget_curve.csv", _famkey(r["family"]), r["metric"], r["mean"],
                          regime=r["regime"], threshold=r["threshold"], k=str(int(r["k"])),
                          sd=r["sd"], n_seeds=r["n_seeds"]))
    if lbs is not None:
        summap = {"crossover_target_only": ("target_only", "crossover_k"),
                  "crossover_full_revision": ("full_revision", "crossover_k"),
                  "ceiling_target_only": ("target_only", "ceiling_k"),
                  "ceiling_full_revision": ("full_revision", "ceiling_k"),
                  "flip_target_over_full": ("target_only", "flip_k"),
                  "calib_usable_target_only": ("target_only", "calib_usable_k"),
                  "calib_usable_full_revision": ("full_revision", "calib_usable_k"),
                  "prec5_delta_naive_to_target_only_k500":
                      ("target_only", "prec5_delta_vs_naive_k500"),
                  "prec5_delta_naive_to_full_revision_k500":
                      ("full_revision", "prec5_delta_vs_naive_k500")}
        for _, r in lbs.iterrows():
            for col, (regime, metric) in summap.items():
                if col in r:
                    rows.append(R("label_budget_summary.csv", _famkey(r["family"]), metric,
                                  r[col], regime=regime, threshold=r["threshold"]))
    if pfc is not None:
        for _, r in pfc.iterrows():
            rows.append(R("per_family_calibration.csv", _famkey(r["family"]), r["metric"],
                          r["mean"], regime=r["regime"], threshold=r["threshold"],
                          sd=r["sd"], n_seeds=r["n_seeds"]))
    if sgc is not None:
        for _, r in sgc.iterrows():
            rows.append(R("subgroup_calibration.csv", _famkey(r["family"]), r["metric"],
                          r["mean"], regime=r["regime"], threshold=r["threshold"], sd=r["sd"],
                          n_seeds=r["n_seeds"], subgroup=r["cell"],
                          notes=("suppressed" if r.get("suppressed") else "")))
    if sbf is not None:
        for _, r in sbf.iterrows():
            rows.append(R("subgroup_by_family.csv", _famkey(r["family"]), "auc", r["auc_mean"],
                          arm=r["arm"], regime="naive", threshold=r["threshold"],
                          sd=r["auc_sd"], n_seeds=r["n_seeds"], subgroup=r["cell"],
                          notes=f"rank={r.get('rank')};gap={r.get('gap')}"))
    if cpf is not None:
        for _, r in cpf.iterrows():
            fam = _famkey(r["family"])
            for metric, mc, sc in (("coverage", "coverage_mean", "coverage_sd"),
                                   ("abstention", "abstention_mean", "abstention_sd")):
                rows.append(R("conformal_per_family.csv", fam, metric, r[mc],
                              arm=("frozen" if fam == "CatBoost" else "tuned"),
                              regime=f"conformal:{r['config']}", threshold=">=1", sd=r[sc],
                              n_seeds=r["n_seeds"], subgroup=r["scope"],
                              notes=(f"cal_n={r.get('cal_n_mean')};"
                                     f"flag_lt50={r.get('flag_lt50')};"
                                     f"supp={r.get('suppressed')}")))
    if ct2 is not None:
        for _, r in ct2.iterrows():
            for metric, mc, sc in (("coverage", "coverage_mean", "coverage_sd"),
                                   ("abstention", "abstention_mean", "abstention_sd")):
                rows.append(R("conformal_threshold2.csv", "CatBoost", metric, r[mc],
                              arm="frozen", regime=f"conformal:{r['config']}",
                              threshold=r["threshold"], sd=r[sc], n_seeds=r["n_seeds"],
                              subgroup=r["scope"],
                              notes=(f"cal_n={r.get('cal_n_mean')};"
                                     f"flag_lt50={r.get('flag_lt50')};"
                                     f"supp={r.get('suppressed')}")))
    for tag, tab, regime, k in (("paper_table_IV", PAPER_IV, "naive", "NA"),
                                ("paper_table_V", PAPER_V, "target_only", "500")):
        for fam, d in tab.items():
            for thr, (auc, prauc) in d.items():
                rows.append(R(tag, fam, "auc", auc, regime=regime, threshold=thr, k=k,
                              notes="VERIFY vs paper"))
                rows.append(R(tag, fam, "prauc", prauc, regime=regime, threshold=thr, k=k,
                              notes="VERIFY vs paper"))
    return pd.DataFrame(rows)


# Written by a live script, read by nothing, and therefore absent from all_results.csv.
# Recorded here rather than in a comment so a caller can assert on it.
INGEST_GAPS: Mapping[str, str] = {
    "calibration_correction_sweep.csv":
        "sole source of five claims in the paper's SS VI-D; ingested by nothing (PROVENANCE Q6)",
    "subgroup_precision_at_capacity.csv":
        "read by nothing; SS VI-F uses subgroup_operational.csv instead (PROVENANCE Q6)",
}



# 6b. CANONICAL.csv — the other lineage
CANONICAL_COLUMNS: Sequence[str] = (
    "claim_id", "metric", "value", "sd", "ci_lo", "ci_hi", "prevalence_null", "cohort",
    "cell", "family", "regime", "arm", "threshold", "k", "lineage", "n_seeds",
    "source_file", "source_script", "generation", "status", "finding", "paper_facing", "note",
)


def _slug(*parts) -> str:
    return ":".join(str(p) for p in parts if p not in (None, "", "nan"))


# `claim_id` IS THE JOIN KEY and is unique per row, which is what lets a rebuilt table be
# compared with a stored one at all. `classify_canonical_row` supplies `finding` and
# `paper_facing` and is the only real logic here; everything else is lookup and assembly.
# Comparing against a stored table needs the same CSV round trip as the consolidation above.
def build_canonical_table(tables_dir=None, *,
                          frames: Mapping[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Assemble $THESIS_WORK_DIR/tables/CANONICAL.csv: one row per reportable number, with provenance.

    LOOKUP AND ASSEMBLY ON FROZEN OUTPUTS. No re-runs, no refits, no metric recomputed —
    every value is copied from an existing summary table. The two exceptions are explicitly
    derived and labelled `source_file="(derived)"`: the max-minus-min subgroup AUC gap and the
    flag-rate gap, both of which are arithmetic on rows this function has already ingested.

    THE STATUS RULES ARE THE POINT OF THE TABLE, not decoration:
      * S4 is canonical wherever it covers a method.
      * Freeze-week is canonical only where S4 does not reach.
      * A backfill COHORT-STD number supersedes the freeze-week RAW number, and the raw one is
        retained as `status="reproduction"` with its lineage stated rather than deleted. A raw
        and a cohort-std number for the same method are DIFFERENT LINEAGES, not a conflict —
        the source says so explicitly about leaf_refresh 0.7296 vs 0.7207.
      * Four things are demoted by name: degenerate BBSE (`excluded`), EN_LR tuned >=1
        (`defect` — C=0.01 sat at the grid boundary and every seed underfits),
        pseudo_label_thresh at >=1 (`spec-artefact` — a symmetric 0.8/0.2 threshold against a
        0.727 prevalence), and the depth-2 grid (`mechanism-only`, never headline).
      * Ventile monotonicity: the CI-based count (0 of 111) is canonical; exact (51) and SD (6)
        are `supplementary`. The criterion changes the headline by a factor of fifty, so which
        one is canonical has to be recorded rather than assumed.

    ONE SUPERSESSION WORTH KNOWING ABOUT, from the source's closing note: naive CatBoost
    untuned >=1 is 0.7461 here (S4, `thread_count=1`, deterministic). The pre-S4
    `catboost.csv` value 0.7464 is SUPERSEDED by thread nondeterminism and is deliberately
    not ingested. 0.7464 also appears separately as a valid-cell-only subgroup marginal, which
    is a different quantity — so seeing it in the repository is not evidence of a conflict.

    NOT A SUPERSET OF `all_results.csv`, AND NOT A SUBSET. Different provenance, different
    scope, no shared rows. See `consolidate_all_results`.

    Returns the frame. Does NOT write — `CANONICAL.csv` is read-only.
    """
    from pathlib import Path
    tables_dir = Path(tables_dir) if tables_dir is not None else _work_tables()

    def load(name):
        if frames is not None and name in frames:
            return frames[name]
        p = tables_dir / name
        return pd.read_csv(p) if p.exists() else None

    def g(row, col, default=""):
        return row[col] if col in row and pd.notna(row[col]) else default

    rows: list = []

    def add(metric, value, *, sd="", ci_lo="", ci_hi="", null="", cohort="", cell="", family="",
            regime="", arm="", threshold="", k="", lineage="", n_seeds=20, src="", script="",
            gen="", status="canonical", note=""):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return
        cid = _slug(gen, cohort, cell, family, regime, arm, threshold, lineage,
                    f"k{k}" if str(k) else "", metric)
        rows.append(dict(
            claim_id=cid, cell=cell, metric=metric, value=round(float(value), 6),
            sd=(round(float(sd), 6) if sd not in ("", None) and sd == sd else ""),
            ci_lo=(round(float(ci_lo), 6) if ci_lo not in ("", None) and ci_lo == ci_lo else ""),
            ci_hi=(round(float(ci_hi), 6) if ci_hi not in ("", None) and ci_hi == ci_hi else ""),
            prevalence_null=(round(float(null), 6)
                             if null not in ("", None) and null == null else ""),
            cohort=cohort, family=family, regime=regime, arm=arm, threshold=threshold, k=k,
            lineage=lineage, n_seeds=n_seeds, source_file=src, source_script=script,
            generation=gen, status=status, note=note))

    # ---- S4 battery: the canonical bulk ----
    s4 = load("regime_battery_summary.csv")
    if s4 is not None:
        for _, r in s4.iterrows():
            fam, arm, reg, thr = r["family"], r["arm"], r["regime"], r["threshold"]
            cohort = "MCS" if reg == "mcs_internal" else "YRBS"
            status, note = "canonical", ""
            if reg == "bbse" and bool(g(r, "degenerate", False)):
                status, note = "excluded", "degenerate BBSE (w clipped to 0)"
            elif fam == "EN_LR" and arm == "tuned" and thr == ">=1":
                status, note = "defect", "EN_LR C=0.01 at grid boundary; all-seed underfit"
            elif reg == "pseudo_label_thresh" and thr == ">=1":
                status, note = "spec-artefact", "symmetric threshold at 0.727 prevalence"
            base = dict(cohort=cohort, family=fam, regime=reg, arm=arm, threshold=thr,
                        lineage="cs", k=g(r, "k", ""), n_seeds=int(g(r, "n_seeds", 20)),
                        src="regime_battery_summary.csv",
                        script="run_s4_regime_battery.py", gen="S4", status=status, note=note)
            add("auc", g(r, "auc_mean", np.nan), sd=g(r, "auc_sd"), ci_lo=g(r, "auc_plo"),
                ci_hi=g(r, "auc_phi"), **base)
            add("prauc_lift", g(r, "prauc_lift_mean", np.nan), sd=g(r, "prauc_lift_sd"),
                ci_lo=g(r, "prauc_lift_plo"), ci_hi=g(r, "prauc_lift_phi"),
                null=g(r, "prevalence"), **base)
            add("brier", g(r, "brier_mean", np.nan), sd=g(r, "brier_sd"),
                ci_lo=g(r, "brier_plo"), ci_hi=g(r, "brier_phi"), **base)
            add("ece", g(r, "ece_mean", np.nan), sd=g(r, "ece_sd"), ci_lo=g(r, "ece_plo"),
                ci_hi=g(r, "ece_phi"), **base)

    # ---- backfill: cohort-std canonical, raw retained as reproduction ----
    bf = load("backfill_summary.csv")
    if bf is not None:
        for _, r in bf.iterrows():
            fam, method, arm = r["family"], r["method"], r["arm"]
            thr, lin = r["threshold"], r["lineage"]
            status, note = "canonical", ""
            if lin == "raw":
                status, note = "reproduction", \
                    "raw-lineage reproduction of frozen freeze-week value"
            if method == "ablation:depth=2":
                status, note = "mechanism-only", "depth-2 grid peek; never headline"
            base = dict(cohort="YRBS", family=fam, regime=method, arm=arm, threshold=thr,
                        lineage=lin, k=g(r, "k", ""), n_seeds=int(g(r, "n_seeds", 20)),
                        src="backfill_summary.csv",
                        script="backfill_*.py + backfill_consolidate.py", gen="backfill",
                        status=status, note=note)
            add("auc", g(r, "auc_mean", np.nan), sd=g(r, "auc_sd"), ci_lo=g(r, "auc_plo"),
                ci_hi=g(r, "auc_phi"), **base)
            add("prauc_lift", g(r, "prauc_lift_mean", np.nan), sd=g(r, "prauc_lift_sd"),
                ci_lo=g(r, "prauc_lift_plo"), ci_hi=g(r, "prauc_lift_phi"),
                null=g(r, "prevalence"), **base)
            add("brier", g(r, "brier_mean", np.nan), sd=g(r, "brier_sd"), **base)
            add("ece", g(r, "ece_mean", np.nan), sd=g(r, "ece_sd"), **base)

    bhs = load("backfill_headline_significance.csv")
    if bhs is not None:
        for _, r in bhs.iterrows():
            add(f"HL_{r['metric']}_leaf_refresh_{r['comparison']}", g(r, "hl", np.nan),
                ci_lo=g(r, "hl_lo"), ci_hi=g(r, "hl_hi"), cohort="YRBS", family="XGB",
                regime="leaf_refresh", arm=r["arm"], threshold=r["threshold"], lineage="cs",
                k=500, src="backfill_headline_significance.csv",
                script="backfill_consolidate.py", gen="backfill", status="canonical",
                note=f"p_holm={g(r, 'p_holm')} reject={g(r, 'holm_reject')}")

    # ---- subgroup: YRBS intersectional, both-cohort marginal panels, operational ----
    sd_ = load("subgroup_discrimination_summary.csv")
    if sd_ is not None:
        for _, r in sd_.iterrows():
            if r["scope"] != "cell":
                continue
            add("auc", g(r, "auc_mean", np.nan), sd=g(r, "auc_sd"), ci_lo=g(r, "auc_plo"),
                ci_hi=g(r, "auc_phi"), cohort="YRBS", cell=r["cell"], family=g(r, "family"),
                regime=f"subgroup4:{str(r['config']).split(':')[0]}", arm=g(r, "arm"),
                threshold=r["threshold"], lineage="cs",
                src="subgroup_discrimination_summary.csv",
                script="subgroup_discrimination.py", gen="subgroup",
                status=("unstable" if bool(g(r, "unstable", False)) else "canonical"),
                note=f"prev={g(r, 'prevalence_mean')}")

    sp = load("subgroup_panels_summary.csv")
    if sp is not None:
        for _, r in sp.iterrows():
            if r["panel"] == "marginal":
                continue
            pnote = f"prev={g(r, 'prevalence_mean')} n={g(r, 'n_mean')}"
            if r["cohort"] == "MCS" and r["panel"] in ("2_binary", "3_ethnicity"):
                pnote += " | FDCE0600 verified (UKDS SN8156 dict, 2026-08-04)"
            add("auc", g(r, "auc_mean", np.nan), sd=g(r, "auc_sd"), ci_lo=g(r, "auc_lo"),
                ci_hi=g(r, "auc_hi"), cohort=r["cohort"], cell=r["cell"],
                family=g(r, "family"), regime=f"panel:{r['panel']}:{r['config']}",
                arm=g(r, "arm"), threshold=r["threshold"], lineage=g(r, "lineage", "cs"),
                src="subgroup_panels_summary.csv", script="subgroup_panels.py",
                gen="subgroup",
                status=("suppressed" if bool(g(r, "suppressed", False)) else "canonical"),
                note=pnote)

    so = load("subgroup_operational.csv")
    if so is not None:
        for _, r in so.iterrows():
            if r["cell"] == "__marginal__":
                continue
            for m in ("fpr15", "flagrate15", "precision15", "recall15"):
                add(m, g(r, f"{m}_mean", np.nan), sd=g(r, f"{m}_sd"), ci_lo=g(r, f"{m}_lo"),
                    ci_hi=g(r, f"{m}_hi"), cohort="YRBS", cell=r["cell"], family="CatBoost",
                    regime=f"screenin:{r['config']}", threshold=r["threshold"], lineage="cs",
                    src="subgroup_operational.csv", script="operating_metrics.py",
                    gen="operational",
                    status=("unstable" if bool(g(r, "unstable", False)) else "canonical"),
                    note=f"prev={g(r, 'prevalence_mean')}")

    # ---- operating points ----
    op = load("operating_point_metrics_summary.csv")
    if op is not None:
        for _, r in op.iterrows():
            base = dict(cohort="YRBS", family=g(r, "model"),
                        regime=f"{r['config']}@{r['operating_point']}", arm=g(r, "arm"),
                        threshold=r["threshold"], lineage=g(r, "lineage"), k=g(r, "k", ""),
                        src="operating_point_metrics_summary.csv",
                        script="operating_metrics.py", gen="operational",
                        status="canonical", note=f"no_skill={g(r, 'no_skill_precision')}")
            add("precision", g(r, "precision_mean", np.nan), sd=g(r, "precision_sd"),
                ci_lo=g(r, "precision_lo"), ci_hi=g(r, "precision_hi"),
                null=g(r, "no_skill_precision"), **base)
            add("recall", g(r, "recall_mean", np.nan), sd=g(r, "recall_sd"),
                ci_lo=g(r, "recall_lo"), ci_hi=g(r, "recall_hi"), **base)
            add("op_lift", g(r, "lift_mean", np.nan), sd=g(r, "lift_sd"), ci_lo=g(r, "lift_lo"),
                ci_hi=g(r, "lift_hi"), null=g(r, "no_skill_precision"), **base)

    # ---- ventile monotonicity ----
    vent = load("ventile_stratification_full.csv")
    if vent is not None:
        cells = vent.drop_duplicates(["label", "threshold"])
        n_cells = len(cells)
        n_ci = int((~cells["monotone_ci"].astype(bool)).sum())
        n_sd = int((~cells["monotone_sd"].astype(bool)).sum())
        n_ex = int((~cells["monotone_exact"].astype(bool)).sum())
        add("monotone_fail_ci", n_ci, cohort="YRBS", regime="ventile_monotonicity",
            gen="operational", src="ventile_stratification_full.csv",
            script="ventile_monotonicity.py", status="canonical",
            note=f"CI-based inversions across {n_cells} config x threshold cells "
                 f"(headline figure)")
        add("monotone_fail_exact", n_ex, cohort="YRBS", regime="ventile_monotonicity",
            gen="operational", src="ventile_stratification_full.csv",
            script="ventile_monotonicity.py", status="supplementary",
            note="exact adjacent-difference test; noise-inflated")
        add("monotone_fail_sd", n_sd, cohort="YRBS", regime="ventile_monotonicity",
            gen="operational", src="ventile_stratification_full.csv",
            script="ventile_monotonicity.py", status="supplementary",
            note="SD-tolerant; none survive CI")
        add("ventile_spearman_median", float(cells["spearman_mean"].median()), cohort="YRBS",
            regime="ventile_monotonicity", gen="operational",
            src="ventile_stratification_full.csv", script="ventile_monotonicity.py",
            status="canonical", note="criterion-free summary (~0.98)")
        for _, r in cells[cells.label.astype(str).str.startswith("r1_rule_head")
                          & (cells.k == 100)].iterrows():
            add("ventile_spearman", g(r, "spearman_mean", np.nan),
                ci_lo=g(r, "spearman_ci_lo"), ci_hi=g(r, "spearman_ci_hi"), cohort="YRBS",
                family="XGB", regime="rule_extraction@ventile", threshold=r["threshold"],
                lineage="raw", k=100, src="ventile_stratification_full.csv",
                script="ventile_monotonicity.py", gen="operational", status="caveat",
                note="k=100 rule-head stratifies unreliably (Spearman CI spans 0); "
                     "recovers by k=500")
        for lab in ("naive:CatBoost:untuned", "naive:CatBoost:tuned",
                    "conformal:3_finetuned_k500"):
            for _, r in cells[cells.label == lab].iterrows():
                add("ventile_rr", g(r, "rr_top_bottom_mean", np.nan), ci_lo=g(r, "rr_lo"),
                    ci_hi=g(r, "rr_hi"), cohort="YRBS", family="CatBoost",
                    regime=f"ventile_rr:{lab}", threshold=r["threshold"],
                    src="ventile_stratification_full.csv",
                    script="ventile_monotonicity.py", gen="operational",
                    status="canonical", note="top:bottom ventile risk ratio")

    # ---- conformal ----
    cc = load("conformal_coverage.csv")
    if cc is not None:
        for _, r in cc.iterrows():
            if r["config"] == "1_MCS_internal":
                continue   # replaced by conformal_mcs_marginal.csv (2026-08-04)
            cohort = "MCS" if r["config"] == "1_MCS_internal" else "YRBS"
            flag = bool(g(r, "flag_lt50", False))
            cbase = dict(cohort=cohort, cell=f"{g(r, 'sex')}|{g(r, 'ethnicity')}",
                         family="CatBoost", regime=f"conformal:{r['config']}",
                         threshold=">=1", src="conformal_coverage.csv",
                         script="split_conformal.py", gen="freeze-week",
                         status=("unstable" if flag else "canonical"),
                         note=(f"target=0.90; cal_n<50 descriptive only (T2), "
                               f"cal_n≈{g(r, 'cal_n_mean')}" if flag else "target=0.90"))
            add("conformal_coverage", g(r, "coverage_mean", np.nan),
                sd=g(r, "coverage_sd"), **cbase)
            add("conformal_abstention", g(r, "abstention_mean", np.nan), **cbase)

    cm = load("conformal_mcs_marginal.csv")
    if cm is not None:
        for _, r in cm.iterrows():
            supp = bool(g(r, "suppressed", False))
            mbase = dict(cohort="MCS", cell=r["cell"], family="CatBoost",
                         regime=f"conformal_marginal:1_MCS_internal:{r['panel']}",
                         threshold=">=1", src="conformal_mcs_marginal.csv",
                         script="mcs_marginal.py", gen="freeze-week",
                         status=("suppressed" if supp else "canonical"),
                         note=(f"cal_n<50 SUPPRESSED ({g(r, 'seeds_below_50')} seeds), "
                               f"cal_n≈{g(r, 'cal_n_mean')}" if supp else
                               f"marginal; cal_n≈{g(r, 'cal_n_mean')} "
                               f"test_n≈{g(r, 'test_n_mean')}"))
            add("conformal_coverage", g(r, "coverage_mean", np.nan),
                ci_lo=g(r, "coverage_lo"), ci_hi=g(r, "coverage_hi"), **mbase)
            add("conformal_abstention", g(r, "abstention_mean", np.nan),
                ci_lo=g(r, "abstention_lo"), ci_hi=g(r, "abstention_hi"), **mbase)
            add("conformal_singleton_prec", g(r, "singleton_prec_mean", np.nan),
                ci_lo=g(r, "singleton_prec_lo"), ci_hi=g(r, "singleton_prec_hi"), **mbase)

    f3 = load("conformal_split_summary.csv")
    if f3 is not None:
        m = f3[f3.scope == "marginal"]
        if len(m):
            r = m.iloc[0]
            add("conformal_coverage_cleansplit", g(r, "coverage_mean", np.nan),
                sd=g(r, "coverage_sd"), ci_lo=g(r, "coverage_lo"), ci_hi=g(r, "coverage_hi"),
                cohort="YRBS", cell="marginal", regime="conformal:2_cleansplit",
                threshold=">=1", src="conformal_split_summary.csv",
                script="exchangeability_split.py", gen="freeze-week", status="canonical",
                note="footnote-3 clean iso/quantile split; marginal")
            add("conformal_abstention_cleansplit", g(r, "abstention_mean", np.nan),
                cohort="YRBS", cell="marginal", regime="conformal:2_cleansplit",
                threshold=">=1", src="conformal_split_summary.csv",
                script="exchangeability_split.py", gen="freeze-week", status="canonical",
                note="marginal")

    f5 = load("conformal_weighted_summary.csv")
    if f5 is not None:
        m = f5[f5.scope == "marginal"]
        if len(m):
            r = m.iloc[0]
            add("conformal_coverage_weighted2c", g(r, "coverage_mean", np.nan),
                sd=g(r, "coverage_sd"), cohort="YRBS", cell="marginal",
                regime="conformal:2c_weighted", threshold=">=1",
                src="conformal_weighted_summary.csv",
                script="weighted_conformal.py", gen="freeze-week", status="canonical",
                note=f"density-ratio weighted; under-covers "
                     f"(ESS collapse {g(r, 'ess_median')})")
            add("conformal_abstention_weighted2c", g(r, "abstention_mean", np.nan),
                cohort="YRBS", cell="marginal", regime="conformal:2c_weighted",
                threshold=">=1", src="conformal_weighted_summary.csv",
                script="weighted_conformal.py", gen="freeze-week", status="canonical",
                note="marginal")

    # ---- E1 calibration headline + freeze-week PR-AUC ----
    e1 = load("headline_metrics_summary.csv")
    if e1 is not None:
        for _, r in e1.iterrows():
            add("ece", g(r, "ece_mean", np.nan), sd=g(r, "ece_sd"), ci_lo=g(r, "ece_lo"),
                ci_hi=g(r, "ece_hi"), cohort="YRBS", family=g(r, "model"),
                regime=f"e1:{r['config']}", k=g(r, "k", ""),
                src="headline_metrics_summary.csv", script="headline_metrics.py",
                gen="freeze-week", status="canonical",
                note="calibration headline (isotonic where conformal)")

    fz = load("freezeweek_pr_backfill.csv")
    if fz is not None:
        for _, r in fz.iterrows():
            add("prauc", g(r, "prauc_mean", np.nan), sd=g(r, "prauc_sd"), null=g(r, "null"),
                cohort="YRBS", family=g(r, "model"), regime=r["item"],
                threshold=r["threshold"], k=g(r, "k", ""), src="freezeweek_pr_backfill.csv",
                script="pr_completion.py", gen="freeze-week", status="canonical",
                note="PR-AUC (e2 per-seed reproduction)")

    # ---- derived: max-min gaps, arithmetic on rows already ingested ----
    _df = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    _sub = _df[(_df.generation == "subgroup") & (_df.metric == "auc")
               & (_df.family == "CatBoost") & (_df.arm == "untuned")
               & _df.cell.astype(str).ne("") & _df.value.notna()]
    for (cohort, regime, thr), grp in _sub.groupby(["cohort", "regime", "threshold"],
                                                   dropna=False):
        if grp["value"].nunique() >= 2:
            hi = grp.loc[grp["value"].astype(float).idxmax()]
            lo = grp.loc[grp["value"].astype(float).idxmin()]
            add("auc_gap_maxmin", float(hi.value) - float(lo.value), cohort=cohort,
                family="CatBoost", regime=f"gap:{regime}", arm="untuned", threshold=thr,
                lineage="cs", src="(derived)", script="build_canonical.py", gen="subgroup",
                status="canonical",
                note=f"max {hi.cell}={hi.value} - min {lo.cell}={lo.value}")
    _fr = _df[(_df.generation == "operational") & (_df.metric == "flagrate15")
              & _df.regime.astype(str).str.startswith("screenin:naive:CatBoost:untuned")
              & _df.cell.astype(str).ne("") & _df.value.notna()]
    for (regime, thr), grp in _fr.groupby(["regime", "threshold"], dropna=False):
        if grp["value"].nunique() >= 2:
            hi = grp.loc[grp["value"].astype(float).idxmax()]
            lo = grp.loc[grp["value"].astype(float).idxmin()]
            add("flagrate15_gap_maxmin", float(hi.value) - float(lo.value), cohort="YRBS",
                family="CatBoost", regime=f"gap:{regime}", threshold=thr, lineage="cs",
                src="(derived)", script="build_canonical.py", gen="operational",
                status="canonical",
                note=f"max {hi.cell}={hi.value} - min {lo.cell}={lo.value}")

    canon = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    cls = canon.apply(lambda r: pd.Series(classify_canonical_row(r),
                                          index=["finding", "paper_facing"]), axis=1)
    canon["finding"] = cls["finding"]
    canon["paper_facing"] = cls["paper_facing"]
    return canon


def classify_canonical_row(r) -> tuple:
    """(finding, paper_facing) — which rows could appear in the paper's text or tables.

    The seven findings F1-F7 are the paper's argument structure, and tagging every canonical
    row with one is how an omission becomes visible: a finding with no paper-facing rows is a
    claim with no artefact behind it.

    Demoted rows are never paper-facing regardless of what else matches — `defect`,
    `spec-artefact` and `excluded` return early. That ordering matters: EN_LR tuned >=1 would
    otherwise qualify as F1/F4 on its regime alone.
    """
    gen, fam, reg, arm, m, cell = (r["generation"], r["family"], str(r["regime"]), r["arm"],
                                   r["metric"], str(r["cell"]))
    if r["status"] in ("defect", "spec-artefact", "excluded"):
        return ("", False)
    extract = fam in ("L1_LR", "CatBoost")               # the canonical-family extract
    if gen == "S4" and extract and reg in ("naive", "mcs_internal", "yrbs_ceiling",
                                           "fine_tune", "target_only") \
            and m in ("auc", "prauc_lift"):
        return (("F1" if reg in ("naive", "mcs_internal", "yrbs_ceiling") else "F4"), True)
    if gen == "backfill" and reg in ("leaf_refresh", "rule_extraction") \
            and str(r["k"]) in ("500", "500.0") and m in ("auc", "prauc_lift"):
        return ("F3", True)
    if gen == "backfill" and m.startswith("HL_"):
        return ("F3", True)
    if m.startswith("conformal_coverage") or m.startswith("conformal_abstention") \
            or m.startswith("conformal_singleton"):
        return ("F5", True)
    if gen == "operational" and "@top_" in reg and fam == "CatBoost" \
            and m in ("precision", "recall") \
            and (reg.startswith("naive@top_") or reg.startswith("2_naive_transfer@")
                 or reg.startswith("3_finetuned_k500@")):
        return ("F6", True)
    if m == "ventile_rr":
        return ("F6", True)
    if reg == "ventile_monotonicity" and m in ("monotone_fail_ci", "ventile_spearman_median"):
        return ("F6", True)
    if m == "ventile_spearman" and r["status"] == "caveat":
        return ("F3", True)
    if gen == "freeze-week" and m == "ece" and fam == "CatBoost":
        return ("F2", True)
    if gen == "S4" and fam == "CatBoost" and m in ("ece", "brier") \
            and reg in ("naive", "isotonic_recal", "platt_frozen", "coef_freeze_intercept",
                        "bbse"):
        return ("F2", True)
    if m in ("auc_gap_maxmin", "flagrate15_gap_maxmin"):
        return ("F7", True)
    if gen == "subgroup" and reg == "subgroup4:naive" and fam == "CatBoost" \
            and arm == "untuned" and m == "auc" and "Black" in cell:
        return ("F7", True)
    if gen == "operational" and reg.startswith("screenin:naive:CatBoost:untuned") \
            and m in ("fpr15", "flagrate15") and cell not in ("", "__marginal__"):
        return ("F7", True)
    return ("", False)


