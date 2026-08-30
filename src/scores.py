"""
src/scores.py — the per-method reporting for notebook 02.

Two things live here and nothing else does:

  write_yrbs_scores    the one person-level artefact notebook 02 leaves behind: the YRBS
                       scores notebook 03 computes from.
  method_table /       the per-method views: the full metric battery for one regime, its
  method_gap /         paired change against a baseline, the significance of that change,
  method_significance  and the rank-invariance check §V-B's prior-correction claim needs.
  check_rank_invariant

THE EXPERIMENT LOOP IS THE NOTEBOOK'S, not this module's. Notebook 02 runs one procedure per
cell and holds each result in a named dataframe. Nothing is written while it runs and nothing
is read back from disk: a procedure that has not been run is a name that is not bound, which
fails at the cell rather than being silently filled from a file an earlier run left.

TIER 1a: every table this module writes is metrics. It cannot emit a row. The one
person-level artefact it touches is the YRBS score frame, which is open CDC data and still
resolves under the secure root because it is person-level — see `write_yrbs_scores`.
"""

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


# The methods each reporting section covers, in the paper's own order
# SOURCE: §V-B "ordered by how much of the source-target discrepancy each attempts to correct";
# §V-C "ordered by how many parameters each re-estimates". Reordering these misstates what the
# sections claim.
#
# Cohort standardisation is one of §V-B's and is not here: it is applied in notebook 01, before
# any model is fitted, so every method below already runs on it.
#
# PRIOR CORRECTION IS NOT HERE EITHER, and its absence is statistical rather than cosmetic.
# These tuples are the reporting groups AND the multiplicity families: `method_significance`
# Holm-corrects over the set it is given. §V-D states that "probability adjustment was
# evaluated separately from adaptation", so prior correction forms its own family in
# POST_TRAINING below. Including it here would correct three adaptation procedures for a
# comparison that does not belong to them, making each harder to pass for no reason.
LABEL_FREE: Sequence[str] = (
    "unadapted", "quantile_map", "importance_weight", "pseudo_label", "pseudo_label_thresh",
)

# Post-training probability adjustment (§V-D, §VI-D). These change the probabilities a fitted
# model emits without adapting the model. Prior correction is the only label-free one and is
# the only member here: `platt_frozen` and `isotonic_recal` remain in LABEL_USING because the
# paper's recalibration is the cross-fitted sweep that FOLLOWS target-only and full revision,
# which is a different object from those two standalone regimes. Recorded rather than assumed —
# Moving either procedure would change the label-using multiplicity family and its p_holm.
POST_TRAINING: Sequence[str] = ("bbse",)

# ELEVEN, and §V-C says "Eleven are evaluated at k=500". `pseudo_label_thresh` used to be the
# twelfth: it drew the k slice, which made it label-using by implementation while every other
# declaration already called it label-free. It consumes no target label now, so it has moved to
# LABEL_FREE above and this family is the eleven the manuscript describes.
LABEL_USING: Sequence[str] = (
    "platt_frozen", "isotonic_recal", "coef_freeze_intercept", "sign_support",
    "leaf_refresh_global", "feature_set", "raw_l1_head", "ensemble_same_family",
    "ensemble_catboost_source", "target_only", "fine_tune",
)


# Platt sits at |dAUC| = 1e-6 on the frozen battery rather than 1e-9. That is float32 score
# rounding through the sigmoid, not a monotonicity failure, so the tolerance for strictly
# monotone methods is set there rather than at machine epsilon.
RANK_TOL = 1e-5


# The persisted handoff, as a contract. `arm` is NOT here: notebook 02 drops it before writing
# because it is constant across every score-bearing regime, so the frame `metric_rows` builds
# has eight columns and the one that lands has these seven.
PERSISTED_COLUMNS: Sequence[str] = (
    "threshold", "family", "regime", "seed", "row_id", "y_true", "score")
PERSISTED_KEY: Sequence[str] = ("threshold", "family", "regime", "seed", "row_id")


def check_handoff(frame) -> None:
    """Refuse a score frame that notebook 03 could not take at face value.

    Checked here because this is the last point at which the frame is still in memory beside
    the run that produced it: a dtype that has quietly become float32, or a label that has
    become a float, is a defect of THIS run and not of the notebook that reads the file
    tomorrow. Each failure names the column and what was wrong with it.

    WHAT IS DELIBERATELY NOT CHECKED: whether every declared cell is present. A recalibration
    cell that could not be estimated writes no person-level rows by design, so a completeness
    rule here would refuse a legitimate run. Notebook 02 accounts for those cells by key
    against the non-estimable record, which is where that question belongs.
    """
    columns = list(frame.columns)
    if columns != list(PERSISTED_COLUMNS):
        raise ValueError(f"the score handoff carries {columns}, expected "
                         f"{list(PERSISTED_COLUMNS)} in that order")

    if frame["score"].dtype != np.float64:
        raise ValueError(f"score is {frame['score'].dtype}; the handoff carries the float64 "
                         f"vector the metrics were computed from, not a narrowed copy")
    for column in ("y_true", "seed", "row_id"):
        if frame[column].dtype.kind != "i":
            raise ValueError(f"{column} is {frame[column].dtype}, expected an integer kind")
    for column in ("threshold", "family", "regime"):
        if not pd.api.types.is_string_dtype(frame[column]):
            raise ValueError(f"{column} is {frame[column].dtype}, expected a string column")

    labels = set(pd.unique(frame["y_true"]))
    if not labels <= {0, 1}:
        raise ValueError(f"y_true takes {sorted(labels)}; the outcome is a binary label")

    if "mcs_internal" in set(frame["regime"]):
        raise ValueError("an MCS-evaluated regime reached the handoff. MCS scores are "
                         "row-level Tier 1a material and are excluded at source")

    score = frame["score"].to_numpy()
    if not np.isfinite(score).all():
        raise ValueError(f"{int((~np.isfinite(score)).sum())} score(s) are not finite; an "
                         f"unusable vector writes no rows rather than writing blanks")
    if score.size and (score.min() < 0.0 or score.max() > 1.0):
        raise ValueError(f"scores span [{score.min():.3g}, {score.max():.3g}], outside [0, 1], "
                         f"so they are not the probabilities the evaluation treats them as")

    duplicated = int(frame.duplicated(list(PERSISTED_KEY)).sum())
    if duplicated:
        raise ValueError(f"{duplicated} row(s) share {list(PERSISTED_KEY)}, so a respondent "
                         f"appears twice in one cell")


# The one persisted artefact
def write_yrbs_scores(frame, *, quiet: bool = False) -> Path:
    """Write the person-level YRBS scores notebook 03 computes from, once.

    WHAT IT CARRIES is the scope notebook 02 declares: the two references, unadapted transfer,
    the four focal adapted pipelines and their four recalibrated counterparts. The remaining
    procedures score the same respondents, and notebook 02 derives their metrics and discards
    the per-person predictions rather than persisting predictions no analysis consumes.

    `check_handoff` runs first, so the frame that lands is one notebook 03 can take at face
    value: the seven declared columns, a float64 score, an integer binary label, and no
    MCS-evaluated regime.

    A PERSON-LEVEL FRAME, so it resolves under the secure root — not the working root, and
    never inside the repository. YRBS is open CDC data; the location is about the grain, not
    the licence. MCS scores never reach here: `transfer.metric_rows` excludes `mcs_internal`
    at source, which is the Tier 1a boundary.

    Takes the finished frame the notebook already holds. There is no shard directory, no
    reassembly step and no scope argument: what is written is what the run computed.
    """
    import config as _C

    check_handoff(frame)
    out = Path(_C.MCS_SCORES) / "yrbs_scores.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    if not quiet:
        print(f"   {len(frame):,} rows over {frame.regime.nunique()} regime(s) x "
              f"{frame.family.nunique()} families -> {out.name} "
              f"({out.stat().st_size / 1e6:.0f} MB)")
    return out


# Reporting
# Every method cell reports ALL FOUR groups. A cell that showed discrimination alone would
# hide the paper's actual finding twice over: prior correction cannot move AUC by construction
# but does move calibration, and several label-using regimes buy calibration at no
# discrimination cost. Selecting metrics per method would make those invisible.
METRIC_GROUPS: Mapping[str, tuple] = {
    "discrimination": ("auc", "prauc", "prauc_lift"),
    "calibration":    ("brier", "ece", "cal_slope", "cal_intercept"),
    "decision_decile":   ("dec_precision", "dec_recall", "dec_specificity",
                          "dec_f1", "dec_mcc", "dec_flag_rate"),
    "decision_quintile": ("qui_precision", "qui_recall", "qui_specificity",
                          "qui_f1", "qui_mcc", "qui_flag_rate"),
}
ALL_METRICS: tuple = tuple(m for g in METRIC_GROUPS.values() for m in g)
CONTEXT_COLS = ("n_test", "prevalence", "n_pos", "degenerate", "note",
                # theory vs observation, kept apart
                "monotone_by_construction", "strict_rank_preservation_expected",
                # BBSE feasibility: what was estimated, and whether it was usable
                "solver_status", "valid_correction", "weights_strictly_positive",
                "tpr", "fpr", "target_predicted_positive_rate", "source_prevalence",
                "target_prevalence_unconstrained", "target_prevalence_constrained",
                "w0", "w1", "n_distinct_before", "n_distinct_after", "new_tie_fraction",
                # the resource-rich benchmark's own search, kept apart from the BBSE and
                # recalibration diagnostics above
                "target_selection_status", "target_selection_folds",
                "target_selection_candidates", "target_selection_cv_auc")


def method_table(per_seed: pd.DataFrame, method: str, *, arm="tuned", threshold=None,
                 metrics=ALL_METRICS) -> pd.DataFrame:
    """The full metric battery for ONE method across all families: mean and sd over seeds.

    One row per (family, threshold). `sd` is the across-seed standard deviation, which is what
    §VI's stability claim is about. Aggregate by construction: `groupby.agg` over seeds cannot
    return a row.

    THE OTHER SUMMARISER IS `evaluation.summarise_seeds`, which takes the whole battery at
    once in the published summary's shape. Use that one for a frame that stands on its own;
    this one for a single regime alongside others in a grid.

    THE NESTED TARGET SENSITIVITY IS BLANKED WHERE IT IS INCOMPLETE, under the same rule the
    other summariser uses — `evaluation.blank_incomplete_benchmark`. A cell whose per-split
    search failed on any seed would otherwise be averaged over its survivors here too.

    THE TWO LOCAL REFERENCES ARE NEVER BLANKED BY THAT RULE. Each takes one fixed configuration
    for all twenty splits, so every seed is configured by construction and the rule could only
    ever remove a complete row.
    """
    import evaluation as _E

    d = per_seed[per_seed["regime"] == method]
    if arm is not None:
        d = d[d["arm"] == arm]
    if threshold is not None:
        d = d[d["threshold"] == threshold]
    if d.empty:
        return pd.DataFrame()
    have = [m for m in metrics if m in d.columns]
    g = d.groupby(["family", "threshold"], dropna=False)
    out = g[have].agg(["mean", "std"])
    out.columns = [f"{m}_{'sd' if s == 'std' else s}" for m, s in out.columns]
    out.insert(0, "n_seeds", g.size())
    for c in CONTEXT_COLS:
        if c in d.columns:
            out[c] = g[c].first()
    out = out.reset_index().assign(regime=method)
    if method == _E.NESTED_TARGET_SENSITIVITY:
        keys = ("family", "threshold")
        # The seed count comes from the WHOLE input frame, before the narrowing above: a
        # benchmark cell missing a seed outright would otherwise be judged against the rows it
        # happens to have and come out complete.
        expected = per_seed["seed"].nunique() if "seed" in per_seed.columns else None
        out = _E.blank_incomplete_benchmark(
            out, _E.benchmark_complete_cells(d, keys=keys, seeds_expected=expected), keys=keys)
    return out


def method_gap(per_seed: pd.DataFrame, method: str, *, baseline="unadapted", arm="tuned",
               metrics=ALL_METRICS, aggregate: bool = True) -> pd.DataFrame:
    """One method's change against a baseline regime, paired within (family, arm, threshold, seed).

    Pairing is what makes the signed-rank test in `method_significance` valid — an unpaired
    difference of means over a grid where families are gated differently would compare
    different sets of cells.

    `aggregate=True` returns the across-seed mean and sd per (family, threshold), which is
    what the reporting grids display. `aggregate=False` returns the per-seed deltas
    themselves, one row per (family, arm, threshold, seed), which is what a paired test needs.
    """
    keys = ["family", "arm", "threshold", "seed"]
    have = [m for m in metrics if m in per_seed.columns]
    a = per_seed[(per_seed["regime"] == method) & (per_seed["arm"] == arm)].set_index(keys)[have]
    b = per_seed[(per_seed["regime"] == baseline) & (per_seed["arm"] == arm)].set_index(keys)[have]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return pd.DataFrame()
    delta = (a.loc[common] - b.loc[common]).reset_index()
    if not aggregate:
        return delta.assign(regime=method, baseline=baseline)
    g = delta.groupby(["family", "threshold"])
    out = g[have].agg(["mean", "std"])
    out.columns = [f"d_{m}_{'sd' if s == 'std' else s}" for m, s in out.columns]
    out.insert(0, "n_paired", g.size())
    return out.reset_index().assign(regime=method, baseline=baseline)


def _hodges_lehmann(deltas, alpha: float = 0.05) -> tuple:
    """The signed-rank point estimate and its distribution-free interval.

    The median of the Walsh averages, and the order statistics of those averages at the
    signed-rank critical value. This is the estimator that belongs beside a Wilcoxon test: the
    mean difference does not, because the test is on ranks.
    """
    import numpy as np
    from scipy.stats import norm

    v = np.asarray(deltas, float)
    v = v[~np.isnan(v)]
    n = v.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    walsh = np.sort(((v[:, None] + v[None, :]) / 2.0)[np.triu_indices(n)])
    hl = float(np.median(walsh))
    m = walsh.size
    z = norm.ppf(1 - alpha / 2)
    k = int(np.floor(m / 2 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    if k < 0:
        return hl, float(walsh[0]), float(walsh[-1])
    return hl, float(walsh[k]), float(walsh[m - 1 - k])


def method_significance(per_seed: pd.DataFrame, methods: Sequence[str], *, baseline="unadapted",
                        arm="tuned", metric: str = "auc") -> pd.DataFrame:
    """Paired signed-rank tests of each method against `baseline`, Holm-corrected.

    One row per (threshold, family, arm, regime, vs). The test is a two-sided Wilcoxon
    signed-rank over the per-seed deltas `method_gap(..., aggregate=False)` produces, so it is
    paired within the split — which is what the twenty seeds buy.

    HOLM IS APPLIED WITHIN (threshold, family, arm), over the methods this call was given.
    `m` records how many tests that was, so a reader can see the family of tests rather than
    infer it. A different `methods` list is a different family and a different correction;
    that is a property of Holm, not a defect here, and it is why the list is a parameter
    rather than a module constant.

    WHAT THIS DOES NOT REPRODUCE. `outputs/paper/data/transfer_significance.csv` carries
    m = 11 per cell over a fixed eight regimes against `naive` plus three against
    `target_only` and one against a `CatBoost_naive` cross-family baseline. Which eight were
    pre-declared, and where the cross-family comparison came from, is not recorded anywhere in
    this repository — so this function computes the family it is asked for and does not claim
    to be that table. Confirm the intended scope before republishing over it.
    """
    import numpy as np
    from scipy.stats import wilcoxon

    rows = []
    for method in methods:
        if method == baseline:
            continue
        d = method_gap(per_seed, method, baseline=baseline, arm=arm,
                       metrics=(metric,), aggregate=False)
        if d.empty or metric not in d.columns:
            continue
        for (thr, fam), grp in d.groupby(["threshold", "family"], sort=False):
            v = grp[metric].to_numpy(float)
            v = v[~np.isnan(v)]
            if v.size == 0:
                continue
            if np.allclose(v, 0.0):
                # Every seed identical to the baseline. `wilcoxon` raises on an all-zero
                # difference vector; the answer is p = 1 and it should be recorded, not
                # skipped — prior correction lands here by construction.
                W, p = 0.0, 1.0
            else:
                res = wilcoxon(v, zero_method="wilcox", alternative="two-sided")
                W, p = float(res.statistic), float(res.pvalue)
            hl, lo, hi = _hodges_lehmann(v)
            rows.append(dict(threshold=thr, family=fam, arm=arm, regime=method, vs=baseline,
                             metric=metric, n_paired=int(v.size),
                             delta_mean=round(float(v.mean()), 6),
                             W=W, p_raw=p, hl=round(hl, 6),
                             hl_lo=round(lo, 6), hl_hi=round(hi, 6)))
    if not rows:
        return pd.DataFrame()

    # HOLM IS STEP-DOWN, and the direction matters. Sort ascending, scale p_(i) by the number
    # of hypotheses still live at that step (m - i), then take a RUNNING MAXIMUM so an adjusted
    # p can never fall below one that came before it. A running minimum from the other end is
    # Benjamini-Hochberg, which controls a different thing.
    groups = []
    for _, g in pd.DataFrame(rows).groupby(["threshold", "family", "arm"], sort=False):
        g = g.sort_values("p_raw").copy()
        m = len(g)
        adj = np.maximum.accumulate((m - np.arange(m)) * g["p_raw"].to_numpy(float))
        g["m"] = m
        g["p_holm"] = np.clip(adj, 0.0, 1.0)
        g["holm_reject"] = g["p_holm"] < 0.05
        groups.append(g)
    return (pd.concat(groups, ignore_index=True)
              .sort_values(["threshold", "family", "regime"])
              .reset_index(drop=True))


def check_rank_invariant(per_seed: pd.DataFrame, method: str, *, baseline="unadapted",
                         tol=RANK_TOL) -> dict:
    """§V-B: prior correction is 'a monotone transformation that cannot alter ranking'.

    So its rank-based metrics must equal the baseline's to floating point while calibration
    moves. Returns the largest observed |delta| per metric group and whether the invariant holds.
    """
    d = method_gap(per_seed, method, baseline=baseline)
    if d.empty:
        # THE KEY SET IS THE SAME EITHER WAY. Callers read `rank_invariant_holds` and
        # `max_abs_rank_delta` to print a line; returning a shorter dict here turned "the
        # procedure has not been run yet" into a KeyError several cells later, which named
        # neither the procedure nor the reason. `checked` is the field that says whether
        # there was anything to check, and it is False rather than the invariant being True.
        return {"method": method, "checked": False, "reason": "no paired cells",
                "cells": 0, "max_abs_rank_delta": float("nan"),
                "max_abs_calibration_delta": float("nan"), "rank_invariant_holds": False}
    rank_cols = [f"d_{m}_mean" for m in ("auc", "prauc") if f"d_{m}_mean" in d.columns]
    cal_cols = [f"d_{m}_mean" for m in ("brier", "ece") if f"d_{m}_mean" in d.columns]
    max_rank = float(d[rank_cols].abs().to_numpy().max()) if rank_cols else float("nan")
    max_cal = float(d[cal_cols].abs().to_numpy().max()) if cal_cols else float("nan")
    return {"method": method, "checked": True, "reason": "",
            "cells": int(d["n_paired"].sum()),
            "max_abs_rank_delta": max_rank, "max_abs_calibration_delta": max_cal,
            "rank_invariant_holds": bool(max_rank <= tol)}
