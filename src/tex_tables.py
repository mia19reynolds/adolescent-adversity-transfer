#!/usr/bin/env python3
r"""The four appendix tables, as LaTeX fragments.

Each file holds only tabular environment(s) with booktabs rules — no \begin{table}, caption
or label — so the manuscript wraps them itself. Lookup and formatting only; nothing is
refitted. Called by notebook 03's fragment cell, writing through `paper.save_fragment`.

Any value that cannot be sourced is emitted as a dash and listed as UNAVAILABLE in the
returned report. Rounding is 3 dp unless stated, p < 0.001 prints as "<0.001", and MCS counts
are rounded to the nearest 10 under Tier 1a.

The search grids are frozen transcriptions of the screening search: the grids were expressed as
code rather than as a CSV, so they are transcribed here rather than read from a spec.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from regime_names import FAMILIES as FAM_ORDER
from regime_names import frozen_keys

# The significance fragment is no longer built. It rendered Hodges-Lehmann estimates,
# distribution-free intervals and Holm-adjusted p-values from `transfer_significance.csv`, and
# that inference assumes the twenty seed-level differences are independent. They are not: the
# splits are overlapping 75% draws from one sample. `build_significance` is left in this module
# — nothing else references it, and deleting it is a separate decision — but it is not called.
# Only the hyperparameter fragment is written automatically: it is built from the tracked
# specs and carries no MCS-derived value. The full grid is produced as a CANDIDATE by
# `fullgrid_candidate` and promoted deliberately after review; the subgroup fragment is
# deferred until its live producer lands.
TABLES = ("hyperparameters.tex",)

FAM_DISP = {"L1_LR": "L1-LR", "L2_LR": "L2-LR", "EN_LR": "EN-LR", "RF": "RF", "ET": "ET",
            "HistGB": "HistGB", "LightGBM": "LightGBM", "XGB": "XGB", "CatBoost": "CatBoost"}
GE1, GE2, GE3 = ">=1", ">=2", ">=3"
# Two different absences, kept apart. `--` is the ordinary not-applicable dash: the battery
# never ran that (family, regime). `NE` is a cell that WAS attempted and could not be estimated
# on all twenty seeds. No disclosure marker is introduced here — none has been reviewed.
NOT_APPLICABLE = "--"
NOT_ESTIMABLE = r"\textsc{ne}"
# The reported thresholds, in order. >=2 is primary; >=1 and >=3 are secondary. The renderer
# takes this as an argument rather than assuming two, so a third block does not require a
# second implementation.
THRESHOLDS_REPORTED = (GE1, GE2, GE3)
report = {}          # file -> {"rows": int, "unavailable": [str,...]}


# --------------------------------------------------------------------------- helpers
def num(x, dp=3, dash="--"):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    return dash if pd.isna(v) else f"{v:.{dp}f}"


def pval(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    if pd.isna(v):
        return "--"
    return "$<$0.001" if v < 0.001 else f"{v:.3f}"


def ci(lo, hi, dp=3):
    a, b = num(lo, dp), num(hi, dp)
    return "--" if a == "--" or b == "--" else f"[{a}, {b}]"


def truthy(x):
    """Robust bool: handles real bools, NaN, and 'True'/'False' strings (bool('False') is True)."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    try:
        if pd.isna(x):
            return False
    except (TypeError, ValueError):
        pass
    return str(x).strip().lower() in ("true", "1", "yes")


def texttt(s):
    """Escape a dict/params string for \\texttt{}."""
    s = str(s)
    for a, b in [("\\", r"\textbackslash "), ("{", r"\{"), ("}", r"\}"), ("_", r"\_"),
                 ("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("^", r"\^{}"), ("~", r"\~{}")]:
        s = s.replace(a, b)
    return r"\texttt{" + s + "}"


def row(cells):
    return " & ".join(cells) + r" \\"


def write(name, lines, outdir):
    # The write goes through `paper.save_fragment` — the one sanctioned fragment writer — so
    # `outdir` must be `paper.fragments_dir()`, which resolves under the configured working
    # root on the call. The parameter survives only for call-site compatibility and is checked
    # rather than obeyed, so there is one destination rather than two.
    import paper
    destination = paper.fragments_dir()
    if Path(outdir) != destination:
        raise ValueError(f"fragments are written only to {destination}, not {outdir}")
    return paper.save_fragment("\n".join(lines) + "\n", name)


def thr_disp(t):
    """`>=1` -> `$\\geq$1`, for any threshold rather than the two that used to exist."""
    return r"$\geq$" + str(t).replace(">=", "")


# =========================================================================== 1. FULL GRID
def fullgrid_candidate(read, *, thresholds=THRESHOLDS_REPORTED):
    """The full-grid fragment's TEXT, for any number of thresholds. Writes nothing.

    IT IS A CANDIDATE, NOT AN OUTPUT. The grid carries MCS-derived aggregates — the
    within-source reference column — so generating it is not approval to publish it. The caller
    reviews the complete table and promotes it deliberately; structural validity is not
    disclosure clearance.

    Returns `(lines, unavailable, n_rows)`.
    """
    df = read("paper/data/transfer_grid.csv")
    df = df[df["lineage"] == "cs"]
    KREG = {"target_only", "fine_tune"}          # k=500 regimes
    # LIVE KEYS, WITH THE FROZEN SPELLING AS A FALLBACK. A run's summary carries `unadapted`;
    # the published tables under `outputs/` were written before that rename and carry `naive`.
    # `frozen_keys` returns the live key first and any frozen alias after it, so this renderer
    # reads either without holding its own alias table.
    #
    # `yrbs_local` has no frozen counterpart — the frozen tables carry `yrbs_internal`, which was
    # a DIFFERENT FIT (YRBS-trained under the MCS-selected configuration) and is retired rather
    # than renamed. So a frozen table renders the target local reference as absent, which is
    # correct and is reported rather than silently filled from the retired arm.
    REGIMES = [("mcs_internal", "MCS local reference"),
               ("yrbs_local", "YRBS local reference"),
               ("unadapted", "unadapted transfer"), ("quantile_map", "quantile map"),
               ("importance_weight", "importance weight"), ("bbse", "BBSE"),
               ("pseudo_label", "pseudo-label"), ("target_only", "target-only ($k$=500)"),
               ("fine_tune", "fine-tune ($k$=500)")]
    unavail = []
    thresholds = list(thresholds)

    def cell(fam, arm, regime, t):
        """`(auc, prauc, degenerate, status)`.

        THREE STATES, NOT TWO. `absent` means the battery never ran this (family, regime) — the
        procedure does not apply, and the ordinary not-applicable dash is right. `incomplete`
        means it WAS attempted and could not be estimated on all twenty seeds, so notebook 02
        blanked the across-seed values; rendering that as the same dash would make an
        unestimable result indistinguishable from an inapplicable one.
        """
        q = df[(df.family == fam) & (df.arm == arm) & (df.threshold == t)
               & df.regime.isin(frozen_keys(regime))]
        if regime in KREG:
            q = q[q["k"] == 500]
        if not len(q):
            return None, None, False, "absent"
        r = q.iloc[0]
        auc, prauc = r["auc_mean"], r["prauc_mean"]
        return auc, prauc, r.get("degenerate", False), ("incomplete" if pd.isna(auc) else "ok")

    ncol = 2 + 2 * len(thresholds)
    header = ["family", "arm"] + [rf"AUC {thr_disp(t)}" for t in thresholds] \
             + [rf"PR-AUC {thr_disp(t)}" for t in thresholds]
    L = [r"\begin{tabular}{ll" + "c" * (ncol - 2) + "}", r"\toprule",
         row(header), r"\midrule"]
    n = 0
    for regime, disp in REGIMES:
        L.append(row([rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{disp}}}}}"]))
        for fam in FAM_ORDER:
            for arm in ["untuned", "tuned"]:
                got = [cell(fam, arm, regime, t) for t in thresholds]
                if all(st == "absent" for *_, st in got):
                    unavail.append(f"{regime}:{fam}:{arm} (every threshold)")
                # EN_LR tuned >=1 grid-boundary defect: target-side cells only.
                en_defect = (fam == "EN_LR" and arm == "tuned" and regime != "mcs_internal")
                marks = [(("$\\dagger$" if en_defect and t == GE1 else "")
                          + ("$\\ddagger$" if truthy(d) else ""))
                         for t, (_, _, d, _) in zip(thresholds, got)]

                def fmt(v, status, mark):
                    if status == "absent":
                        return NOT_APPLICABLE
                    if status == "incomplete":
                        return NOT_ESTIMABLE
                    return num(v) + mark

                L.append(row([FAM_DISP[fam], arm]
                             + [fmt(a, st, m) for (a, _, _, st), m in zip(got, marks)]
                             + [fmt(pv, st, m) for (_, pv, _, st), m in zip(got, marks)]))
                n += 1
        L.append(r"\midrule")
    if L[-1] == r"\midrule":
        L[-1] = r"\bottomrule"
    else:
        L.append(r"\bottomrule")
    _note = (r"\multicolumn{" + str(ncol) + r"}{p{0.92\linewidth}}{\footnotesize "
             r"$\dagger$ EN-LR tuned $\geq$1 "
             r"(target side): the tuned configuration selects $C$=0.01, the strongest-regularisation "
             r"endpoint of the elastic-net search grid, yielding an all-seed underfit model that "
             r"transfers poorly (e.g.\ target-only AUC 0.704 / PR-AUC 0.833 vs 0.764--0.766 AUC for "
             r"the other linear families). These cells are excluded from the summary claims in the "
             r"main text; the within-source \emph{MCS internal} value is unaffected and carries no "
             r"marker. $\ddagger$ degenerate BBSE cell (label-shift prior collapsed). "
             r"\textsc{ne}: not estimable under the prespecified 20-seed procedure --- the "
             r"cell was attempted and at least one seed could not be estimated, so no "
             r"across-seed value is reported. \texttt{--}: the procedure does not apply to "
             r"that family.}")
    L += [row([_note]), r"\end{tabular}"]
    return L, unavail, n


# =========================================================================== 2. HYPERPARAMETERS
#
# THE CANDIDATE SPACE IS NOT TRANSCRIBED HERE. It used to be, and the hand copy drifted from the
# selector it was printed beside. `_search_space_tex` renders it from `models.TREE_SEARCH_DIST`
# and `models.LR_SEARCH_GRID` instead, so the appendix and the search cannot disagree.


def _search_space_tex(family: str) -> str:
    """One family's candidate space, rendered FROM LIVE CODE rather than transcribed.

    The grids used to be written out by hand beside the table. That hand copy drifted: it
    described a narrower space than the `search_method` strings it was printed next to, so the
    published appendix contradicted itself. Deriving the rendering from `models.TREE_SEARCH_DIST`
    and `models.LR_SEARCH_GRID` means the table cannot say one thing while the selector does
    another.
    """
    import models

    def render(value):
        if value is None:
            return r"None"
        if isinstance(value, str):
            return rf"\text{{{value}}}"
        return f"{value:g}"

    if family in models.LR_SEARCH_GRID:
        keys = sorted({k for cand in models.LR_SEARCH_GRID[family] for k in cand})
        parts = []
        for key in keys:
            values = sorted({cand[key] for cand in models.LR_SEARCH_GRID[family] if key in cand})
            parts.append(rf"{key.replace('_', chr(92) + '_')}$\in\{{"
                         + ",".join(render(v) for v in values) + r"\}$")
        return "; ".join(parts)
    dist = models.TREE_SEARCH_DIST[family]
    return "; ".join(rf"{k.replace('_', chr(92) + '_')}$\in\{{"
                     + ",".join(render(v) for v in vals) + r"\}$"
                     for k, vals in dist.items())


def build_hyperparameters(read, outdir):
    """The candidate space and fixed untuned arm, then the selected configuration per cohort.

    Everything comes from the tracked specification `spec/local_model_settings.csv` and from live
    code, so this fragment builds in a clone with no working root. `read` is unused here and kept
    for the shared builder signature.

    NO CROSS-VALIDATED SCORE IS RENDERED. The three seed-level AUCs behind each selection, their
    mean and their standard deviation live in the private records under the working root; the MCS
    ones are MCS-derived and need separate disclosure review. What a published appendix needs is
    which configuration each model was fitted under, and that is what this prints.
    """
    import config
    import models

    settings = pd.read_csv(config.LOCAL_MODEL_SETTINGS, dtype=str, keep_default_na=False)
    unavail = []
    # (a) candidate space + fixed untuned arm
    La = [r"% (a) candidate space and fixed (untuned) configuration",
          r"\begin{tabular}{lp{0.46\linewidth}p{0.26\linewidth}}", r"\toprule",
          row(["family", "candidate space", "fixed untuned arm"]), r"\midrule"]
    na = 0
    for fam in FAM_ORDER:
        canonical = models.FAM[fam][1]
        arm_a = models.FROZEN[fam] if canonical else {}
        arm_a_disp = texttt(str(arm_a)) if arm_a else r"\emph{library default}"
        La.append(row([FAM_DISP[fam], _search_space_tex(fam), arm_a_disp]))
        na += 1
    La += [r"\bottomrule",
           row([r"\multicolumn{3}{p{0.92\linewidth}}{\footnotesize Fixed untuned arm: XGB and "
                r"CatBoost use the specified frozen configuration; L2-LR's frozen configuration "
                r"is the library default (empty override); L1-LR, EN-LR, RF, ET, HistGB and "
                r"LightGBM use library defaults. The tuned configurations in panel (b) were "
                r"selected independently within each cohort by the same procedure: development "
                r"seeds 0, 1 and 2; five-fold stratified cross-validation within each "
                r"development seed's outer-training partition; AUC; the mean of the three "
                r"seed-level mean AUCs, with ties broken by the lower across-seed standard "
                r"deviation and then by candidate-pool order. Selection read no outer test "
                r"partition of either cohort.}"]),
           r"\end{tabular}"]

    # (b) the selected configuration per cohort, family and threshold
    Lb = [r"% (b) selected configuration per cohort, family and threshold",
          r"\begin{tabular}{lllp{0.44\linewidth}}", r"\toprule",
          row(["cohort", "family", "threshold", "selected configuration"]), r"\midrule"]
    nb = 0
    for cohort, cohort_disp in (("mcs", "MCS"), ("yrbs", "YRBS")):
        for fam in FAM_ORDER:
            for t in THRESHOLDS_REPORTED:
                q = settings[(settings.cohort == cohort) & (settings.family == fam)
                             & (settings.threshold == t)]
                if not len(q):
                    unavail.append(f"selected config {cohort} {fam} {t}")
                    Lb.append(row([cohort_disp, FAM_DISP[fam], thr_disp(t), "--"]))
                    continue
                Lb.append(row([cohort_disp, FAM_DISP[fam], thr_disp(t),
                               texttt(q.iloc[0]["settings"])]))
                nb += 1
    Lb += [r"\bottomrule", r"\end{tabular}"]

    p = write("hyperparameters.tex", La + ["", ""] + Lb, outdir)
    report["hyperparameters.tex"] = {"rows": na + nb, "unavailable": unavail, "path": p}


# ==================================================== 3. RESOURCE-RICH TARGET BENCHMARK
# The columns the selection record must carry for this block. `cv_auc` is deliberately NOT
# among them: an inner-CV AUC is measured on training folds and would be read as comparable to
# the test-set AUCs elsewhere in the appendix. It stays in the working record.
BENCHMARK_SELECTION_COLUMNS: tuple = (
    "family", "threshold", "seed", "status", "selected_configuration",
    "complexity_param", "complexity_value")


def target_benchmark_candidate(selection, *, thresholds=THRESHOLDS_REPORTED):
    """The resource-rich benchmark's selected configurations, as a candidate fragment's TEXT.

    ONE CONFIGURATION PER FAMILY AND THRESHOLD WOULD BE FALSE. The benchmark selects inside each
    seed's own YRBS training partition, so there are twenty selections per cell and quoting one
    of them as "the" configuration would invent a stability the search did not have. This
    reports the modal configuration, how many seeds chose it, the range the complexity parameter
    took across seeds, and how many seeds produced a selection at all.

    NO CROSS-VALIDATED PERFORMANCE. The inner-CV AUC is a working diagnostic on training folds
    and is not comparable to the held-out AUCs the rest of the appendix reports, so it is not
    rendered here and `BENCHMARK_SELECTION_COLUMNS` does not include it.

    A CANDIDATE, NOT AN OUTPUT. Returns `(lines, unavailable, n_rows)` and writes nothing.
    """
    missing = [c for c in BENCHMARK_SELECTION_COLUMNS if c not in selection.columns]
    if missing:
        raise ValueError(
            f"the benchmark selection record has no column(s) {missing}; expected "
            f"{list(BENCHMARK_SELECTION_COLUMNS)}. Notebook 02's target-selection stage "
            f"writes them.")

    unavail = []
    L = [r"% resource-rich target benchmark: configuration selected per seed inside YRBS",
         r"\begin{tabular}{llp{0.40\linewidth}cc}", r"\toprule",
         row(["family", "threshold", "modal configuration", "seeds", "complexity range"]),
         r"\midrule"]
    n = 0
    for fam in FAM_ORDER:
        for t in list(thresholds):
            cell = selection[(selection.family == fam) & (selection.threshold == t)]
            if not len(cell):
                unavail.append(f"benchmark selection {fam} {t}")
                L.append(row([FAM_DISP[fam], thr_disp(t), NOT_APPLICABLE, "--", "--"]))
                continue
            estimable = cell[cell.status == "selected"]
            if not len(estimable):
                # Attempted on every seed and estimable on none. Rendered as not estimable
                # rather than not applicable — the two are different states.
                L.append(row([FAM_DISP[fam], thr_disp(t), NOT_ESTIMABLE,
                              f"0/{len(cell)}", "--"]))
                n += 1
                continue
            counts = estimable["selected_configuration"].value_counts()
            modal = str(counts.index[0])
            values = estimable["complexity_value"].dropna().unique().tolist()
            param = str(estimable["complexity_param"].iloc[0])
            if not values:
                span = "--"
            else:
                lo, hi = min(map(str, values)), max(map(str, values))
                span = (rf"{texttt(param)} = {texttt(lo)}" if lo == hi
                        else rf"{texttt(param)} $\in$ [{texttt(lo)}, {texttt(hi)}]")
            L.append(row([FAM_DISP[fam], thr_disp(t), texttt(modal),
                          f"{int(counts.iloc[0])}/{len(cell)}", span]))
            n += 1
    L += [r"\bottomrule",
          row([r"\multicolumn{5}{p{0.92\linewidth}}{\footnotesize Configurations were selected "
               r"independently within each seed's YRBS training partition by five-fold "
               r"cross-validation on AUC, over the same search spaces used for source "
               r"selection. \emph{seeds} counts how many of the twenty seeds chose the modal "
               r"configuration, out of those the search was attempted on. Cross-validated "
               r"scores are working diagnostics measured on training folds and are not "
               r"reported. \textsc{ne}: attempted, and no seed produced a selection.}"]),
          r"\end{tabular}"]
    return L, unavail, n


def build_significance(read, outdir):
    sg = read("paper/data/transfer_significance.csv")
    unavail = []
    LABELFREE = {"quantile_map": "quantile map", "importance_weight": "importance weight",
                 "bbse": "BBSE", "pseudo_label": "pseudo-label"}
    rows = []   # (finding, comparison, family, arm, threshold, metric, hl, lo, hi, p)

    # F1: each label-free regime vs naive (AUC; s4 significance is AUC-only)
    for _, r0 in sg[(sg["vs"] == "naive") & (sg["regime"].isin(LABELFREE))].iterrows():
        rows.append(("F1", f"{LABELFREE[r0['regime']]} vs naive", r0["family"], r0["arm"],
                     r0["threshold"], "AUC", r0["hl"], r0["hl_lo"], r0["hl_hi"], r0["p_holm"]))
    # F4: target_only vs fine_tune per family (regime=fine_tune, vs=target_only)
    for _, r0 in sg[(sg["vs"] == "target_only") & (sg["regime"] == "fine_tune")].iterrows():
        rows.append(("F4", "fine-tune vs target-only", r0["family"], r0["arm"],
                     r0["threshold"], "AUC", r0["hl"], r0["hl_lo"], r0["hl_hi"], r0["p_holm"]))
    # The XGB leaf-refresh structure verdicts (F3) are not built. They existed only in a frozen
    # working-root table whose producer is no longer part of this repository, the current
    # significance output carries no leaf-refresh comparison, and the manuscript makes no
    # leaf-refresh claim.
    unavail.append("XGB leaf-refresh structure verdicts (retired with the structure-transfer "
                   "branch; no comparison exists in the current significance output)")

    fam_rank = {f: i for i, f in enumerate(FAM_ORDER)}
    rows.sort(key=lambda x: (x[0], fam_rank.get(x[2], 99), x[3], x[4], x[5]))

    L = [r"\begin{tabular}{llllcccc}", r"\toprule",
         row(["finding", "comparison", "family", "arm", "threshold", "metric", "HL [95\\% CI]", "Holm $p$"]),
         r"\midrule"]
    cur = None
    for f, comp, fam, arm, t, metric, hl, lo, hi, p in rows:
        if f != cur:
            if cur is not None:
                L.append(r"\midrule")
            cur = f
        L.append(row([f, comp, FAM_DISP.get(fam, fam), arm, thr_disp(t), metric,
                      f"{num(hl)} {ci(lo, hi)}", pval(p)]))
    L.append(r"\bottomrule")
    # UNAVAILABLE: linear-vs-tree fine_tune contrast
    unavail.append("linear-vs-tree fine_tune contrast (not computed in any frozen table; "
                   "only cross-family contrast that exists is RF-naive vs CatBoost-naive)")
    L.append(row([r"\multicolumn{8}{p{0.92\linewidth}}{\footnotesize A linear-vs-tree fine-tune "
                  r"contrast is \emph{not available}: no such paired comparison exists in the frozen "
                  r"significance tables. The only cross-family contrast computed is RF naive vs "
                  r"CatBoost naive (AUC), reported separately. All estimates are Hodges--Lehmann with "
                  r"Holm-adjusted $p$; ``reject'' at $\alpha$=0.05.}"]))
    L.append(r"\end{tabular}")
    p_ = write("transfer_significance.tex", L, outdir)
    report["transfer_significance.tex"] = {"rows": len(rows), "unavailable": unavail, "path": p_}


# =========================================================================== 4. SUBGROUP
def r10(x):
    """Round a count to the nearest 10 (Tier 1a, MCS side)."""
    try:
        return str(int(round(float(x) / 10.0) * 10))
    except (TypeError, ValueError):
        return "--"


def build_subgroup(read, outdir):
    disc = read("paper/data/subgroup_performance.csv")
    op = read("paper/data/subgroup_capacity.csv")
    pan = read("subgroup_panels_summary.csv")
    unavail = []

    # ---- target side: 8 sex x ethnicity cells, naive:CatBoost:tuned, both thresholds
    d = disc[(disc.config == "naive:CatBoost:tuned") & (disc.scope == "cell")]
    CELLS = ["female|White", "female|Black", "female|Hispanic", "female|Other",
             "male|White", "male|Black", "male|Hispanic", "male|Other"]
    LT = [r"% target side (YRBS): naive CatBoost tuned",
          r"\begin{tabular}{llrccc}", r"\toprule",
          row(["cell", "threshold", "$n$", "prev.", "AUC [95\\% CI]", "flag / FPR @15\\%"]), r"\midrule"]
    nt = 0
    supp_t = False
    for cell in CELLS:
        for t in [GE1, GE2]:
            dr = d[(d.cell == cell) & (d.threshold == t)]
            orr = op[(op.config == "naive:CatBoost:tuned") & (op.cell == cell) & (op.threshold == t)]
            if not len(dr):
                unavail.append(f"YRBS {cell} {t}")
                LT.append(row([cell.replace("|", " $|$ "), thr_disp(t), "--", "--", "--", "--"]))
                continue
            r0 = dr.iloc[0]
            if truthy(r0.get("unstable", False)):     # suppressed: dash metric AND count
                supp_t = True
                LT.append(row([cell.replace("|", " $|$ "), thr_disp(t), "--", "--", "--", "--"]))
                continue
            auc = f"{num(r0['auc_mean'])} {ci(r0['auc_plo'], r0['auc_phi'])}"
            if len(orr):
                o0 = orr.iloc[0]
                flagfpr = f"{num(o0['flagrate15_mean'])} / {num(o0['fpr15_mean'])}"
            else:
                flagfpr = "--"
                unavail.append(f"YRBS operational {cell} {t}")
            LT.append(row([cell.replace("|", " $|$ "), thr_disp(t), str(int(round(float(r0["n_mean"])))),
                           num(r0["prevalence_mean"]), auc, flagfpr]))
            nt += 1
    LT += [r"\bottomrule", r"\end{tabular}"]

    # ---- source side: 3 MCS marginal panels, mcs_internal CatBoost tuned, cohort-std
    pm = pan[(pan.cohort == "MCS") & (pan.config == "mcs_internal") & (pan.family == "CatBoost") &
             (pan.arm == "tuned")]
    PANELS = [("1_sex", ["female", "male"]),
              ("2_binary", ["White", "non-White"]),
              ("3_ethnicity", ["White", "Asian", "Black", "Mixed-or-Other"])]
    PDISP = {"1_sex": "sex", "2_binary": "White / non-White", "3_ethnicity": "ethnicity (4-way)"}
    LS = [r"% source side (MCS): mcs_internal CatBoost tuned, cohort-std; counts rounded to 10 (Tier 1a)",
          r"\begin{tabular}{lllrcc}", r"\toprule",
          row(["panel", "cell", "threshold", "$n$", "prev.", "AUC [95\\% CI]"]), r"\midrule"]
    ns = 0
    supp_s = False
    for panel, cells in PANELS:
        for cell in cells:
            for t in [GE1, GE2]:
                pr = pm[(pm.panel == panel) & (pm.cell == cell) & (pm.threshold == t)]
                if not len(pr):
                    unavail.append(f"MCS {panel}/{cell} {t}")
                    LS.append(row([PDISP[panel], cell, thr_disp(t), "--", "--", "--"]))
                    continue
                r0 = pr.iloc[0]
                if truthy(r0.get("suppressed", False)):
                    supp_s = True
                    LS.append(row([PDISP[panel], cell, thr_disp(t), "--", "--", "--"]))
                    continue
                auc = f"{num(r0['auc_mean'])} {ci(r0['auc_lo'], r0['auc_hi'])}"
                LS.append(row([PDISP[panel], cell, thr_disp(t), r10(r0["n_mean"]),
                               num(r0["prevalence_mean"]), auc]))
                ns += 1
    LS += [r"\bottomrule", r"\end{tabular}"]

    foot = (r"% footnote: dashes denote suppressed cells (per-seed calibration/eval $n<50$: metric AND "
            r"count withheld). MCS counts rounded to the nearest 10 (UKDS Safeguarded Tier 1a). "
            r"Ethnicity is 4-way; YRBS \{White, Black, Hispanic, Other\}, MCS \{White, Black, Asian, "
            r"Mixed-or-Other\} (Asian excludes Chinese).")
    p = write("subgroup_performance.tex", LT + ["", ""] + LS + ["", foot], outdir)
    report["subgroup_performance.tex"] = {"rows": nt + ns, "unavailable": unavail, "path": p,
                                  "note": f"suppressed(target)={supp_t} suppressed(source)={supp_s}"}


def build_appendix_tables(read, outdir) -> list[Path]:
    """Write the automatically-written fragment(s) into `outdir`.

    ONE, NOT FOUR. `build_significance` is not called — the inference it rendered is not
    supported on overlapping splits. `fullgrid_candidate` and `build_subgroup` are not called
    either: both carry MCS-derived values, and generating a table is not approval to publish
    it. See the note on `TABLES`.
    """
    report.clear()
    for build in (build_hyperparameters,):
        build(read, outdir)

    written = []
    for name in TABLES:
        r = report[name]
        written.append(Path(r["path"]))
        unavailable = f"{len(r['unavailable'])} unavailable" if r["unavailable"] else "complete"
        print(f"  appendix  {name:<26} {r['rows']:>4} rows  ({unavailable})")
        for u in r["unavailable"]:
            print(f"              - {u}")
    return written
