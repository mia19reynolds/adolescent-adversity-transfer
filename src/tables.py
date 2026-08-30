"""
config/consolidated_tables.py — the public analysis tables, declared.

A file grid of 934 tracked outputs, 696 of which differ from a sibling only by model family
and 176 only by threshold, is not a set of results — it is one set of results written out
many times. Family, threshold, regime, seed, budget, metric, subgroup, lineage and tuning
arm are VALUES, so they belong in columns. Every table below is long or wide over its own
dimensions, and none of them is split across files.

WHAT A TABLE IS. One coherent analysis at one aggregation level, backed by one or more
working aggregates the experiments produce under `$THESIS_WORK_DIR/`. Tables are NOT merged
merely to reduce the count: `label_budget` (family x k x threshold x metric) and
`label_budget_summary` (one row per family x threshold, carrying crossover scalars) stay
separate because a crossover point is not a metric value and a table that held both would
have a meaningless unique key.

WHAT THE PUBLISHER MAY DO with these. Concatenate, reshape, validate, filter to a declared
manuscript scope, format, render. It may not fit anything, choose families opportunistically,
or drop rows that disagree with the manuscript. `paper.save` is the live writer.

This module imports nothing from the repository and performs no I/O.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Consolidated:
    key: str
    investigation: str          # which of the project's investigations, A-J, the table belongs to
    question: str               # the scientific question the table answers
    sources: tuple[str, ...]    # working aggregates it is built from
    unique_key: tuple[str, ...] # columns that must be jointly unique
    required: tuple[str, ...]   # columns that must be present and non-empty
    grain: str                  # one row per ...
    disclosure: str             # "public-aggregate" | "person-level"
    consumers: tuple[str, ...]  # manuscript locations, or "supplementary"
    replaces: tuple[str, ...]   # current files this retires
    surface: str = "paper"      # "paper" | "supplementary"
    notes: str = ""
    min_rows: int = 1


# `replaces` names STEMS, not paths: a stem stands for every per-family and per-format copy
# of it in the current tree. The exact file list is resolved at cleanup time from the
# inventory, and reported in the removal manifest.

TABLES: tuple[Consolidated, ...] = (

    Consolidated(
        key="transfer_grid", investigation="C",
        question="How much discrimination survives the border, by family and regime?",
        sources=("regime_battery_summary.csv",),
        unique_key=("family", "arm", "regime", "threshold", "lineage", "k"),
        required=("family", "arm", "regime", "threshold", "auc_mean", "prauc_mean", "n_seeds"),
        grain="one row per family x arm x regime x threshold x lineage x k",
        disclosure="public-aggregate",
        consumers=("Table II", "Table VI", "Table VII", "SS VI-A", "SS VI-B", "SS VI-C"),
        replaces=("full_regime_grid", "full_regime_grid_sd", "full_regime_grid_pr",
                  "s4_regime_battery_summary", "regime_profile", "screening_profile",
                  "transfer_gap", "regime_deltas", "CANONICAL"),
        notes="The 63-column battery summary is already the canonical wide table; "
              "full_regime_grid is a narrow projection of it and is retired.",
        min_rows=500),

    Consolidated(
        key="transfer_significance", investigation="C",
        question="Which regime differences survive Holm correction?",
        sources=("regime_significance.csv",),
        unique_key=("family", "arm", "regime", "vs", "threshold"),
        required=("family", "arm", "regime", "vs", "threshold", "p_holm", "holm_reject"),
        grain="one row per family x arm x regime x comparison x threshold",
        disclosure="public-aggregate",
        consumers=("Table III dagger marks", "SS VI-A", "SS VI-C"),
        replaces=("s4_regime_significance", "regime_significance", "d1_naive_headline_significance",
                  "e5_lineage_significance", "e8_prauc_significance", "e10_outcome_significance"),
        notes="`vs` is the comparison dimension. The per-family figure copies were this "
              "table filtered to vs=naive; that filter is now a column value.",
        min_rows=300),

    Consolidated(
        key="label_budget", investigation="H",
        question="What does a fixed budget of target labels buy?",
        sources=("label_budget_curve.csv",),
        unique_key=("family", "regime", "threshold", "k", "metric"),
        required=("family", "regime", "threshold", "k", "metric", "mean", "n_seeds"),
        grain="one row per family x regime x threshold x budget x metric",
        disclosure="public-aggregate",
        consumers=("Table III", "SS VI-C"),
        replaces=("label_budget_curve", "budget_curve", "calibration_budget",
                  "finetune_curves", "budget_prauc"),
        notes="Already long format. R2 is decided on this table and the summary below.",
        min_rows=1000),

    Consolidated(
        key="label_budget_summary", investigation="H",
        question="Where does each family cross the target-only reference?",
        sources=("label_budget_summary.csv",),
        unique_key=("family", "threshold"),
        required=("family", "threshold"),
        grain="one row per family x threshold",
        disclosure="public-aggregate",
        consumers=("SS VI-C budget crossover (R2)",),
        replaces=("label_budget_summary",),
        notes="A DIFFERENT aggregation level from `label_budget` — crossover points and "
              "ceilings, not metric values — so it is a separate table, not merged. "
              "R2 is open: the crossover claim is contradicted by LightGBM and CatBoost "
              "at k=500 and every family stays in this table.",
        min_rows=18),

    Consolidated(
        key="calibration", investigation="E",
        question="How badly does transfer distort calibration, and what repairs it?",
        sources=("calibration_correction_sweep.csv",),
        unique_key=("family", "regime", "threshold", "correction", "calib_n",
                    "update_n", "metric"),
        required=("family", "regime", "threshold", "correction", "calib_n", "metric", "mean"),
        grain="one row per family x regime x threshold x correction x calibration size x metric",
        disclosure="public-aggregate",
        consumers=("Table IV", "SS VI-D (R1)"),
        replaces=("calibration_correction_sweep", "per_family_calibration",
                  "calibration_sweep", "e6_recal_ladder", "e6_prevalence_sweep",
                  "recalibration_ladder", "discrimination_vs_calibration"),
        notes="R1 is decided on this table: the manuscript's ECE range must be located "
              "here, at a stated family x correction x threshold x calib_n scope.",
        min_rows=1000),

    Consolidated(
        key="subgroup_performance", investigation="G",
        question="Does the model fail unevenly across demographic groups?",
        sources=("subgroup_discrimination_summary.csv",),
        unique_key=("config", "family", "arm", "regime", "threshold", "cell"),
        required=("family", "threshold", "cell", "auc_mean", "n_seeds", "n_min"),
        grain="one row per configuration x family x arm x regime x threshold x subgroup cell",
        disclosure="public-aggregate",
        consumers=("Table V", "SS VI-F"),
        replaces=("subgroup_by_family", "subgroup_discrimination_summary", "subgroup_auc",
                  "subgroup_brier", "subgroup_ece", "subgroup_cal_intercept",
                  "subgroup_gap_summary", "subgroup_forest", "subgroup_rank",
                  "subgroup_panels", "subgroup_calibration"),
        notes="Carries the small-cell suppression flags. `n_min` below 50 must remain "
              "visible rather than being filtered out of the public table.",
        min_rows=300),

    Consolidated(
        key="operating_points", investigation="G",
        question="What precision and recall does the model deliver at a deployable capacity?",
        sources=("operating_point_metrics_summary.csv",),
        unique_key=("config", "model", "arm", "lineage", "k", "threshold", "operating_point"),
        required=("config", "model", "threshold", "operating_point", "precision_mean",
                  "recall_mean", "n_seeds"),
        grain="one row per configuration x family x arm x lineage x budget x threshold x capacity",
        disclosure="public-aggregate",
        consumers=("SS VI-F",),
        replaces=("operating_point_metrics", "operating_point_metrics_summary",
                  "operating_points", "capture_rates"),
        notes="The per-family `figures/operating_points.csv` copies were this table filtered "
              "to config=naive, the first three capacities and the two named arms — proven "
              "identical on every metric. That filter is now the `selected_for_plot` column.",
        min_rows=2000),

    Consolidated(
        key="subgroup_capacity", investigation="G",
        question="Does a fixed screening capacity flag some groups more than others?",
        sources=("subgroup_operational.csv",),
        unique_key=("config", "threshold", "cell"),
        required=("config", "threshold", "cell", "fpr15_mean", "flagrate15_mean"),
        grain="one row per configuration x threshold x subgroup cell, at 15% capacity",
        disclosure="public-aggregate",
        consumers=("SS VI-F FPR and flag-rate spread",),
        replaces=("subgroup_operational", "subgroup_precision_at_capacity"),
        notes="A different aggregation level from `operating_points`: by subgroup cell at "
              "one fixed capacity, not by capacity.",
        min_rows=50),

    Consolidated(
        key="ventile_stratification", investigation="G",
        question="Does observed risk rise monotonically across score twentieths?",
        sources=("ventile_stratification_full.csv",),
        unique_key=("config", "model", "arm", "lineage", "k", "threshold", "ventile"),
        required=("config", "model", "threshold", "ventile", "obs_prevalence_mean",
                  "spearman_mean", "spearman_ci_lo", "spearman_ci_hi",
                  "n_inversions_exact", "n_inversions_sd", "n_inversions_ci",
                  "monotone_exact", "monotone_sd", "monotone_ci"),
        grain="one row per configuration x family x arm x budget x threshold x ventile",
        disclosure="public-aggregate",
        consumers=("Figure 2", "SS VI-F"),
        replaces=("ventile_stratification_full", "ventile_curve"),
        notes="Backs the manuscript's only generated figure.",
        min_rows=2000),

    Consolidated(
        key="conformal_coverage", investigation="F",
        question="Can the model abstain and still achieve its nominal coverage?",
        sources=("conformal_threshold2.csv", "conformal_per_family.csv"),
        unique_key=("grain", "family", "config", "scope", "threshold"),
        required=("config", "scope", "coverage_mean", "n_seeds"),
        grain="one row per grain x family x configuration x scope x threshold",
        disclosure="public-aggregate",
        consumers=("SS VI-E",),
        replaces=("conformal_threshold2", "conformal_per_family", "conformal_full_grid",
                  "conformal_full_grid_untuned", "conformal_coverage", "f3_conformal_split",
                  "f5_weighted_conformal", "e4_conformal_metrics"),
        notes="The two sources report at different grains — one by threshold, one by family. "
              "They are concatenated with an explicit `grain` column rather than joined, "
              "because joining them would invent rows neither source measured.",
        min_rows=150),

    Consolidated(
        key="conformal_cells", investigation="F",
        question="Are the Mondrian cells large enough to support a coverage claim at all?",
        sources=("conformal_prereq_cells.csv",),
        unique_key=("cohort", "slice", "sex", "ethnicity"),
        required=("cohort", "slice", "n_mean", "n_min", "flag_lt50"),
        grain="one row per cohort x slice x sex x ethnicity",
        disclosure="public-aggregate",
        consumers=("SS VI-E source-side cell sizes",),
        replaces=("conformal_prereq_cells", "e7e_cell_counts_geq2"),
        notes="The mandatory first output before any conformal claim. `flag_lt50` marks "
              "cells too small to support one, and those rows stay in the public table.",
        min_rows=40),

    Consolidated(
        key="outcome_variants", investigation="D",
        question="Does the finding survive the outcome definition we chose?",
        sources=("threshold_sensitivity_summary.csv",),
        unique_key=("threshold", "config", "model", "k"),
        required=("threshold", "config", "model", "auc_mean", "n_seeds"),
        grain="one row per threshold x configuration x family x budget",
        disclosure="public-aggregate",
        consumers=("SS VI-C",),
        replaces=("e2_threshold_sensitivity", "e2_threshold_sensitivity_summary",
                  "e10_outcome_variants", "s1_deductive_sensitivity", "outcome_variants"),
        min_rows=30),

    Consolidated(
        key="outcome_exclusions", investigation="D",
        question="Which outcome variants were excluded, and on what stated ground?",
        sources=("outcome_variants_excluded.csv",),
        unique_key=("outcome",),
        required=("outcome", "reason"),
        grain="one row per excluded outcome variant",
        disclosure="public-aggregate",
        consumers=("SS VI-C the >=4 prevalence guard",),
        replaces=("e10_excluded_variants",),
        notes="Eight variants, each with the ground it was dropped on. Small, and the "
              "point of it is that the exclusions are stated rather than silent.",
        min_rows=8),

    Consolidated(
        key="xgboost_rule_transfer", investigation="C",
        question="Does transferring structure beat transferring parameters?",
        sources=("rule_head.csv",),
        unique_key=("k",),
        required=("k", "n_rules", "r1_mean_auc", "n_seeds"),
        grain="one row per label budget",
        disclosure="public-aggregate",
        consumers=("SS VI-C",),
        replaces=("r1_rule_head", "r1_rule_head_per_seed", "f1_catboost_leaf_head",
                  "f4_lr_raw_discriminator", "catboost_ablations", "structure_transfer"),
        surface="supplementary",
        notes="Supplementary: the manuscript states the verdict in prose without a table.",
        min_rows=3),

    Consolidated(
        key="sample_characteristics", investigation="A",
        question="Are the two analytic samples comparable?",
        sources=("participant_characteristics.csv",),
        unique_key=("cohort", "characteristic", "category"),
        required=("cohort", "characteristic", "n"),
        grain="one row per cohort x characteristic x category",
        disclosure="public-aggregate",
        consumers=("Table I", "SS IV sample flow"),
        replaces=("table2_participant_characteristics", "missingness_audit"),
        notes="MCS counts are SDC-rounded at source. The publisher does not round again; "
              "it verifies the rounding is already applied.",
        min_rows=100),

    Consolidated(
        key="pillar_prevalence", investigation="A",
        question="Which adverse-experience categories differ between cohorts?",
        sources=("pillar_prevalence.csv",),
        # keyed on `pillar`, not `pillar_key`: the three summary rows all carry
        # pillar_key = "(summary)", which is correct data and a wrong key
        unique_key=("pillar",),
        required=("pillar", "mcs_prev", "yrbs_prev", "ratio"),
        grain="one row per pillar, plus three composite summary rows",
        disclosure="public-aggregate",
        consumers=("Table VIII", "SS IV"),
        replaces=("pillar_prevalence", "table3_pillar_composition"),
        min_rows=8),
    # ---- published because the draft check needs them -----------------------------------
    # These five are not new science. They are the tables the verification layer reads to
    # check the manuscript's 189 numbers, and they were the last reason a fresh clone could
    # not verify itself. Publishing them costs 1.4 MB and removes an undocumented dependency
    # on ignored local state.

    Consolidated(
        key="regime_grid", investigation="C",
        question="What is the reported AUC/PR-AUC grid, in the shape the appendix renders?",
        sources=("full_regime_grid.csv",),
        unique_key=("arm", "threshold", "family", "regime"),
        required=("arm", "threshold", "family", "regime", "auc", "prauc"),
        grain="one row per arm x threshold x family x regime",
        disclosure="public-aggregate",
        consumers=("Table VI", "Table VII", "pipeline_headline.tally"),
        replaces=(),
        notes="A projection of transfer_grid in the appendix's column vocabulary. Both are "
              "published because the tally checks the manuscript against THIS spelling; "
              "reshaping it inside the check would put a transformation between the "
              "manuscript and the number that verifies it.",
        min_rows=500),

    Consolidated(
        key="regime_grid_sd", investigation="C",
        question="What is the across-seed spread on each cell of the reported grid?",
        sources=("full_regime_grid_sd.csv",),
        unique_key=("arm", "threshold", "family", "regime"),
        required=("arm", "threshold", "family", "regime", "auc"),
        grain="one row per arm x threshold x family x regime, with seed spread",
        disclosure="public-aggregate",
        consumers=("SS VI preamble across-seed variance", "pipeline_headline.tally"),
        replaces=(),
        min_rows=500),

    Consolidated(
        key="canonical_claims", investigation="J",
        question="Which artifact and value backs each manuscript claim?",
        sources=("CANONICAL.csv",),
        unique_key=("claim_id",),
        required=("claim_id", "metric", "value"),
        grain="one row per traced manuscript claim",
        disclosure="public-aggregate",
        consumers=("pipeline_headline.tally — the 174 / 0 / 0 / 174 check reads this",),
        replaces=(),
        notes="The claim-to-value crosswalk. It is what makes the manuscript checkable from "
              "a clone, so it is tracked despite being the largest public file at 1.2 MB.",
        min_rows=4000),

    Consolidated(
        key="subgroup_ranking", investigation="G",
        question="Which subgroup is served worst, and is that ranking stable across seeds?",
        sources=("subgroup_by_family.csv",),
        unique_key=("family", "arm", "threshold", "cell"),
        required=("family", "arm", "threshold", "cell", "auc_mean", "rank", "gap"),
        grain="one row per family x arm x threshold x subgroup cell",
        disclosure="public-aggregate",
        consumers=("SS VI-F worst-served cells and Kendall tau",),
        replaces=(),
        surface="supplementary",
        notes="Carries `rank` and `gap`, which subgroup_performance does not.",
        min_rows=200),

    Consolidated(
        key="conformal_by_cell", investigation="F",
        question="Does conformal coverage hold within each demographic cell?",
        # qualified: a bare `conformal_coverage.csv` also names the CONSOLIDATED table in
        # work/aggregates/, which is a different grain. The raw per-cell file is the one meant.
        sources=("tables/conformal_coverage.csv",),
        unique_key=("config", "sex", "ethnicity"),
        required=("config", "target_coverage", "coverage_mean"),
        grain="one row per configuration x sex x ethnicity",
        disclosure="public-aggregate",
        consumers=("SS VI-E", "pipeline_headline.tally"),
        replaces=(),
        surface="supplementary",
        notes="A third grain alongside conformal_coverage (by family, by threshold) and "
              "conformal_cells (cell sizes). Kept separate because a coverage rate and a "
              "cell count are not the same measurement.",
        min_rows=30),
)


BY_KEY: dict[str, Consolidated] = {t.key: t for t in TABLES}
