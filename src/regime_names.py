"""Names used for model families and transfer regimes.

Stored battery results use ``fine_tune`` while the manuscript calls the same
regime ``full_revision``. The conversion functions keep that translation at the
reporting boundary. Other regime keys pass through unchanged; ``PAPER_DISPLAY``
provides readable labels without turning prose labels into storage keys.

``$THESIS_WORK_DIR/label_budget_curve.csv`` is already written with manuscript names,
so it needs no conversion when read.
"""

from typing import Mapping

__all__ = [
    "PAPER_TO_PIPELINE", "PIPELINE_TO_PAPER",
    "PAPER_DISPLAY", "SHORT_DISPLAY", "UNSPECIFIED", "to_pipeline",
]

# What the manuscript calls each of the nineteen regime keys
# SOURCE: working_draft.tex, read at the line numbers given. This table is a
# RECORD, not a translation layer — `to_pipeline`/`to_paper` still map only the three
# identifiers in PAPER_TO_PIPELINE below, because those are the only three where both
# vocabularies use an identifier. Everything here is prose the manuscript prints, which is a
# different kind of thing and must not be fed back in as a key.
#
# The pass-through in `to_pipeline` is a pass-through over an ABSENCE, not over an agreement.
# Eleven of the sixteen unmapped regimes have no manuscript spelling at all, and two of them
# collide with a manuscript term for something else entirely. Passing through is still the
# right behaviour — there is nothing to translate to — but the difference matters a great
# deal to whoever writes Appendix D.
UNSPECIFIED = "UNSPECIFIED — deferred to Appendix D (app:regimes), which has no source"

# KEYED TO working_draft.tex, and to nothing else. Every line number below is the draft's and
# every name is the draft's own wording, which is not always the obvious wording:
#
#   * `unadapted`      was `naive` until the keys were corrected. The draft says "unadapted
#                      transfer" throughout; "naive" survived only as the LaTeX label
#                      `tab:naive` and never appeared in its prose. Frozen tables under
#                      `outputs/` still carry `naive` in their `regime` column — see the note
#                      at the foot of this module.
#   * `yrbs_internal`  was `yrbs_ceiling`. "Ceiling" asserted an upper bound nothing
#                      guarantees; `internal` describes what it is, and pairs with
#                      `mcs_internal`. The draft says "target-trained reference".
#   * `mcs_internal`   not "within-source reference" — the draft says "source reference".
#   * `platt_frozen`   UNSPECIFIED, and deliberately. The draft's logistic recalibration
#                      (§IV-E, line 394) "followed target-only fitting and full revision" and
#                      is cross-fitted; `platt_frozen` is Platt scaling on a FROZEN, unadapted
#                      model. Two different objects, so naming one with the other would be a
#                      claim the draft does not make.
#   * `isotonic_recal` UNSPECIFIED for the same kind of reason. "Isotonic" appears nowhere in
#                      the draft's main text — only in the unwritten appendix placeholder at
#                      line 1305.
#
# FULL REVISION HAS TWO IMPLEMENTATIONS AND THE DRAFT NAMES BOTH (§IV-D, lines 377-382):
# "pooled revision" for the logistic families, random forest, extra trees and HistGB, which
# refit on pooled MCS and YRBS records; "continued training" for XGBoost, LightGBM and CatBoost,
# which warm-start from the fitted MCS model. They are not two regimes — they are one regime
# (`fine_tune`) behaving differently by family, and the pipeline records which in `ft_mechanism`.
PAPER_DISPLAY: Mapping[str, str] = {
    # named in the draft, identifier matches
    "target_only": "target-only fitting (§IV-D, line 375)",
    "quantile_map": "quantile mapping (§IV-D, line 363)",
    "importance_weight": "importance weighting (§IV-D, line 364)",
    "pseudo_label": "pseudo-labelling (§IV-D, line 366)",
    # named in the draft, identifier differs
    "unadapted": "unadapted transfer (§IV-D, line 362)",
    "fine_tune": "full revision (§IV-D, line 377)",
    "mcs_internal": "source reference (§IV-C, line 348)",
    "bbse": "prior correction (§IV-E, line 658)",
    # Post-adaptation cross-fitted logistic recalibration. §V-F line 285 describes it and
    # §VI-D reports it; the two keys are the two adaptations it follows.
    "target_only_logistic_recal": "logistic recalibration after target-only fitting (§V-F, line 285)",
    "fine_tune_logistic_recal": "logistic recalibration after full revision (§V-F, line 285)",
    # not named anywhere in the draft
    # The rung beneath `unadapted`: the source model's own preprocessing carried across, with
    # no target-cohort statistic anywhere in it. The draft has no term for it because the
    # published results do not contain it — see `transfer.source_scaled_scores`.
    "source_scaled": UNSPECIFIED,
    # The YRBS local reference: a model developed and evaluated inside YRBS, configured by the
    # YRBS consensus selection. It is the symmetric partner of `mcs_internal` and is new work,
    # so it has no draft wording yet. Whoever revises §IV-C adds it.
    "yrbs_local": UNSPECIFIED,
    # The nested per-split target redevelopment. A SENSITIVITY ANALYSIS, off by default, which
    # supplies no headline reference and no headline gap. It is not a third local reference.
    "yrbs_resource_rich": UNSPECIFIED,
    "platt_frozen": UNSPECIFIED,
    "isotonic_recal": UNSPECIFIED,
    "raw_l1_head": UNSPECIFIED,
    "coef_freeze_intercept": UNSPECIFIED,
    "sign_support": UNSPECIFIED,
    "ensemble_same_family": UNSPECIFIED,
    "ensemble_catboost_source": UNSPECIFIED,
    "pseudo_label_thresh": UNSPECIFIED,
    # COLLISION — §V-F line 492 describes 'Leaf refresh' as holding the MCS tree structure
    # fixed and re-estimating each leaf's value. THAT IS NOT THIS REGIME. The battery lists
    # plain `leaf_refresh` in its own NOT_RUN set (tree-internal, no cross-family analogue);
    # `leaf_refresh_global` is the R2 addition, a one-degree-of-freedom global intercept
    # offset on the source raw logit, which its own code comment calls a '1-DOF analogue of
    # leaf_refresh'. The manuscript's leaf-refresh numbers come from
    # `backfill_leaf_refresh_per_seed.csv` via the CANONICAL.csv HL rows, not from here.
    "leaf_refresh_global": UNSPECIFIED,
    # COLLISION — 'feature set' appears three times in the manuscript (lines 388, 555, 979)
    # and every one of them means the harmonised feature schema, not the R5 regime. The
    # regime re-fits the target family on the active columns of an L1 source. A sentence
    # about 'the feature set' in this paper is about the schema.
    "feature_set": UNSPECIFIED,
}

def paper_label(key: str) -> str:
    """The draft's word for a regime, or the pipeline key where the draft has none.

    Nine of the nineteen are named in the draft; the rest are deferred to an appendix with
    no content, so there is no term to use and inventing one would put a word in the
    manuscript's mouth. Those show their pipeline key, which is at least true.
    """
    display = PAPER_DISPLAY[key]
    return key if display.startswith("UNSPECIFIED") else display.split(" (§")[0]


# Short display labels for notebook tables and figures — prose, never keys. Distinct from
# PAPER_DISPLAY, which records the draft's own wording with line numbers; these are the
# compact spellings a pivot-table column header can afford, and they must ABBREVIATE the
# draft's wording rather than depart from it.
#
# The three anchors used to read "within-source", "naive" and "target ceiling", which departed
# from it on all three counts: the draft says "source reference", "unadapted transfer" and
# "target-trained reference", and PAPER_DISPLAY records that "naive" never appears in the
# draft's prose at all — it survives only as the LaTeX label `tab:naive`. They are now the
# draft's words, shortened.
SHORT_DISPLAY: Mapping[str, str] = {
    # The two local references, and the labels say what they are: each was developed AND
    # evaluated inside one cohort. Neither is a ceiling and neither label may imply one.
    "mcs_internal": "MCS local", "yrbs_local": "YRBS local",
    # The nested sensitivity. The label says "nested" so it cannot be mistaken in a table for a
    # third local reference; it is off by default and anchors nothing.
    "yrbs_resource_rich": "nested target (sens.)",
    # "(cohort-std)" IS NOT DECORATION. Both YRBS frames are z-scored against themselves, test
    # frame included, so this regime already carries one label-free adaptation and is
    # transductive on the target side. A header that read "unadapted" alone invited every
    # reader to take it as the raw source model applied unchanged, which it is not.
    # `source_scaled` below is that quantity and is the rung beneath this one.
    "unadapted": "unadapted (cohort-std)",
    "source_scaled": "source-scaled (raw)",
    "quantile_map": "quantile map", "importance_weight": "importance wt.",
    "bbse": "prior corr.", "pseudo_label": "pseudo-label", "target_only": "target-only",
    "fine_tune": "full revision", "isotonic_recal": "isotonic", "platt_frozen": "logistic recal",
    "leaf_refresh_global": "leaf refresh (1-DOF)", "raw_l1_head": "raw-L1 head",
    "coef_freeze_intercept": "coef freeze", "sign_support": "sign support",
    "feature_set": "feature set (R5)", "ensemble_same_family": "ensemble (same)",
    "ensemble_catboost_source": "ensemble (CatBoost src)", "pseudo_label_thresh": "threshold PL",
    # `platt_frozen` above recalibrates the FROZEN SOURCE model in-sample on the k slice.
    # These two follow an adaptation and fit the mapping out-of-fold. Different experiments,
    # so the short labels must not collide.
    "target_only_logistic_recal": "target-only + recal.",
    "fine_tune_logistic_recal": "full revision + recal.",
}

# The label-using set as the notebooks report it: the eleven of §V-F. `pseudo_label_thresh` is
# not among them — it consumes no target label and is reported with the label-free group.
LABEL_USING: tuple = ("target_only", "fine_tune", "isotonic_recal", "platt_frozen",
                      "leaf_refresh_global", "raw_l1_head", "coef_freeze_intercept",
                      "sign_support", "feature_set", "ensemble_same_family",
                      "ensemble_catboost_source")

# The three regimes whose manuscript name and pipeline key differ. Every other key is spelled
# the same way in both vocabularies and needs no entry.
PAPER_TO_PIPELINE: Mapping[str, str] = {
    "unadapted": "unadapted",
    "target_only": "target_only",
    "full_revision": "fine_tune",
}

# The inverse. A bijection on these three, so inverting is safe rather than lossy.
PIPELINE_TO_PAPER: Mapping[str, str] = {v: k for k, v in PAPER_TO_PIPELINE.items()}

if len(PIPELINE_TO_PAPER) != len(PAPER_TO_PIPELINE):
    raise RuntimeError(
        "PAPER_TO_PIPELINE is not injective — two paper names collapsed onto one pipeline key, "
        "so PIPELINE_TO_PAPER would silently drop one of them.")


def to_pipeline(name: str, *, strict: bool = True) -> str:
    """Paper name -> pipeline key. Use when reading an artefact or calling pipeline code.

    Regimes not in the mapping pass through unchanged. **That is a pass-through over an
    absence, not over an agreement**: of the nineteen keys in `regime_battery.csv`, three
    are mapped here, eight more have a manuscript name this module deliberately does not carry
    (`bbse` is "prior correction", `platt_frozen` is "logistic recalibration", and so on), and
    eight are not named in the manuscript at all pending Appendix D. `PAPER_DISPLAY` records
    which is which, with line numbers, and `COLLIDES` names the two keys whose obvious reading
    is the wrong one. None of that is translatable — prose is not an identifier — so the
    pass-through stays.

    `strict=True` additionally refuses a name that is ALREADY a pipeline key whose paper
    spelling differs — i.e. `to_pipeline("fine_tune")` raises. That looks unhelpful and is
    the point: it is the signature of a double translation, which is the bug this module
    exists to prevent. Pass `strict=False` at a genuine mixed-vocabulary boundary, such as
    reading a hand-maintained column that could hold either.
    """
    if name in PAPER_TO_PIPELINE:
        return PAPER_TO_PIPELINE[name]
    if name in PIPELINE_TO_PAPER:
        if strict and PIPELINE_TO_PAPER[name] != name:
            raise ValueError(
                f"to_pipeline({name!r}): that is already the pipeline key; the paper name is "
                f"{PIPELINE_TO_PAPER[name]!r}. Translating it again means the mapping is being "
                f"applied twice. Translate once, at the boundary, or pass strict=False.")
        return name
    return name


if not set(PAPER_TO_PIPELINE.values()) <= set(PAPER_DISPLAY):
    raise RuntimeError(
        "a mapped pipeline key is missing from PAPER_DISPLAY, so the two tables disagree about "
        "which keys exist.")


# Family vocabulary
# These five names are vocabulary in exactly the sense the regime names above are: a fixed
# set, a spelling normaliser, and the alias table for the one lineage that spells four of
# them differently.
#
# They live in this module because it is the lowest layer — it imports nothing from the
# repository — so a metrics module can depend on them without reaching up into the reporting
# module and dragging matplotlib along behind it.
#
# SOURCE: $THESIS_WORK_DIR/tables/CANONICAL.csv `family`, restricted to the nine that
#         $THESIS_WORK_DIR/tables/full_regime_grid.csv carries. CANONICAL.csv also holds
#         `CatBoost-Plain`, `LR` and `ensemble`, which are ablation labels rather
#         than families and are deliberately not among the nine.
# The nine families and what kind each is. This is the one declaration: the order is the
# reporting order, and everything else about the nine is read off it.
# `spec/local_model_settings.csv` carries the same nine in the same order, in each cohort, and
# notebook 02 checks that they agree.
FAMILY_CLASS: Mapping[str, str] = {
    "L1_LR": "linear", "L2_LR": "linear", "EN_LR": "linear",
    "RF": "bagged", "ET": "bagged",
    "HistGB": "boosted", "LightGBM": "boosted", "XGB": "boosted", "CatBoost": "boosted",
}

FAMILIES: tuple = tuple(FAMILY_CLASS)

# Several regimes re-estimate part of a linear model and are defined only for the linear
# three; one relearns leaf values and needs the other six. Read off the partition rather
# than written out again, so the two cannot disagree.
LR_FAMS: frozenset = frozenset(f for f, kind in FAMILY_CLASS.items() if kind == "linear")

# How a family class is drawn. Read off the same partition, so a figure legend and a table
# column cannot disagree about which families are boosted.
CLASS_COLOUR: Mapping[str, str] = {"linear": "#01418F", "bagged": "#C69D00", "boosted": "#5A6580"}
CLASS_MARKER: Mapping[str, str] = {"linear": "o", "bagged": "^", "boosted": "s"}

# All three are computed and reported. `spec/local_model_settings.csv` carries nine families at
# each of them, in each cohort, and the frozen transfer grid is 298 rows per threshold across
# all three.
THRESHOLDS: tuple = (1, 2, 3)


# The reverse direction's three roles
#
# The forward battery fits in MCS and evaluates in YRBS. The reverse reading fits in YRBS and
# evaluates in MCS, and needs three names because it asks the same three questions the forward
# side asks — what a transferred model reaches, what the same configuration reaches when the
# training cohort changes, and what the evaluating cohort reaches choosing its own
# configuration. The correspondence is exact:
#
#     forward                    reverse                 what each is
#     unadapted            <->   reverse_transfer        developed in one cohort, evaluated in the other
#     yrbs_local           <->   mcs_local_reference     developed and evaluated in the same cohort
#
# THESE ARE NOT REGIME KEYS AND DO NOT BELONG IN `PAPER_DISPLAY`. Nothing in the manuscript
# names them, no published table carries them, and `to_pipeline` must not translate them. They
# are a supplementary reading's vocabulary, kept here with the rest of the vocabulary so the
# analysis has one spelling rather than one per display.
#
# NONE OF THE THREE IS A CEILING, for the same reason the forward references are not: each is a
# model fitted somewhere, and one arm exceeding another is a result rather than a contradiction.
#
# `mcs_local_reference` AND `mcs_internal` ARE ONE ESTIMAND UNDER TWO NAMES. The reverse reading
# reuses the battery's rows for it rather than refitting, so the two keys coexist deliberately:
# the regime key says which experiment produced the row, and the role key says what part it
# plays here. See `transfer.mcs_local_reference_rows`.
REVERSE_ROLES: tuple = ("reverse_transfer", "mcs_local_reference")

REVERSE_ROLE_DISPLAY: Mapping[str, str] = {
    "reverse_transfer": "YRBS-developed, tested on MCS",
    "mcs_local_reference": "MCS-developed, tested on MCS",
}

if set(REVERSE_ROLE_DISPLAY) != set(REVERSE_ROLES):
    raise RuntimeError(
        "the reverse roles and their display labels disagree about which roles exist.")
if set(REVERSE_ROLES) & set(PAPER_DISPLAY):
    raise RuntimeError(
        "a reverse role collides with a regime key; the two vocabularies must stay separate.")

# The `analysis/` lineage spells four of the nine differently. Normalising on the way
# in means a caller passes one vocabulary whichever file is being read.
FAMILY_ALIASES: dict = {
    "l1_lr": "L1_LR", "l2_lr": "L2_LR", "en_lr": "EN_LR",
    "random_forest": "RF", "extra_trees": "ET", "histgb": "HistGB",
    "lightgbm": "LightGBM", "xgboost": "XGB", "catboost": "CatBoost",
}


def canonical_family(name: str) -> str:
    """Normalise a family spelling to the canonical nine, or raise naming what is accepted."""
    if name in FAMILIES:
        return name
    alias = FAMILY_ALIASES.get(str(name).strip().lower())
    if alias in FAMILIES:
        return alias
    raise ValueError(
        f"unknown family {name!r}. The canonical nine are {list(FAMILIES)}; the analysis "
        f"lineage's spellings ({sorted(FAMILY_ALIASES)}) are accepted and normalised. "
        f"Do not invent a name — spec/local_model_settings.csv and every `family` column "
        f"under outputs/ are keyed on these.")


def resolve_families(families=None) -> list:
    """`None` means all nine. A bare string means one. Anything else is a sequence."""
    if families is None:
        return list(FAMILIES)
    if isinstance(families, str):
        return [canonical_family(families)]
    return [canonical_family(f) for f in families]


# THE FROZEN TABLES PREDATE THE KEY RENAME. Everything under `outputs/` was published while the
# two anchors were called `naive` and `yrbs_ceiling`, so their `regime`, `config` and
# `eval_config` columns still say that. Code that READS a frozen table must use the frozen
# spelling; code that computes from a live run uses the keys above. The two meet only at a
# republication, which is where the frozen side catches up.
RENAMED_KEYS: Mapping[str, str] = {"naive": "unadapted"}

# RETIRED, NOT RENAMED, and the difference matters when reading a frozen table. `yrbs_internal`
# (spelled `yrbs_ceiling` before the rename) was a YRBS-trained model under the MCS-selected
# configuration. Under the symmetric design a model uses the settings selected in the cohort it
# is trained on, so that arm no longer exists: the YRBS-trained reference is `yrbs_local`, which
# is configured by the YRBS consensus selection and is A DIFFERENT FIT, not a new name for the
# same one. `mcs_fixed_reference` is retired for the same reason, being its reverse mirror.
#
# Nothing maps these onto a live key. A frozen table carrying one describes a model this
# analysis no longer fits, and silently translating it to `yrbs_local` would put a
# source-configured number under a target-configured name.
RETIRED_KEYS: Mapping[str, str] = {
    "yrbs_internal": "retired: YRBS-trained under MCS-selected settings",
    "yrbs_ceiling": "retired: the pre-rename spelling of yrbs_internal",
    "mcs_fixed_reference": "retired: MCS-trained under YRBS-selected settings",
}

# The same problem one level down, on COLUMNS rather than regimes. The frozen tables carry
# `recovered` — `(auc - 0.5) / (yrbs_internal - 0.5)`, attainment above chance of a reference
# this analysis no longer fits. It is retired rather than renamed, for the same reason
# `yrbs_internal` is: there is no live column that means what it meant.
#
# The live reference-relative columns are `transfer_loss`, `target_resource_gap`,
# `adaptation_gain` and `target_gap_recovered` — see `evaluation.add_reference_gaps`.
RENAMED_COLUMNS: Mapping[str, str] = {}

RETIRED_COLUMNS: Mapping[str, str] = {
    "recovered": "retired with yrbs_internal; chance-anchored attainment of a retired reference",
    "target_attainment": "retired: the post-rename spelling of `recovered`",
    "gap_a": "renamed: `transfer_loss`",
    "gap_b": "retired with yrbs_internal",
    "fixed_configuration_gap": "retired with yrbs_internal",
    "resource_gap": "renamed and re-anchored: `target_resource_gap`, against yrbs_local",
    "remaining_resource_gap": "retired: the distance to the local reference, now read off it",
    "resource_gap_recovered": "renamed and re-anchored: `target_gap_recovered`",
    "resource_gap_reason": "renamed: `target_gap_reason`",
}


# The live regime key for a spelling that may be either. Frozen tables predate two renames, so
# a reader that has to accept both asks here rather than carrying its own alias table.
def live_key(name: str) -> str:
    """`naive` -> `unadapted`; anything else unchanged.

    `yrbs_ceiling` is deliberately NOT mapped. It and its later spelling `yrbs_internal` name a
    retired arm — a YRBS-trained model under the MCS-selected configuration — that this analysis
    no longer fits, so there is no live key to translate it to. See `RETIRED_KEYS`.
    """
    return RENAMED_KEYS.get(str(name), str(name))


def frozen_keys(name: str) -> tuple:
    """Every spelling `name` may appear under: the live key first, then any frozen alias.

    For reading a table that could have been written before or after the rename. The order is
    the point — a live run's key wins, and the frozen spelling is a fallback rather than an
    equal alternative.
    """
    live = live_key(name)
    aliases = [frozen for frozen, current in RENAMED_KEYS.items() if current == live]
    return (live, *aliases)
