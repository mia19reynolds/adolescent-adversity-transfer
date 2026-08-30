"""
src/data.py — data acquisition, harmonisation, outcome and splits.

Backs notebook 01, and supplies notebook 02 with the artefacts notebook 01 persisted.

THE SCHEMAS ARE DERIVED, NOT RESTATED. `FEATURE_COLUMNS` is `tuple(features.FEATURE_MAP)`,
whose own docstring names it the single source of truth for the harmonised features.
`MODEL_FEATURE_COLUMNS` is that schema with each declared nominal predictor replaced by its
indicators, derived from `NOMINAL_FEATURES` in turn. A second hand-written list here would be a
second source of truth, and the two would drift the first time a feature moved.

TWO SCHEMAS, ONE TRANSFORMATION. The raw schema is what notebook 01 persists and what the
univariate association section reads; the model schema is what every fitted model is on.
`model_features` is the only route between them and `build_splits` refuses a frame that has not
been through it.

THE HANDOFF. Notebook 01 owns harmonisation and the outcome definition and persists three
frames per cohort: features, the five ACE pillars, and the evaluation-only attributes. It writes
them to the `config` constants that name them; notebooks 02 and 04 read the same constants back.
This module supplies the builders and the schema those frames are built to — `FEATURE_COLUMNS`
and `SHARED_PILLARS` — and nothing else about the handoff.

The artefacts are working intermediates rather than results, a deterministic recode of a raw
cohort, so a rerun replaces them. The frozen material in this project is the published tables
under `outputs/`, which no notebook writes.

The outcome series are derived from the pillars through `make_outcome` rather than stored, so
that function stays the one definition of the outcome. Split bundles are not stored either.

TIER 1a: MCS row-level material never enters the repo and is never printed. Functions that touch
MCS return frames held in memory or write only to paths under $MCS_DATA_DIR (config.MCS_FEATURES,
config.MCS_ATTRIBUTES, config.MCS_PILLARS). Every diagnostic this module returns or raises is an
aggregate — shapes, dtypes, column names, counts, prevalences. No function here may return, print
or persist a row, and no exception message may quote one.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import features
import recode_yrbs as _Y     # the YRBS coarse-ethnicity crosswalk, for section 6

Side = Literal["mcs", "yrbs"]
Threshold = Literal[1, 2, 3]


def _check_side(side: str) -> None:
    """The one place `side` is validated. `load_raw` and `build_pillars` keep their own
    fall-through raise instead: there the raise is the else of a two-way dispatch, not a
    validation prelude, and folding it in would leave an unreachable return path."""
    if side not in ("mcs", "yrbs"):
        raise ValueError(f"side must be 'mcs' or 'yrbs', got {side!r}")


# 0. SplitBundle — the frozen split, named
# The eighteen frames one split bundle holds, in the order they are built. Both the names and
# the order are load-bearing: `SplitBundle.__getitem__` is keyed on them, and `build_splits`
# below is checked against this tuple rather than against a second hand-typed list.
SPLIT_KEYS: Sequence[str] = (
    "Xm_trm", "ym_trm", "Xm_te", "ymte", "Xm_te_cs",
    "Xy_tr", "Xy_te", "yte", "Xy_trm", "yy_trm",
    "Xm_cs", "Xy_te_cs", "Xy_tr_cs",
    "Xy_tr_cs2", "Xy_te_cs2", "y_idx",
    "pool_cs", "pool_raw",
)


@dataclass(frozen=True)
class SplitBundle:
    """The eighteen frames `build_splits` returns, named and kept together.

    This is a RECORD OF A FACT, not a new design. `build_splits` returns exactly these
    eighteen keys; every regime, every metric and every paired comparison in the paper is
    defined relative to them. Naming the object does not change what a split is. It does three
    things:

      1. Makes the drift LOUD. `from_s4_dict` asserts the key set is exactly the eighteen and
         raises with the specific difference otherwise. If the construction ever gains, loses
         or renames a key, the next run fails with a message naming it, instead of a
         downstream function
         silently reading a `None`.
      2. Gives the k=500 anchor ONE definition. `pool_cs.sample(500, random_state=seed)` is
         currently written out in four scripts; `k_slice()` below is that line, once.
      3. Keeps transcription literal. `__getitem__` means source code written against the
         dict — `S["Xm_cs"]`, `S["yte"]` — runs unchanged against the bundle. Every function
         consolidated from S4 in this repository is a verbatim transcription for that reason,
         not a rewrite.

    WHAT EACH FIELD IS. The naming looks arbitrary and is not: `_cs` means cohort-standardised
    against the MCS training side, `_cs2` means standardised against the YRBS-native training
    side, `_trm`/`_m` means NaN-outcome rows already dropped.

      MCS side
        Xm_trm     MCS train features, NaN-outcome rows dropped        (raw scale)
        ym_trm     matching MCS train outcome
        Xm_te      MCS test features                                   (raw scale)
        ymte       MCS test outcome, as a numpy array (S4 calls it `ymte`, not `ym_te`)
        Xm_cs      Xm_trm standardised against itself    <- what every base model trains on
        Xm_te_cs   Xm_te standardised against ITSELF     <- the `mcs_internal` eval slice
      YRBS side
        Xy_tr      YRBS train/pool features                            (raw scale)
        Xy_te      YRBS test features                                  (raw scale)
        yte        YRBS test outcome, numpy array        <- what almost every metric uses
        Xy_trm     YRBS train features, NaN-outcome rows dropped
        yy_trm     matching YRBS train outcome
        Xy_te_cs   Xy_te standardised against itself     <- the naive-transfer eval slice
        Xy_tr_cs   Xy_tr standardised against itself, the target pool
        Xy_tr_cs2  YRBS-native pair, train  }  the target-trained regimes. Each is standardised
        Xy_te_cs2  YRBS-native pair, test   }  against itself, as everything here is.
        y_idx      Xy_te.index as an array — the join key for the persisted score parquets
      label pools
        pool_cs    Xy_tr_cs + y, NaN-y dropped   <- median-imputed, so only NaN-y is lost
        pool_raw   Xy_tr + y, dropna()           <- complete-case, so it is strictly smaller

    EVERY `_cs` FRAME IS STANDARDISED AGAINST ITSELF, TEST FRAMES INCLUDED. Between cohorts that
    is the reported adaptation. Within one cohort it means a model fitted on `Xm_cs` — training
    units — predicts on `Xm_te_cs`, which is centred on the test frame's own mean rather than the
    training frame's. The two scalers are close, because the splits are a random 75/25 cut of one
    population, but they are not the same scaler and the protocol is transductive: how a test row
    is transformed depends on the other test rows.

    A consequence worth knowing before reading the two reference arms: `Xy_te_cs2` EQUALS
    `Xy_te_cs`, cell for cell, because both are `Xy_te` standardised against itself. The
    `_cs` / `_cs2` distinction is real on the train side (`Xm_cs` vs `Xy_tr_cs2`) and empty on
    the test side. A target-trained regime therefore fits on a YRBS-train scaling and predicts on a
    YRBS-test one.

    This is recorded, not corrected. Whether the reference arms should use a train-fitted scaler
    is a question about the method, and the method is fixed for this analysis.

    THE TWO POOLS ARE NOT INTERCHANGEABLE and that is the single easiest thing to get wrong
    here. `pool_cs` and `pool_raw` differ in size because the two lineages drop different
    rows, which is exactly why nominal k and effective k diverge (see
    the battery's `effective_k` column). A regime that draws its k slice from the wrong pool
    produces a number that looks right and is not comparable to anything published.

    `seed` and `threshold` are METADATA, not part of the eighteen. S4's dict does not carry
    them — they are the arguments that produced it. They are held here because `k_slice()`
    needs the seed to reproduce the published anchor, and carrying it beats passing it
    alongside the bundle everywhere and eventually passing the wrong one.
    """

    # --- the eighteen, in SPLIT_KEYS order ---
    Xm_trm: pd.DataFrame
    ym_trm: pd.Series
    Xm_te: pd.DataFrame
    ymte: np.ndarray
    Xm_te_cs: pd.DataFrame
    Xy_tr: pd.DataFrame
    Xy_te: pd.DataFrame
    yte: np.ndarray
    Xy_trm: pd.DataFrame
    yy_trm: pd.Series
    Xm_cs: pd.DataFrame
    Xy_te_cs: pd.DataFrame
    Xy_tr_cs: pd.DataFrame
    Xy_tr_cs2: pd.DataFrame
    Xy_te_cs2: pd.DataFrame
    y_idx: np.ndarray
    pool_cs: pd.DataFrame
    pool_raw: pd.DataFrame
    # --- metadata: the arguments that produced the eighteen ---
    seed: int | None = field(default=None, compare=False)
    threshold: int | None = field(default=None, compare=False)

    # -- construction ---------------------------------------------------
    @classmethod
    def from_s4_dict(cls, d: dict, *, seed: int | None = None,
                     threshold: int | None = None) -> "SplitBundle":
        """Wrap the dict S4's `build_splits` returns, asserting the key set is exactly the 18.

        This is the drift alarm. It is deliberately intolerant in both directions — a MISSING
        key and an EXTRA key are both errors — because both mean the same thing: the frozen
        split construction has changed and every paired comparison downstream is now paired
        within something different from what the paper reports.

        The error names the specific keys. "SplitBundle: S4 returned 19 keys, expected 18;
        unexpected: {'Xy_va'}" is a five-minute fix. A silent `None` is a week.
        """
        if not isinstance(d, dict):
            raise TypeError(f"SplitBundle.from_s4_dict expects a dict, got {type(d).__name__}. "
                            f"If you already hold a SplitBundle, pass it straight through.")
        got, want = set(d), set(SPLIT_KEYS)
        if got != want:
            missing, extra = sorted(want - got), sorted(got - want)
            raise ValueError(
                f"SplitBundle: S4's build_splits returned {len(got)} keys, expected "
                f"{len(want)}. missing={missing} unexpected={extra}. "
                f"The frozen split construction has changed — every paired comparison in the "
                f"paper is defined relative to it, so this must be reconciled deliberately "
                f"rather than by widening this check. "
                f"`SPLIT_KEYS` is the declaration this is checked against.")
        return cls(**{k: d[k] for k in SPLIT_KEYS}, seed=seed, threshold=threshold)

    def __getitem__(self, key: str):
        """`S["Xm_cs"]` works, so S4 source transcribes verbatim instead of being rewritten."""
        if key not in set(SPLIT_KEYS):
            raise KeyError(
                f"{key!r} is not one of the eighteen split keys. Available: "
                f"{', '.join(SPLIT_KEYS)}. (`seed` and `threshold` are attributes, not keys.)")
        return getattr(self, key)

    # -- derived views --------------------------------------------------
    @property
    def feat_cols(self) -> list:
        """The feature columns, derived from the pool rather than passed in.

        Both pools are built as `[features..., 'y']`, so the feature list is recoverable from
        the bundle rather than having to be computed once and passed around — which means a
        function needs the bundle and nothing else.
        """
        return [c for c in self.pool_cs.columns if c != "y"]

    # The two lineages are an identical construction over a different pool.
    def k_slice(self, k: int = 500, *, lineage: Literal["cs", "raw"] = "cs") -> pd.DataFrame:
        """The k-label target slice: `pool.sample(min(k, len(pool)), random_state=seed)`.

        THE PUBLISHED ANCHOR. `k_slice(500, lineage='cs')` IS the slice behind every
        label-using number in `regime_battery.csv`. The whole label-budget curve is built
        as nested prefixes of an ordering whose first 500 rows are this draw, so if this
        changes, the k=500 column of the curve stops reconciling with the battery — silently,
        because both still produce plausible numbers.

        Returns the slice WITH its `y` column; split it with `self.feat_cols`.

        `min(k, len(pool))` is S4's, not a safety addition: at large k the pool runs out, and
        the realised size is what `effective_k` reports.
        """
        if self.seed is None:
            raise ValueError(
                "k_slice needs the seed to reproduce the published draw, and this bundle "
                "carries seed=None. Build it with build_splits(...) or pass seed= to "
                "from_s4_dict(). Guessing a seed here would produce a different k slice from "
                "the one every published label-using number is computed on.")
        pool = self.pool_cs if lineage == "cs" else self.pool_raw
        if lineage not in ("cs", "raw"):
            raise ValueError(f"lineage must be 'cs' or 'raw', got {lineage!r}")
        return pool.sample(min(k, len(pool)), random_state=self.seed)

    def nested_draw(self, *, anchor_k: int = 500, max_k: int = 2000) -> pd.DataFrame:
        """The budget-curve ordering, whose first `anchor_k` rows ARE `k_slice(anchor_k)`.

            ordered = [pool.sample(500, seed)] ++ [(pool - those).sample(1500, seed)]

        Every k <= 500 is therefore a prefix of the published anchor and every k > 500 extends
        it. Rebuild the ordering any other way — one `sample(2000)` call, say — and the k=500
        row of the curve is a different 500 records from the battery's, so the two artefacts
        disagree while both look internally consistent.
        """
        if self.seed is None:
            raise ValueError("nested_draw needs the seed; this bundle carries seed=None.")
        pool = self.pool_cs
        anchor = pool.sample(min(anchor_k, len(pool)), random_state=self.seed)
        rest = pool.drop(anchor.index)
        n_extra = min(max(0, max_k - len(anchor)), len(rest))
        extra = rest.sample(n_extra, random_state=self.seed) if n_extra else rest.iloc[:0]
        return pd.concat([anchor, extra])

    def shapes(self) -> dict:
        """Aggregate description — shapes only, for logging. Tier 1a safe by construction."""
        out = {}
        for k in SPLIT_KEYS:
            v = getattr(self, k)
            out[k] = tuple(v.shape) if hasattr(v, "shape") else len(v)
        out["seed"] = self.seed
        out["threshold"] = self.threshold
        return out


# 1. Acquisition
def download_yrbs_2023(dest: Path, *, force: bool = False) -> Path:
    """Return the supplied YRBS 2023 national high-school file, or raise saying how to get it.

    THIS FUNCTION DOES NOT DOWNLOAD ANYTHING. No script in this repository fetches YRBS; the
    parquet is placed by hand from the CDC distribution, and this returns its path when it is
    already there. YRBS is open data, so that is a missing convenience rather than a licence
    restriction — unlike MCS Sweep 6, which is UKDS Safeguarded Tier 1a and must be obtained
    under an accepted application for SN 8156 (see README.md).

    `force=False` and the file already present is the normal case and returns the existing
    path. Only a missing file (or `force=True`) needs a downloader, and there is none.
    """
    # UNRESOLVED, and still is: there is no source to transcribe. No script in the repository
    # fetches YRBS; the raw parquet under $THESIS_WORK_DIR/data/raw/ was placed by hand. A downloader
    # would be new code, not consolidation, and no paper number depends on a download step.
    #
    # The no-op case is not blocked, though. `force=False` means "do not re-fetch what is
    # already here", so when the file exists this function has nothing to do and returning its
    # path is the whole of its contract. Raising there would fail a correctly-provisioned
    # repository, saying "not implemented" when the answer is "already done".
    dest = Path(dest)
    target = dest / "yrbs2023_raw.parquet"
    if target.exists() and not force:
        return target
    raise NotImplementedError(
        f"no YRBS downloader exists in this repository, and {target} is "
        f"{'absent' if not target.exists() else 'present but force=True was passed'}. "
        f"Fetch the 2023 national high-school file from the CDC YRBSS data-files page and "
        f"convert it to parquet at that path; see README.md. YRBS is open data, so this is a "
        f"missing convenience rather than a licence restriction.")


# Both locations come from `src/config.py`; nothing here hard-codes a path.
def load_raw(side: Side) -> pd.DataFrame:
    """Load the raw cohort frame: MCS Sweep 6 core, or YRBS 2023 national.

    MCS resolves through config.MCS_CORE, under $MCS_DATA_DIR.
    YRBS resolves through config.YRBS_RAW, under $THESIS_WORK_DIR — open data, but still a
    working input rather than a release file, so it is not in the repository either.

    A FileNotFoundError on the MCS path is the Tier 1a safeguard working, not a bug.
    """
    import config as C  # imported here, not at module scope, to keep this module importable
                        # without either root configured; `config.__getattr__` resolves a path
                        # on access, so it is the attribute below that needs credentials.
    if side == "mcs":
        return pd.read_parquet(C.MCS_CORE)
    if side == "yrbs":
        return pd.read_parquet(C.YRBS_RAW)
    raise ValueError(f"side must be 'mcs' or 'yrbs', got {side!r}")


# 2. Harmonisation to the 31-feature set
# Built through `features.build_frame`, which owns the registry-driven recoder dispatch.
def build_harmonised_features(raw: pd.DataFrame, side: Side, *,
                              verbose: bool = False) -> pd.DataFrame:
    """Recode one cohort onto the shared harmonised schema.

    Builds every column of `features.FEATURE_MAP` — the single source of truth for the
    mapping — through the per-feature recoders, then encodes the declared nominal column from
    level names to codes so the frame is numeric throughout. A recoder that raises stops the
    build rather than returning a frame short of a predictor.

    THIS IS THE RAW SCHEMA, AND IT IS NOT WHAT A MODEL IS FITTED ON. The nominal predictor
    leaves here as the coded column `features.encode_categoricals` writes, which is what the
    association section needs and what the parquet files carry. `model_features` turns it into
    the model schema, and the split protocol refuses a frame that has not been through it.

    `verbose=True` prints `recode_utils.sanity_report` for every column: coverage and
    distribution, one line per declared predictor. Off by default — the harmonisation section reports
    the same coverage in one summary.

    WHICH RECODERS THE FEATURES COME FROM. The predictor matrix is built by
    `features.build_frame`, which imports `recode_mcs` and `recode_yrbs` — the un-versioned
    recoders — and writes to `config.MCS_FEATURES` / `config.YRBS_FEATURES`. The `_v6` recoders
    supply the demographic ATTRIBUTES only; see `build_attributes` below.
    """
    _check_side(side)
    frame = features.build_frame(raw, side=side, verbose=verbose)
    return features.encode_categoricals(frame)


# Assembled by `features.build_attributes`, which calls each cohort's `attr_*` recoders.
def build_attributes(raw: pd.DataFrame, side: Side) -> pd.DataFrame:
    """Build the evaluation-only attribute table: sex, age, ethnicity, orientation.

    These are never model features. The 4-category coarse ethnicity used for the Mondrian
    conformal cells and the subgroup panels is NOT built here — notebook 01 §3 applies
    `features.COARSE_ATTRIBUTE_MAP` itself, so the collapse is reported where it is decided.

    MCS attributes are written to config.MCS_ATTRIBUTES, under $MCS_DATA_DIR. Nothing here
    writes inside the repository.

    DELEGATES to features.build_attributes, which owns the attribute registry.
    This function does not call the recoders itself: one registry, one column order, one
    error contract. A recoder that raises arrives here as a named, chained `RecodeError`
    identifying the attribute and the cohort, and carrying no row values.

    The earlier ambiguity about which recoder generation supplied
    `attr_ethnicity_coarse` is resolved. A versioned and an unversioned pair once
    coexisted, but their shared functions were identical and have been merged.
    `recode_mcs` and `recode_yrbs` are now the single definitions; nothing remains
    version-split.
    
    SIDE EFFECTS: computes only. Reads no file, fits nothing, writes nothing, prints
    nothing. Persisting the result is notebook 01 section 7's job, not this one's.
    """
    _check_side(side)
    return features.build_attributes(raw, side=side)


# 3. Five-pillar ACE outcome
# The pillar definitions live in `src/outcomes.py` — `build_mcs_pillars` and
# `build_yrbs_pillars`. This function only dispatches on the cohort.
def build_pillars(raw: pd.DataFrame, side: Side) -> pd.DataFrame:
    """Build the ACE pillars for one cohort.

    One binary column per pillar. The frame is WIDER than the shared set on the YRBS side,
    which also carries `ace_neglect`, `ace_incarceration` and `ace_ipv` — constructs MCS
    does not measure. `SHARED_PILLARS` is the symmetric five, and every consumer selects
    it explicitly rather than taking the frame whole, because training on one construct
    and testing on another is not transfer.
    """
    import outcomes as BO
    if side == "mcs":
        return BO.build_mcs_pillars(raw)
    if side == "yrbs":
        return BO.build_yrbs_pillars(raw)
    raise ValueError(f"side must be 'mcs' or 'yrbs', got {side!r}")


# The outcome vocabulary, from the module that owns it. `outcomes` imports numpy, pandas and
# the MCS recodes and nothing from here, so there is no cycle — the same arrangement this file
# already uses for `features`. Both names are re-exposed because every notebook reads them
# through `data`, which is where the analysis-ready artefacts live.
from outcomes import SHARED_PILLARS, make_outcome    # noqa: F401 — re-exported


# 4. Analytic sample, splits and seeds
# UNVERIFIED: needs MCS data to check. The YRBS side needs the processed parquets under
#             $THESIS_WORK_DIR/data/processed/, and those are out of bounds under the
#             data-safety rule (do not open any .parquet), so neither side was executed.
#             The transcription is line-for-line against the source, and where it deviates
#             the deviation is stated below.
def analytic_sample_mask(features: pd.DataFrame, outcome: pd.Series, side: Side,
                         *, pillars: pd.DataFrame | None = None) -> pd.Series:
    """Boolean mask for the analytic sample: valid outcome and required-module coverage.

    YRBS applies the strict ACE-module restriction (~12,764 on record). Returns a mask,
    never a filtered frame, so the caller controls what is materialised.

    THE RESTRICTION IS THE STRICT COMPOSITE, and nothing else. The source
    (missingness_audit.py:52-55) defines the analytic sample as the rows where
    `compose_outcome(pillars5, threshold=1, strict=True)` is non-NaN — that is, rows with ALL
    FIVE pillars present. There is no separate feature-coverage criterion: the source's own
    sample-flow block records `analytic_N == final_N` with the note "no feature-based drop"
    (:128). A mask that also dropped incomplete-feature rows would be a smaller, different
    sample from the one every published number uses.

    THRESHOLD 1 IS NOT A CHOICE HERE. The mask is taken at `threshold=1` for BOTH reported
    thresholds, because `strict=True` makes non-NaN mean "all five pillars observed", which
    does not depend on the cut. Passing the >=2 outcome and taking `~isna()` off it gives the
    same rows — but going through the pillars, as the source does, makes that explicit rather
    than incidental.

    `pillars` is optional but preferred. The mask is derived from the five-pillar frame rather
    than from a composed outcome, because `outcome` alone cannot distinguish "pillar missing"
    from "pillar observed and zero" once composed. Omitting it falls back to `~outcome.isna()`,
    which is equivalent WHEN the outcome was composed with `strict=True` and silently wrong
    when it was not — so the fallback checks and says so.

    `features` is used only to fix the index the mask is aligned to, exactly as in the source
    (`.reindex(index)`), so the caller can apply it to a frame it has not filtered yet.
    """
    _check_side(side)
    index = features.index
    if pillars is not None:
        y1 = make_outcome(pillars, 1)
        return y1.notna().reindex(index).fillna(False)
    # Fallback: derive from the composed outcome. Only valid under strict=True.
    if outcome is None:
        raise ValueError("pass either pillars= (preferred) or a strict-composed outcome")
    mask = pd.Series(outcome).notna().reindex(index).fillna(False)
    if bool(mask.all()) and len(mask):
        raise ValueError(
            "analytic_sample_mask: every row has a non-NaN outcome, which means the outcome "
            "was composed with strict=False (the documented fillna(0) label leak — see "
            "make_outcome). The analytic sample cannot be recovered from it. Pass pillars=.")
    return mask


# VERIFIED: the returned key set is exactly SPLIT_KEYS — 18 of 18, in that order.
# UNVERIFIED: needs MCS data to check the SPLITS THEMSELVES. Nothing on disk records which
#             row indices each seed drew, so "seed 7 yields these 2,478 test rows" cannot be
#             confirmed without $MCS_DATA_DIR. Structure is verified; contents are not.

# The predictor and attribute schemas, derived from their registries
# SOURCE: `features.FEATURE_MAP`, which that module's own docstring names as the single
# source of truth — harmonised name -> (mcs_fn, yrbs_fn). Its key order IS the column order both
# cohorts are built in, so the schema is derived rather than transcribed: a second hand-typed
# list would be a second source of truth, and the two would drift the first time a feature moved.
#
# The import is safe at module level: `features` pulls in pandas and the two recode
# modules and nothing from this one, so there is no cycle.
from features import (ATTRIBUTE_MAP as _ATTRIBUTE_MAP,
                      COARSE_ATTRIBUTE_MAP as _COARSE_ATTRIBUTE_MAP,
                      FEATURE_MAP as _FEATURE_MAP)

FEATURE_COLUMNS: Sequence[str] = tuple(_FEATURE_MAP)

# THE TWO SCHEMAS, AND WHICH ONE A FRAME IS ON
#
# `FEATURE_COLUMNS` is the RAW harmonised schema: the columns notebook 01 recodes and persists,
# one per entry of `FEATURE_MAP`. It is what the parquet files carry and what the univariate
# association section reads, because that section needs the nominal predictor as codes so it
# can name the levels.
#
# `MODEL_FEATURE_COLUMNS` is the schema every fitted model is on: the raw schema with each
# declared nominal predictor replaced, in place, by one indicator per non-reference level. Both
# are DERIVED rather than transcribed — a second hand-written list would drift the first time a
# level was added — and every count in the pipeline reads one of these two names rather than a
# number written down beside it. `model_features` is the one transformation between them, and
# `build_splits` refuses a frame that is not on the model schema.

# The nominal predictors: those whose encoded values are level names rather than a scale. Their
# codes come from the encoder that wrote them — `features.RECENT_SEX_CONDOM_CODES` — and are not
# retyped here. `h_recent_sex_condom_status` is the only one; `h_weight_status_iotf` also takes
# three codes and is NOT here, because not overweight / overweight / obese is ordered and enters
# the model frames as one ordered column.
#
# THE FIRST DECLARED LEVEL IS THE REFERENCE and gets no column of its own, so the two indicators
# are contrasts against a respondent reporting no recent intercourse — which is the contrast
# `_feature_rows` estimates for the same predictor, so the model and the association reading
# agree on what a coefficient is against.
NOMINAL_FEATURES: dict[str, dict[str, int]] = {
    "h_recent_sex_condom_status": features.RECENT_SEX_CONDOM_CODES,
}


def _indicator_names(name: str) -> list[str]:
    """The indicator columns a declared nominal predictor contributes, in declared order."""
    return [f"{name}_{level}" for level in list(NOMINAL_FEATURES[name])[1:]]


def _expanded_schema(columns: Sequence[str]) -> list[str]:
    """`columns` with every declared nominal name replaced, in place, by its indicator names."""
    out: list[str] = []
    for column in columns:
        out += _indicator_names(column) if column in NOMINAL_FEATURES else [column]
    return out


MODEL_FEATURE_COLUMNS: Sequence[str] = tuple(_expanded_schema(FEATURE_COLUMNS))

# Persisted artefacts record this version and must be refused when the meaning
# or schema of the model features changes.
#
# v2 corrects the YRBS mappings for sugary drinks, binge frequency and lifetime
# cannabis frequency. v3 excludes the contraception predictor. v4 replaces the
# recent-intercourse predictor with the combined three-state
# `h_recent_sex_condom_status`; spec/harmonisation_spec_v5.csv carries the
# derivation and the reference-period and routing limitations.
PREPROCESSING_VERSION = "harmonisation-v4"

# THE ONE TRANSFORMATION FROM THE RAW SCHEMA TO THE MODEL SCHEMA
#
# The nominal predictor's codes are level NAMES, not quantities: the step from
# `no_recent_intercourse` to `recent_no_condom` is not the same size as the step from
# `recent_no_condom` to `recent_condom`, and neither is in a direction. A model handed the
# coded column reads an order the levels do not have — a linear model fits one slope through
# three unordered labels, and a tree splits them at a cut point that means nothing. So the
# coded column does not reach a model. `model_features` replaces it with the declared
# indicators, once, before the split protocol, and `build_splits` refuses a frame that still
# carries it.
#
# BOTH COHORTS PASS THROUGH THE SAME CALL. The encoding is fitted from the declared level map
# and from nothing else — not from the data, not from the two cohorts pooled, and not from a
# test frame — so there is no quantity here that a split could leak across.


def nominal_indicator_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """`name`'s coded column replaced, in place, by one indicator per non-reference level.

    ONE PREDICTOR AT A TIME. `model_features` is the caller and applies this to every declared
    nominal predictor in turn; that is the entry point a notebook uses.

    The first declared level is the reference and gets no column of its own, which is the
    contrast `_feature_rows` already estimates for the same predictor, so the two readings of
    the nominal predictor agree on what a coefficient is against. The indicators are
    `<name>_<level>` for the remaining levels in declared order, and they take the position the
    coded column held. Every other column keeps its values, its dtype and its position, and the
    row order and the index are the frame's own.

    MISSINGNESS IS CARRIED, NOT FILLED. A respondent whose value is missing gets a missing
    value in each indicator, so the frame arrives at `standardise_cohort`'s median imputation in
    the state the coded column would have arrived in. Filling zeros here would silently move
    those respondents into the reference level, which is an answer rather than an absence — and
    for this predictor the reference level is a substantive state, not a residual.

    REFUSES AN UNDECLARED VALUE, AND NAMES NEITHER THE VALUE NOR THE ROWS. A value the level map
    does not cover is a recoding defect: it cannot be given a level name, and guessing one would
    put respondents behind the wrong contrast. This frame is built from restricted data, so the
    message says which column and which levels are declared and nothing about what was in it.
    Anything that is neither missing nor a declared code is refused, an infinity included.
    """
    if name not in NOMINAL_FEATURES:
        raise ValueError(
            f"{name!r} is not a declared nominal predictor. The declared ones are "
            f"{sorted(NOMINAL_FEATURES)}; re-encoding a column that is a scale would assert a "
            f"level structure it does not have.")
    if name not in frame.columns:
        raise ValueError(
            f"the frame does not carry {name!r}, so there is no coded column to re-encode. It "
            f"is one of the declared predictors and a frame built to `FEATURE_COLUMNS` has it.")

    codes = NOMINAL_FEATURES[name]
    values = frame[name]
    observed = values.notna()
    if bool((observed & ~values.isin(list(codes.values()))).any()):
        raise ValueError(
            f"{name} carries a value outside its declared levels ({', '.join(codes)}), so it "
            f"cannot be re-encoded as a contrast against the first of them. Neither the value "
            f"nor the rows carrying it are named here. Correct the recoding rather than "
            f"widening this check.")

    contrasts = list(codes)[1:]          # the first declared level is the reference
    collisions = [f"{name}_{level}" for level in contrasts if f"{name}_{level}" in frame.columns]
    if collisions:
        raise ValueError(
            f"the frame already carries {collisions}, so the indicators would overwrite "
            f"columns that mean something else. Re-encode a frame that carries the coded "
            f"column alone.")

    position = frame.columns.get_loc(name)
    out = frame.drop(columns=[name])
    for offset, level in enumerate(contrasts):
        indicator = (values == codes[level]).astype(float).where(observed)
        out.insert(position + offset, f"{name}_{level}", indicator.to_numpy())
    return out


def model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """A harmonised frame on the model schema: every declared nominal predictor re-encoded.

    THE ONE PLACE THE MODEL SCHEMA IS BUILT. Both cohorts go through this call and nothing
    else re-encodes a predictor downstream, so every training, validation, test and
    target-label slice inherits the encoding from the frame this returns rather than deriving
    its own.

    Row order and the index are the frame's own, every column that is not a declared nominal
    predictor keeps its values, its dtype and its position, and the indicators take the
    position their coded column held. Missingness is carried rather than filled — see
    `nominal_indicator_frame`, which does the work.

    Refuses a frame short of a raw predictor, and refuses an undeclared code. Extra columns
    beyond the declared schema are preserved and are not part of the model schema.

    SIDE EFFECTS: computes only. Reads no file, fits nothing, writes nothing, prints nothing.
    """
    _require(frame, FEATURE_COLUMNS, "model_features")
    out = frame
    for name in NOMINAL_FEATURES:
        out = nominal_indicator_frame(out, name)
    check_model_features(out, "model_features")
    return out


def check_model_features(frame: pd.DataFrame, what: str) -> None:
    """Refuse a frame that is not on the model schema. Names columns, never values.

    Three ways to fail, and the first is the one that matters: a frame still carrying a
    declared nominal predictor as its coded column has not been through `model_features`, and
    fitting on it would put an unordered predictor into a model as a scale.
    """
    coded = [name for name in NOMINAL_FEATURES if name in frame.columns]
    if coded:
        raise ValueError(
            f"{what}: the frame carries {coded} as coded column(s). Those codes are level names "
            f"rather than a scale, so a model fitted on them reads an order the levels do not "
            f"have. Pass the frame through `model_features` before the split protocol.")
    absent = [c for c in MODEL_FEATURE_COLUMNS if c not in frame.columns]
    if absent:
        raise ValueError(
            f"{what}: the frame has no column(s) {absent}. The model schema is "
            f"{len(MODEL_FEATURE_COLUMNS)} columns and a frame short of one fits a different "
            f"model from the one the manuscript reports.")
    declared = [c for c in frame.columns if c in set(MODEL_FEATURE_COLUMNS)]
    if declared != list(MODEL_FEATURE_COLUMNS):
        raise ValueError(
            f"{what}: the declared model columns are present but not in the declared order. "
            f"The order is `MODEL_FEATURE_COLUMNS`, and a model fitted on a reordered frame "
            f"cannot be compared coefficient for coefficient with one that was not.")


# Analysis-ready artefacts: notebook 01 writes them, notebooks 02 and 04 read them.
#
# Notebook 01 owns everything up to and including the outcome definition and persists three
# frames per cohort. Notebook 02 reads the features and the pillars and owns the split
# protocol. Nothing downstream re-reads a raw cohort: a second route from raw would be a
# second definition of the outcome, and the two would drift.
#
# Both cohorts resolve outside the repository. MCS frames are person-level and go under
# $MCS_DATA_DIR; YRBS is open CDC data but its frames are still generated rather than release
# files, so they go under $THESIS_WORK_DIR/data/processed/.
#
# The outcome series are not stored. They are a deterministic collapse of the pillars through
# `make_outcome`, and a stored copy could disagree with it.
#
# A persisted feature frame records the preprocessing version and the schema that built it, in
# a sidecar beside the parquet. `read_harmonised_features` refuses a frame built under another
# version, so a recoder change cannot reach a model through a stale parquet.
#
# The pillar and attribute artefacts carry no sidecar. `PREPROCESSING_VERSION` covers the model
# feature schema and the feature recoders, and neither the pillars (`outcomes.MCS_PILLARS` /
# `YRBS_PILLARS`) nor the attributes (`features.ATTRIBUTE_MAP`) are built from either. A change
# that does alter their construction needs a stamp of its own rather than this one, which would
# otherwise claim to cover something it does not check.
FEATURE_PROVENANCE_SUFFIX = ".provenance.json"


def feature_provenance_path(path) -> Path:
    """The sidecar recording which preprocessing version wrote `path`."""
    path = Path(path)
    return path.with_name(path.name + FEATURE_PROVENANCE_SUFFIX)


def write_harmonised_features(frame: pd.DataFrame, path) -> Path:
    """Write a harmonised feature frame and the sidecar naming its preprocessing version."""
    import json
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=True)
    os.replace(tmp, path)                   # os.replace is atomic; to_parquet is not
    feature_provenance_path(path).write_text(
        json.dumps({"preprocessing_version": PREPROCESSING_VERSION,
                    "feature_columns": list(frame.columns)}, indent=2, sort_keys=True),
        encoding="utf-8")
    return path


def read_harmonised_features(path, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read a harmonised feature frame, refusing one built under another preprocessing version.

    An absent sidecar is refused for the same reason: a frame that does not record which
    version recoded it is not evidence that the live one did. The recorded schema is checked
    against `FEATURE_COLUMNS` as well, because a version name alone would pass a frame whose
    columns had moved without the name being bumped.
    """
    import json

    path = Path(path)
    sidecar = feature_provenance_path(path)
    if not sidecar.exists():
        raise ValueError(
            f"{path.name} has no {FEATURE_PROVENANCE_SUFFIX} sidecar, so the preprocessing "
            f"version that recoded it is unrecorded. Rebuild it with notebook 01 under "
            f"{PREPROCESSING_VERSION!r}.")
    recorded = json.loads(sidecar.read_text(encoding="utf-8"))
    version = recorded.get("preprocessing_version")
    if version != PREPROCESSING_VERSION:
        raise ValueError(
            f"{path.name} was built under preprocessing version {version!r}; the live version "
            f"is {PREPROCESSING_VERSION!r}. Its feature values are an earlier recoding, so "
            f"rebuild it with notebook 01 rather than reading it.")
    schema = list(recorded.get("feature_columns") or ())
    if schema != list(FEATURE_COLUMNS):
        raise ValueError(
            f"{path.name} records a harmonised schema of {len(schema)} column(s) that is not "
            f"the live `FEATURE_COLUMNS` ({len(FEATURE_COLUMNS)} columns, in a declared order).")
    return pd.read_parquet(path, columns=list(columns) if columns is not None else None)


# The frozen split fraction. Every published number is paired within a 75/25 cut, so this is
# a protocol constant rather than a parameter: it was a `test_size=` keyword no caller ever
# passed, and the docstring's caveat about justifying another value described a caller that
# did not exist.
TEST_SIZE = 0.25


def build_splits(seed: int, y_mcs: pd.Series, y_yrbs: pd.Series, *,
                 X_mcs: pd.DataFrame, X_yrbs: pd.DataFrame,
                 threshold: int | None = None) -> SplitBundle:
    """The frozen split construction: 25% test, plus the disjoint target-label pool.

    Transcribed VERBATIM from S4. This is the single most important function in the
    repository to keep unchanged: every published number is paired within these splits, and
    the k=500 anchor is a prefix the whole label-budget curve is nested inside. Changing any
    line here silently invalidates every paired comparison in the paper — silently, because
    the pipeline still runs and still produces plausible numbers.

    FOUR PRECONDITIONS, CHECKED HERE rather than by each caller. Both feature frames must
    already be on `MODEL_FEATURE_COLUMNS` — a frame still carrying a nominal predictor as codes
    has not been through `model_features`, and every fitting route in the project passes through
    this function, so refusing here is what makes that route the only one. The index must be
    unique —
    a repeated label puts one respondent in both the pool and the test set, and `pool_cs`'s
    concat joins the duplicate instead of raising. The features and the outcome must be on the
    same index, because `train_test_split` slices by position. And the outcome must carry both
    classes once `fillna(0)` has been applied, because that is what `stratify` receives.

    Four details that look incidental and are load-bearing:

      * `stratify=y.fillna(0)`. The stratification treats a MISSING outcome as a negative.
        That is not a rounding of the data — it keeps the NaN-outcome rows spread evenly
        across train and test instead of clustering, and it happens BEFORE those rows are
        dropped from training. `dropna()` first and the split changes.
      * NaN-outcome rows are dropped from TRAIN ONLY (`mtr`, `tmask`). They stay in the test
        frames and are masked at metric time by `_mask`, which is why `n_test` in the battery
        is smaller than `len(Xy_te)`.
      * `standardise_cohort` runs FIVE times, each fitted on a different frame. The
        standardisation is part of the split, not a preprocessing step applied after it, and
        each exists for one regime — see SplitBundle's field docs. A sixth frame,
        `Xy_te_cs2`, is bound to `Xy_te_cs` rather than recomputed, because today they are
        the same standardisation of the same frame.
      * The two pools drop different rows. `pool_cs` is built from the median-imputed frame
        so only NaN-y is lost; `pool_raw` is `dropna()` on unimputed features and is strictly
        smaller. This is the source of the nominal-vs-effective k gap.

    The four inputs are named for their cohort. S4 held them as module globals called `Xm` and
    `Xy_all`; they are keyword arguments here so the function has no hidden global dependency,
    and the `_all` in the old name meant "the full frame, not the analytic sample" — which is
    true of BOTH cohorts, so it said nothing that distinguished them. The eighteen frames the
    bundle returns keep S4's names, because `SplitBundle` pins that key set. The return is a
    `SplitBundle` rather than a dict: the
    standardisation is fitted inside the split, so the eighteen frames it returns are not
    derivable from row counts. The 75/25 cut is `TEST_SIZE` and is not a parameter.

    `threshold` is recorded on the bundle for bookkeeping only. It does not enter the
    computation — `y_mcs` and `y_yrbs` already encode it.

    """
    # `train_test_split` slices by POSITION, so features and outcome must already be in the
    # same row order — they are two different parquet files and nothing upstream guarantees it.
    # Out of order, every row is paired with another respondent's outcome. Today the only thing
    # that notices is `Xm_tr[m_obs]` three lines below, which raises pandas' "Unalignable
    # boolean Series provided as indexer" — a message that names neither the frames nor the
    # problem, and which does not fire at all if both carry a bare RangeIndex.
    for side, X_side, y_side in (("MCS", X_mcs, y_mcs), ("YRBS", X_yrbs, y_yrbs)):
        # The fourth precondition. Both cohorts must already be on the model schema, so no
        # regime, notebook or model family can reach a fit through a frame that still carries
        # a nominal predictor as codes. This checks columns and nothing else: it reads no
        # value, and it changes no frame.
        check_model_features(X_side, f"build_splits ({side} features)")
        for what, frame in (("features", X_side), ("outcome", y_side)):
            if not frame.index.is_unique:
                raise ValueError(
                    f"{side} {what}: repeated index labels. The same respondent would land in "
                    f"both the training pool and the test set, and the pool is built with a "
                    f"concat that joins the duplicate rather than raising.")
        if not X_side.index.equals(y_side.index):
            raise ValueError(
                f"{side}: the feature frame and the outcome are not on the same index, so "
                f"splitting them by position would pair each respondent with someone else's "
                f"outcome. Features {X_side.shape[0]} rows, outcome {y_side.shape[0]}; "
                f"reindex the outcome onto the features before calling build_splits.")

        # The third precondition, checked on what `stratify` will actually see. `y.fillna(0)`
        # counts a missing outcome as a negative, so an outcome that looks single-class by
        # `nunique()` can still stratify — checking the raw series rejects a cut that splits
        # perfectly well. One class AFTER the fill is unsplittable, and sklearn's own message
        # names neither the cohort nor the cut.
        if y_side.fillna(0).nunique() < 2:
            raise ValueError(
                f"{side}: the outcome has one class once missing values are counted as "
                f"negatives, so it cannot be stratified. Every respondent falls on the same "
                f"side of this cut — choose a threshold this cohort supports.")

    # `stratify=y.fillna(0)` treats a MISSING outcome as a negative FOR THE PURPOSE OF THE CUT
    # ONLY. It is not a rounding of the data: it spreads the NaN-outcome rows evenly across
    # train and test instead of letting them cluster, and it happens BEFORE those rows are
    # dropped from training two lines below. Dropping them first changes the split.
    m_train_rows, m_test_rows = _stratified_split_rows(y_mcs, seed)
    Xm_tr, Xm_te = X_mcs.loc[m_train_rows], X_mcs.loc[m_test_rows]
    ym_tr, ym_te = y_mcs.loc[m_train_rows], y_mcs.loc[m_test_rows]
    m_obs = ym_tr.notna(); Xm_trm, ym_trm = Xm_tr[m_obs], ym_tr[m_obs]
    y_train_rows, y_test_rows = _stratified_split_rows(y_yrbs, seed)
    Xy_tr, Xy_te = X_yrbs.loc[y_train_rows], X_yrbs.loc[y_test_rows]
    yy_tr, yy_te = y_yrbs.loc[y_train_rows], y_yrbs.loc[y_test_rows]
    y_obs = yy_tr.notna(); Xy_trm, yy_trm = Xy_tr[y_obs], yy_tr[y_obs]
    # Every frame is standardised against itself — see `standardise_cohort` and the `_cs` note
    # in SplitBundle's docstring.
    Xm_cs = standardise_cohort(Xm_trm)
    Xm_te_cs = standardise_cohort(Xm_te)
    Xy_tr_cs = standardise_cohort(Xy_tr)                     # target pool, standardised
    Xy_te_cs = standardise_cohort(Xy_te)
    Xy_tr_cs2 = standardise_cohort(Xy_trm)                   # YRBS-native train side
    # The target-trained arms' test frame. Both it and `Xy_te_cs` are `Xy_te` standardised against
    # itself, so they were equal cell for cell and one of the two fits was thrown away. Bound
    # rather than recomputed. If the open question in SplitBundle's docstring resolves toward
    # scaling their test side on the YRBS TRAINING frame, this is the line that changes.
    Xy_te_cs2 = Xy_te_cs
    # cohort-std k-pool (median-imputed -> only NaN-y dropped) and raw k-pool (complete-case)
    pool_cs = pd.concat([Xy_tr_cs, yy_tr.rename("y")], axis=1).dropna()
    pool_raw = pd.concat([Xy_tr, yy_tr.rename("y")], axis=1).dropna()
    return SplitBundle.from_s4_dict(
        dict(Xm_trm=Xm_trm, ym_trm=ym_trm, Xm_te=Xm_te, ymte=ym_te.values, Xm_te_cs=Xm_te_cs,
             Xy_tr=Xy_tr, Xy_te=Xy_te, yte=yy_te.values, Xy_trm=Xy_trm, yy_trm=yy_trm,
             Xm_cs=Xm_cs, Xy_te_cs=Xy_te_cs, Xy_tr_cs=Xy_tr_cs,
             Xy_tr_cs2=Xy_tr_cs2, Xy_te_cs2=Xy_te_cs2, y_idx=Xy_te.index.values,
             pool_cs=pool_cs, pool_raw=pool_raw),
        seed=seed, threshold=threshold)


def _stratified_split_rows(outcome: pd.Series, seed: int) -> tuple:
    """The frozen 75/25 cut, as `(train_rows, test_rows)` index labels in split order.

    THE ONE PLACE THE CUT IS DEFINED. `build_splits` slices its four frames with these labels
    and `yrbs_test_index` reads the target half out of them, so there is no second copy of the
    rule to drift.

    The partition depends on the row count and on the stratification vector alone, not on how
    many frames are handed to `train_test_split`, so splitting the outcome by itself returns
    the same members in the same order. `tests/test_split_index.py` pins that against the
    two-frame call this was extracted from.

    `stratify=outcome.fillna(0)` counts a missing outcome as a negative FOR THE CUT ONLY, which
    spreads the NaN-outcome rows evenly instead of letting them cluster. `TEST_SIZE` is 0.25 and
    is not a parameter.
    """
    train_side, test_side = train_test_split(
        outcome, test_size=TEST_SIZE, stratify=outcome.fillna(0), random_state=seed)
    return train_side.index, test_side.index


def yrbs_test_index(outcome: pd.Series, seed: int) -> pd.Index:
    """The YRBS respondents `build_splits(seed, ...)` holds out as that seed's test set.

    THE EVALUATION POPULATION, RECOVERABLE WITHOUT REFITTING ANYTHING. Every score-bearing
    procedure in notebook 02 predicts on this seed's test frame and on nothing else, so this is
    the set a persisted score cell should cover. Notebook 03 asks for it directly rather than
    inferring the population from whichever respondents turned up in the handoff.

    It needs the outcome and the seed, not the feature frame: the cut is stratified on the
    outcome and sized by the row count. **The outcome is threshold-specific**, so a different
    threshold at the same seed holds out a different set of respondents.

    Rows whose outcome is undefined stay in the test frame and are dropped at metric time, so
    the evaluable slice is this index intersected with `outcome.notna()`.
    """
    return _stratified_split_rows(outcome, seed)[1]


def standardise_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    """Median-impute, then z-score — one imputer and one scaler, both fitted on this frame.

    The reported headline lineage ('cs'). This is the live definition: `build_splits` calls it
    for every standardised frame in the bundle and `transfer.py` twice more. The archived
    pipeline copy-pasted the same arithmetic into roughly twenty runners under the names
    `cohort_std` / `make_scaler`. This is the one definition; nothing else in `src/` repeats it.

    FITTED ON WHATEVER IT IS GIVEN, INCLUDING A TEST FRAME. Each frame is standardised against
    itself and never against another. Between cohorts that is the point — it is the simplest
    label-free adaptation and the paper reports it as one (Section V-B); do not refit the source
    scaler on the target. Between the splits of ONE cohort it is a consequence rather than a
    design: a test frame is centred on its own mean rather than on the training frame's, so a
    model fitted in training units predicts in test units. See `SplitBundle`'s note on `_cs`
    keys. Nothing here changes that; it is recorded so a reader is not surprised by it.

    `StandardScaler` substitutes 1.0 for a zero variance, so a column with no variance
    standardises to 0.0 rather than to NaN. A hand-written `(x - mean) / x.std(ddof=0)` does not
    do this and is not a substitute.
    """
    imputed = SimpleImputer(strategy="median").fit_transform(frame)
    return pd.DataFrame(StandardScaler().fit_transform(imputed),
                        columns=frame.columns, index=frame.index)


MISSINGNESS_SUPPRESS = 10      # Tier 1a: subgroup cells below this are withheld entirely
MISSINGNESS_DIFF_FLAG = 10.0   # percentage-point flag threshold
MISSINGNESS_TRAIN_FRAC = 1 - TEST_SIZE   # the split protocol's train side, not a second figure
MISSINGNESS_DOC_POOL = 502     # documented ~502 raw fine-tune pool (effective_k cap)
MISSINGNESS_COLUMNS: Sequence[str] = (
    "section", "cohort", "subgroup_type", "subgroup", "key", "n",
    "pct_missing", "compare_pct", "diff_pp", "flag", "note",
)


def pct_missing(frame: pd.DataFrame) -> pd.Series:
    """Per-column percent missing, rounded to 2 dp. Aggregate by construction."""
    return (frame.isna().mean() * 100).round(2)


# UNVERIFIED: needs MCS data to check. $THESIS_WORK_DIR/tables/missingness_audit.csv exists and is the
#             frozen output, but it cannot be REBUILT without both cohorts' feature frames —
#             this function computes from row-level data rather than consolidating a CSV, so
#             unlike regime_grid there is no frame-in/frame-out path to diff. What has been
#             checked against the frozen file is its SHAPE: MISSINGNESS_COLUMNS and the six
#             section labels match its header and `section` column. The values have not been.
def missingness_audit(mcs_features: pd.DataFrame, yrbs_features: pd.DataFrame, *,
                      mcs_pillars: pd.DataFrame, yrbs_pillars: pd.DataFrame,
                      mcs_attributes: pd.DataFrame | None = None,
                      yrbs_attributes: pd.DataFrame | None = None,
                      mcs_raw_n: int | None = None,
                      yrbs_raw_n: int | None = None,
                      eth_levels: dict | None = None) -> pd.DataFrame:
    """Cross-cohort and subgroup missingness plus the TRIPOD participant-flow counts.

    Tier 1a: MCS cell counts suppressed below 10 and rounded to the nearest 10.

    Six sections, in the source's order, emitted long-format into one frame:

      1. `feature_cohort`               per-feature % missing, MCS vs YRBS, |diff| desc
      2. `pillar_cohort`                per outcome-pillar % missing, on the RAW sample
      3. `sample_flow`                  TRIPOD item 20: raw N -> strict restriction -> final N
      4. `feature_subgroup`             by sex and coarse ethnicity, within cohort
      5. `complete_case`                rows with no missing feature (the raw-lineage pool)
      6. `feature_outcome_conditional`  missingness conditioned on the outcome (MCAR read)

    SECTIONS 4 AND 6 ARE OPTIONAL and degrade honestly. Section 4 needs the attribute tables
    and is skipped with a recorded `note` row if they are not passed; section 6 needs the
    strict outcome, which is derived here from the pillars. A skipped section leaves an
    explicit row saying it was skipped, never a silent absence — an absent section reads as
    "no differential missingness found", which is the opposite of "not computed".

    The pillars, the attribute tables and the raw row counts are all keywords: sections 2, 3
    and 6 need the pillars, section 4 the attributes, and section 3's flow the counts.
    `mcs_raw_n` and `yrbs_raw_n` are COUNTS rather than frames deliberately — the audit needs
    `len(raw)` and nothing else, and a function that never receives MCS row-level data cannot
    leak it.

    THE ONE THING THE SOURCE ESTIMATES RATHER THAN MEASURES, carried across unchanged: the
    YRBS train-split pool is `round(0.75 * complete_case)`, not a real split. It is
    cross-checked against the documented ~502 and the delta reported. The real number is
    `len(SplitBundle.pool_raw)`, which varies by seed; the 0.75 figure is the seed-free
    approximation the source quotes, and replacing it here would change a published count.
    """
    feat_cols = [c for c in mcs_features.columns
                 if c != "y" and c in yrbs_features.columns]
    rows: list = []

    cohorts = {
        "MCS": dict(feat=mcs_features, pil=mcs_pillars, attr=mcs_attributes, raw_n=mcs_raw_n),
        "YRBS": dict(feat=yrbs_features, pil=yrbs_pillars, attr=yrbs_attributes, raw_n=yrbs_raw_n),
    }
    an = {}
    for ck, d in cohorts.items():
        mask = analytic_sample_mask(d["feat"], None, ck.lower(), pillars=d["pil"])
        y1 = make_outcome(d["pil"], 1)
        y1 = y1.reindex(d["feat"].index)[mask.values]
        an[ck] = dict(X=d["feat"].loc[mask.values, feat_cols], y1=y1.values, mask=mask)

    # -- 1. per-feature missingness, MCS vs YRBS, sorted by |diff| -------------
    pm_mcs, pm_yrbs = pct_missing(an["MCS"]["X"]), pct_missing(an["YRBS"]["X"])
    feat_tbl = pd.DataFrame({"feature": feat_cols,
                             "mcs_pct_missing": pm_mcs[feat_cols].values,
                             "yrbs_pct_missing": pm_yrbs[feat_cols].values})
    feat_tbl["abs_diff_pp"] = (feat_tbl.mcs_pct_missing - feat_tbl.yrbs_pct_missing).abs().round(2)
    feat_tbl["flag_gt10pp"] = feat_tbl.abs_diff_pp > MISSINGNESS_DIFF_FLAG
    feat_tbl = feat_tbl.sort_values("abs_diff_pp", ascending=False).reset_index(drop=True)
    for _, r in feat_tbl.iterrows():
        rows.append(dict(section="feature_cohort", cohort="MCS_vs_YRBS", key=r.feature,
                         pct_missing=r.mcs_pct_missing, compare_pct=r.yrbs_pct_missing,
                         diff_pp=r.abs_diff_pp, flag=bool(r.flag_gt10pp)))

    # -- 2. per outcome-pillar missingness, on the RAW sample -----------------
    pil_mcs = (mcs_pillars[list(SHARED_PILLARS)].isna().mean() * 100).round(2)
    pil_yrbs = (yrbs_pillars[list(SHARED_PILLARS)].isna().mean() * 100).round(2)
    for p in SHARED_PILLARS:
        rows.append(dict(section="pillar_cohort", cohort="MCS_vs_YRBS", key=p,
                         pct_missing=float(pil_mcs[p]), compare_pct=float(pil_yrbs[p]),
                         diff_pp=round(abs(pil_mcs[p] - pil_yrbs[p]), 2)))

    # -- 3. analytic-sample flow (TRIPOD item 20) -----------------------------
    for ck, d in cohorts.items():
        pil = d["pil"][list(SHARED_PILLARS)]
        raw_n = d["raw_n"] if d["raw_n"] is not None else len(d["feat"])
        miss_any = int(pil.isna().any(axis=1).sum())
        miss_all = int(pil.isna().all(axis=1).sum())
        analytic_n = raw_n - miss_any
        note_n = ("all rows in cohort parquet" if d["raw_n"] is not None
                  else "raw N not supplied — feature-frame length used instead")
        rows.append(dict(section="sample_flow", cohort=ck, key="raw_N", n=raw_n, note=note_n))
        rows.append(dict(section="sample_flow", cohort=ck, key="excluded_strict", n=miss_any,
                         note=f">=1 of 5 ACE pillars missing ({miss_all} missing all 5)"))
        rows.append(dict(section="sample_flow", cohort=ck, key="analytic_N", n=analytic_n,
                         note="strict=True: all 5 pillars present; == final N (no feature-based drop)"))

    # -- 4. differential missingness by subgroup ------------------------------
    subgroup_spread: dict = {}
    for ck, d in cohorts.items():
        attr = d["attr"]
        if attr is None:
            rows.append(dict(section="feature_subgroup", cohort=ck, key="(all features)",
                             note="SKIPPED — no attribute table passed; section 4 not computed"))
            continue
        X = an[ck]["X"]
        m = an[ck]["mask"].values
        sex = attr["attr_sex"].reindex(d["feat"].index).map(SEX_LABELS)[m].values
        eth = attr["attr_ethnicity_coarse"].reindex(d["feat"].index)[m].values
        levels_eth = (eth_levels or {}).get(ck)
        if levels_eth is None:
            levels_eth = [v for v in pd.unique(eth) if v == v]
        for stype, vec, levels in (("sex", sex, ["male", "female"]),
                                   ("ethnicity", eth, list(levels_eth))):
            per_feat_vals = {f: [] for f in feat_cols}
            for g in levels:
                gmask = (vec == g)
                n_g = int(gmask.sum())
                if n_g < MISSINGNESS_SUPPRESS:
                    rows.append(dict(section="feature_subgroup", cohort=ck, subgroup_type=stype,
                                     subgroup=g, key="(all features)",
                                     note=f"SUPPRESSED n<{MISSINGNESS_SUPPRESS}"))
                    continue
                pmg = pct_missing(X[gmask])
                for f in feat_cols:
                    v = float(pmg[f])
                    per_feat_vals[f].append(v)
                    rows.append(dict(section="feature_subgroup", cohort=ck, subgroup_type=stype,
                                     subgroup=g, key=f, n=n_g, pct_missing=v))
            for f in feat_cols:
                vals = per_feat_vals[f]
                if len(vals) >= 2:
                    spread = round(max(vals) - min(vals), 2)
                    subgroup_spread[(ck, stype, f)] = spread
                    if spread > MISSINGNESS_DIFF_FLAG:
                        rows.append(dict(section="feature_subgroup", cohort=ck,
                                         subgroup_type=stype, subgroup="(spread)", key=f,
                                         diff_pp=spread, flag=True,
                                         note="max-min across subgroups > 10pp"))

    # -- 5. complete-case pool + the YRBS ~502 cross-check --------------------
    for ck in cohorts:
        X = an[ck]["X"]
        n_an = len(X)
        any_missing = int(X.isna().any(axis=1).sum())
        complete_case = n_an - any_missing
        pct_any = round(100 * any_missing / n_an, 2) if n_an else float("nan")
        note = (f"rows with NO missing feature across all {len(feat_cols)} "
                f"(raw-lineage pool)")
        if ck == "YRBS":
            train_pool = int(round(MISSINGNESS_TRAIN_FRAC * complete_case))
            note += (f"; train-split est {train_pool} = 0.75 x {complete_case}, "
                     f"documented ~{MISSINGNESS_DOC_POOL}, "
                     f"delta {train_pool - MISSINGNESS_DOC_POOL:+d}")
        rows.append(dict(section="complete_case", cohort=ck, key="complete_case_N",
                         n=complete_case, pct_missing=pct_any, note=note))

    # -- 6. missingness conditioned on the outcome (MCAR/MAR/MNAR read) -------
    for ck in cohorts:
        X, y1 = an[ck]["X"], an[ck]["y1"]
        pos, neg = X[y1 == 1], X[y1 == 0]
        if len(pos) < MISSINGNESS_SUPPRESS or len(neg) < MISSINGNESS_SUPPRESS:
            rows.append(dict(section="feature_outcome_conditional", cohort=ck,
                             key="(all features)",
                             note=f"SUPPRESSED — an outcome arm has n<{MISSINGNESS_SUPPRESS}"))
            continue
        pm_pos, pm_neg = pct_missing(pos), pct_missing(neg)
        for f in feat_cols:
            diff = round(float(pm_pos[f] - pm_neg[f]), 2)
            rows.append(dict(section="feature_outcome_conditional", cohort=ck, key=f,
                             pct_missing=float(pm_pos[f]), compare_pct=float(pm_neg[f]),
                             diff_pp=diff, flag=bool(abs(diff) > MISSINGNESS_DIFF_FLAG)))

    audit = pd.DataFrame(rows)
    for c in MISSINGNESS_COLUMNS:
        if c not in audit.columns:
            audit[c] = np.nan
    return audit[list(MISSINGNESS_COLUMNS)]




# 5. Cohort comparison — how each predictor relates to the outcome, in each cohort
#
# ONE QUESTION. Transfer from MCS to YRBS loses discrimination. This asks whether the
# predictors relate to the outcome differently in the two cohorts, which is one observable
# thing that could sit behind such a loss.
#
# THE OTHER TWO DESCRIPTIVE COMPARISONS ARE NOT HERE. Pillar prevalence by cohort is notebook
# 01's own section, and per-predictor missingness by cohort is `missingness_audit`'s
# `feature_cohort` rows, displayed and published from that notebook. Recomputing either would
# be a second copy of a table that already exists.
#
# WHAT THIS IS NOT. Not a decomposition of the transfer gap: nothing here attributes any part
# of the loss to any difference it finds. Not a test of measurement invariance: two cohorts can
# reach the same coefficient from different constructs. And it identifies nothing — country,
# age range, instrument and sampling frame differ at once between MCS Sweep 6 and YRBS 2023.
# Nothing is tuned, split or scored, and no estimate here is a predictive performance.
#
# THE SAMPLE, AND MISSING VALUES. ONE CONVENTION FOR EVERY PREDICTOR: each association is
# estimated on the respondents who answered that predictor, within the cohort's analytic
# sample. The modelling frames median-impute instead (`standardise_cohort`), and that route
# cannot cover the nominal predictor — the median of a set of level names is not one of them.
# Imputing the scales and dropping rows for the nominal one would make one estimate answer a
# different question from the rest, so all of them answer the same one, and the convention is
# stated wherever the estimates are displayed. Standardisation is within the cohort and within
# the predictor's answered rows.
#
# TWO STATUSES, BECAUSE THEY ANSWER TWO DIFFERENT QUESTIONS. Whether a term could be estimated
# is a statistical fact about the data; whether the estimate may be released is a disclosure
# decision about the small cells behind it. Collapsing them makes the scientific reading wrong
# in one direction — a term stopped by the small-count rule is not a term the data could not
# answer — so every term is fitted, and the two are recorded in separate columns:
#
#   estimation_status   `estimated`, or `not_estimable` for one observed level, a declared
#                       level nobody is in, or a fit that did not converge. Nothing else.
#   disclosure_status   `reportable`, or `not_reportable` when a cell the estimate rests on is
#                       small under `publication`'s rule, checked from BOTH tails — the group
#                       and its complement. A predictor that fails any of them has all of its
#                       terms marked together, and which cell failed does not leave this
#                       section.
#
# TWO ROUTES OUT, AND THE RELEASE-SAFE ONE IS THE DEFAULT. `cohort_associations` and
# `compare_cohort_associations` return the release-safe view: one `status` column, and a blank
# estimate wherever the term is not reportable. `cohort_associations_internal` and
# `compare_cohort_associations_internal` return the internal view, which keeps the estimate a
# withheld term has and both statuses beside it. Neither view carries a count, a denominator
# or a suppression reason, and passing the rule is not disclosure clearance.

import publication as _P

# The nominal predictors are declared with the schemas, beside `FEATURE_COLUMNS`, because the
# model schema is derived from them. This section reads `NOMINAL_FEATURES` to decide which
# predictors get a contrast against a reference level instead of a slope.

# What an estimate is on. Both are log-odds and they are not interchangeable, so the scale
# travels with every row and the summary never mixes them.
PER_SD = "log_odds_per_sd"                  # one SD of the predictor, within that cohort
PER_LEVEL = "log_odds_vs_reference_level"   # one nominal level against the reference level

# Whether the term could be estimated at all. Degenerate and non-convergent fits share one
# status because the difference between them is a debugging detail.
ESTIMATED = "estimated"
NOT_ESTIMABLE = "not_estimable"             # one observed level, or a fit that did not converge

# Whether the estimate may be released. A separate axis, and never a reason.
REPORTABLE = "reportable"
NOT_REPORTABLE = "not_reportable"

# What the release-safe view puts in place of a withheld estimate: `publication`'s single
# marker, deliberately indistinguishable from any other withheld value it writes.
NOT_REPORTED = _P.NOT_REPORTED_STATUS

TOP_K = 5
MAX_ITER = 1000

# The two per-cohort schemas. Neither carries a count, a denominator or a reason.
INTERNAL_ASSOCIATION_COLUMNS: Sequence[str] = ("feature", "term", "scale", "estimate",
                                               "estimation_status", "disclosure_status")
ASSOCIATION_COLUMNS: Sequence[str] = ("feature", "term", "scale", "estimate", "status")

# The two side-by-side schemas, in the same order.
INTERNAL_COMPARISON_COLUMNS: Sequence[str] = (
    "feature", "term", "scale", "mcs_estimate", "yrbs_estimate", "difference",
    "mcs_estimation_status", "yrbs_estimation_status",
    "mcs_disclosure_status", "yrbs_disclosure_status")
COMPARISON_COLUMNS: Sequence[str] = (
    "feature", "term", "scale", "mcs_estimate", "yrbs_estimate", "difference",
    "mcs_status", "yrbs_status")


def _require(frame: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    """A named absence rather than a KeyError from three frames down. Names columns, not values."""
    absent = [c for c in columns if c not in frame.columns]
    if absent:
        raise ValueError(f"{what}: the frame has no column(s) {absent}. A comparison of the "
                         f"declared columns stops rather than silently narrowing.")


def _reportable(cells: Sequence[tuple[int, int]], cohort: str) -> bool:
    """Whether every cell an estimate rests on is safe to report.

    `cells` are (count, denominator) pairs. `publication.withhold_rate` checks BOTH tails of
    each, so one pair covers a group and its complement together. Which pair failed does not
    leave this function.
    """
    return not any(_P.withhold_rate(count, total, cohort) for count, total in cells)


def _fit(design: np.ndarray, outcome: np.ndarray) -> np.ndarray | None:
    """Effectively unpenalised logistic coefficients, or None if the fit did not converge.

    `C=1e8` is this repository's way of asking for that — `evaluation.cal_slope_intercept` fits
    the calibration slope the same way — and avoids `penalty=None`, which scikit-learn 1.8
    deprecates and 1.10 removes. Non-convergence is usually separation; the estimate is dropped
    rather than reported at whatever value the optimiser stopped at.
    """
    model = LogisticRegression(C=1e8, solver="lbfgs", max_iter=MAX_ITER)
    model.fit(design, outcome)
    if int(np.max(model.n_iter_)) >= MAX_ITER:
        return None
    return model.coef_[0]


def _as_levels(name: str, values: pd.Series) -> pd.Series:
    """A nominal predictor's encoded column as level names, or a refusal naming the codes.

    The codes are labels: entering the predictor as one number would assert that the step from
    `no_recent_intercourse` to `recent_no_condom` is the same size as the step from
    `recent_no_condom` to `recent_condom`, and in some direction. A code the map does not cover
    cannot be given a level name, and guessing one would report a contrast against the wrong
    reference.
    """
    codes = {code: level for level, code in NOMINAL_FEATURES[name].items()}
    unknown = sorted(set(values.dropna().unique()) - set(codes))
    if unknown:
        raise ValueError(f"{name}: encoded value(s) {unknown} are not in the declared level "
                         f"map {sorted(codes)}.")
    return values.map(codes)


def _disclosure_status(answered: pd.Series, values: pd.Series, outcome: np.ndarray,
                       present: Sequence, *, nominal: bool, cohort: str) -> str:
    """Whether this predictor's estimates may be released. Never why, and never how close.

    THE CELLS AN ESTIMATE RESTS ON: who answered (against the analytic sample), the two outcome
    arms among them, and — where the predictor has a level structure the coefficient could be
    inverted back to — each level and each level's outcome split. A predictor taking two values
    is exactly its 2x2 table up to the standardisation; one taking more is a slope through
    several cells and is not, so the level checks apply to the two-level and nominal predictors.

    The counts assembled here are local and are reduced to one word before returning. This
    decides release only: no caller consults it to decide whether to fit.
    """
    cells = [(len(answered), len(values)), (int(outcome.sum()), len(answered))]
    if nominal or len(present) == 2:
        for level in present:
            in_level = (answered == level).to_numpy()
            cells += [(int(in_level.sum()), len(answered)),
                      (int(outcome[in_level].sum()), int(in_level.sum()))]
    return REPORTABLE if _reportable(cells, cohort) else NOT_REPORTABLE


def _not_estimable(rows: Sequence[tuple[dict, object]]) -> list[dict]:
    """Every one of this predictor's terms, marked as the data could not answer it."""
    for row, _ in rows:
        row["estimation_status"] = NOT_ESTIMABLE
    return [row for row, _ in rows]


def _feature_rows(name: str, values: pd.Series, outcome: np.ndarray, cohort: str) -> list[dict]:
    """Every row this predictor contributes: one for a scale, one per non-reference level.

    A nominal predictor becomes indicators against its first declared level; everything else
    becomes one standardised column. Both are estimated on the respondents who answered.

    EVERY DECLARED TERM IS ATTEMPTED. The disclosure status is settled first, and settles only
    whether the estimate may leave the section — the fit runs either way, so a term withheld
    for release still carries a scientific answer here.
    """
    nominal = name in NOMINAL_FEATURES
    if nominal:
        values = _as_levels(name, values)
        declared = list(NOMINAL_FEATURES[name])
        reference, contrasts = declared[0], declared[1:]
        rows = [(dict(feature=name, term=f"{name}={level} vs {reference}", scale=PER_LEVEL,
                      estimate=None, estimation_status=ESTIMATED,
                      disclosure_status=REPORTABLE), level) for level in contrasts]
    else:
        declared, reference, contrasts = [], None, []
        rows = [(dict(feature=name, term=name, scale=PER_SD, estimate=None,
                      estimation_status=ESTIMATED, disclosure_status=REPORTABLE), None)]

    answered = values.notna()
    x, y = values[answered], outcome[answered.to_numpy()]
    present = [level for level in declared if (x == level).any()] if nominal \
        else sorted(pd.unique(x))

    disclosure = _disclosure_status(x, values, y, present, nominal=nominal, cohort=cohort)
    for row, _ in rows:
        row["disclosure_status"] = disclosure

    if len(present) < 2 or (nominal and reference not in present):
        return _not_estimable(rows)

    if nominal:
        used = [level for level in contrasts if level in present]
        design = np.column_stack([(x == level).to_numpy(float) for level in used])
    else:
        used, design = [], ((x - x.mean()) / x.std(ddof=0)).to_numpy().reshape(-1, 1)

    coefficients = _fit(design, y)
    if coefficients is None:
        return _not_estimable(rows)
    for row, level in rows:
        if nominal and level not in used:
            row["estimation_status"] = NOT_ESTIMABLE   # a declared level nobody in this cohort is in
        else:
            row["estimate"] = float(coefficients[used.index(level) if nominal else 0])
    return [row for row, _ in rows]


def _release_status(estimation_status: str, disclosure_status: str) -> str:
    """The one word the release-safe view shows for a term.

    A term the data could not answer says so, whatever the cells behind it look like: reporting
    it as withheld would assert that an estimate exists to withhold. Everything else that is
    not reportable carries `publication`'s single marker.
    """
    if estimation_status != ESTIMATED:
        return NOT_ESTIMABLE
    if disclosure_status != REPORTABLE:
        return NOT_REPORTED
    return ESTIMATED


def cohort_associations_internal(X: pd.DataFrame, y, *, cohort: str,
                                 columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Univariate feature–outcome associations within one cohort, as estimated. One row per term.

    THE INTERNAL VIEW. Every declared term is attempted and every estimate that exists is
    present, including the estimates the small-count rule will not let out; `disclosure_status`
    says which those are. FOR USE INSIDE THE APPROVED ENVIRONMENT ONLY — nothing returned here
    may be displayed, copied or published without disclosure review. `cohort_associations` is
    the route for anything else.

    Each predictor is related to the outcome ON ITS OWN — one one-predictor fit per predictor,
    never one model of the whole schema — so an estimate carries whatever the predictor shares
    with the others and is not an adjusted effect. For a scalar predictor the estimate is the
    change in log-odds of the outcome per within-cohort standard deviation of it; for the
    nominal predictor it is the log-odds of one level against the reference level.

    `y` is the outcome on the analytic sample and may not carry missing values: the strict
    composite has none there, and dropping rows here would estimate over a sample nothing else
    in the pipeline uses.
    """
    if cohort not in ("MCS", "YRBS"):
        raise ValueError(f"cohort must be 'MCS' or 'YRBS', got {cohort!r} — the label decides "
                         f"whether the small-count rule applies")
    columns = list(columns or FEATURE_COLUMNS)
    _require(X, columns, f"cohort_associations ({cohort})")

    outcome = pd.Series(np.asarray(y, dtype=float))
    if len(outcome) != len(X):
        raise ValueError(f"cohort_associations ({cohort}): {len(X)} feature rows and "
                         f"{len(outcome)} outcome values.")
    if outcome.isna().any():
        raise ValueError(f"cohort_associations ({cohort}): the outcome carries missing values. "
                         f"Restrict to the analytic sample first — `analytic_sample_mask` is "
                         f"the definition.")
    if outcome.nunique() < 2:
        raise ValueError(f"cohort_associations ({cohort}): the outcome has one class, so no "
                         f"association is defined.")

    values = outcome.to_numpy()
    rows = [row for column in columns
            for row in _feature_rows(column, X[column].reset_index(drop=True), values, cohort)]
    frame = pd.DataFrame(rows, columns=list(INTERNAL_ASSOCIATION_COLUMNS))
    frame["estimate"] = pd.to_numeric(frame["estimate"])
    return frame


def cohort_associations(X: pd.DataFrame, y, *, cohort: str,
                        columns: Sequence[str] | None = None) -> pd.DataFrame:
    """The release-safe view of `cohort_associations_internal`. One row per term.

    Same estimand, same sample, same missingness convention. The two statuses are collapsed
    into one `status` column and the estimate is blank wherever the term is not reportable, so
    nothing a small cell stands behind reaches a caller.
    """
    internal = cohort_associations_internal(X, y, cohort=cohort, columns=columns)
    out = internal.copy()
    out["status"] = [_release_status(estimation, disclosure) for estimation, disclosure
                     in zip(out["estimation_status"], out["disclosure_status"])]
    out.loc[out["status"] == NOT_REPORTED, "estimate"] = np.nan
    return out[list(ASSOCIATION_COLUMNS)]


def compare_cohort_associations_internal(mcs_features: pd.DataFrame,
                                         yrbs_features: pd.DataFrame, *,
                                         mcs_pillars: pd.DataFrame, yrbs_pillars: pd.DataFrame,
                                         threshold: int = 2) -> pd.DataFrame:
    """The two cohorts' univariate associations side by side, as estimated. One row per term.

    THE INTERNAL VIEW, with the same warning `cohort_associations_internal` carries: it keeps
    the MCS estimates the small-count rule will not release, so it is for use inside the
    approved environment and its contents need disclosure review before they go anywhere.

    Restricts each cohort with `analytic_sample_mask` and composes the outcome at `threshold`
    through `make_outcome`. There is no route through this function that builds a sample or an
    outcome of its own, and it reads and writes nothing.

    A term estimable in one cohort and not the other keeps its row with that side blank, so a
    predictor that dropped out is visible rather than absent.
    """
    estimates = {}
    for side, cohort, X, pillars in (("mcs", "MCS", mcs_features, mcs_pillars),
                                     ("yrbs", "YRBS", yrbs_features, yrbs_pillars)):
        _require(pillars, SHARED_PILLARS, f"compare_cohort_associations ({side} pillars)")
        _require(X, FEATURE_COLUMNS, f"compare_cohort_associations ({side} features)")
        mask = analytic_sample_mask(X, None, side, pillars=pillars).to_numpy()
        estimates[side] = cohort_associations_internal(
            X.loc[mask, list(FEATURE_COLUMNS)],
            make_outcome(pillars, threshold).reindex(X.index)[mask], cohort=cohort)

    merged = estimates["mcs"].merge(estimates["yrbs"], on=["feature", "term", "scale"],
                                    how="outer", suffixes=("_mcs", "_yrbs"), sort=False)
    merged = merged.rename(columns={
        "estimate_mcs": "mcs_estimate", "estimate_yrbs": "yrbs_estimate",
        "estimation_status_mcs": "mcs_estimation_status",
        "estimation_status_yrbs": "yrbs_estimation_status",
        "disclosure_status_mcs": "mcs_disclosure_status",
        "disclosure_status_yrbs": "yrbs_disclosure_status"})

    # THE ROW ORDER IS THE SCHEMA'S, RESTORED RATHER THAN INHERITED. An outer merge does not
    # promise to preserve either side's order, and which order it happens to produce varies by
    # pandas version — so the declared order is reimposed here from `_declared_terms` rather
    # than left to the merge. Sorting alphabetically would be a different order that also looks
    # stable, and a second hand-written list would be a second source of truth.
    expected = _declared_terms()
    order = {term: position for position, term in enumerate(expected)}
    present = list(merged["term"])
    unknown = sorted(set(present) - set(order))
    if unknown:
        raise ValueError(
            f"the comparison carries term(s) {unknown} that the declared schema does not "
            f"produce, so the two cohorts were not built to the same schema.")
    absent = [term for term in expected if term not in set(present)]
    if absent:
        raise ValueError(
            f"the comparison is missing declared term(s) {absent}. A comparison short of one "
            f"describes a different schema from the one the manuscript reports.")
    repeated = sorted(term for term in set(present) if present.count(term) > 1)
    if repeated:
        raise ValueError(
            f"the comparison carries more than one row for term(s) {repeated}, so the merge "
            f"joined two rows that describe the same term.")

    # `map` then sort then drop, so `feature` stays ordinary object data — a categorical column
    # would carry the ordering into every frame built from this one.
    merged = (merged.assign(_position=merged["term"].map(order))
                    .sort_values("_position", kind="stable")
                    .drop(columns="_position")
                    .reset_index(drop=True))

    merged["difference"] = merged["yrbs_estimate"] - merged["mcs_estimate"]
    return merged[list(INTERNAL_COMPARISON_COLUMNS)]


def compare_cohort_associations(mcs_features: pd.DataFrame, yrbs_features: pd.DataFrame, *,
                                mcs_pillars: pd.DataFrame, yrbs_pillars: pd.DataFrame,
                                threshold: int = 2) -> pd.DataFrame:
    """The release-safe view of `compare_cohort_associations_internal`. One row per term.

    Each side's two statuses are collapsed into one `status` column, an estimate that is not
    reportable is blanked, and `difference` is recomputed afterwards so it cannot carry back a
    value neither estimate shows. This is what every caller gets unless it asks for the
    internal route by name.
    """
    internal = compare_cohort_associations_internal(
        mcs_features, yrbs_features, mcs_pillars=mcs_pillars, yrbs_pillars=yrbs_pillars,
        threshold=threshold)
    out = internal.copy()
    for side in ("mcs", "yrbs"):
        estimation = out[f"{side}_estimation_status"].fillna(NOT_ESTIMABLE)
        disclosure = out[f"{side}_disclosure_status"].fillna(NOT_REPORTABLE)
        out[f"{side}_status"] = [_release_status(e, d) for e, d in zip(estimation, disclosure)]
        out.loc[out[f"{side}_status"] == NOT_REPORTED, f"{side}_estimate"] = np.nan
    out["difference"] = out["yrbs_estimate"] - out["mcs_estimate"]
    return out[list(COMPARISON_COLUMNS)]


def _declared_terms() -> list[str]:
    """Every term the comparison should carry, in declared order.

    One per predictor, or one per non-reference level for a declared nominal predictor — the
    same sequence `_feature_rows` emits, derived from the same two registries rather than
    written out a second time.
    """
    terms: list[str] = []
    for feature in FEATURE_COLUMNS:
        if feature in NOMINAL_FEATURES:
            declared = list(NOMINAL_FEATURES[feature])
            terms += [f"{feature}={level} vs {declared[0]}" for level in declared[1:]]
        else:
            terms.append(feature)
    return terms


def _both_estimated(comparison: pd.DataFrame) -> pd.DataFrame:
    return comparison[comparison[["mcs_estimate", "yrbs_estimate"]].notna().all(axis=1)]


def _directional(both: pd.DataFrame) -> pd.DataFrame:
    """The terms carrying two estimates that have a direction: exactly zero has no sign."""
    return both[(both["mcs_estimate"] != 0) & (both["yrbs_estimate"] != 0)]


def _agreement(both: pd.DataFrame) -> dict:
    """How far two sets of estimates point the same way, and how far they track each other.

    The one arithmetic both summaries share, so the release-safe and internal frames cannot
    drift apart in what a direction or a correlation means. `both` is whatever the caller
    counts as having two estimates.

    Sign agreement covers every term with two estimates, whatever its scale, because a sign
    does not depend on units. The two correlations cover the per-SD terms alone: a coefficient
    per standard deviation and a contrast between two nominal levels are both log-odds and are
    not measurements of the same thing.
    """
    from scipy import stats

    signed = _directional(both)
    per_sd = both[both["scale"] == PER_SD]
    same = int((np.sign(signed["mcs_estimate"]) == np.sign(signed["yrbs_estimate"])).sum())
    correlated = len(per_sd) >= 3
    return dict(
        n_direction_terms=len(signed),
        n_same_direction=same,
        n_reversed_direction=len(signed) - same,
        sign_agreement=round(same / len(signed), 4) if len(signed) else None,
        n_correlated_terms=len(per_sd),
        pearson_r=round(float(np.corrcoef(per_sd["mcs_estimate"],
                                          per_sd["yrbs_estimate"])[0, 1]), 4)
        if correlated else None,
        spearman_rho=round(float(stats.spearmanr(per_sd["mcs_estimate"],
                                                 per_sd["yrbs_estimate"]).statistic), 4)
        if correlated else None)


def association_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """How far the two cohorts agree, over the terms this view may report. One row.

    EVERY NUMBER IN IT IS ABOUT TERMS, and every one of them is about REPORTABLE terms, because
    that is all this view carries. A term is reportable when both cohorts returned an estimate
    the small-count rule allows out; `n_withheld_terms` says how many terms lost an estimate to
    that rule and nothing further, so the size of the release is establishable without the cell
    that decided it. WHAT IS COUNTED HERE IS NOT A SCIENTIFIC TALLY: a withheld term was
    estimated, and `association_summary_internal` is the frame that counts it.

    THE ACCOUNTING IS COMPLETE, so a term that drops out of a later column can be found in an
    earlier one. A reportable term enters the direction accounting unless an estimate is
    exactly zero, which has no sign to agree or disagree with. So `n_same_direction` and
    `n_reversed_direction` sum to `n_direction_terms`, and `sign_agreement` is the first of
    those two over their total.

    AGREEMENT IS NOT EQUIVALENCE. Two cohorts can reach the same coefficient from different
    instruments, age ranges and sampling frames.
    """
    _require(comparison, ("mcs_status", "yrbs_status"), "association_summary")
    both = _both_estimated(comparison)
    withheld = (comparison[["mcs_status", "yrbs_status"]] == NOT_REPORTED).any(axis=1)
    return pd.DataFrame([dict(
        n_attempted_terms=len(comparison),
        n_reportable_terms=len(both),
        reportable_fraction=round(len(both) / len(comparison), 4) if len(comparison) else None,
        **_agreement(both),
        n_withheld_terms=int(withheld.sum()))])


def association_summary_internal(comparison: pd.DataFrame) -> pd.DataFrame:
    """How far the two cohorts agree, over the terms that were actually estimated. One row.

    THE SCIENTIFIC TALLY, and the reason the two summaries are separate: a term withheld for
    release was estimated, and it belongs in the direction accounting and in the correlations
    exactly as any other estimated term does. `n_not_estimable` counts the terms the data could
    not answer in at least one cohort; `n_not_reportable_for_release` counts the estimated
    terms the release-safe view would blank, which is a disclosure fact and not a scientific
    one. Neither is a count of respondents, and no cell, denominator or reason appears here.

    Sign agreement and the two correlations mean what `_agreement` says they mean: every
    estimated non-zero term has a direction whatever its scale, and only the per-SD terms enter
    the correlations. FOR USE INSIDE THE APPROVED ENVIRONMENT ONLY.
    """
    _require(comparison, ("mcs_estimation_status", "yrbs_estimation_status",
                          "mcs_disclosure_status", "yrbs_disclosure_status"),
             "association_summary_internal")
    both = _both_estimated(comparison)
    # Exactly what the release-safe view would blank, so the two summaries agree on the size of
    # the gap between them: a side that was estimated and may not be released.
    withheld = pd.Series(False, index=comparison.index)
    for side in ("mcs", "yrbs"):
        withheld = withheld | ((comparison[f"{side}_estimation_status"] == ESTIMATED)
                               & (comparison[f"{side}_disclosure_status"] == NOT_REPORTABLE))
    return pd.DataFrame([dict(
        n_attempted_terms=len(comparison),
        n_estimable_terms=len(both),
        **_agreement(both),
        n_not_estimable=len(comparison) - len(both),
        n_not_reportable_for_release=int(withheld.sum()))])


def largest_association_differences(comparison: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    """The `k` per-SD terms whose two estimates differ most, largest first.

    Per-SD terms only, for the reason `_agreement` gives. Ties break on the term name, so the
    order is a property of the estimates rather than of the order the predictors were built in.
    A LARGE DIFFERENCE IS NOT A CAUSE OF TRANSFER LOSS: no model is refitted without the term,
    and nothing here measures what a difference is worth.
    """
    both = _both_estimated(comparison)
    per_sd = both[both["scale"] == PER_SD].assign(
        absolute_difference=lambda f: f["difference"].abs())
    ranked = per_sd.sort_values(["absolute_difference", "term"], ascending=[False, True])
    return ranked.head(k)[["term", "mcs_estimate", "yrbs_estimate", "difference",
                           "absolute_difference"]].reset_index(drop=True)


def reversed_associations(comparison: pd.DataFrame) -> pd.DataFrame:
    """The terms carrying two estimates that point opposite ways, largest gap first.

    Every scale, for the reason `_agreement` gives about signs: a direction does not depend on
    units, so a level contrast belongs here beside a per-SD coefficient. There is no magnitude
    threshold — a reversal is a reversal, and one row here is one term counted in
    `n_reversed_direction`. Ties break on the term name, so the order is a property of the
    estimates rather than of the order the predictors were built in.

    A REVERSAL IS NOT A CONTRADICTION BETWEEN THE COHORTS. Each estimate is univariate and
    carries whatever its predictor shares with the other thirty, and nothing here tests whether
    the two differ by more than they differ from zero.
    """
    signed = _directional(_both_estimated(comparison))
    reversed_ = signed[np.sign(signed["mcs_estimate"]) != np.sign(signed["yrbs_estimate"])]
    ranked = reversed_.assign(absolute_difference=lambda f: f["difference"].abs()) \
        .sort_values(["absolute_difference", "term"], ascending=[False, True])
    return ranked[["term", "mcs_estimate", "yrbs_estimate", "difference"]].reset_index(drop=True)


def unreported_features(comparison: pd.DataFrame) -> list[str]:
    """The predictors carrying no reportable estimate in the release-safe view, named only.

    ONE LIST, and deliberately not two. Whether a predictor is absent because a cell was small
    or because it was degenerate is the suppression reason, and a reason names the rule that
    applied, which names the size of what it hid. THIS IS NOT A LIST OF WHAT COULD NOT BE
    ESTIMATED — some of these terms have an estimate in the internal view.
    """
    missing = comparison[comparison[["mcs_estimate", "yrbs_estimate"]].isna().any(axis=1)]
    return sorted(set(missing["feature"]))


# 6. YRBS analytic inclusion — who the outcome rule leaves out, and how they differ
#
# THE COMPARISON. The analytic sample is the respondents whose harmonised outcome can be
# defined: all five shared pillars observed. Everyone else is excluded. This sets the two
# groups side by side on the demographics the cohort already carries, so that any selection
# associated with outcome availability is visible rather than assumed absent.
#
# WHAT EXCLUSION MEANS, AND WHAT IT DOES NOT. It means the harmonised outcome could not be
# defined for that respondent, and nothing more. The 2023 adverse-experience module was
# OPTIONAL, so a site that did not field it contributes excluded respondents in bulk — but
# the data record only that the answers are absent. NOTHING HERE ESTABLISHES WHY A SITE DID
# NOT ADMINISTER A MODULE, and no row of the returned frame should be read as evidence about
# a school, a district or a state.
#
# YRBS ALONE, AND DELIBERATELY. YRBS is open CDC data and is reported in full. The MCS
# equivalent is not simply this function called twice: the included and excluded breakdowns
# are two tables over the same categories on nested populations, which is exactly the case
# `publication.paired_withholding` exists for, and it would have to be decided together with
# the breakdowns notebook 01 already publishes. So there is no `side` argument to get wrong.
#
# NO COUNTS COME OUT. Group sizes are denominators inside this function and stay there. The
# returned frame carries means, percentages and differences and nothing a count can be read
# off, and writing it is not disclosure clearance.

# The one sex crosswalk. `missingness_audit` reads it too; notebook 01's characteristics cell
# still carries the same literal inline.
SEX_LABELS = {0.0: "male", 1.0: "female"}

# The declared coarse-ethnicity categories, READ FROM THE CROSSWALK rather than retyped. Both
# YRBS source maps land on the same four, `Other` among them, so the set is what the recoder
# can produce and not a second list that could drift from it.
YRBS_ETHNICITY_LEVELS: Sequence[str] = tuple(sorted(
    set(_Y._RACE4_TO_COARSE.values()) | set(_Y._RACEETH8_TO_COARSE.values())))

# What is compared, and what the comparison is allowed to say. `attr_orientation` is NOT here:
# `recode_yrbs.attr_orientation` records that its grouping of the "some other way", "not sure"
# and "do not know" codes is an open decision made when nothing consumed the attribute, and a
# selection table is not the place to start consuming it.
INCLUSION_ATTRIBUTES: Sequence[str] = ("attr_age", "attr_sex", "attr_ethnicity_coarse")
INCLUSION_COLUMNS: Sequence[str] = ("variable", "level", "included_value", "excluded_value",
                                    "difference", "measure")

MEAN_YEARS = "mean_years"                   # attr_age is the YRBS age band's midpoint in years
PERCENT = "percent"                         # of that group; the difference is in points
SMD = "standardised_mean_difference"        # (included - excluded) / pooled SD

# Continuous rows have no category, and this is what stands in their `level`.
WHOLE_GROUP = "all"


def _age_rows(included: pd.Series, excluded: pd.Series) -> list[dict]:
    """The mean age in each group, their difference, and the standardised difference.

    `attr_age` is the principal age measure on the YRBS side — `recode_yrbs.attr_age` maps the
    Q1 age band to its midpoint in years — so it is the one continuous field here.

    The standardised mean difference divides by the pooled standard deviation,
    `sqrt((s_included^2 + s_excluded^2) / 2)`, which is the balance convention: it makes the
    difference readable without the scale, not significant. A group with no observed age, or
    with no spread in either arm, yields no standardised difference rather than a division that
    would report one.
    """
    means = {"included": float(included.mean()), "excluded": float(excluded.mean())}
    pooled = float(np.sqrt((float(included.std()) ** 2 + float(excluded.std()) ** 2) / 2))
    gap = means["included"] - means["excluded"]
    standardised = gap / pooled if np.isfinite(pooled) and pooled > 0 else None
    return [
        dict(variable="attr_age", level=WHOLE_GROUP,
             included_value=round(means["included"], 2),
             excluded_value=round(means["excluded"], 2),
             difference=round(gap, 2), measure=MEAN_YEARS),
        dict(variable="attr_age", level=WHOLE_GROUP, included_value=None, excluded_value=None,
             difference=None if standardised is None else round(standardised, 4), measure=SMD),
    ]


def _percentage_rows(variable: str, included: pd.Series, excluded: pd.Series) -> list[dict]:
    """One row per category, as a percentage of its own group, on the public grouping.

    THE GROUPING IS `publication.breakdown`'S. A respondent who did not answer is a `missing`
    category rather than a dropped row, so the categories are exhaustive and each group's
    percentages add to a hundred. Every other category is one the crosswalk declares:
    `_check_levels` has already stopped the comparison if the extract carries anything else.

    The order is the categories' own, with `missing` last, so it does not depend on which
    group happened to be counted first.
    """
    counts = {"included": included.fillna("missing").value_counts(),
              "excluded": excluded.fillna("missing").value_counts()}
    levels = sorted(set(counts["included"].index) | set(counts["excluded"].index),
                    key=lambda level: (str(level) == "missing", str(level)))

    rows = []
    for level in levels:
        percent = {group: _P.public_rate(int(counts[group].get(level, 0)), len(series), "YRBS",
                                         places=_P.DISPLAYED_PLACES)
                   for group, series in (("included", included), ("excluded", excluded))}
        rows.append(dict(
            variable=variable, level=str(level),
            included_value=percent["included"], excluded_value=percent["excluded"],
            difference=round(percent["included"] - percent["excluded"], _P.DISPLAYED_PLACES),
            measure=PERCENT))
    return rows


def _check_levels(variable: str, values: pd.Series, declared) -> None:
    """Stop if `variable` carries a non-missing value the crosswalk does not declare.

    FAIL CLOSED. An unrecognised code is a crosswalk that no longer matches the extract, and
    reporting it as a category of its own would publish a level nobody defined — while mapping
    it into `missing` or into `Other` would file a respondent under a category they are not in.
    Neither is a decision this function may take.

    The message names the variable and stops there: the value itself, how many rows carry it
    and where they are would each say something about respondents.
    """
    if set(values.dropna().unique()) - set(declared):
        raise ValueError(
            f"yrbs_inclusion_comparison: {variable} carries a value outside its declared "
            f"levels. The crosswalk that produced the column no longer covers the extract, so "
            f"the comparison stops rather than reporting or reassigning a category.")


def yrbs_inclusion_comparison(attributes: pd.DataFrame, analytic: pd.Series) -> pd.DataFrame:
    """Included against excluded YRBS respondents, on the demographics the cohort carries.

    `analytic` is THE MASK, not a rule to reapply: pass the Series `analytic_sample_mask`
    returned for this cohort. Nothing here re-derives eligibility, so the two groups are the
    analysis's own and cannot drift from it. They partition the attribute table — every
    respondent is in exactly one — because one boolean mask defines both.

    `difference` is always the included value minus the excluded one, in years for the mean
    age, in percentage points for a category, and unitless for the standardised difference.

    NO HYPOTHESIS IS TESTED. There is no p-value here and none is wanted: the question is how
    large the differences are and which way they point, and a test statistic on a sample this
    size would answer a different one.
    """
    _require(attributes, INCLUSION_ATTRIBUTES, "yrbs_inclusion_comparison")
    mask = pd.Series(analytic)
    if not pd.api.types.is_bool_dtype(mask):
        raise ValueError("yrbs_inclusion_comparison: `analytic` is not a boolean mask. Pass "
                         "the Series `analytic_sample_mask` returned, not a filtered frame or "
                         "an index.")
    if not mask.index.equals(attributes.index):
        raise ValueError("yrbs_inclusion_comparison: the mask and the attribute table describe "
                         "different rows, so the two groups would not partition the cohort. "
                         "Both are built from the same raw frame and share its index.")

    groups = {"included": attributes.loc[mask], "excluded": attributes.loc[~mask]}
    empty = [name for name, group in groups.items() if not len(group)]
    if empty:
        raise ValueError(
            f"yrbs_inclusion_comparison: no respondent is {' or '.join(empty)}, so there is "
            f"nothing to compare. Every respondent falls on one side of the outcome rule, "
            f"which is a finding about the extract rather than a table to build.")

    _check_levels("attr_sex", attributes["attr_sex"], SEX_LABELS)
    _check_levels("attr_ethnicity_coarse", attributes["attr_ethnicity_coarse"],
                  YRBS_ETHNICITY_LEVELS)

    rows = _age_rows(groups["included"]["attr_age"], groups["excluded"]["attr_age"])
    for variable in ("attr_sex", "attr_ethnicity_coarse"):
        labelled = {name: group[variable].map(SEX_LABELS) if variable == "attr_sex"
                    else group[variable] for name, group in groups.items()}
        rows += _percentage_rows(variable, labelled["included"], labelled["excluded"])
    return pd.DataFrame(rows, columns=list(INCLUSION_COLUMNS))
