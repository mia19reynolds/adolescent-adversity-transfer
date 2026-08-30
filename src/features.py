"""
src/features.py — orchestrates recoders into aligned MCS & YRBS frames.

FEATURE_MAP is the single source of truth: harmonised name → (mcs_fn, yrbs_fn, group).
Both cohorts pass through the SAME target, so the columns line up for transfer.
"""
import pandas as pd
import recode_mcs as M
import recode_yrbs as Y
from recode_utils import (NO_RECENT_INTERCOURSE, RECENT_CONDOM, RECENT_NO_CONDOM,
                          sanity_report)

# harmonised_name : (mcs function, yrbs function, group)
FEATURE_MAP = {
    # tobacco
    "h_ever_smoked":            (M.ever_smoked,        Y.ever_smoked,        "substance"),
    "h_current_cigarette":      (M.current_cigarette,  Y.current_cigarette,  "substance"),
    "h_age_first_smoked":       (M.age_first_smoked,   Y.age_first_smoked,   "substance"),
    "h_ecig_ever":              (M.ecig_ever,          Y.ecig_ever,          "substance"),
    "h_ecig_current":           (M.ecig_current,       Y.ecig_current,       "substance"),
    # alcohol
    "h_ever_drank":             (M.ever_drank,         Y.ever_drank,         "substance"),
    "h_age_first_drank":        (M.age_first_drank,    Y.age_first_drank,    "substance"),
    "h_past_month_alcohol":     (M.past_month_alcohol, Y.past_month_alcohol, "substance"),
    "h_ever_binge":             (M.ever_binge,         Y.ever_binge,         "substance"),
    "h_past_year_binge_freq":   (M.past_year_binge_freq, Y.past_year_binge_freq, "substance"),
    # cannabis / drugs
    "h_cannabis_freq":          (M.cannabis_freq,      Y.cannabis_freq,      "substance"),
    "h_other_drugs":            (M.other_drugs,        Y.other_drugs,        "substance"),
    # violence / bullying
    "h_weapon_carrying":        (M.weapon_carrying,    Y.weapon_carrying,    "violence"),
    "h_weapon_victim":          (M.weapon_victim,      Y.weapon_victim,      "violence"),
    "h_physical_fight":         (M.physical_fight,     Y.physical_fight,     "violence"),
    "h_bullied_school":         (M.bullied_school,     Y.bullied_school,     "violence"),
    "h_cyberbullied":           (M.cyberbullied,       Y.cyberbullied,       "violence"),
    # mental health
    "h_depression":             (M.depression_smfq,    Y.depression_smfq,    "mental_health"),
    "h_self_harm":              (M.self_harm,          Y.self_harm,          "mental_health"),
    # sexual
    "h_recent_sex_condom_status": (M.recent_sex_condom_status,
                                   Y.recent_sex_condom_status, "sexual"),
    # diet
    "h_breakfast_days":         (M.breakfast_days,     Y.breakfast_days,     "diet"),
    "h_sugary_drinks":          (M.sugary_drinks,      Y.sugary_drinks,      "diet"),
    "h_fruit_intake":           (M.fruit_intake,       Y.fruit_intake,       "diet"),
    "h_veg_intake":             (M.veg_intake,         Y.veg_intake,         "diet"),
    # activity / sleep
    "h_mvpa_days":              (M.mvpa_days,          Y.mvpa_days,          "activity"),
    "h_sleep_duration":         (M.sleep_duration,     Y.sleep_duration,     "activity"),
    # weight / healthcare
    "h_weight_perception":      (M.weight_perception,  Y.weight_perception,  "health"),
    "h_social_media":           (M.social_media,       Y.social_media,       "health"),
    "h_dentist_12mo":           (M.dentist_12mo,       Y.dentist_12mo,       "health"),
    "h_weight_status_iotf":     (M.weight_status_iotf, Y.weight_status_iotf, "health"),
}

# Fairness attributes: excluded as predictors, kept for subgroup analysis.
#
# ATTRIBUTE_MAP is what `build_attributes` builds. COARSE_ATTRIBUTE_MAP is applied
# by the caller — `attr_ethnicity_coarse` is the ONS four-level collapse the
# Mondrian conformal cells and the subgroup panels are defined over, and notebook
# 01 §3 applies it in its own cell so it can report which YRBS source column was
# used and raise on a code the map does not cover.
#
# Column order in the built frame follows these dicts, core first. That is what
# makes the order a property of the registry rather than of the order somebody
# happened to write the calls in.
ATTRIBUTE_MAP = {
    "attr_sex":         (M.attr_sex,         Y.attr_sex),
    "attr_age":         (M.attr_age,         Y.attr_age),
    "attr_ethnicity":   (M.attr_ethnicity,   Y.attr_ethnicity),
    "attr_orientation": (M.attr_orientation, Y.attr_orientation),
}

COARSE_ATTRIBUTE_MAP = {
    "attr_ethnicity_coarse": (M.attr_ethnicity_coarse, Y.attr_ethnicity_coarse),
}

class RecodeError(RuntimeError):
    """A recoder failed. Names what and where; never carries a value.

    The message is built from the harmonised name, the cohort side and the
    ORIGINAL EXCEPTION'S TYPE — deliberately not its message, because a pandas
    error can quote the data that provoked it and MCS Sweep 6 is safeguarded.
    The original is chained (`raise ... from`) so a traceback still reaches it
    when someone is debugging interactively against data they may see.

    ONE NARROW EXCEPTION: a `KeyError`. Its argument is the name of a column the
    recoder asked for and the cohort does not have, which is schema rather than
    respondent data, and it is the single most useful thing to know when a raw
    extract is the wrong vintage. `_missing_column` extracts it and nothing else;
    every other exception type still contributes only its class name.
    """


def _missing_column(exc):
    """The column name a `KeyError` names, as a phrase, or an empty string.

    Restricted to `KeyError` with a single string argument. A KeyError raised for
    any other reason, or carrying anything other than a plain column name, adds
    nothing to the message rather than risking a value in it.
    """
    if isinstance(exc, KeyError) and len(exc.args) == 1 and isinstance(exc.args[0], str):
        return f" The cohort has no column {exc.args[0]!r}."
    return ""


def build_frame(df, side="mcs", verbose=True):
    """side in {'mcs','yrbs'}. Returns the harmonised feature DataFrame.

    Every feature in FEATURE_MAP must build. A recoder that raises stops the
    build: a frame silently missing a predictor trains a different model from
    the one the manuscript reports, and nothing downstream would notice.
    """
    idx = 0 if side == "mcs" else 1
    out = {}
    for name, fns in FEATURE_MAP.items():
        fn = fns[idx]
        try:
            col = fn(df)
        except Exception as e:                     # noqa: BLE001 — re-raised clean below
            raise RecodeError(
                f"feature {name!r} failed for cohort {side!r}: "
                f"{type(e).__name__} raised by {getattr(fn, '__name__', fn)!r}."
                f"{_missing_column(e)} "
                f"A harmonised frame missing a predictor is not usable, so this "
                f"is fatal rather than skipped."
            ) from e
        out[name] = col
        if verbose:
            sanity_report(name, col if hasattr(col, "notna") else col.iloc[:, 0])

    frame = pd.DataFrame(out, index=df.index)
    missing = [n for n in FEATURE_MAP if n not in frame.columns]
    if missing:
        raise RecodeError(f"harmonised frame for {side!r} is missing {len(missing)} "
                          f"predictor(s): {missing}")
    return frame[list(FEATURE_MAP)]


def build_attributes(df, side="mcs"):
    """Fairness attributes. Excluded as predictors, required for subgroup analysis.

    Builds the four core attributes. `attr_ethnicity_coarse` is not one of them:
    the ONS four-level collapse is applied by the caller from
    `COARSE_ATTRIBUTE_MAP`, because deciding to collapse is an analytic choice
    that wants reporting alongside it.

    THIS IS THE ONE IMPLEMENTATION. `data.build_attributes` delegates
    here rather than calling the recoders itself, so the attribute set is
    declared once, the column order comes from the registry rather than from
    call order, and one error contract covers every caller.

    Fatal on failure for the same reason as `build_frame`: a subgroup table
    built over a silently absent attribute reports on a different population
    than it claims to.

    SIDE EFFECTS: computes only. Reads no file, fits nothing, writes nothing,
    prints nothing.
    """
    if side not in ("mcs", "yrbs"):
        raise ValueError(f"side must be 'mcs' or 'yrbs', got {side!r}")
    idx = 0 if side == "mcs" else 1

    out = {}
    for name, fns in ATTRIBUTE_MAP.items():
        fn = fns[idx]
        try:
            out[name] = fn(df)
        except Exception as e:                     # noqa: BLE001 — re-raised clean below
            raise RecodeError(
                f"attribute {name!r} failed for cohort {side!r}: "
                f"{type(e).__name__} raised by {getattr(fn, '__name__', fn)!r}."
                f"{_missing_column(e)} "
                f"A subgroup analysis missing an attribute is not usable, so this "
                f"is fatal rather than skipped."
            ) from e

    frame = pd.DataFrame(out, index=df.index)
    missing = [n for n in ATTRIBUTE_MAP if n not in frame.columns]
    if missing:
        raise RecodeError(f"attribute frame for {side!r} is missing: {missing}")
    return frame[list(ATTRIBUTE_MAP)]


# The one nominal predictor's level codes, as a constant rather than a literal inside the
# encoder. The three levels are NAMES, not quantities — `recent_no_condom` is not "less" than
# `recent_condom`, and neither is a step away from `no_recent_intercourse` — so anything that
# reads the encoded column has to know which code means what rather than treat 0/1/2 as a
# scale. `data.cohort_associations` is the reader that does. THE FIRST LEVEL IS THE REFERENCE.
RECENT_SEX_CONDOM_CODES = {NO_RECENT_INTERCOURSE: 0, RECENT_NO_CONDOM: 1, RECENT_CONDOM: 2}

NOMINAL_LEVEL_CODES = {"h_recent_sex_condom_status": RECENT_SEX_CONDOM_CODES}


def encode_categoricals(frame):
    """Encode the declared nominal columns from level names to numeric codes.

    The recoders emit level names, which is what makes them readable; the persisted frame and
    every reader downstream take codes. A name the level map does not cover is a recoding
    defect and is refused rather than silently mapped to missing.
    """
    frame = frame.copy()
    for name, codes in NOMINAL_LEVEL_CODES.items():
        if name not in frame.columns:
            continue
        values = frame[name]
        unknown = sorted(set(values.dropna().unique()) - set(codes))
        if unknown:
            raise ValueError(
                f"{name} carries level name(s) {unknown} that are not declared in its level "
                f"map ({', '.join(codes)}). Correct the recoding rather than widening this.")
        frame[name] = values.map(codes).astype("float")
    return frame
