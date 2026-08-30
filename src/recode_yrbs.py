"""
src/recode_yrbs.py — YRBS 2023 → harmonised features and attributes.

YRBS columns are Q1..Q107 (raw). Each function maps a YRBS question onto the
SAME target scale `recode_mcs` produces, so the two cohort frames align
column-for-column and a model fitted on one can score the other.

  * the harmonised FEATURES, assembled by `features.FEATURE_MAP`, which is the
    authority for the set.
  * 5 fairness ATTRIBUTES (`attr_*`), excluded as predictors and retained for
    subgroup analysis. `attr_ethnicity_coarse` is the four-level collapse the
    conformal subgroup cells are defined over. It is NOT the same four levels as
    the MCS side: YRBS collapses to White/Black/Hispanic/Other, MCS to the ONS
    White/Asian/Black/Mixed-or-Other. The cells are per-cohort for that reason.

The ACE outcome pillars are not functions on this side: YRBS pillars are defined
as question-code pairs in `outcomes.YRBS_PILLARS`.

Includes IOTF weight-status computed from raw height and weight, matching the
CLS-derived MCS reference rather than re-deriving a separate standard.
"""
import numpy as np
import pandas as pd
from recode_utils import (NO_RECENT_INTERCOURSE, RECENT_CONDOM, RECENT_NO_CONDOM,
                          to_binary, map_ordinal, collapse_bins, yrbs_freq_to_binary)


# ---- TOBACCO ----
def ever_smoked(df):
    """Q31 ever smoked Yes(1)/No(2) → binary."""
    return to_binary(df["Q31"], [1])

def current_cigarette(df):
    """Q33 days smoked past 30 (1=0 days .. up) → current smoker if >1 day."""
    return yrbs_freq_to_binary(df["Q33"], none_code=1)

def age_first_smoked(df, ref="Q1"):
    """Q32 age-band first cigarette. AGE-STANDARDISE: (age - age_first) using Q1.
    Q32 codes → approx age; Q1 codes → age years. Return years-before-current."""
    q32_age = {1: np.nan, 2: 8, 3: 10, 4: 12, 5: 14, 6: 16, 7: 18}   # band midpoints
    q1_age  = {1: 12, 2: 13, 3: 14, 4: 15, 5: 16, 6: 17, 7: 18}
    af = df["Q32"].map(q32_age)
    age = df[ref].map(q1_age)
    return age - af

def ecig_ever(df):
    """Q35 ever vape Yes(1)/No(2) → binary."""
    return to_binary(df["Q35"], [1])

def ecig_current(df):
    """Q36 days vaped past 30 → current if >1 day."""
    return yrbs_freq_to_binary(df["Q36"], none_code=1)


# ---- ALCOHOL ----
def ever_drank(df):
    """Q41 1=never, 2-7=age bands → ever if >=2."""
    out = (df["Q41"] >= 2).astype("float")
    out[df["Q41"].isna()] = np.nan
    return out

def age_first_drank(df, ref="Q1"):
    """Q41 age bands (2-7) → midpoint age; (Q1 age - age_first)."""
    q41_age = {1: np.nan, 2: 8, 3: 10, 4: 12, 5: 14, 6: 16, 7: 18}
    q1_age  = {1: 12, 2: 13, 3: 14, 4: 15, 5: 16, 6: 17, 7: 18}
    af = df["Q41"].map(q41_age)
    age = df[ref].map(q1_age)
    return age - af

def past_month_alcohol(df):
    """Q42 days drank past 30 (bands) → common 4-level none/1-2/3-9/10+."""
    # Q42: 1=0 days,2=1-2,3=3-5,4=6-9,5=10-19,6=20-29,7=all 30
    m = {1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3}
    return map_ordinal(df["Q42"], m)

def ever_binge(df):
    """Flag any binge-drinking day in Q43. Reference period differs: Q43 covers 30 days,
    MCS FCALFV00 asks ever. Q43 also uses a sex-specific drink threshold (4 female,
    5 male) against MCS's flat 5."""
    return yrbs_freq_to_binary(df["Q43"], none_code=1)

def past_year_binge_freq(df):
    """Map Q43 binge-drinking days to the shared none, one-to-two and three-or-more
    scale. Reference period differs: Q43 covers 30 days, MCS FCALFN00 12 months."""
    # Q43: 1=0 days; 2=1 day; 3=2 days; 4-7=3 or more days.
    mapping = {1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2}
    return map_ordinal(df["Q43"], mapping)


# ---- CANNABIS / DRUGS ----
def cannabis_freq(df):
    """Map Q46 lifetime cannabis use to the shared never, one-to-two and three-or-more
    scale. Both cohorts are lifetime; the underlying bands differ above 3 uses, so the
    three-level collapse is the common resolution."""
    # Q46: 1=0 times; 2=1-2; 3=3-9; 4=10-19; 5=20-39; 6=40-99; 7=100 or more.
    mapping = {1: 0, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2}
    return map_ordinal(df["Q46"], mapping)

def other_drugs(df, qs=("Q49","Q50","Q51","Q52","Q53","Q54","Q55")):
    """Aggregate Q49-Q55 (each drug) → any-drug binary (union of ever-use)."""
    present = [q for q in qs if q in df.columns]
    any_use = pd.Series(False, index=df.index)
    all_missing = pd.Series(True, index=df.index)
    for q in present:
        used = df[q] > 1          # 1 = 0 times / never
        any_use = any_use | used.fillna(False)
        all_missing = all_missing & df[q].isna()
    out = any_use.astype("float")
    out[all_missing] = np.nan
    return out


# ---- VIOLENCE / BULLYING ----
def weapon_carrying(df):
    """NEW. (Q12 days>0 OR Q13 days>0) → ever-carried binary."""
    q12 = df["Q12"] > 1 if "Q12" in df.columns else pd.Series(False, index=df.index)
    q13 = df["Q13"] > 1 if "Q13" in df.columns else pd.Series(False, index=df.index)
    out = (q12.fillna(False) | q13.fillna(False)).astype("float")
    both_missing = (df.get("Q12", pd.Series(np.nan, index=df.index)).isna() &
                    df.get("Q13", pd.Series(np.nan, index=df.index)).isna())
    out[both_missing] = np.nan
    return out

def weapon_victim(df):
    """Q15 threatened/injured with weapon, times → binary (>0)."""
    return yrbs_freq_to_binary(df["Q15"], none_code=1)

def physical_fight(df):
    """Q16 in a physical fight, times → binary (>0). Bidirectional vs MCS perpetration."""
    return yrbs_freq_to_binary(df["Q16"], none_code=1)

def bullied_school(df):
    """Q24 bullied on school property Yes(1)/No(2) → binary."""
    return to_binary(df["Q24"], [1])

def cyberbullied(df):
    """Q25 electronically bullied Yes(1)/No(2) → binary."""
    return to_binary(df["Q25"], [1])


# ---- MENTAL HEALTH ----
def depression_smfq(df):
    """Q26 sad/hopeless 2+ weeks Yes(1)/No(2) → binary (matches MCS binary target)."""
    return to_binary(df["Q26"], [1])

def self_harm(df):
    """Q29 attempted suicide, times → binary (>0). CONSTRUCT caveat: attempt vs MCS self-harm."""
    return yrbs_freq_to_binary(df["Q29"], none_code=1)


# ---- SEXUAL ----
def recent_sex_condom_status(df):
    """Combine Q59 and Q61 into the shared three-state scale.

    Q59: 1 never had intercourse, 2 had intercourse but not in the past 3 months,
    3-8 one to six-or-more partners in the past 3 months.
    Q61: 1 never had intercourse, 2 condom used at last intercourse, 3 not used.

    The active-and-answered combination is CDC's own QN61 denominator. Only explicit,
    internally consistent states are coded: a respondent reporting partners in the past
    3 months alongside Q61 = 1 is contradictory, and a value CDC's edit rules blanked is
    absent, so both stay missing.
    """
    missing = [question for question in ("Q59", "Q61") if question not in df.columns]
    if missing:
        raise ValueError(
            f"recent_sex_condom_status requires ['Q59', 'Q61']; missing columns: {missing}")
    partners = pd.to_numeric(df["Q59"], errors="coerce")
    condom = pd.to_numeric(df["Q61"], errors="coerce")
    active = partners.isin([3, 4, 5, 6, 7, 8])
    out = pd.Series(np.nan, index=df.index, dtype="object")
    out[partners.isin([1, 2])] = NO_RECENT_INTERCOURSE
    out[active & (condom == 3)] = RECENT_NO_CONDOM
    out[active & (condom == 2)] = RECENT_CONDOM
    return out

# ---- DIET ----
def breakfast_days(df):
    """Q75 breakfast days 0-7 (8-level) → 3-level none/some/every to match MCS.
    0 days→1(never), 1-6→2(some), 7→3(every)."""
    m = {1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 3}   # Q75 1=0d..8=7d
    return map_ordinal(df["Q75"], m)

def sugary_drinks(df):
    """Map Q74 past-week soda frequency to the shared rare, weekly, daily and
    multiple-times-daily scale used for MCS alignment."""
    # Q74: 1 none; 2=1-3/week; 3=4-6/week; 4=once/day; 5-7=at least twice/day.
    mapping = {1: 0, 2: 1, 3: 1, 4: 2, 5: 3, 6: 3, 7: 3}
    return map_ordinal(df["Q74"], mapping)

def fruit_intake(df):
    """Q69 fruit times past 7 → collapse to 3-level low/med/high to match MCS threshold-scale.
    {1}→1(never), {2,3,4}→2(some), {5,6,7}→3(high). Construct caveat documented."""
    m = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3}
    return map_ordinal(df["Q69"], m)

def veg_intake(df, qs=("Q70", "Q71", "Q72", "Q73")):
    """Sum the four vegetable items as times per week and collapse the result
    to the shared three-level scale.

    All four items are required because the cut points apply to the complete
    four-item total.
    """
    midpoints = {1: 0, 2: 2, 3: 5, 4: 7, 5: 14, 6: 21, 7: 28}
    missing = [question for question in qs if question not in df.columns]
    if missing:
        raise ValueError(
            f"veg_intake requires {list(qs)}; missing columns: {missing}"
        )
    total = sum(df[question].map(midpoints) for question in qs)
    return collapse_bins(
        total,
        bins=[-np.inf, 3, 10, np.inf],
        labels=[1, 2, 3],
    )


# ---- ACTIVITY / SLEEP ----
def mvpa_days(df):
    """Q76 days active 60min (1=0d..8=7d) → ascending 5-level to match MCS flipped scale.
    {1}→0,{2,3}→1,{4,5}→2,{6,7}→3,{8}→4."""
    m = {1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4}
    return map_ordinal(df["Q76"], m)

def sleep_duration(df):
    """Q85 hours sleep (1=<=4hr..7=10+hr) → keep as 7-level (MCS derives to same)."""
    return df["Q85"].astype("float")


# ---- WEIGHT / HEALTHCARE ----
def weight_perception(df):
    """Q66 describe weight (1=very under..5=very over) → align to MCS 4-level.
    The five levels collapse onto MCS FCWEGT00's four by merging the two
    underweight codes: {1,2}→1 under, {3}→2 about right, {4}→3 slightly over,
    {5}→4 very over. That merge is forced rather than chosen — MCS pools the
    underweight end and keeps the overweight split, so no other alignment
    preserves the levels. Both label sets are quoted in recode_mcs.py, above
    ::weight_perception."""
    m = {1: 1, 2: 1, 3: 2, 4: 3, 5: 4}
    return map_ordinal(df["Q66"], m)

def social_media(df):
    """Q80 social media use frequency → align ascending to MCS FCSOME00 bands."""
    return df["Q80"].astype("float")

def dentist_12mo(df):
    """Q83 A(<12mo)→1, B/C/D→0, E(not sure)→NaN."""
    m = {1: 1, 2: 0, 3: 0, 4: 0, 5: np.nan}
    return map_ordinal(df["Q83"], m)

def weight_status_iotf(df, height_col="Q6", weight_col="Q7", age_col="Q1", sex_col="Q2"):
    """
    YRBS: compute IOTF (Cole 2000/2012) weight-for-age category from raw
    height (m), weight (kg), age (Q1 code), sex (Q2 code).

    Cutoff table interpolated to integer ages 12-18 from Cole 2000/2012
    adolescent grid. Uses sex-specific overweight and obese BMI thresholds.

    RAW HEIGHT AND WEIGHT, NOT CDC `bmipct`. The parquet carries a CDC-derived
    percentile, and it is deliberately not the primary source: MCS supplies
    `FCOBFLG6`, a CLS-derived IOTF category on the Cole reference, so computing
    IOTF here from raw measurements is what puts both cohorts on the SAME
    reference. Taking `bmipct` instead would harmonise the variable name while
    silently comparing two different growth standards.

    Returns 0/1/2 categorical:
      0 = not overweight, 1 = overweight, 2 = obese.
    NaN if any of height/weight/age/sex missing.

    Coding: Q1 age = code + 11; Q2 sex 1=female, 2=male (confirmed empirically
    via median height by age × sex).
    """
    import numpy as np
    import pandas as pd

    # Cole 2000/2012 IOTF cutoffs, interpolated to integer ages 12-18.
    # Columns: age (int, in years), overweight_threshold, obese_threshold.
    IOTF_BOYS = {
        12: (21.22, 26.02),
        13: (21.91, 26.84),
        14: (22.62, 27.63),
        15: (23.29, 28.30),
        16: (23.90, 28.88),
        17: (24.46, 29.41),
        18: (25.00, 30.00),
    }
    IOTF_GIRLS = {
        12: (21.68, 26.67),
        13: (22.58, 27.76),
        14: (23.34, 28.57),
        15: (23.94, 29.11),
        16: (24.37, 29.43),
        17: (24.70, 29.69),
        18: (25.00, 30.00),
    }

    height_m = pd.to_numeric(df[height_col], errors="coerce")
    weight_kg = pd.to_numeric(df[weight_col], errors="coerce")
    q1 = pd.to_numeric(df[age_col], errors="coerce")
    q2 = pd.to_numeric(df[sex_col], errors="coerce")

    # Age = Q1 code + 11, clipped to [12, 18]
    age = (q1 + 11).clip(lower=12, upper=18)

    bmi = weight_kg / (height_m ** 2)

    out = pd.Series(np.nan, index=df.index, dtype="float")
    for age_int in range(12, 19):
        for sex_code, table in [(2, IOTF_BOYS), (1, IOTF_GIRLS)]:
            ow_thresh, ob_thresh = table[age_int]
            mask = (age == age_int) & (q2 == sex_code) & bmi.notna()
            out.loc[mask & (bmi < ow_thresh)] = 0
            out.loc[mask & (bmi >= ow_thresh) & (bmi < ob_thresh)] = 1
            out.loc[mask & (bmi >= ob_thresh)] = 2

    # Anyone with missing height, weight, age, or sex is NaN (default)
    return out

# ---- FAIRNESS ATTRIBUTES ----
def attr_sex(df):
    """Q2 1=Female/2=Male → 0=male,1=female (align to MCS coding)."""
    return (df["Q2"] == 1).astype("float").where(df["Q2"].notna())

def attr_age(df):
    """Q1 age band → midpoint years. Stratifier/covariate."""
    m = {1: 12, 2: 13, 3: 14, 4: 15, 5: 16, 6: 17, 7: 18}
    return df["Q1"].map(m)

def attr_ethnicity(df):
    """Prefer CDC-derived raceeth (single col). Else Q5. FAIRNESS ONLY."""
    if "raceeth" in df.columns:
        return df["raceeth"].astype("float")
    if "Q5" in df.columns:
        return df["Q5"].astype("float")
    return pd.Series(np.nan, index=df.index)


# `raceeth` (8-cat) → coarse 4-way. THIS PARQUET DOES NOT USE THE STANDARD CDC
# ORDERING. The coding below was decoded from the data itself, by crosstabbing
# raceeth against the Q4 Hispanic item, and it is what the map assumes:
#   codes 6,7 = 100% Hispanic (Q4=Yes) → Hispanic
#   non-Hispanic single races in ALPHABETICAL order:
#     1 AI/AN, 2 Asian, 3 Black, 4 Native Hawaiian/OPI, 5 White
#   8 = Multiple non-Hispanic
# (Q5 race item is entirely NaN in this parquet — raceeth is the only source.)
_RACEETH8_TO_COARSE = {5: "White", 3: "Black", 6: "Hispanic", 7: "Hispanic",
                       1: "Other", 2: "Other", 4: "Other", 8: "Other"}
# CDC `race4` is already the target 4-way: 1 White, 2 Black, 3 Hispanic, 4 Other.
_RACE4_TO_COARSE = {1: "White", 2: "Black", 3: "Hispanic", 4: "Other"}


def attr_ethnicity_coarse(df):
    """4-category coarse ethnicity (White/Black/Hispanic/Other). FAIRNESS ONLY.

    Source preference: CDC `race4` (already 4-cat) > `raceeth` (8-cat, collapsed).
    Returns an object Series of category strings; NaN where the source is missing
    or unmapped.

    `race4` follows the standard CDC coding. `raceeth` in this parquet does NOT —
    see the decoded map above — so if the input file is ever regenerated, check
    the raw value counts the prereq diagnostic prints before trusting it. Raw Q5
    is deliberately unused: its coding is unverified for this collapse, and if
    neither race4 nor raceeth is present the diagnostic flags it.
    """
    if "race4" in df.columns:
        s = pd.to_numeric(df["race4"], errors="coerce")
        return s.map(_RACE4_TO_COARSE).astype("object")
    if "raceeth" in df.columns:
        s = pd.to_numeric(df["raceeth"], errors="coerce")
        return s.map(_RACEETH8_TO_COARSE).astype("object")
    return pd.Series(np.nan, index=df.index, dtype="object")


def attr_orientation(df):
    """Q64 identity → any-same-sex binary (gay/lesbian/bi → 1, hetero → 0). FAIRNESS ONLY."""
    # Q64 (2023 Users Guide codebook, verbatim):
    #   1 Heterosexual (straight)
    #   2 Gay or lesbian
    #   3 Bisexual
    #   4 I describe my sexual identity some other way
    #   5 I am not sure about my sexual identity (questioning)
    #   6 I do not know what this question is asking
    # Codes 4, 5 and 6 currently fall to the non-minority side, which is a real grouping
    # decision and not an obvious one. It is recorded rather than resolved because nothing
    # consumes this attribute: the subgroup cells are sex x ethnicity, and no analysis reads
    # attr_orientation. Revisit the grouping before any analysis starts to.
    out = df["Q64"].isin([2, 3]).astype("float")
    out[df["Q64"].isna()] = np.nan
    return out
