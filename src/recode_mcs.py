"""
src/recode_mcs.py — MCS Sweep 6 → harmonised features, pillars, attributes.

One function per variable, in three groups:

  * the harmonised FEATURES, assembled by `features.FEATURE_MAP`, which is the
    authority for the set. The YRBS side maps its own questions onto these same
    names so the two frames align column-for-column.
  * 5 ACE outcome PILLARS (`ace_*`), composed into the ≥1 and ≥2 outcomes by
    `outcomes.MCS_PILLARS`.
  * 5 fairness ATTRIBUTES (`attr_*`), excluded as predictors and retained for
    subgroup analysis. `attr_ethnicity_coarse` is the ONS-style four-level
    collapse the conformal subgroup cells are defined over.

Each docstring states the spec rule it implements. Reversed scales (FCPHEX00,
FCSWTD00, FCHURT00, FCCYBU00) are flipped HERE by label mapping, never by code
order. Each flip is stated in the docstring of the recoder that performs it.

MCS missing codes (-9, -8, -1) become NaN via `recode_utils.clean_mcs`.
# VERIFIED: FCOBFLG6 carries only {-1, 0, 1, 2} in Sweep 6 (11,852 rows; -1 on
# 773; no -9/-8; no values outside that set), so `weight_status_iotf` mapping
# -1 alone to NaN covers the actual domain.
"""
import numpy as np
import pandas as pd
from recode_utils import (NO_RECENT_INTERCOURSE, RECENT_CONDOM, RECENT_NO_CONDOM,
                          clean_mcs, to_binary, map_ordinal, collapse_bins)

# SMFQ depression: recode raw 1/2/3 -> 0/1/2 (canonical Angold), sum 0-26, cut >=12.
# Precedent: Kelly et al. 2018 (MCS-S6), Kwong 2021 (validated ROC), CORC/NHS guidance.
# A fact about the scale, so it lives with the recoder that applies it.
SMFQ_ITEMS = [f"FCMDS{ch}00" for ch in "ABCDEFGHIJKLM"]  # 13 items A..M
SMFQ_CUT   = 12   # on the 0-26 canonical scale (NOT on raw 13-39)


# SUBSTANCE — TOBACCO
def ever_smoked(df):
    """FCSMOK00 {2,3,4,5,6}=ever(1), {1}=never(0). Collapse 6-level→binary."""
    x = clean_mcs(df["FCSMOK00"])
    return to_binary(x, [2, 3, 4, 5, 6])

def current_cigarette(df):
    """FCSMOK00 {4,5,6}=current smoker(1), {1,2,3}=not(0)."""
    x = clean_mcs(df["FCSMOK00"])
    return to_binary(x, [4, 5, 6])

def age_first_smoked(df, reference_age=14):
    """FCAGSM00 direct age. AGE-STANDARDISE: return (reference_age - age_first).
    Do NOT return raw age (MCS all-14). Early-onset (<13) also derivable downstream."""
    x = clean_mcs(df["FCAGSM00"])
    return reference_age - x

def ecig_ever(df):
    """FCECIG00 {2,3,4}=ever(1), {1}=never(0)."""
    x = clean_mcs(df["FCECIG00"])
    return to_binary(x, [2, 3, 4])

def ecig_current(df):
    """FCECIG00 {3,4}=current(1)."""
    x = clean_mcs(df["FCECIG00"])
    return to_binary(x, [3, 4])


# SUBSTANCE — ALCOHOL
def ever_drank(df):
    """FCALCD00 Yes(1)/No(2) → binary 1/0."""
    x = clean_mcs(df["FCALCD00"])
    return to_binary(x, [1])

def age_first_drank(df, reference_age=14):
    """FCALAG00 direct age → (reference_age - age_first). AGE-STANDARDISE."""
    x = clean_mcs(df["FCALAG00"])
    return reference_age - x

def past_month_alcohol(df):
    """FCALNF00 (4wk, 1=Never..7=40+) → common 4-level none/1-2/3-9/10+.
    MCS: 1→none, 2→1-2, {3,4}→3-9, {5,6,7}→10+."""
    x = clean_mcs(df["FCALNF00"])
    m = {1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3}   # 0=none,1=1-2,2=3-9,3=10+
    return map_ordinal(x, m)

def ever_binge(df):
    """FCALFV00 ever 5+ Yes(1)/No(2) → binary. Caveat: MCS-ever vs YRBS-30d, thresh 5 vs 4F/5M."""
    x = clean_mcs(df["FCALFV00"])
    return to_binary(x, [1])

def past_year_binge_freq(df):
    """FCALFN00 (12mo, 1=Never..5=10+) → common 3-level none/1-2/3+.
    MCS: 1→none, 2→1-2, {3,4,5}→3+."""
    x = clean_mcs(df["FCALFN00"])
    m = {1: 0, 2: 1, 3: 2, 4: 2, 5: 2}
    return map_ordinal(x, m)


# SUBSTANCE — CANNABIS / DRUGS
def cannabis_freq(df):
    """FCCANO00 lifetime freq (1=once/twice..4=>10) + FCCANB00 ever-flag for 'never'.
    Common 3-level: never/1-2/3+. FCCANB00=No→never(0); else FCCANO00 {1}→1-2, {2,3,4}→3+."""
    ever = clean_mcs(df["FCCANB00"])          # 1=Yes,2=No
    freq = clean_mcs(df["FCCANO00"])          # 1..4
    out = pd.Series(np.nan, index=df.index)
    out[ever == 2] = 0                         # never
    out[(ever == 1) & (freq == 1)] = 1         # 1-2
    out[(ever == 1) & (freq.isin([2, 3, 4]))] = 2  # 3+
    return out

def other_drugs(df):
    """FCOTDR00 any other illegal drug Yes(1)/No(2) → binary.
    Caveat: MCS aggregate may under-capture Rx/inhalants vs YRBS Q49-55 union."""
    x = clean_mcs(df["FCOTDR00"])
    return to_binary(x, [1])


# VIOLENCE / BULLYING   (two reversed scales here!)
def weapon_carrying(df):
    """NEW. FCKNIF00 ever carried weapon Yes(1)/No(2) → binary."""
    x = clean_mcs(df["FCKNIF00"])
    return to_binary(x, [1])

def weapon_victim(df):
    """FCVICC00 hit/weapon used against CM Yes(1)/No(2) → binary."""
    x = clean_mcs(df["FCVICC00"])
    return to_binary(x, [1])

def physical_fight(df):
    """Aggregate perpetration: (FCHITT00 OR FCWEPN00)=Yes → 1.
    Caveat: MCS perpetration vs YRBS bidirectional 'in a fight'."""
    hit = clean_mcs(df["FCHITT00"])
    wep = clean_mcs(df["FCWEPN00"])
    both_missing = hit.isna() & wep.isna()
    out = (((hit == 1) | (wep == 1))).astype("float")
    out[both_missing] = np.nan
    return out

def bullied_school(df):
    """REVERSED. FCHURT00 1=Most days..6=Never. Map 6(Never)→0, {1,2,3,4,5}→1.
    DO NOT process this by raw code order: 6 is the low-risk end, not the high one."""
    x = clean_mcs(df["FCHURT00"])
    out = x.isin([1, 2, 3, 4, 5]).astype("float")
    out[x.isna()] = np.nan
    out[x == 6] = 0
    return out

def cyberbullied(df):
    """REVERSED. FCCYBU00 1=Most days..6=Never. Map 6(Never)→0, {1..5}→1.
    This is a TOP feature, so an inversion here corrupts transfer without saying so."""
    x = clean_mcs(df["FCCYBU00"])
    out = x.isin([1, 2, 3, 4, 5]).astype("float")
    out[x.isna()] = np.nan
    out[x == 6] = 0
    return out


# Mental health
def depression_smfq(df, cut=SMFQ_CUT):
    """Sum 13 SMFQ items using STANDARD scoring: recode raw 1/2/3 -> 0/1/2, sum (0-26),
    cut at >=12 (Kelly et al. 2018, MCS-S6 precedent; Kwong 2021 validated ROC).
    Items are stored 1-3 in MCS; the -1 shift makes them the canonical 0-2 Angold coding."""
    items = df[SMFQ_ITEMS].apply(clean_mcs)
    items_0_2 = items - 1                 # 1/2/3 -> 0/1/2 (canonical sMFQ)
    score = items_0_2.sum(axis=1, min_count=len(SMFQ_ITEMS))   # 0-26
    out = (score >= cut).astype("float")
    out[score.isna()] = np.nan
    return out

def self_harm(df):
    """RELABELLED self_harm (not suicide attempt). FCHARM00 Yes(1)/No(2) → binary.
    Construct caveat vs YRBS Q29 (attempt, implies intent). MCS = self-harm incl NSSI."""
    x = clean_mcs(df["FCHARM00"])
    return to_binary(x, [1])


# SEXUAL
def recent_sex_condom_status(df):
    """Combine FCSEXX00 and FCCONP0A into the shared three-state scale.

    FCSEXX00: 1 intercourse in the last 12 months, 2 none.
    FCCONP0A: 1 used a condom, 2 used another form of contraceptive, 3 used none.

    Read raw, before `clean_mcs`. FCSEXX00 = -1 marks structural routing and
    administration-mode skipping rather than an answer, so it is not read as no recent
    intercourse. Only explicit, internally consistent states are coded; -1, -8, -9 and system
    missingness stay missing.
    """
    sex = pd.to_numeric(df["FCSEXX00"], errors="coerce")
    condom = pd.to_numeric(df["FCCONP0A"], errors="coerce")
    out = pd.Series(np.nan, index=df.index, dtype="object")
    out[sex == 2] = NO_RECENT_INTERCOURSE
    out[(sex == 1) & condom.isin([2, 3])] = RECENT_NO_CONDOM
    out[(sex == 1) & (condom == 1)] = RECENT_CONDOM
    return out

# DIET
def breakfast_days(df):
    """FCBRKN00 1=Never/2=Some/3=Every → 3-level ordinal (aligned direction)."""
    x = clean_mcs(df["FCBRKN00"])
    return x  # already 3-level ascending; YRBS 0-7 days binned to match

def sugary_drinks(df):
    """REVERSED. FCSWTD00 1=>once/day..7=Never. FLIP + collapse to common 4-level:
    none-rare / weekly / daily / multi-daily.
    MCS: {7,6}→0 none-rare, {5,4,3}→1 weekly-ish, {2}→2 daily, {1}→3 multi-daily."""
    x = clean_mcs(df["FCSWTD00"])
    m = {7: 0, 6: 0, 5: 1, 4: 1, 3: 1, 2: 2, 1: 3}
    return map_ordinal(x, m)

def fruit_intake(df):
    """FCFRUT00 1=Never/2=Some/3=Every (threshold-days). 3-level ordinal.
    Construct caveat: MCS days≥2portions vs YRBS times-eaten. Provisional."""
    return clean_mcs(df["FCFRUT00"])

def veg_intake(df):
    """FCVEGI00 1=Never/2=Some/3=Every (threshold-days). 3-level ordinal.
    Provisional (construct gap; test importance)."""
    return clean_mcs(df["FCVEGI00"])


# ACTIVITY / SLEEP   (one reversed, one derived)
def mvpa_days(df):
    """REVERSED. FCPHEX00 1=Every day..5=Not at all. FLIP so higher=more active.
    Map to ascending 5-level: 5(Not at all)→0 .. 1(Every day)→4."""
    x = clean_mcs(df["FCPHEX00"])
    m = {5: 0, 4: 1, 3: 2, 2: 3, 1: 4}
    return map_ordinal(x, m)

def sleep_duration(df):
    """DERIVED. duration = wake_midpoint - bed_midpoint (+24 if cross-midnight).
    Bin to YRBS Q85 hours: <=4→1,5→2,6→3,7→4,8→5,9→6,10+→7.
    Document ±1hr band error + MCS-derived vs YRBS-self-reported mode diff."""
    bed = clean_mcs(df["FCSLWK00"])
    wake = clean_mcs(df["FCWUWK00"])
    bed_mid  = {1: 20.5, 2: 21.5, 3: 22.5, 4: 23.5, 5: 24.5}
    wake_mid = {1: 5.5, 2: 6.5, 3: 7.5, 4: 8.5, 5: 9.5}
    dur = wake.map(wake_mid) - bed.map(bed_mid)
    dur = dur.where(dur > 0, dur + 24)
    bins   = [-np.inf, 4, 5, 6, 7, 8, 9, np.inf]
    labels = [1, 2, 3, 4, 5, 6, 7]
    return collapse_bins(dur, bins, labels)


# Weight and healthcare

# FCWEGT00 — CM's perception of their weight (MCS6, Young Person
# questionnaire, CAPI code WEGT). UKDS dictionary value labels, verbatim:
#   (-9.0) Don't want to answer  (-8.0) Don't know  (-1.0) Not applicable
#   (1.0) Underweight  (2.0) About the right weight
#   (3.0) Slightly overweight  (4.0) Very overweight
# YRBS 2023 Q66 ("How do you describe your weight?"):
#   1 Very underweight  2 Slightly underweight  3 About the right weight
#   4 Slightly overweight  5 Very overweight
# Harmonisation: YRBS levels 1 and 2 merge into MCS level 1 ("Underweight").
# This is the unique level-preserving alignment: MCS pools the underweight
# end but retains the slightly/very overweight split, so merging on the
# overweight side would discard a distinction MCS records. Resulting 4-level
# scale (both datasets): underweight < about right < slightly overweight
# < very overweight. MCS codes -9/-8/-1 -> missing.
def weight_perception(df):
    """FCWEGT00 1=Under..4=Very over (aligned ascending). Keep ordinal."""
    return clean_mcs(df["FCWEGT00"])

def social_media(df):
    """FCSOME00 1=None..8=7+hrs (ascending). Keep ordinal; align bands to YRBS Q80."""
    return clean_mcs(df["FCSOME00"])

def dentist_12mo(df):
    """FCDENY00 Yes(1)/No(2) dentist in last 12mo → binary."""
    x = clean_mcs(df["FCDENY00"])
    return to_binary(x, [1])


# FAIRNESS ATTRIBUTES (excluded as predictors; retained for subgroup analysis)
def attr_sex(df):
    """FCCSEX00 1=M/2=F → 0=male,1=female (align to common coding)."""
    x = clean_mcs(df["FCCSEX00"])
    return (x == 2).astype("float").where(x.notna())

def attr_age(df):
    """FCMCS6AG continuous ~14 (near-constant). Stratifier/covariate, NOT student feature."""
    return clean_mcs(df["FCMCS6AG"])

def attr_ethnicity(df):
    """FDCE0600 6-cat. Map to common coarse set downstream. FAIRNESS ONLY."""
    return clean_mcs(df["FDCE0600"])


# FDCE0600 — "S6 DV CM ethnic group classification - 6 categories". CODING VERIFIED (2026-08-04)
# against the local UKDS SN 8156 MCS Safeguarded Data Dictionary, sheet MCS_Safeguarded_DataDict
# (dataset mcs6_cm_derived.sav; Sweep 6, 2015/6, age 14; Topic Demographics/Ethnic group; Derived).
# Value labels (verbatim from the dictionary):
#   1 White | 2 Mixed | 3 Indian | 4 Pakistani and Bangladeshi |
#   5 Black or Black British | 6 Other Ethnic group (inc Chinese, Other)
#  -1 Not applicable - Not codeable | -8 Don't know | -9 Refusal   (all -> NaN via clean_mcs)
# ("FDC06E00", which appears in the CLS online docs, is not in the dictionary at all. Ignore it.)
#
# 6->4 ONS collapse (DOCUMENTED, not assumed): White=1; Asian=3+4; Black=5; Mixed-or-Other=2+6.
# CAVEAT for the paper: "Asian" here EXCLUDES Chinese — the MCS derived variable places Chinese in
# category 6 (Other), so it falls under Mixed-or-Other, a deviation from some ONS groupings.
_FDCE6_TO_COARSE = {1: "White", 3: "Asian", 4: "Asian", 5: "Black",
                    2: "Mixed-or-Other", 6: "Mixed-or-Other"}


def attr_ethnicity_coarse(df):
    """ONS-style coarse ethnicity (White/Asian/Black/Mixed-or-Other). FAIRNESS ONLY.

    Collapses FDCE0600 (6-cat; coding VERIFIED against the UKDS SN8156 dictionary — see block above)
    by ONS high-level grouping. Missing codes (-1,-8,-9) become NaN via clean_mcs, and a negative
    value surviving that would map into a real ethnicity category, so it raises rather than
    recoding. Returns all-NaN if FDCE0600 is absent.
    """
    if "FDCE0600" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="object")
    x = clean_mcs(df["FDCE0600"])
    if (x < 0).any():
        raise ValueError(
            "FDCE0600 negative missing code survived clean_mcs — would mis-map")
    return x.map(_FDCE6_TO_COARSE).astype("object")


def attr_orientation(df):
    """Same-sex attraction -> 1, else 0. Needs sex to interpret. FAIRNESS ONLY."""
    sex = clean_mcs(df["FCCSEX00"])
    attr_f = clean_mcs(df["FCROMG00"]) == 1
    attr_m = clean_mcs(df["FCROMB00"]) == 1
    same_sex = ((sex == 1) & attr_m) | ((sex == 2) & attr_f)
    out = same_sex.astype("float")
    out[sex.isna()] = np.nan
    return out

# ACE outcome pillars
def ace_sexual_abuse(df):
    """FCVICF0A Yes(1)/No(2) → 1. Caveat: MCS any-perpetrator vs YRBS adult 5+yrs."""
    return to_binary(clean_mcs(df["FCVICF0A"]), [1])

def ace_emotional_abuse(df):
    """FCVICG00 Yes(1)/No(2) → 1. Caveat: perpetrator-unspecified vs YRBS caregiver."""
    return to_binary(clean_mcs(df["FCVICG00"]), [1])

def ace_physical_abuse(df):
    """FCVICA00 Yes(1)/No(2) → 1. Caveat: perpetrator-unspecified vs YRBS caregiver."""
    return to_binary(clean_mcs(df["FCVICA00"]), [1])

def ace_household_substance(df):
    """Household substance ACE (parent-report).
    FDAUDIT (AUDIT composite) is absent from mcs6_core, so use:
      recreational drug use (FPDRUG00 occasionally/regularly)  OR
      frequent drinking (FPALDR00 == 1, i.e. 4+/week — scale is REVERSED, 1=most).
    Frequency alone is a weak proxy for 'problem' use — documented limitation."""
    from recode_utils import clean_mcs
    import numpy as np
    drug = clean_mcs(df["FPDRUG00"])          # 1=occ, 2=reg, 3=never
    drug_flag = drug.isin([1, 2])
    alc = clean_mcs(df["FPALDR00"])           # 1=4+/wk (MOST) ... 5=never (REVERSED)
    alc_flag = (alc == 1)                     # heaviest-frequency band only
    out = (drug_flag | alc_flag).astype("float")
    all_missing = drug.isna() & alc.isna()
    out[all_missing] = np.nan
    return out

def ace_household_mental_illness(df):
    """FPDEAN00 diagnosed depression/anxiety Yes(1)/No(2) → 1.
    Optionally combine with high FDKESSL. Parent-report caveat."""
    return to_binary(clean_mcs(df["FPDEAN00"]), [1])


def weight_status_iotf(df):
    """
    MCS: FCOBFLG6 is a CLS-derived IOTF (Cole 2000/2012) weight-for-age category.
    Native coding: -1 = missing, 0 = not overweight, 1 = overweight, 2 = obese.
    All MCS respondents are ~14 years old at Sweep 6, so age-conditioning is
    already baked into FCOBFLG6.

    Returns 0/1/2 categorical (NaN for missing).
    """
    import numpy as np
    v = df["FCOBFLG6"].astype("float").copy()
    v[v == -1] = np.nan
    return v
