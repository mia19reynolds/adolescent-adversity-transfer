"""
src/recode_utils.py — shared building blocks for feature recoding.

Design rule (from the spec): where the two cohorts differ in granularity,
harmonise DOWN to the coarser side. Always remap ordinals by their *labels*,
never by raw code order — several MCS scales are reversed.
"""
import numpy as np
import pandas as pd

# The MCS missing-value sentinels. A recoding fact rather than configuration: it says how a
# raw answer is read, so it lives with the readers.
MCS_MISSING = [-9, -8, -1]        # "don't want", "don't know", "not applicable"

# The three states of the combined recent-intercourse and condom-use predictor. Both recoders
# emit these names, which is why they sit in the module the two share;
# `features.RECENT_SEX_CONDOM_CODES` gives them their numeric codes. The first is the
# reference level.
NO_RECENT_INTERCOURSE = "no_recent_intercourse"
RECENT_NO_CONDOM = "recent_no_condom"
RECENT_CONDOM = "recent_condom"


def clean_mcs(s: pd.Series) -> pd.Series:
    """Replace MCS missing codes (-9,-8,-1) with NaN."""
    return s.replace(MCS_MISSING, np.nan)


def to_binary(s: pd.Series, positive_values, missing_to_nan=True) -> pd.Series:
    """Map a set of 'positive' codes to 1, everything else present to 0."""
    s = s.copy()
    out = s.isin(positive_values).astype("float")
    if missing_to_nan:
        out[s.isna()] = np.nan
    return out


def map_ordinal(s: pd.Series, mapping: dict) -> pd.Series:
    """Remap by explicit label→value dict. Unmapped→NaN. Use this, not code order."""
    return s.map(mapping)


def collapse_bins(s: pd.Series, bins, labels) -> pd.Series:
    """Cut a numeric/ordinal series into labelled bins (both cohorts to common set)."""
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype("float")


def yrbs_freq_to_binary(s: pd.Series, none_code=1) -> pd.Series:
    """YRBS frequency where code `none_code` = none/never → 0, anything else → 1."""
    out = (s != none_code).astype("float")
    out[s.isna()] = np.nan
    return out


def sanity_report(name: str, s: pd.Series) -> None:
    """Print coverage + distribution — call after each recode to catch surprises."""
    cov = s.notna().mean()
    if s.dropna().nunique() <= 12:
        dist = s.value_counts(normalize=True, dropna=True).round(3).to_dict()
    else:
        dist = {"min": float(np.nanmin(s)), "median": float(np.nanmedian(s)),
                "max": float(np.nanmax(s))}
    print(f"  [{name}] coverage={cov:.1%}  dist={dist}")
