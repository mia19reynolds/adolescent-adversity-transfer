"""
src/outcomes.py — constructs y_harmonised from the 5 shared ACE pillars.

Pillars (binary each): sexual, emotional, physical abuse; household substance;
household mental illness. y = 1 once `threshold` pillars are positive — one by
default, and the paper also reports two. Neglect (Q99) and separation (Q102) are
YRBS-only, so they stay out of the SHARED outcome.
"""
import numpy as np
import pandas as pd
import recode_mcs as M


MCS_PILLARS = {
    "ace_sexual_abuse":            M.ace_sexual_abuse,
    "ace_emotional_abuse":         M.ace_emotional_abuse,
    "ace_physical_abuse":          M.ace_physical_abuse,
    "ace_household_substance":     M.ace_household_substance,
    "ace_household_mental_illness": M.ace_household_mental_illness,
}

# YRBS pillar source questions → binary (built inline; simple enough not to need a module)
YRBS_PILLARS = {
    "ace_sexual_abuse":            ("Q88", [1]),                 # Yes=1
    "ace_emotional_abuse":         ("Q89", [2, 3, 4, 5]),        # >=Rarely
    "ace_physical_abuse":          ("Q90", [2, 3, 4, 5]),
    "ace_household_substance":     ("Q100", [1]),
    "ace_household_mental_illness": ("Q101", [1]),
    # YRBS-only pillars (no MCS counterpart):
    "ace_neglect":                 ("Q99", [1, 2]),              # reverse-coded: Never/Rarely = neglect
    "ace_incarceration":           ("Q102", [1]),                # Yes=1
    "ace_ipv":                     ("Q91", [2, 3, 4, 5]),        # any experience
}


# THE FIVE SHARED PILLARS, DERIVED RATHER THAN DECLARED. "Shared" means a pillar both cohorts
# build, so it is the intersection of the two maps above and not a third hand-typed list.
# It was typed out three times — here, in CATEGORY_SETS["flat5"], and in src/data.py — with
# nothing checking that the three agreed.
#
# Order follows MCS_PILLARS. It does not reach the outcome: `compose_outcome` sums across
# columns and asks whether any is missing, and both are order-independent.
SHARED_PILLARS = tuple(k for k in MCS_PILLARS if k in YRBS_PILLARS)


# CATEGORY-SET GROUPINGS (Felitti/Hughes taxonomies mapped to the 5 pillars).
# On the available pillars the two frameworks assign the same categories; the
# meaningful comparison is the grouping scheme. neglect is schema-defined but has
# NO MCS pillar (no neglect construct in MCS training) — kept to reproduce the
# below-chance transfer result honestly.
S, E, P, SUB, MI = SHARED_PILLARS      # short names for the groupings below

CATEGORY_SETS = {
    # composite
    "flat5":              list(SHARED_PILLARS),
    # Felitti/Kaiser domains
    "felitti_abuse":      [S, E, P],
    "felitti_neglect":    [],                 # no MCS pillar -> expect degenerate/below-chance
    "felitti_household":  [SUB, MI],
    # Hughes clusters (relational = direct interpersonal harm; household = context)
    "hughes_relational":  [S, E, P],
    "hughes_household":   [SUB, MI],
    # per-pillar (diagnostic)
    "pillar_sexual":      [S],
    "pillar_emotional":   [E],
    "pillar_physical":    [P],
    "pillar_substance":   [SUB],
    "pillar_mental":      [MI],
}


def build_mcs_pillars(df):
    return pd.DataFrame({name: fn(df) for name, fn in MCS_PILLARS.items()}, index=df.index)


def build_yrbs_pillars(df):
    out = {}
    for name, (q, pos) in YRBS_PILLARS.items():
        s = df[q]
        col = s.isin(pos).astype("float")
        col[s.isna()] = np.nan
        out[name] = col
    return pd.DataFrame(out, index=df.index)


def compose_outcome(pillars: pd.DataFrame, threshold=1, strict=True):
    """Standardised outcome constructor for use throughout the pipeline.

    strict=True (default): outcome is NaN if ANY pillar is missing. Restricts
      to the analytic sample (respondents who answered all outcome questions).
      Prevents the missingness-proxy leak that inflates YRBS ceilings.
    strict=False: legacy fillna(0) — treats missing as no-ACE. Diagnostic only.
    """
    if strict:
        y = (pillars.sum(axis=1) >= threshold).astype("float")
        y[pillars.isna().any(axis=1)] = np.nan
    else:
        y = (pillars.fillna(0).sum(axis=1) >= threshold).astype("float")
        y[pillars.isna().all(axis=1)] = np.nan
    return y


def make_outcome(pillars: pd.DataFrame, threshold=1):
    """The paper's outcome: the five shared pillars, cut at `threshold`, strictly composed.

    THE ONE DEFINITION, and every caller that wants the five-pillar outcome goes through it —
    notebook 02's splits, notebook 04, `data.analytic_sample_mask`, `data.missingness_audit`
    and `transfer.leave_one_pillar_out`'s full variant. `strict=True` is fixed here and
    nowhere else: `strict=False` is the documented fillna(0) label leak that inflated the
    YRBS ceiling, and a hand-typed `strict=` at six call sites is six chances to get it wrong
    in silence.

    The exception is `transfer.outcome_variant_battery`, which composes over `CATEGORY_SETS`
    subsets of one or two pillars. It calls `compose_outcome` directly because it is
    deliberately building a different construct, and that difference should be visible rather
    than hidden behind a shared helper.

    >=1, >=2 and >=3 are all reported. >=4 was dropped when the E2 prevalence guard tripped.
    THAT IS HISTORY, NOT A CHECK: the guard's threshold is not in this repository, so nothing
    here re-derives the exclusion, and a reuse with a different extract has to settle the cut
    for itself from the prevalences. What a run can enforce is intrinsic rather than policy —
    `build_splits` stratifies, so a single-class cut cannot be split at all.
    The prevalence at each cut is the no-skill PR-AUC null and must travel with every PR-AUC
    number; notebook 01 prints them for both cohorts and every battery row carries its own in
    `prevalence`. No figure is repeated here, because a number written into a docstring is a
    number that stops being recomputed.
    """
    missing = [c for c in SHARED_PILLARS if c not in pillars.columns]
    if missing:
        raise ValueError(f"pillar frame is missing shared pillars: {missing}")
    return compose_outcome(pillars[list(SHARED_PILLARS)], threshold=threshold, strict=True)


def compose_outcome_loo(pillars: pd.DataFrame, drop_pillar: str, threshold=1, strict=True):
    """Leave-one-pillar-out outcome: compose_outcome over ALL pillars except `drop_pillar`.

    Decomposes the transfer gap into construct non-equivalence in one specific pillar
    versus relational shift. It does not touch compose_outcome — it drops the named
    column and delegates — so `strict` behaves exactly as it does there (NaN if any
    remaining pillar is missing).
    """
    if drop_pillar not in pillars.columns:
        raise ValueError(f"{drop_pillar!r} not in pillars {list(pillars.columns)}")
    return compose_outcome(pillars.drop(columns=[drop_pillar]),
                           threshold=threshold, strict=strict)


