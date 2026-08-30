"""
src/publication.py — what the notebooks are allowed to show and to write.

Three things, and nothing else:

  show            display a frame through an explicit column allow-list
  save_table      write one named candidate table, allow-listed the same way
  save_figure     write one named candidate figure

plus the MCS small-count rule the three notebooks share (`public_count`,
`withhold`, `breakdown`).

THE ALLOW-LIST IS THE MECHANISM. A publication table names the columns it means to publish;
anything else stays behind. Nothing here strips dangerous columns out of a working frame,
because that only protects against the columns somebody remembered.

WRITING A FILE HERE IS NOT DISCLOSURE CLEARANCE. Candidate tables and figures go to the
working root, not into the repository, because a generated result is a manuscript working file
rather than a release artefact — and because a table is only ever released together with every notebook,
figure and table that carries a related value. Whether an MCS-derived value may leave the
approved environment is a decision for the researcher, the institution and the UK Data Service,
and no check in this module makes it.
"""

import math
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# NOTHING HERE IS A REPOSITORY FILE. Candidate tables and candidate figures alike are
# manuscript working files: reviewed, revised, and only ever released deliberately and as a
# set. Both resolve under the working root with everything else this pipeline generates, so
# the project directory holds source and nothing a run produced.
#
# Both destinations are functions rather than module constants, because resolving the working
# root is a check that can refuse. Importing this module must not need a configured root; only
# writing does.
CANDIDATE_SUBDIR = "publication_candidates"
FIGURE_SUBDIR = ("publication_outputs", "figures")


def candidates_dir() -> Path:
    """Where candidate tables are written, under the configured working root."""
    import config as _C

    # `work_path(create=True)` makes the PARENT, so the directory itself is made here.
    path = _C.work_path(CANDIDATE_SUBDIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir() -> Path:
    """Where candidate figures are written, under the configured working root.

    Resolved and created on the call, not at import, for the reason above.
    """
    import config as _C

    path = _C.work_path(*FIGURE_SUBDIR)
    path.mkdir(parents=True, exist_ok=True)
    return path

# Below this an MCS count is withheld. YRBS is open CDC data and is reported in full.
SUPPRESS_BELOW = 10

# ONE PUBLIC MARKER FOR EVERY WITHHELD VALUE. A marker that said "<10" would disclose the range
# it was hiding, and a per-row reason would say which rule applied — so a reader could tell a
# small cell from one withheld to stop a subtraction. Both are the same mark here, and the
# reasons live in the restricted private file and in internal mode.
NOT_REPORTED = "\u2014"
NOT_REPORTED_STATUS = "not_reported"

# Columns a publication table may never carry, whatever the allow-list asks for. This is a
# refusal, not a filter: naming one of these is a mistake in the call, and it stops rather than
# being quietly dropped.
NEVER_PUBLISHED: Sequence[str] = (
    "row_id", "score", "y_true",
    "n_test", "n_pos",
    "dec_TP", "dec_FP", "dec_FN", "dec_TN",
    "qui_TP", "qui_FP", "qui_FN", "qui_TN",
    "TP", "FP", "FN", "TN",
    "solver_status", "recal_slope", "recal_intercept", "recal_clipped_fraction",
    "recal_score_range", "w0", "w1", "tpr", "fpr",
    # The resource-rich benchmark's SEARCH, as opposed to its result. An inner-CV AUC is a
    # working diagnostic measured on training folds; publishing it beside a test-set AUC invites
    # the two to be read as comparable, and they are not. The candidate count and fold count are
    # the same kind of thing. `target_selection_status` is NOT here — a compact estimability
    # status is exactly what a reader needs to know a cell was attempted.
    "target_selection_cv_auc", "target_selection_candidates", "target_selection_folds",
    "selected_configuration", "target_params",
)


def _checked(frame: pd.DataFrame, columns: Sequence[str], where: str) -> pd.DataFrame:
    """The named columns, in the order named, or a refusal saying which and why."""
    columns = list(columns)
    refused = [c for c in columns if c in NEVER_PUBLISHED]
    if refused:
        raise ValueError(f"{where}: {refused} may not be published. Exact counts, person-level "
                         f"fields and working diagnostics stay in the internal artefacts.")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{where}: the frame has no column(s) {missing}. The allow-list names "
                         f"what to publish, so a renamed column has to be noticed here.")
    return frame.loc[:, columns]


def _rounded(frame: pd.DataFrame, places: int) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].round(places)
    return out


def show(frame: pd.DataFrame, columns: Sequence[str], *, places: int = 4) -> pd.DataFrame:
    """The named columns of `frame`, rounded, for display. Nothing else is rendered."""
    return _rounded(_checked(frame, columns, "show"), places)


def save_table(frame: pd.DataFrame, name: str, columns: Sequence[str], *,
               places: int = 4) -> None:
    """Write one candidate table under `publication_candidates/` in the working root.

    Prints the filename and nothing else — not a row count, not a path, and not a claim that
    what was written may be released.
    """
    if not name.endswith(".csv"):
        raise ValueError(f"save_table: {name!r} should name a .csv")
    out = _rounded(_checked(frame, columns, f"save_table({name})"), places)
    out.to_csv(candidates_dir() / name, index=False)
    # The filename only. Not the path, not a row count, and not a claim about release.
    print(f"  wrote  {name}  (working candidate — not a repository file, and not reviewed)")


def save_figure(fig, name: str) -> None:
    """Write one candidate figure under `publication_outputs/figures/` in the working root.

    Prints the filename and nothing else — not the path, and not a claim that what was written
    may be released.
    """
    if not name.endswith(".png"):
        raise ValueError(f"save_figure: {name!r} should name a .png")
    fig.savefig(figures_dir() / name, format="png", dpi=300, bbox_inches="tight")
    print(f"  wrote  {name}  (candidate — disclosure review still required)")


# ---- the MCS small-count rule ----------------------------------------------

def public_count(n, cohort: str, total=None, *, share_only: bool = False) -> str:
    """A count for display, with its share of `total` when one is given.

    A withheld MCS count becomes the neutral marker, and the percentage goes with it — a
    percentage on a printed denominator is the count.

    `share_only` reports the share and withholds the count. It is for the case where the
    DENOMINATOR is itself withheld: the counts of an exhaustive breakdown sum to it, so
    printing them would hand back the total that was withheld elsewhere.
    """
    n = int(n)
    if cohort == "MCS" and 0 < n < SUPPRESS_BELOW:
        return NOT_REPORTED
    if total is None:
        return f"{n:,}"
    if share_only:
        return f"{100 * n / total:.1f}%"
    return f"{n:,} ({100 * n / total:.1f}%)"


def withhold(counts: pd.Series, cohort: str, *, also_withheld=()) -> dict:
    """Which categories of an exhaustive breakdown are withheld, and why.

    Primary: any MCS count of 1 to 9.

    Secondary: a breakdown prints its total, so a single hidden category is given back by
    subtraction — the smallest remaining category goes with it. Ties break on the label, so the
    choice does not depend on row order.

    `also_withheld` names categories a DIFFERENT table already withheld on the same population.
    Two tables over the same categories can be differenced, so a category hidden in one and
    shown in the other is not hidden at all; passing the first table's withheld set here keeps
    the two consistent.
    """
    if cohort != "MCS":
        return {}
    reasons = {c: "primary" for c, n in counts.items() if 0 < n < SUPPRESS_BELOW}
    for category in also_withheld:
        if category in counts.index and category not in reasons:
            reasons[category] = "consistency_with_a_related_table"
    if len(reasons) == 1:
        others = sorted((c for c in counts.index if c not in reasons),
                        key=lambda c: (counts[c], str(c)))
        if others:
            reasons[others[0]] = "secondary"
    return reasons


def paired_withholding(counts_a: pd.Series, counts_b: pd.Series, cohort: str) -> dict:
    """One withheld set for TWO tables over the same categories, hidden in both.

    Two breakdowns of the same categories on nested populations — the full cohort and the
    analytic sample — are not protected by suppressing each on its own. A category of 100 in one
    and 95 in the other passes every single-cell rule and still says that five respondents in
    that category were excluded.

    So the pair is decided together, and the same set is hidden in both:

      * a small count in EITHER table;
      * a small DIFFERENCE between the two, which is the case a per-table rule cannot see;
      * the complementary pass, because one hidden category is given back by subtraction from
        either total.

    Hiding the same set in both is what stops the differencing: the pair then yields the sum of
    the hidden differences, not any one of them.

    THE RESIDUAL CASE, recorded rather than hidden. If a hidden category happens to have the
    same count in both tables its difference is zero, and the remaining hidden difference is
    the whole residual. The secondary choice prefers a category whose difference is non-zero to
    avoid that; where no such category exists, the reason is
    `secondary_no_nonzero_difference_available` and the pair needs manual review before release.
    """
    if cohort != "MCS":
        return {}
    categories = list(dict.fromkeys(list(counts_a.index) + list(counts_b.index)))
    a = {c: int(counts_a.get(c, 0)) for c in categories}
    b = {c: int(counts_b.get(c, 0)) for c in categories}

    reasons = {}
    for c in categories:
        if 0 < a[c] < SUPPRESS_BELOW or 0 < b[c] < SUPPRESS_BELOW:
            reasons[c] = "primary"
    for c in categories:
        if c not in reasons and 0 < abs(a[c] - b[c]) < SUPPRESS_BELOW:
            reasons[c] = "small_difference_between_the_two_tables"

    if len(reasons) == 1:
        remaining = [c for c in categories if c not in reasons]
        moves = [c for c in remaining if a[c] != b[c]]
        if moves:
            reasons[min(moves, key=lambda c: (min(a[c], b[c]), str(c)))] = "secondary"
        elif remaining:
            reasons[min(remaining, key=lambda c: (min(a[c], b[c]), str(c)))] = (
                "secondary_no_nonzero_difference_available")
    return reasons


UNRESOLVED = "secondary_no_nonzero_difference_available"


def require_resolved(reasons: dict, what: str, *, public: bool = True) -> None:
    """Stop before display when a paired withholding could not be made safe.

    `paired_withholding` hides a second category so the pair yields the SUM of the hidden
    differences rather than any one of them. That fails when every remaining category has the
    same count in both tables: the second hidden difference is then zero, and the first is the
    whole residual.

    The gate is unconditional in the pipeline: the notebook calls it without an argument, and an
    unresolved pair stops the run whether or not that run is destined for a public copy. Two
    tables that give a withheld value back by subtraction are a disclosure problem in the
    approved environment as much as outside it, and the notebook prints the detail beside this
    call before it fires. `public=False` exists for the tests that exercise the permissive path.
    """
    if not public or UNRESOLVED not in set(reasons.values()):
        return
    raise ValueError(
        f"{what}: these two tables cannot be released together as they stand, because the "
        f"components they would display allow a withheld value to be recovered by subtraction. "
        f"Revise the categories or the totals the two share, or seek disclosure guidance, "
        f"before publishing either. Re-run in internal mode to see which components are "
        f"involved.")


def withhold_rate(numerator, denominator, cohort: str) -> bool:
    """Whether a rate must be withheld because the cell behind it is small.

    A percentage on a published denominator IS the count, however few decimal places it carries,
    so dropping the numerator is not enough. BOTH TAILS COUNT: a rate of 99.7% on a denominator
    of 1,000 says three respondents are in the complementary cell, which is the same disclosure
    from the other end.
    """
    if cohort != "MCS" or not denominator:
        return False
    numerator = int(round(float(numerator)))
    complement = int(denominator) - numerator
    return (0 < numerator < SUPPRESS_BELOW) or (0 < complement < SUPPRESS_BELOW)


def public_rate(numerator, denominator, cohort: str, *, places: int = 2):
    """A rate for display, or `None` where the cell behind it is small.

    Rounded to `places` everywhere, so two tables cannot be differenced at different precisions.
    Rounding is NOT the protection — `withhold_rate` is; rounding only keeps the published
    figures consistent with one another.
    """
    if not denominator:
        return None
    if withhold_rate(numerator, denominator, cohort):
        return None
    return round(100 * float(numerator) / float(denominator), places)


# The precision a rate is displayed at, here and in `public_rate`, and the precision
# `data.pct_missing` has already rounded to before a rate reaches this module.
DISPLAYED_PLACES = 2

# What a rate whose cell cannot be worked out is treated as. It is not an estimate of anything:
# it is the smallest positive count, which `withhold_rate` always withholds, so a rate this
# module cannot check is a rate that is not published.
UNCHECKABLE_CELL = 1


def _positive_count(value, where: str) -> int:
    """A denominator, or a refusal. A rate on a denominator that is not a count of respondents
    cannot be turned back into a cell, and guessing one would withhold the wrong rows."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{where}: {value!r} is not a denominator. A count of respondents is "
                         f"a number, and a string or a flag reaching here is a mistaken call.")
    number = float(value)
    if not math.isfinite(number) or number <= 0 or number != int(number):
        raise ValueError(f"{where}: the denominator has to be a positive whole number of "
                         f"respondents, not {value!r}. A rate whose denominator is unknown, "
                         f"zero or fractional cannot be checked against the small-count rule.")
    return int(number)


def _tail_margin(numerator: int, denominator: int) -> int:
    """How near a count sits to either end of its denominator.

    Below `SUPPRESS_BELOW` is exactly the condition `withhold_rate` withholds on, from whichever
    tail. An empty cell and a full one are not near an end — they are not small counts.
    """
    low = numerator if numerator > 0 else SUPPRESS_BELOW
    high = denominator - numerator if denominator - numerator > 0 else SUPPRESS_BELOW
    return min(low, high)


def _compatible_counts(percent: float, denominator: int, places: int) -> range:
    """Every count on `denominator` that displays as `percent` at `places` decimal places.

    Exact arithmetic throughout: a binary float that is a hair under x.5 rounds the wrong way
    and hands back a count that is off by one, which is the one error this must not make.
    """
    half = Fraction(1, 2 * 10 ** places)          # in percentage points
    value = Fraction(str(float(percent)))
    scale = Fraction(denominator, 100)
    first, last = math.ceil((value - half) * scale), math.floor((value + half) * scale)
    if first > last:
        # No count on this denominator displays as this rate — the interval falls between two
        # integers. That means the rate was measured on a DIFFERENT denominator, so take the
        # pair the interval sits between rather than inferring nothing.
        first, last = math.floor((value - half) * scale), math.ceil((value + half) * scale)
    return range(max(0, first), min(denominator, last) + 1)


def implied_numerator(percent, denominator, *, places: int = DISPLAYED_PLACES) -> int:
    """The count behind a rate that is already published as a rounded percentage.

    `percent` is on the 0-100 scale, like `public_rate` returns and `data.pct_missing` produces,
    and it is assumed to carry `places` decimal places already. `denominator` is the count the
    rate was measured over.

    ROUNDING DOES NOT ALWAYS LEAVE ONE ANSWER. A rate shown to two places stands for an interval
    of counts whose width is the denominator over ten thousand, so on a large sample several
    counts display identically. This returns the most disclosive of them — the one nearest
    either end of the denominator — so that passing the result to `withhold_rate` withholds the
    rate whenever ANY count compatible with it is a small cell. A rate that is compatible with
    both a safe count and a small one is a rate that could be hiding the small one.

    Where the cell cannot be worked out at all — a missing or non-finite percentage — the answer
    is `UNCHECKABLE_CELL`, which `withhold_rate` withholds. A denominator that is not a positive
    whole number, or a percentage off the 0-100 scale, is a mistake in the call and stops here.
    """
    denominator = _positive_count(denominator, "implied_numerator")
    if percent is None:
        return UNCHECKABLE_CELL
    try:
        value = float(percent)
    except (TypeError, ValueError):
        raise ValueError(f"implied_numerator: {percent!r} is not a percentage.") from None
    if not math.isfinite(value):
        return UNCHECKABLE_CELL
    if not 0.0 <= value <= 100.0:
        raise ValueError(f"implied_numerator: {value} is not a percentage from 0 to 100. The "
                         f"rate is expected on the same scale `public_rate` returns.")
    candidates = _compatible_counts(value, denominator, places)
    return min(candidates, key=lambda n: (_tail_margin(n, denominator), n))


def breakdown(series: pd.Series, cohort: str, total, *, also_withheld=(),
              share_only: bool = False) -> str:
    """An exhaustive categorical breakdown, with withheld categories marked and no share."""
    counts = series.fillna("missing").value_counts()
    hidden = withhold(counts, cohort, also_withheld=also_withheld)
    return ", ".join(f"{category} {NOT_REPORTED}" if category in hidden
                     else f"{category} {public_count(n, cohort, total, share_only=share_only)}"
                     for category, n in counts.items())


def withheld_rows(series: pd.Series, cohort: str, measure: str, *, also_withheld=()) -> list:
    """The exact values a breakdown withheld, for the private disclosure-review file."""
    counts = series.fillna("missing").value_counts()
    total = len(series)
    return [dict(measure=measure, category=str(c), count=int(counts[c]), denominator=total,
                 percent=round(100 * counts[c] / total, 4) if total else None, reason=why)
            for c, why in withhold(counts, cohort, also_withheld=also_withheld).items()]


def recovers_by_subtraction(parts: Sequence[int], total: int, withheld: Sequence[int]) -> bool:
    """Whether the shown parts and the printed total give a withheld part back exactly.

    An exhaustive breakdown with one category hidden is the case this catches: `total` minus the
    shown parts IS the hidden one. With two or more hidden, only their sum is recoverable.
    """
    if len(withheld) != 1:
        return False
    return int(total) - int(sum(parts)) == int(withheld[0])


# ---- what a manuscript-facing table has to be -------------------------------

THRESHOLD_ROLES = {">=1": "lower_sensitivity", ">=2": "primary", ">=3": "higher_sensitivity"}


def require_primary_threshold(frame: pd.DataFrame, threshold: str, name: str) -> pd.DataFrame:
    """A main-facing table is non-empty and carries that threshold alone, or it is not written.

    A main table that quietly came out empty, or that picked up a sensitivity row, is worse than
    no table: it reads as a result. Non-estimable and not-applicable rows STAY — they are
    scientific statuses carried in their own columns, not something to filter out.
    """
    if "threshold" not in frame.columns:
        raise ValueError(f"{name}: the frame has no threshold column to check")
    if frame.empty:
        raise ValueError(f"{name}: no rows at {threshold}. A main table is not written empty.")
    other = sorted(set(frame["threshold"]) - {threshold})
    if other:
        raise ValueError(f"{name}: rows at {other} reached a table that carries {threshold} "
                         f"alone; the sensitivity thresholds belong in threshold_sensitivity")
    return frame


def add_threshold_roles(frame: pd.DataFrame, *, keys=("regime", "family")) -> pd.DataFrame:
    """Label each threshold with the part it plays, and refuse a gap in the grid.

    The sensitivity table is the one place `>=1` and `>=3` appear, and it is only readable if
    every model carries all three. A model-threshold that is genuinely absent must say so
    through its own status column rather than by not being there.
    """
    unknown = sorted(set(frame["threshold"]) - set(THRESHOLD_ROLES))
    if unknown:
        raise ValueError(f"threshold_sensitivity: unrecognised threshold(s) {unknown}")
    present = frame.groupby(list(keys))["threshold"].nunique()
    incomplete = present[present != len(THRESHOLD_ROLES)]
    if len(incomplete):
        raise ValueError(f"threshold_sensitivity: {len(incomplete)} model(s) do not carry all "
                         f"three thresholds. A missing row has to be recorded, not absent.")
    out = frame.copy()
    out["threshold_role"] = out["threshold"].map(THRESHOLD_ROLES)
    return out
