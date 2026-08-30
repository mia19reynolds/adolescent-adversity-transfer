"""Resolve aggregate inputs for checks and figures.

Tracked public tables are preferred so verification works in a clean clone. If
no public copy exists, ``resolve`` looks in the external working directory. It
raises ``MissingInput`` rather than substituting a different analysis when
neither copy is available.
"""

from pathlib import Path


import config as C

__all__ = ["save_table", "save_figure", "load_frozen", "resolve", "published_path", "PUBLISHED",
           "MissingInput"]


class MissingInput(FileNotFoundError):
    """Neither a public nor a working copy exists. The message names the producer."""


# The tracked public tables, keyed as the repository holds them, and the working filename each
# was published from. A clone has only the left-hand side, so callers name the published table.
#
# Every pair is a rename rather than a transformation — the two are equivalent in content.
# Anything that reshapes a frame belongs in a publisher, not in this table. Two working files
# can publish to one table, so the value is a tuple.
PUBLISHED: dict[str, tuple[str, ...]] = {
    "paper/data/transfer_grid.csv":            ("regime_battery_summary.csv",),
    "paper/data/transfer_significance.csv":    ("regime_significance.csv",),
    "paper/data/label_budget.csv":             ("label_budget_curve.csv",),
    "paper/data/label_budget_summary.csv":     ("label_budget_summary.csv",),
    "paper/data/calibration.csv":              ("calibration_correction_sweep.csv",),
    "paper/data/subgroup_performance.csv":     ("subgroup_discrimination_summary.csv",),
    "paper/data/operating_points.csv":         ("operating_point_metrics_summary.csv",),
    "paper/data/subgroup_capacity.csv":        ("subgroup_operational.csv",),
    "paper/data/ventile_stratification.csv":   ("ventile_stratification_full.csv",),
    "paper/data/conformal_cells.csv":          ("conformal_prereq_cells.csv",),
    "paper/data/model_screening.csv":          ("screening_summary.csv",),
    "paper/data/outcome_variants.csv":         ("threshold_sensitivity_summary.csv",),
    "paper/data/outcome_exclusions.csv":       ("outcome_variants_excluded.csv",),
    "paper/data/sample_characteristics.csv":   ("participant_characteristics.csv",),
    "paper/data/pillar_prevalence.csv":        ("pillar_prevalence.csv",),
    "paper/data/regime_grid.csv":              ("full_regime_grid.csv",),
    "paper/data/regime_grid_sd.csv":           ("full_regime_grid_sd.csv",),
    "paper/data/canonical_claims.csv":         ("CANONICAL.csv",),
    "paper/data/conformal_coverage.csv":       ("conformal_per_family.csv", "conformal_threshold2.csv"),
    "supplementary/data/xgboost_rule_transfer.csv": ("rule_head.csv",),
    "supplementary/data/subgroup_ranking.csv":      ("subgroup_by_family.csv",),
    "supplementary/data/conformal_by_cell.csv":     ("conformal_coverage.csv",),
}

# The reverse lookup, for a caller that still names a working file.
_WORKING_TO_PUBLISHED: dict[str, str] = {
    w: pub for pub, works in PUBLISHED.items() for w in works
}

# Local-only by design. Too large or too fine-grained to track, and the checks that need
# them need a working root, unlike everything else here.
LOCAL_ONLY: frozenset[str] = frozenset({
    "conformal_full_grid.csv",      # 97,920 rows, 7 MB, one row per seed x cell
    "all_results.csv",              # 7,686 rows, budget lineage
    "regime_battery.csv",        # the per-seed battery; the public copy is its summary
})


def published_path(rel: str) -> str | None:
    """The tracked public path for `rel`, whether it is already one or a working filename."""
    rel = str(rel)
    if rel in PUBLISHED:
        return rel
    return _WORKING_TO_PUBLISHED.get(Path(rel).name)


def working_names(rel: str) -> tuple[str, ...]:
    """The working-root filename(s) `rel` was published from, or just `rel` if it is one."""
    pub = published_path(rel)
    return PUBLISHED[pub] if pub else (Path(str(rel)).name,)


def resolve(rel: str | Path, *, root: Path | None = None, required: bool = True,
            allow_frozen: bool = False) -> Path:
    """The working copy. The tracked public copy only if the caller asks for it.

    `allow_frozen=False` is the default and the important part. A tracked table under
    `outputs/` is a RESULT, and handing one back to a caller that believes it is deriving
    something makes a check compare a file with itself. Callers that genuinely want the
    published copy — `load_frozen`, and the private draft check under `$THESIS_WORK_DIR` —
    pass `allow_frozen=True` and say so at the call site.
    """
    rel = str(rel)
    base = Path(root) / "outputs" if root else Path(C.OUTPUTS_DIR)
    pub = published_path(rel)

    if allow_frozen and pub is not None:
        p = base / pub
        if p.exists():
            return p

    # then the working root, which is OUTSIDE the repository. A clone has no working root
    # configured and does not need one: everything published is already here.
    candidates = []
    if pub is None:
        candidates.append(Path(root) / rel if root else Path(C.PROJECT_ROOT) / rel)
    try:
        w = C.work_path()
        for name in working_names(rel):
            candidates += [w / "aggregates" / name, w / "tables" / name, w / name]
    except RuntimeError:
        pass
    for p in candidates:
        if p.exists():
            return p

    if not required:
        return candidates[0] if candidates else base / (pub or rel)

    name = working_names(rel)[0]
    where = "local-only by design; run notebook 02" if name in LOCAL_ONLY else \
            "run notebooks 02 and 03"
    frozen = "" if allow_frozen or pub is None else (
        f" A tracked copy exists at outputs/{pub}, and was NOT used: this call did not ask for "
        f"a frozen result. Use inputs.load_frozen({pub!r}) if that is what you want.")
    raise MissingInput(
        f"{rel} is not under the working root. Nothing here recomputes it — {where}.{frozen}")


# ---------------------------------------------------------------- the two operations
#
# `save_table` writes what this run derived. `load_frozen` reads a published table that
# nothing here can rebuild, and prints FROZEN when it does — which is the whole reason the
# second verb exists. No notebook currently calls it: the sections that read a frozen table
# were rebuilt live, and it stands for the sections listed in `outputs/FROZEN.md`.

def save_table(frame, name: str, *, subdir: str = "tables", quiet: bool = False) -> None:
    """Write a working table under `$THESIS_WORK_DIR/<subdir>/` and say what landed.

    Writes and prints; returns nothing. `quiet=True` prints the filename alone, for a frame
    whose shape is itself MCS-derived. `scores.write_yrbs_scores` takes the same keyword.

    `name` is a filename inside `subdir`, and `subdir` a directory under the working root.
    Both are checked before `mkdir` runs, because `mkdir` acts on whatever it is handed and
    a `name` carrying a separator would create a directory rather than fail.
    """
    import config as _C

    # `Path("..").name` is `".."`, not the empty string a separator-bearing name reduces to,
    # so the two traversal components are named rather than left to the comparison.
    if not name or name in (".", "..") or Path(name).name != name:
        raise ValueError(
            f"save_table: {name!r} is not a filename. It names one file inside the working "
            f"subdirectory, so it may not be absolute, carry a path separator, or traverse "
            f"upwards.")
    root = _C.work_path()
    directory = _C.work_path(subdir).resolve()
    if not directory.is_relative_to(root):
        raise ValueError(
            f"save_table: subdir={subdir!r} resolves outside the working root at {root}. It "
            f"names a directory under that root, not a location of its own.")

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    frame.to_csv(path, index=False)
    if quiet:
        print(f"  wrote  {name}")
    else:
        print(f"  wrote  {name}  ({len(frame):,} rows x {len(frame.columns)} cols)")


def save_figure(fig, name: str) -> None:
    """Write a working diagnostic figure under `$THESIS_WORK_DIR/figures/`.

    Not `paper.save_figure`, which is the only thing that writes a figure INTO the repository
    and carries the published figure's sizing. These are screen diagnostics.
    """
    import config as _C

    path = _C.work_path("figures", create=True) / f"{name}.png"
    fig.savefig(path, format="png", dpi=200, bbox_inches="tight")
    print(f"  wrote  {path.name}")


def load_frozen(published: str, **read_kwargs):
    """A published table, read from `outputs/` and returned labelled as frozen.

    `published` is the path as this repository holds it — `paper/data/transfer_grid.csv` — not
    the working filename it was published from. That is the one a clone actually has, and it
    says what the table contains rather than which run produced it.

    Recomputes nothing. The label travels with the frame and is printed on the read, so a
    number that came from an earlier run is identifiable as one where it is displayed.
    `read_kwargs` pass through to `pandas.read_csv`, for callers that need exact text
    (`dtype=str`) rather than parsed values.
    """
    import pandas as _pd

    path = resolve(published, allow_frozen=True)
    frame = _pd.read_csv(path, **read_kwargs)
    frame.attrs["provenance"] = "frozen"
    frame.attrs["source"] = str(path)
    print(f"  FROZEN      {published}  ({len(frame):,} rows)")
    return frame
