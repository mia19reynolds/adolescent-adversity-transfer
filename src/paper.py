"""
Checking a result against its declaration, and the two renderings that still write.

**Nothing here writes into the repository.** The notebooks compute and display; the published
tables under `outputs/` are curated by hand from a finished run. There is no CSV writer here
and no caller for one, which is deliberate: a table that reaches `outputs/` has been reviewed,
and an automatic writer would make that step easy to skip.

What remains is the check. `validate()` compares a frame against its declaration in
`tables.py` — required columns present, the unique key actually unique, the row count not
silently collapsed, and no person-level column anywhere near a public table. That last check
is the one that matters: MCS Sweep 6 is UKDS Safeguarded Tier 1a and `outputs/` is the one
tree that leaves this machine, so a score column reaching it would be a licence breach rather
than a bug. It is cheap, so it runs wherever a frame is about to be treated as public.

`save_figure` and `save_fragment` are the two renderings notebook 03 still generates: the
ventile figure and the LaTeX appendix fragments. Both land under the configured working root,
beside every other run product, as review candidates for the manuscript rather than as
repository files. They carry no schema machinery — a bare-filename check and an atomic replace
— because a PNG and a tabular fragment have no columns to validate.

Their destinations are functions rather than module constants. Resolving the working root is a
check that can refuse, so importing this module must not need one; `figures_dir()` and
`fragments_dir()` resolve on the call, and the directory is made by the write itself.

Both write through a temporary file in the same directory and then `os.replace`, so an
interrupted run leaves the previous version intact rather than a half-written file.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd

import config as C
from tables import BY_KEY, Consolidated

# The two generated destinations, under the configured working root. `PAPER_SUBDIR` is shared
# so the pair cannot drift apart.
PAPER_SUBDIR = ("publication_outputs", "paper")


def figures_dir() -> Path:
    """Where `save_figure` writes, under the configured working root."""
    return C.work_path(*PAPER_SUBDIR, "figures")


def fragments_dir() -> Path:
    """Where `save_fragment` writes, under the configured working root."""
    return C.work_path(*PAPER_SUBDIR, "tables")


# Columns that would make an aggregate person-level. None should ever appear; the check is
# cheap and the consequence of missing one is a licence breach, not a bug.
FORBIDDEN_COLUMNS = frozenset({
    "mcsid", "MCSID", "pidp", "respondent_id", "person_id", "row_id",
    "score", "y_score", "y_pred", "prediction", "proba", "p_hat",
})


# ---------------------------------------------------------------- validation

def validate(spec: Consolidated, df: pd.DataFrame) -> list[str]:
    """Every way the table can be wrong, reported together rather than one at a time."""
    problems = []

    missing = [c for c in spec.required if c not in df.columns]
    if missing:
        problems.append(f"required column(s) absent: {missing}")

    present_key = [c for c in spec.unique_key if c in df.columns]
    absent_key = [c for c in spec.unique_key if c not in df.columns]
    if absent_key:
        problems.append(f"unique-key column(s) absent: {absent_key}")
    elif present_key:
        dup = df.duplicated(present_key).sum()
        if dup:
            problems.append(f"{dup} row(s) duplicate the unique key {list(spec.unique_key)}")

    if len(df) < spec.min_rows:
        problems.append(f"{len(df)} rows, fewer than the declared minimum {spec.min_rows} — "
                        f"a source is truncated or a scope silently narrowed")

    leaked = sorted(set(df.columns) & FORBIDDEN_COLUMNS)
    if leaked:
        problems.append(f"DISCLOSURE: person-level column(s) {leaked} in a public table")

    for c in spec.required:
        if c in df.columns and df[c].isna().all():
            problems.append(f"required column {c!r} is entirely empty")

    return problems


# ---------------------------------------------------------------- the write

def _write_atomic_bytes(render, dest: Path) -> str:
    """Render into a temporary sibling file, then replace. Returns what happened."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    before = dest.read_bytes() if dest.exists() else None
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".partial")
    os.close(fd)
    tmp = Path(tmp)
    try:
        render(tmp)
        after = tmp.read_bytes()
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if before is None:
        return "created"
    return "unchanged" if before == after else "replaced"


def save_figure(fig, name: str, *, dpi: int = 300) -> Path:
    """The ONLY sanctioned write into `figures_dir()`.

    No schema machinery — a destination check and an atomic replace. `name` is a bare
    filename; the destination is fixed so a caller cannot route a figure elsewhere. The
    filename alone is printed: the working root is a local location, not a result.
    """
    if Path(name).name != name:
        raise ValueError(f"save_figure takes a bare filename, got {name!r}")
    dest = figures_dir() / name
    action = _write_atomic_bytes(lambda p: fig.savefig(p, format="png", dpi=dpi), dest)
    print(f"  {action:<10}{name}")
    return dest


def save_fragment(text: str, name: str) -> Path:
    """The ONLY sanctioned write into `fragments_dir()`.

    Same contract as `save_figure`."""
    if Path(name).name != name:
        raise ValueError(f"save_fragment takes a bare filename, got {name!r}")
    dest = fragments_dir() / name
    action = _write_atomic_bytes(lambda p: p.write_text(text), dest)
    print(f"  {action:<10}{name}")
    return dest

