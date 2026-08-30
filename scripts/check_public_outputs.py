#!/usr/bin/env python3
"""
scripts/check_public_outputs.py — warn about known risks in an executed public notebook.

TWO KINDS OF FILE, and only one of them is ever a candidate for release.

The four tracked notebooks in `notebooks/` declare `PUBLIC_NOTEBOOK = False`. They are the
internal executed record of a run: they may hold the full stored output of that run, which is
what makes them worth reading, and declaring False is their correct state rather than a fault.
They are NOT publication-ready, and nothing here treats them as such.

A public copy is a SEPARATE file written by `scripts/make_public_notebooks.py`. It declares
`PUBLIC_NOTEBOOK = True` and may retain stored output only in the cells
`spec/public_notebook_cells.json` names explicitly; every other cell is cleared. The flag alone
authorises nothing, and a copy of a known notebook with no allow-list entry is reported.

Source and copy are told apart by location. `check_notebook_outputs.classify` is where the rule
lives; nothing here reimplements it.

This reads the saved notebook files and reports the patterns that would make a stored output
obviously unsuitable: a traceback, an absolute path, a person-level column, an exact count, a
private filename, a table too large to have been meant for a reader, an out-of-order execution,
or a `PUBLIC_NOTEBOOK` declaration that is not the one the file should carry.

It reads STORED OUTPUTS and the declared public-table schemas. It does not scan ordinary source
code for words like `score` or `y_true`, which are legitimate analytical names.

Where it can, it reports the pattern and the cell index rather than the text that matched.

PASSING THIS IS NOT DISCLOSURE CLEARANCE. It is a structural check for known patterns. An
MCS-derived figure needs researcher, institutional and UK Data Service review before it leaves
the approved environment.

Usage:
    python scripts/check_public_outputs.py [path ...]     # default: notebooks/0[123]_*.ipynb
Exit status 0 clean, 1 findings, 2 unreadable.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_notebook_outputs import (      # noqa: E402 — one allow-list, one policy, one rule
    FLAG_DECLARATION, FLAG_POLICIES, FLIP, GENERATED, INTERNAL, INVALID, SPEC, classify,
    load_allowlists, load_flag_policies)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_NOTEBOOKS = ("01_data_and_features.ipynb", "02_models_and_transfer.ipynb",
                    "03_evaluation_and_robustness.ipynb", "04_explorations.ipynb")

DISCLAIMER = ("This is a warning tool. It checks for known patterns. It does not establish "
              "disclosure clearance, which requires researcher, institutional and UK Data "
              "Service review.")

PUBLIC_FLAG = FLAG_DECLARATION      # the one pattern, re-exported for callers of this module
ABSOLUTE_PATH = re.compile(r"(/Users/|/home/|C:\\\\Users\\\\|\$HOME)")
PRIVATE_FILE = re.compile(r"notebook_01_exact_counts|yrbs_scores\.parquet|regime_battery")
PERSON_LEVEL = re.compile(r"\b(row_id|y_true)\b")
EXACT_COUNTS = re.compile(r"\b(n_test|n_pos|dec_TP|dec_FP|dec_FN|dec_TN|"
                          r"qui_TP|qui_FP|qui_FN|qui_TN)\b")
# A count that is not itself a respondent total but from which one is recoverable: a
# denominator, an eligible-set size, or a per-cell tally beside a rate.
RECONSTRUCTABLE_COUNTS = re.compile(
    r"\b(eligible_respondents|n_analytic|n_scored|n_eligible|n_excluded|n_people|"
    r"n_respondents|declared_cells|n_min|denominator|seeds_estimated)\b")
MAX_RENDERED_ROWS = 60


def output_text(cell) -> str:
    """Every piece of text a stored output would show a reader, concatenated."""
    pieces = []
    for out in cell.get("outputs", []):
        pieces.append("".join(out.get("text", "")))
        for value in (out.get("data") or {}).values():
            pieces.append("".join(value) if isinstance(value, list) else str(value))
        pieces.append("\n".join(out.get("traceback", []) or []))
    return "\n".join(p for p in pieces if p)


def is_source_notebook(path: Path) -> bool:
    """Whether this file is the tracked source rather than a generated public copy.

    The source notebooks live in one known directory; a sanitised copy is written to a separate
    tree. The distinction matters because the sanitiser sets the flag True in the copy whatever
    the source declared, so the two are judged against different expectations.
    """
    try:
        return path.resolve().parent == (ROOT / "notebooks").resolve()
    except OSError:
        return False


def flag_findings(cells, *, policy: str | None, is_source: bool) -> list:
    """Whether the `PUBLIC_NOTEBOOK` declaration is the one this file should carry.

    ONE RULE, and `check_notebook_outputs.classify` is where it lives. The tracked SOURCE
    declares False and is the internal notebook: correct, and not a finding, but not cleared for
    release either — the copy is a separate file. A GENERATED copy declares True and is the only
    kind whose stored output may be released after review.
    """
    if policy is not None and policy not in FLAG_POLICIES:
        return [(f"public_flag={policy!r} in {SPEC.name} is not one of {list(FLAG_POLICIES)}",
                 None)]

    kind, why = classify(cells)
    if kind == INVALID:
        return [(f"{why}. The source declares PUBLIC_NOTEBOOK = False exactly once and a "
                 f"generated public copy declares True exactly once", None)]
    if is_source and kind == GENERATED:
        return [("PUBLIC_NOTEBOOK is set to True in the tracked source; the source declares "
                 "False and scripts/make_public_notebooks.py sets True in the copy alone",
                 None)]
    if not is_source and kind == INTERNAL:
        return [("PUBLIC_NOTEBOOK is set to False — this notebook is in internal mode and is "
                 "not cleared for public release; a generated public copy declares True",
                 None)]
    return []


def missing_allowlist_findings(path: Path, allowlists: dict) -> list:
    """A generated copy of a known notebook that the specification does not name.

    THE FLAG ALONE AUTHORISES NOTHING. Every one of the four known notebooks must carry an
    explicit `retain_output_cells` entry; an absent entry is an unfinished decision, and a copy
    made before that decision has nothing permitting any of its stored output.

    Takes the loaded map rather than reading the spec itself, so a caller can supply a synthetic
    one. Returns findings in `audit`'s shape.
    """
    if path.name not in PUBLIC_NOTEBOOKS or allowlists.get(path.name) is not None:
        return []
    cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
    if classify(cells)[0] != GENERATED:
        return []
    return [(f"no entry in {SPEC.name}; a generated public copy keeps output only in the cells "
             f"the spec names, and this notebook has none", None)]


def audit(path: Path, allowed: frozenset | None = None, *,
          policy: str | None = None, is_source: bool | None = None) -> list:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    findings = []

    # An output outside the allow-list is a finding whatever it contains. The allow-list is the
    # reviewed decision; a cell that is not on it was not reviewed.
    if allowed is not None:
        for i, cell in enumerate(cells):
            if cell.get("outputs") and cell.get("id") not in allowed:
                findings.append(("output in a cell the allow-list does not name", i))

    findings += flag_findings(
        cells, policy=policy,
        is_source=is_source_notebook(path) if is_source is None else is_source)

    counts, previous = [], 0
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        count = cell.get("execution_count")
        if cell.get("outputs") and count is None:
            findings.append(("a cell carries output with no execution count", i))
        if count is not None:
            counts.append((i, count))
            if count < previous:
                findings.append(("execution count goes backwards — the notebook was not run "
                                 "in order", i))
            previous = count

        for out in cell.get("outputs", []):
            if out.get("output_type") == "error" or out.get("traceback"):
                findings.append(("a saved error or traceback", i))

        text = output_text(cell)
        if not text:
            continue
        if ABSOLUTE_PATH.search(text):
            findings.append(("an absolute user path in stored output", i))
        if PRIVATE_FILE.search(text):
            findings.append(("a private or internal filename in stored output", i))
        if PERSON_LEVEL.search(text):
            findings.append(("a person-level column name in stored output", i))
        if EXACT_COUNTS.search(text):
            findings.append(("an exact-count column in stored output", i))
        if RECONSTRUCTABLE_COUNTS.search(text):
            findings.append(("a column a respondent count is recoverable from", i))
        rendered = text.count("\n")
        if rendered > MAX_RENDERED_ROWS:
            findings.append((f"a rendered output of {rendered} lines — larger than a table "
                             f"meant for a reader", i))

    if counts and not cells[0].get("outputs") and len(counts) < 2:
        findings.append(("almost nothing was executed", None))
    return findings


# Generated result tables belong under the working root, not in the tree. `.gitignore` is the
# structural defence; this reports one that got past it.
GENERATED_TABLE_DIRS = ("publication_outputs", "publication_candidates")


def stray_result_tables(root: Path) -> list:
    """Generated result CSVs sitting inside the repository, where none should be."""
    found = []
    for directory in GENERATED_TABLE_DIRS:
        base = root / directory
        if base.is_dir():
            found += sorted(p.relative_to(root) for p in base.rglob("*.csv"))
    return found


def main(argv):
    paths = [Path(a) for a in argv] or [ROOT / "notebooks" / n for n in PUBLIC_NOTEBOOKS]
    allowlists = load_allowlists()
    policies = load_flag_policies()
    findings_found = False

    stray = stray_result_tables(ROOT)
    if stray:
        findings_found = True
        print("  FINDINGS  generated result table(s) inside the repository")
        for name in stray:
            print(f"      {name}: candidate tables belong under the working root")
    for path in paths:
        try:
            findings = missing_allowlist_findings(path, allowlists)
            # Judged against the empty set where the spec names nothing, so its outputs are
            # reported too rather than passing unexamined.
            allowed = allowlists.get(path.name)
            findings += audit(path, frozenset() if findings else allowed,
                              policy=policies.get(path.name))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  UNREADABLE  {path.name}: {type(exc).__name__}")
            return 2
        if findings:
            findings_found = True
            print(f"  FINDINGS  {path.name}")
            for what, cell in findings:
                where = f"cell {cell}" if cell is not None else "notebook"
                print(f"      {where}: {what}")
        else:
            print(f"  no known pattern found  {path.name}")

    print(f"\n{DISCLAIMER}")
    return 1 if findings_found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
