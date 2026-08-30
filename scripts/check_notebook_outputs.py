#!/usr/bin/env python3
"""
scripts/check_notebook_outputs.py — assert tracked notebooks carry no saved outputs.

The notebooks are tracked (`.gitignore`: `!notebooks/*.ipynb`)
because a clone is unreadable without them. An executed notebook carries its
output cells, and an output cell is a place an MCS row can hide — a `head()`
that was never meant to survive, a traceback quoting a slice, a repr of a frame.
MCS Sweep 6 is UKDS Safeguarded Tier 1a and git history does not forget.

So the rule is: notebooks are committed with their outputs stripped.

THREE KINDS OF FILE, decided from the `PUBLIC_NOTEBOOK` declaration alone.

  internal   exactly one `= False`. The tracked source, and what the final run executes. Its
             stored output is the internal record and is there to be read, so it is not
             reported — but it is NOT publication-ready, and this says so.
  generated  exactly one `= True`. A copy made by `scripts/make_public_notebooks.py`, which may
             keep output only in the cells `spec/public_notebook_cells.json` names.
  invalid    no declaration, or more than one. Reported.

Any other notebook is cleared, with no exception. Passing both checks is still not disclosure
clearance: an MCS-derived table needs researcher, institutional and UK Data Service review
before it is committed.

Reports counts and cell indices only. It never reads or prints cell output
content, because doing so is the very disclosure it exists to prevent.

Usage:
    python scripts/check_notebook_outputs.py [path ...]      # default: notebooks/*.ipynb
Exit status 0 clean, 1 dirty, 2 unreadable.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOB = "notebooks/*.ipynb"
SPEC = ROOT / "spec" / "public_notebook_cells.json"

# The notebooks that may carry public outputs at all. Matched on the filename so a sanitised
# copy under another directory is recognised as the same notebook.
PUBLIC_NOTEBOOKS = ("01_data_and_features.ipynb", "02_models_and_transfer.ipynb",
                    "03_evaluation_and_robustness.ipynb", "04_explorations.ipynb")
PUBLIC_FLAG = re.compile(r"^\s*PUBLIC_NOTEBOOK\s*=\s*True\s*$", re.M)

# Any whole-line declaration, either value. A mention inside a comment, a markdown cell or a
# longer expression is not a declaration.
FLAG_DECLARATION = re.compile(r"^\s*PUBLIC_NOTEBOOK\s*=\s*(True|False)\s*$", re.M)

# What a notebook file is, decided from its declaration alone.
INTERNAL = "internal"       # the tracked source: exactly one `= False`
GENERATED = "generated"     # a copy made by make_public_notebooks: exactly one `= True`
INVALID = "invalid"         # missing, duplicated, or both values present

DISCLAIMER = ("Structural check only. It does not establish disclosure clearance, which "
              "requires researcher, institutional and UK Data Service review.")


def load_allowlists(path: Path = SPEC) -> dict:
    """Notebook filename -> the cell ids that may keep output.

    A generated public copy may keep output ONLY in the cells named here, so the flag alone
    authorises nothing: `PUBLIC_NOTEBOOK = True` is necessary and never sufficient.

    EVERY KNOWN NOTEBOOK MUST HAVE AN ENTRY. A generated copy of one of the four with no entry
    is reported, because there is then nothing that authorises any of its output — an absent
    entry is an unfinished decision, not permission.
    """
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: frozenset(entry.get("retain_output_cells", []))
            for name, entry in payload.get("notebooks", {}).items()}


FLIP = "flip"
FLAG_POLICIES = (FLIP,)


def load_flag_policies(path: Path = SPEC) -> dict:
    """Notebook filename -> its `public_flag` policy.

    ONE POLICY, "flip". The tracked source declares `PUBLIC_NOTEBOOK = False` and is what the
    final internal run executes; `make_public_notebooks` replaces that one declaration with True
    in a separate copy. A notebook the spec does not list has no policy.

    An unrecognised value is returned as it stands rather than dropped or raised on, so the
    caller reports it as a finding instead of the checker failing to run. `already_true` is
    withdrawn and now reads as unrecognised.
    """
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: entry.get("public_flag", FLIP)
            for name, entry in payload.get("notebooks", {}).items()}


def flag_declarations(cells) -> list:
    """Every whole-line `PUBLIC_NOTEBOOK` declaration in the code cells, in order."""
    return [match.group(1) for cell in cells if cell.get("cell_type") == "code"
            for match in FLAG_DECLARATION.finditer("".join(cell.get("source", [])))]


def classify(cells) -> tuple:
    """What this notebook file is, and why if it is neither kind.

    THE ONE PLACE THE RULE LIVES. Both checkers and the copying script read it here rather than
    each deciding for itself what a declaration means.
    """
    declarations = flag_declarations(cells)
    if not declarations:
        return INVALID, "no PUBLIC_NOTEBOOK declaration in any code cell"
    if len(declarations) > 1:
        return INVALID, (f"{len(declarations)} PUBLIC_NOTEBOOK declarations "
                         f"({', '.join(declarations)}); exactly one is required")
    return (INTERNAL, "") if declarations[0] == "False" else (GENERATED, "")


def in_public_mode(path: Path) -> bool:
    """Whether this notebook file is a generated public copy — one `PUBLIC_NOTEBOOK = True`."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    return classify(nb.get("cells", []))[0] == GENERATED


def execution_problems(cells) -> list[str]:
    """Why the execution counts do not describe one Restart and Run All, if they do not.

    A public copy keeps the counts of the run that produced it. They must be complete over the
    non-blank code cells and strictly increasing; a gap says the notebook was run piecemeal,
    and a repeat or a step backwards says cells were re-run out of order. Either way the
    outputs beside them do not describe a single execution.
    """
    executable = [(i, c) for i, c in enumerate(cells)
                  if c.get("cell_type") == "code" and "".join(c.get("source", [])).strip()]
    counted = [(i, c["execution_count"]) for i, c in executable
               if c.get("execution_count") is not None]
    uncounted = [i for i, c in executable if c.get("execution_count") is None]
    if not counted:
        return []                       # not executed at all; the cleared-notebook rule applies
    problems = []
    if uncounted:
        problems.append(f"{len(uncounted)} executable cell(s) carry no execution count, "
                        f"first at index {uncounted[0]}")
    for (index, previous), (_, current) in zip(counted, counted[1:]):
        if current <= previous:
            problems.append(f"execution count {current} follows {previous} at index {index}")
            break
    return problems


def audit(path: Path, allowed: frozenset | None = None) -> tuple[list[int], list[int]]:
    """Return (cells carrying disallowed output, cells carrying an execution count).

    With `allowed` given, a cell whose id is in it may keep its output and is not reported.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    with_outputs, with_counts = [], []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("outputs") and not (allowed and cell.get("id") in allowed):
            with_outputs.append(i)
        if cell.get("execution_count") is not None:
            with_counts.append(i)
    return with_outputs, with_counts


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or sorted(ROOT.glob(DEFAULT_GLOB))
    if not paths:
        print(f"  no notebooks matched {DEFAULT_GLOB}")
        return 0

    allowlists = load_allowlists()
    dirty = False
    for path in paths:
        try:
            allowed = allowlists.get(path.name)
            cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
            kind, why = classify(cells)
            public = path.name in PUBLIC_NOTEBOOKS and kind == GENERATED
            # THE ALLOW-LIST ONLY RELAXES ANYTHING IN A GENERATED COPY. In the internal source
            # every output is reported, listed cell or not, so a notebook that has not been
            # sanitised cannot keep an output by being named in the spec.
            #
            # A KNOWN NOTEBOOK WITH NO ENTRY IS JUDGED AGAINST THE EMPTY SET, not waved through:
            # nothing authorises its output, so every cell carrying one is reported.
            enforce = allowed if allowed is not None else frozenset()
            with_outputs, with_counts = audit(path, enforce if public else None)
            counts = execution_problems(cells)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  UNREADABLE  {path}: {type(exc).__name__}")
            return 2

        listed = "" if allowed is None else f" [{len(allowed)} allow-listed]"
        # The two new branches apply to the four known notebooks only. Anything else falls
        # through to the strict rule below, where any stored output is reported.
        known = path.name in PUBLIC_NOTEBOOKS
        if known and kind == INVALID:
            dirty = True
            print(f"  DIRTY  {path}{listed} — {why}")
            print("           the source declares PUBLIC_NOTEBOOK = False exactly once; a "
                  "generated copy declares True exactly once")
        elif known and kind == INTERNAL:
            # NOT PUBLICATION-READY, and said so. Its stored output is the internal record of
            # the run and is there to be read; the public copy is a separate file.
            note = (f" — carries output in {len(with_outputs)} cell(s)"
                    if with_outputs else " — no stored output")
            print(f"  internal  {path}{listed}{note}")
            print("           internal source, for inspection. NOT publication-ready: make a "
                  "public copy with scripts/make_public_notebooks.py")
            if counts:
                dirty = True
                print("           the execution counts do not describe one Restart and Run All")
                for problem in counts:
                    print(f"           {problem}")
        elif public and allowed is None:
            # An unfinished decision, not permission. Reported before anything else about the
            # file, because without an entry there is nothing to judge its output against.
            dirty = True
            print(f"  DIRTY  {path} — no entry in {SPEC.name}")
            print(f"           a generated public copy keeps output only in the cells the spec "
                  f"names, and this notebook has none; classify it before copying")
            if with_outputs:
                print(f"           {len(with_outputs)} cell(s) carry output, index "
                      f"{with_outputs[:8]}{' …' if len(with_outputs) > 8 else ''}")
        elif public and counts:
            dirty = True
            print(f"  DIRTY  {path}{listed} — the execution counts do not describe one "
                  f"Restart and Run All")
            for problem in counts:
                print(f"           {problem}")
        elif public and not with_outputs:
            print(f"  public  {path}{listed}  — outputs confined to the allow-list; run "
                  f"scripts/check_public_outputs.py and review before committing")
        elif public:
            dirty = True
            print(f"  DIRTY  {path}{listed}")
            print(f"           {len(with_outputs)} cell(s) carry output that the allow-list "
                  f"does not permit, index "
                  f"{with_outputs[:8]}{' …' if len(with_outputs) > 8 else ''}")
        elif with_outputs or with_counts:
            dirty = True
            print(f"  DIRTY  {path}")
            if with_outputs:
                print(f"           {len(with_outputs)} cell(s) carry outputs, "
                      f"index {with_outputs[:8]}{' …' if len(with_outputs) > 8 else ''}")
            if with_counts:
                print(f"           {len(with_counts)} cell(s) carry an execution count, "
                      f"index {with_counts[:8]}{' …' if len(with_counts) > 8 else ''}")
        else:
            print(f"  clean  {path}")

    if dirty:
        print("\n  Strip them before committing:")
        print("      jupyter nbconvert --clear-output --inplace notebooks/*.ipynb")
        print("  A sanitised public copy is made instead by "
              "scripts/make_public_notebooks.py,")
        print("  which keeps output only in the cells spec/public_notebook_cells.json names.")
        print(f"\n{DISCLAIMER}")
        return 1
    print(f"-- {len(paths)} notebook(s) pass")
    print(f"\n{DISCLAIMER}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
