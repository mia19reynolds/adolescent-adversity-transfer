#!/usr/bin/env python3
"""Copy executed notebooks into a destination tree, keeping only allow-listed outputs.

THE SOURCE IS NEVER MODIFIED. An executed notebook is the internal record of a run: every
suppressed diagnostic, reconciliation and count stays inspectable in it. This reads it, writes
a separate sanitised copy, and touches the original in no way at all. There is no in-place mode
and no `--clear`; a workflow that cleared the only executed copy is the thing this exists to
replace.

WHAT A SANITISED COPY IS. The same notebook with visible output removed from every cell that
`spec/public_notebook_cells.json` does not name, `PUBLIC_NOTEBOOK` set to True, and everything
else — cell ids, order, source, metadata and execution counts — untouched.

ONE FLAG POLICY, "flip", for every notebook. The tracked source declares
`PUBLIC_NOTEBOOK = False` and is what the final internal run executes; the copy declares True.
A source that already declares True is REFUSED: it is either a generated copy being fed back in
or a source that was edited by hand, and in both cases the outputs beside the flag were not
produced by the internal run this tool is meant to sanitise.

THE FLAG IS PRESENTATION ONLY, which is what makes the copy trustworthy. Both modes run the
same calculations, fits, validations and writers, so the scientific files a run produces do not
depend on it. Only what is displayed or printed does.

EXECUTION COUNTS ARE KEPT, AND CHECKED. A public copy carries the counts of the run that
produced it, so a reader can see the notebook ran start to finish. They must be complete and
strictly increasing, which is what Restart and Run All produces; a gap or a repeat means the
notebook was run piecemeal and the outputs beside it do not describe one coherent execution.
Counts are never cleared selectively — clearing them only where output was removed would hide
exactly that.

DEFAULT IS A DRY RUN. Nothing is written without `--apply`. The dry run names notebooks, cell
ids and retain-or-clear decisions, and never an output value.

PASSING THIS IS NOT DISCLOSURE CLEARANCE. It is a structural check. An MCS-derived aggregate
needs researcher, institutional and UK Data Service review before it leaves the approved
environment, and notebook 04's retained cells are MCS-evaluated throughout.

Usage:
    python scripts/make_public_notebooks.py --source SRC --dest DEST [--apply] [--overwrite]

`SRC` is a notebook file or a directory holding `notebooks/`. Exit 0 clean, 1 refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "spec" / "public_notebook_cells.json"
MANIFEST_NAME = "public_notebooks_manifest.json"

PUBLIC_FALSE = re.compile(r"^(\s*)PUBLIC_NOTEBOOK\s*=\s*False\s*$", re.M)
PUBLIC_TRUE = re.compile(r"^\s*PUBLIC_NOTEBOOK\s*=\s*True\s*$", re.M)

NOT_CLEARANCE = ("Structural check only. It does not establish disclosure clearance, which "
                 "requires researcher, institutional and UK Data Service review.")


class Refused(RuntimeError):
    """The copy was not made, and the reason names what to fix."""


# --------------------------------------------------------------------------- the spec

FLIP = "flip"
FLAG_POLICIES = (FLIP,)


def load_allowlists(path: Path = SPEC) -> dict[str, frozenset[str]]:
    """Notebook filename -> the cell ids that may keep output. Absent means unlisted."""
    if not path.exists():
        raise Refused(f"{path.name} is missing; it is the one place the allow-list lives")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: frozenset(entry.get("retain_output_cells", []))
            for name, entry in payload.get("notebooks", {}).items()}


def load_flag_policies(path: Path = SPEC) -> dict[str, str]:
    """Notebook filename -> its `public_flag` policy. Defaults to flipping."""
    if not path.exists():
        raise Refused(f"{path.name} is missing; it is the one place the allow-list lives")
    payload = json.loads(path.read_text(encoding="utf-8"))
    policies = {}
    for name, entry in payload.get("notebooks", {}).items():
        policy = entry.get("public_flag", FLIP)
        if policy not in FLAG_POLICIES:
            raise Refused(f"{name}: public_flag={policy!r} is not one of {list(FLAG_POLICIES)}")
        policies[name] = policy
    return policies


# --------------------------------------------------------------------------- reading

def _text(cell) -> str:
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def _executable(cell) -> bool:
    """A code cell with something in it. A blank cell may legitimately carry no count."""
    return cell.get("cell_type") == "code" and _text(cell).strip() != ""


def _has_error(cell) -> bool:
    return any(out.get("output_type") == "error" or out.get("traceback")
               for out in cell.get("outputs", []))


def _check_execution(cells, name: str) -> None:
    """Complete and strictly increasing, or a refusal naming which of the two failed."""
    executable = [(i, c) for i, c in enumerate(cells) if _executable(c)]
    if not executable:
        raise Refused(f"{name}: no executable code cell carries anything to sanitise")

    counted = [(i, c["execution_count"]) for i, c in executable
               if c.get("execution_count") is not None]
    uncounted = [i for i, c in executable if c.get("execution_count") is None]

    if not counted:
        raise Refused(
            f"{name}: no code cell carries an execution count, so this notebook has not been "
            f"executed. A public copy is made FROM a run; there is nothing here to preserve.")
    if uncounted:
        raise Refused(
            f"{name}: {len(counted)} code cell(s) carry an execution count and "
            f"{len(uncounted)} do not (first at index {uncounted[0]}). A partial execution is "
            f"refused: the outputs that exist do not describe one run of the whole notebook.")

    counts = [n for _, n in counted]
    for (index, previous), (_, current) in zip(counted, counted[1:]):
        if current <= previous:
            raise Refused(
                f"{name}: execution count {current} follows {previous} at cell index {index}. "
                f"Counts must increase strictly, as Restart and Run All produces; a repeat or "
                f"a step backwards means cells were re-run out of order.")
    if len(set(counts)) != len(counts):
        raise Refused(f"{name}: execution counts repeat")


def _check_allowlisted_cells(cells, name: str, allowed: frozenset[str]) -> None:
    """Every allow-listed id occurs exactly once, and no retained cell carries an error."""
    problems = []
    for cell_id in sorted(allowed):
        found = [c for c in cells if c.get("id") == cell_id]
        if len(found) != 1:
            problems.append(f"{cell_id}: found {len(found)}, expected 1")
        elif _has_error(found[0]):
            problems.append(f"{cell_id}: carries a saved error or traceback")
    if problems:
        raise Refused(f"{name}: the allow-list does not match the notebook:\n    "
                      + "\n    ".join(problems))


def _check_no_retained_error(cells, name: str, allowed: frozenset[str]) -> None:
    """No cell that KEEPS its output may carry an error, allow-listed or not."""
    bad = [c.get("id") for c in cells
           if c.get("id") in allowed and _has_error(c)]
    if bad:
        raise Refused(f"{name}: retained cell(s) {bad} carry an error or traceback")


# --------------------------------------------------------------------------- writing

def _check_source_flag(cells, name: str) -> None:
    """The source must be an internal notebook: exactly one `PUBLIC_NOTEBOOK = False`.

    Checked BEFORE anything is replaced, so the refusal names what is wrong rather than
    reporting a replacement count. A source declaring True is either a generated copy being fed
    back in or a hand-edited source; either way its outputs were not produced by the internal
    run this tool sanitises.
    """
    code = [c for c in cells if c.get("cell_type") == "code"]
    declared_false = sum(len(PUBLIC_FALSE.findall(_text(c))) for c in code)
    declared_true = sum(len(PUBLIC_TRUE.findall(_text(c))) for c in code)

    if declared_true and not declared_false:
        raise Refused(
            f"{name}: declares `PUBLIC_NOTEBOOK = True`, so this is a generated public copy or "
            f"a hand-edited source, not the internal notebook. The tracked source declares "
            f"False; sanitise that instead.")
    if declared_true:
        raise Refused(
            f"{name}: declares both `PUBLIC_NOTEBOOK = False` and `= True`. Exactly one False "
            f"declaration is required and no True declaration.")
    if declared_false != 1:
        raise Refused(
            f"{name}: declares `PUBLIC_NOTEBOOK = False` {declared_false} time(s), expected "
            f"exactly once. The internal notebook declares it False and the public copy "
            f"declares it True; anything else means the flag is missing or duplicated.")


def _set_public_flag(cells, name: str) -> None:
    """Replace the source's one False with True — IN THE COPY, which is what `cells` is."""
    replacements = 0
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        new_text, n = PUBLIC_FALSE.subn(r"\1PUBLIC_NOTEBOOK = True", _text(cell))
        if n:
            replacements += n
            cell["source"] = new_text.splitlines(keepends=True)
    if replacements != 1:
        raise Refused(
            f"{name}: `PUBLIC_NOTEBOOK = False` was replaced {replacements} time(s), expected "
            f"exactly once after the source check passed. Nothing was written.")


def sanitise(payload: dict, allowed: frozenset[str], name: str,
             policy: str = FLIP) -> tuple[dict, list, list]:
    """The destination notebook, the ids that keep output, and the ids that lose it.

    `payload` is the parsed COPY; the source file on disk is never opened for writing.
    """
    if policy != FLIP:
        raise Refused(f"{name}: public_flag={policy!r} is not {FLIP!r}. The mixed-policy rule "
                      f"is withdrawn; every notebook declares False in source.")
    cells = payload["cells"]
    retained, cleared = [], []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        cell_id = cell.get("id")
        if cell_id in allowed:
            retained.append(cell_id)
            continue
        if cell.get("outputs"):
            cleared.append(cell_id)
        cell["outputs"] = []
        # The execution count STAYS. Clearing it only where output was removed would disguise
        # which cells had been sanitised.

    _set_public_flag(cells, name)
    return payload, retained, cleared


def _manifest_entry(source: Path, retained, cleared, had_error: bool) -> dict:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "notebook": source.name,
        "retained_output_cells": sorted(retained),
        "cleared_output_cells": sorted(cleared),
        "errors_found": had_error,
        "source_sha256": digest,
        "source_modified_utc": datetime.fromtimestamp(
            source.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "preprocessing_version": _preprocessing_version(),
    }


def _preprocessing_version() -> str | None:
    """Read the version out of `src/data.py` as text.

    Importing `data` would import `config`, which reads a layout file and resolves configured
    roots. This records a version, so it reads a line rather than starting the pipeline.
    """
    path = ROOT / "src" / "data.py"
    if not path.exists():
        return None
    found = re.search(r'^PREPROCESSING_VERSION\s*=\s*"([^"]+)"', path.read_text("utf-8"), re.M)
    return found.group(1) if found else None


# --------------------------------------------------------------------------- the run

def _notebooks(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    folder = source / "notebooks" if (source / "notebooks").is_dir() else source
    found = sorted(p for p in folder.glob("*.ipynb"))
    if not found:
        raise Refused(f"no .ipynb found under {folder}")
    return found


def process(source: Path, destination: Path, *, apply: bool, overwrite: bool) -> list[dict]:
    allowlists = load_allowlists()
    policies = load_flag_policies()
    entries = []
    for notebook in _notebooks(source):
        name = notebook.name
        if name not in allowlists:
            print(f"  skipped   {name}  (not in {SPEC.name})")
            continue
        allowed = allowlists[name]
        policy = policies.get(name, FLIP)
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        cells = payload["cells"]

        _check_source_flag(cells, name)
        _check_allowlisted_cells(cells, name, allowed)
        _check_execution(cells, name)
        _check_no_retained_error(cells, name, allowed)

        target = destination / name
        if target.resolve() == notebook.resolve():
            raise Refused(f"{name}: source and destination are the same file")
        if target.exists() and not overwrite:
            raise Refused(f"{target} already exists; pass --overwrite to replace it")

        payload, retained, cleared = sanitise(payload, allowed, name, policy)
        entry = _manifest_entry(notebook, retained, cleared, had_error=False)
        entry["public_flag"] = policy
        entries.append(entry)

        print(f"  {name}  (public_flag: {policy})")
        print(f"    retain  {len(retained):>3}  {', '.join(sorted(retained)) or '-'}")
        print(f"    clear   {len(cleared):>3}  {', '.join(sorted(cleared)) or '-'}")
        if apply:
            destination.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    if apply and entries:
        (destination / MANIFEST_NAME).write_text(
            json.dumps({"generated_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"), "notebooks": entries,
                "not_clearance": NOT_CLEARANCE}, indent=1) + "\n", encoding="utf-8")
    return entries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True,
                        help="an executed notebook, or a directory holding notebooks/")
    parser.add_argument("--dest", required=True, help="where the sanitised copies are written")
    parser.add_argument("--apply", action="store_true",
                        help="write. Without it nothing is written and this is a dry run.")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace a destination notebook that already exists")
    args = parser.parse_args(argv)

    source, destination = Path(args.source).resolve(), Path(args.dest).resolve()
    if source == destination:
        raise Refused("source and destination resolve to the same path; a sanitised copy is a "
                      "SEPARATE tree, and writing into the source would destroy the internal "
                      "record of the run")
    if source.is_dir() and destination.is_relative_to(source):
        raise Refused(f"{destination} is inside {source}; the public copy belongs in a "
                      f"separate tree")

    print(f"-- {'writing' if args.apply else 'dry run, nothing written'}")
    entries = process(source, destination, apply=args.apply, overwrite=args.overwrite)
    if not entries:
        print("   no listed notebook processed")
        return 0
    if args.apply:
        print(f"   manifest  {MANIFEST_NAME}")
        print("   now run:  python scripts/check_notebook_outputs.py "
              f"{destination}/*.ipynb")
        print("             python scripts/check_public_outputs.py "
              f"{destination}/*.ipynb")
    print(f"\n{NOT_CLEARANCE}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refused as refusal:
        print(f"REFUSED  {refusal}", file=sys.stderr)
        sys.exit(1)
