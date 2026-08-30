#!/usr/bin/env python3
"""Promote two reviewed private selection records into the tracked specification.

THE ONLY THING THAT WRITES `spec/local_model_settings.csv`. No notebook, no test and no library
function may write it: a specification that an ordinary run could rewrite is not a specification,
and the whole point of fixing the configurations is that a run reads them rather than choosing
them. This script is the deliberate, reviewed step in between.

DRY RUN BY DEFAULT. Nothing is written without BOTH `--apply` and `--confirm`. The dry run prints
the complete per-cell diff for both cohorts and writes nothing at all.

    python scripts/promote_local_settings.py                    # dry run
    python scripts/promote_local_settings.py --apply --confirm  # writes

ONE FILE, ONE REPLACE. Both cohorts' 54 configurations go into a single CSV written through a
temporary sibling and one `os.replace`. Two files would need two replacements, and two
replacements are not one transaction: a failure between them would leave the MCS and YRBS halves
describing different selections, with nothing on disk to detect it.

NO RESULT IS PROMOTED. The private records carry each candidate's three seed-level CV AUCs, their
mean and their standard deviation. None of that is written here. The MCS values are MCS-derived
and need separate disclosure review; the YRBS values are not needed to state a configuration. The
tracked file carries the configurations and two provenance identifiers and nothing else.

TWO IDENTIFIERS, AND BOTH ARE NEEDED. `protocol_id` identifies the selection PROCEDURE;
`settings_digest` identifies the 54 SELECTED CONFIGURATIONS. Two different selections run under
the same procedure share the first and differ in the second, so a matching `protocol_id` alone is
not evidence that two sets of results are comparable.

BACKUP AND ROLLBACK. The existing tracked file is copied under the working root before it is
replaced, and the path is printed. Rolling back is copying that one file back. The backup goes to
the working root rather than into the repository because the project directory holds source and
nothing a run generates. There is no dependence on Git here — this project has no repository, and
promotion must not require one.

Exit 0 clean, 1 refused.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import models as PM  # noqa: E402
import regime_names as RN  # noqa: E402

BACKUP_SUBDIR = "spec_backups"


class Refused(RuntimeError):
    """The specification was not written, and the reason names what to fix."""


# --------------------------------------------------------------------------- reading

def _load(cohort: str) -> dict:
    """One cohort's private record, through the ordinary loader so every refusal applies."""
    try:
        return PM.load_selection_record(cohort=cohort)
    except FileNotFoundError as absent:
        raise Refused(
            f"{cohort.upper()}: {absent}. Run Notebook 02's selection section to produce it; "
            f"nothing here re-derives a selection.") from None
    except (ValueError, RuntimeError) as bad:
        raise Refused(f"{cohort.upper()}: {bad}") from None


def _check_pair(records: dict) -> None:
    """The two records must describe ONE procedure, or they are not one specification."""
    shared = ("preprocessing_version", "model_features", "development_seeds", "folds",
              "objective", "tie_rule", "candidate_pool_fingerprint", "protocol_id")
    for field in shared:
        values = {cohort: records[cohort]["payload"][field] for cohort in PM.COHORTS}
        if len({repr(v) for v in values.values()}) != 1:
            raise Refused(
                f"the two records disagree on {field}: "
                + "; ".join(f"{c}={v!r}" for c, v in values.items())
                + ". A pair of selections made under different procedures is not one "
                  "specification, and promoting them would put two protocols in one file.")

    live = PM.protocol_id()
    declared = records["mcs"]["payload"]["protocol_id"]
    if declared != live:
        raise Refused(
            f"the records declare protocol_id {declared!r}; the live procedure is {live!r}. "
            f"They were made under a different candidate pool, development-seed set, fold "
            f"count, objective or tie rule. Re-run Notebook 02's selection section under the "
            f"live procedure.")

    expected = len(RN.THRESHOLDS) * len(RN.FAMILIES)
    for cohort in PM.COHORTS:
        cells = records[cohort]["records"]
        if len(cells) != expected:
            raise Refused(
                f"{cohort.upper()} carries {len(cells)} cell(s); a complete specification has "
                f"{expected} ({len(RN.THRESHOLDS)} thresholds x {len(RN.FAMILIES)} families).")
        unselected = sorted(k for k, r in cells.items()
                            if r["status"] != PM.SELECTION_SELECTED)
        if unselected:
            raise Refused(
                f"{cohort.upper()}: {len(unselected)} cell(s) carry no selected configuration, "
                f"e.g. {unselected[:4]}. A fixed specification has to be complete; nothing here "
                f"substitutes a configuration for a cell the search could not settle.")


# --------------------------------------------------------------------------- the rows

def _rows(records: dict) -> list[dict]:
    """The 54 rows, in cohort x threshold x family order, with both provenance identifiers."""
    base = []
    for cohort in PM.COHORTS:
        cells = records[cohort]["records"]
        for threshold in RN.THRESHOLDS:
            for family in RN.FAMILIES:
                base.append({
                    "cohort": cohort,
                    "threshold": f">={int(threshold)}",
                    "family": str(family),
                    "settings": str(cells[(int(threshold), str(family))]["params"]),
                })
    digest = PM.settings_digest(
        (r["cohort"], r["threshold"], r["family"], r["settings"]) for r in base)
    protocol = PM.protocol_id()
    for row in base:
        row["protocol_id"] = protocol
        row["settings_digest"] = digest
    return base


def _existing() -> dict:
    """The tracked specification as `{(cohort, threshold, family): settings}`, or empty."""
    path = Path(config.LOCAL_MODEL_SETTINGS)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {(r["cohort"], r["threshold"], r["family"]): r["settings"]
                for r in csv.DictReader(handle)}


def _diff(rows: list[dict], existing: dict) -> list[tuple]:
    """Every cell, and whether it changed. Reported in full — a promotion is reviewed, not
    skimmed, and a cell that did NOT change is as much a part of the review as one that did."""
    out = []
    for row in rows:
        key = (row["cohort"], row["threshold"], row["family"])
        was = existing.get(key)
        out.append((key, was, row["settings"], was != row["settings"]))
    return out


# --------------------------------------------------------------------------- writing

def _backup(path: Path) -> Path | None:
    """Copy the current specification under the working root. Returns the backup path."""
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = config.work_path(BACKUP_SUBDIR) / f"{path.stem}.{stamp}{path.suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return destination


def _write(rows: list[dict], path: Path) -> None:
    """One temporary sibling, one `os.replace`. Either the whole file lands or none of it does."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(PM.SETTINGS_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def promote(*, apply: bool, confirm: bool) -> int:
    records = {cohort: _load(cohort) for cohort in PM.COHORTS}
    _check_pair(records)

    rows = _rows(records)
    existing = _existing()
    diff = _diff(rows, existing)
    changed = [entry for entry in diff if entry[3]]

    print(f"-- {'writing' if apply and confirm else 'dry run, nothing written'}")
    print(f"   protocol_id      {rows[0]['protocol_id']}")
    print(f"   settings_digest  {rows[0]['settings_digest']}")
    print(f"   cells            {len(rows)} ({len(PM.COHORTS)} cohorts x "
          f"{len(RN.THRESHOLDS)} thresholds x {len(RN.FAMILIES)} families)")
    print(f"   changed          {len(changed)} of {len(diff)}\n")
    for (cohort, threshold, family), was, now, differs in diff:
        marker = "*" if differs else " "
        print(f" {marker} {cohort:<4} {threshold:<4} {family:<9} {now}")
        if differs and was is not None:
            print(f"      was: {was}")
    for cohort in PM.COHORTS:
        tallies = PM.selection_coverage(records[cohort]["records"])["tie_broken_on"]
        print(f"\n   {cohort.upper()} tie-breaking: "
              + ", ".join(f"{basis}={count}" for basis, count in tallies.items()))

    if not apply:
        print("\n   nothing was written. Re-run with --apply --confirm to promote.")
        return 0
    if not confirm:
        raise Refused(
            "--apply was given without --confirm. Promotion replaces the configuration every "
            "model in the analysis is fitted under, so it takes both flags: review the diff "
            "above, then re-run with --apply --confirm.")

    path = Path(config.LOCAL_MODEL_SETTINGS)
    backup = _backup(path)
    _write(rows, path)

    reloaded = PM.load_local_settings(path)
    expected = len(RN.THRESHOLDS) * len(RN.FAMILIES)
    for cohort in PM.COHORTS:
        if len(reloaded[cohort]) != expected:
            raise Refused(
                f"the written specification reloads {len(reloaded[cohort])} {cohort.upper()} "
                f"cell(s), not {expected}. It has been left in place for inspection; the backup "
                f"is at {backup}.")
        for (threshold, family), params in reloaded[cohort].items():
            want = records[cohort]["records"][(threshold, family)]["params"]
            if not PM._cfg_equal(params, want):
                raise Refused(
                    f"the written specification reloads {cohort} >={threshold} {family} as "
                    f"{params}, but the record selected {want}. The backup is at {backup}.")

    print(f"\n   wrote     {path.name}")
    print(f"   backup    {backup if backup else '(none — no previous specification)'}")
    print("\n   Every result produced before this promotion was fitted under a different "
          "configuration.\n   Re-run notebooks 02, 03 and 04 in order, and do not mix outputs "
          "from either side of it.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true",
                        help="write. Without it nothing is written and this is a dry run.")
    parser.add_argument("--confirm", action="store_true",
                        help="required alongside --apply; confirms the diff has been reviewed")
    args = parser.parse_args(argv)
    return promote(apply=args.apply, confirm=args.confirm)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refused as refusal:
        print(f"REFUSED  {refusal}", file=sys.stderr)
        sys.exit(1)
