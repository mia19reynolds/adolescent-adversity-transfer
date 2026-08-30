"""Paths for the pipeline, and the rules about where they may point.

Two roots are configured outside the repository, and neither has a default:

    export MCS_DATA_DIR=/path/to/mcs_secure       restricted data, and any fitted model
    export THESIS_WORK_DIR=/path/to/thesis_work   YRBS raw, working tables, checkpoints

Either may instead be the first line of `.mcs_data_dir` or `.thesis_work_dir` in the
repository root. Both are git-ignored and hold a path, never data; they exist because VS Code
and JupyterLab do not pass a shell environment to a kernel. README covers the setup.

A root that resolves inside the repository, at a home directory, or at a filesystem root is
refused. MCS Sweep 6 is UKDS Safeguarded Tier 1a, so no row-level material may sit under
version control, and the working tree is meant to hold only what Git tracks. Symlinks resolve
before the check, and a sibling directory whose name merely starts with the repository's is
outside it.

There are two ways to ask for a path, because there are two kinds:

    config.MCS_FEATURES              a location declared in `paths.json`
    config.work_path("runs", rid)    a location assembled by the caller

The first covers the fixed artefacts. The second covers paths known only at run time —
`inputs.save_table` takes its subdirectory and filename from whoever calls it. Both resolve the root
and apply the rules above on every access, so nothing in the pipeline holds a path that skipped
them. Reading `paths.json` and joining the components yourself does skip them.
"""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Both roots resolve the same way, so they resolve through the same function.
_ROOTS = {
    "MCS_DATA_DIR":    PROJECT_ROOT / ".mcs_data_dir",
    "THESIS_WORK_DIR": PROJECT_ROOT / ".thesis_work_dir",
}


def _root(var: str, *, name: str) -> Path:
    """The configured root for `var`, refused if it points somewhere it should not."""
    raw = os.environ.get(var, "").strip()
    dotfile = _ROOTS[var]
    if not raw and dotfile.is_file():
        lines = [line.strip() for line in dotfile.read_text().splitlines() if line.strip()]
        raw = os.path.expanduser(lines[0]) if lines else ""
    if not raw:
        raise RuntimeError(
            f"{var} is not set, and {name} needs it. Export it, or write the path as the "
            f"first line of {dotfile.name}. There is no default — see README.")

    resolved = Path(raw).expanduser().resolve()
    if resolved == PROJECT_ROOT or resolved.is_relative_to(PROJECT_ROOT):
        raise RuntimeError(
            f"{var} points inside the repository:\n    {resolved}\nIt must be a directory "
            f"outside this clone — see README.")
    if resolved == Path.home() or resolved == Path(resolved.anchor):
        raise RuntimeError(
            f"{var} resolves to {resolved}, which is your home directory or the filesystem "
            f"root. The pipeline writes and cleans up inside this path, so give it one of "
            f"its own — see README.")
    return resolved


# An optional local read gate. Where `src/_mcs_gate.py` is present it requires MCS_READ_OK=1
# per invocation before any MCS path is handed out, which keeps a process that merely
# inherited a shell environment from reaching restricted data. It is git-ignored, so a clone
# does not have one and `GATE_ACTIVE` is False; the placement rules above are unaffected
# either way. Only ImportError is caught, so a fault in the gate surfaces rather than
# disabling it.
try:
    from _mcs_gate import require as _acknowledge_mcs
    GATE_ACTIVE = True
except ImportError:
    GATE_ACTIVE = False

    def _acknowledge_mcs(name: str) -> None:
        """No local gate installed."""


# The layout, read once. `paths.json` lists components under each root and nothing else: `mcs`
# entries sit under $MCS_DATA_DIR and are Tier 1a to the same degree the cohort is, `work`
# entries under $THESIS_WORK_DIR. A malformed file is fatal rather than defaulted, because
# guessing a layout would put restricted data somewhere nobody declared.
#
# Every entry has a reader, apart from the `MODELS_*` block, which docs/MODEL_STORAGE.md
# reserves as the only legal destination for the day fitted models are persisted. A key no
# caller names is a second declaration of the layout rather than a record of it.
with open(Path(__file__).with_name("paths.json")) as _fh:
    _LAYOUT = json.load(_fh)

_MCS_PATHS = {k: tuple(v) for k, v in _LAYOUT["mcs"].items()}
_WORK_PATHS = {k: tuple(v) for k, v in _LAYOUT["work"].items()}


def work_path(*parts: str, create: bool = False) -> Path:
    """A path under the working root; with no parts, the root itself.

    For locations the caller assembles at run time. Declared artefacts come from
    `paths.json` by name instead. `create=True` makes the parent directory.
    """
    path = _root("THESIS_WORK_DIR", name="the working root").joinpath(*parts)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def __getattr__(name):
    """Resolve a declared path on access, so importing this module needs no credentials."""
    if name in _WORK_PATHS:
        return _root("THESIS_WORK_DIR", name=f"config.{name}").joinpath(*_WORK_PATHS[name])
    if name in _MCS_PATHS:
        _acknowledge_mcs(f"config.{name}")
        return _root("MCS_DATA_DIR", name=f"config.{name}").joinpath(*_MCS_PATHS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_MCS_PATHS) + list(_WORK_PATHS))


# The tracked public output root, holding `paper/` and `supplementary/` and nothing else. No
# notebook writes a table here — the published CSVs are curated by hand from a finished run,
# and the only writers into the repository are `paper.save_figure` and `paper.save_fragment`.
# It is not a working root: working artifacts belong under $THESIS_WORK_DIR, or the project
# directory stops holding only what Git tracks.
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# The fixed model settings for BOTH cohorts: one combined tracked specification carrying 54
# rows — nine families at three thresholds, in MCS and in YRBS. Each cohort's 27 configurations
# were chosen by the same consensus procedure (three development seeds, five-fold stratified
# inner CV, AUC, mean-of-seed-means with a standard-deviation then pool-order tie rule) inside
# that cohort's own outer-training partitions. See `models.consensus_select`.
#
# ONE FILE, NOT TWO, because promotion is then a single atomic replace. Two files would need two
# replacements, and two replacements are not one transaction: a failure between them would leave
# the MCS and YRBS halves describing different selections with nothing to detect it.
#
# IT CARRIES NO RESULT. The cross-validated scores behind the selection stay in the private
# records under $THESIS_WORK_DIR; the MCS ones are MCS-derived and need separate disclosure
# review. This file holds configurations and two provenance identifiers and nothing else.
#
# It is written ONLY by scripts/promote_local_settings.py, after a reviewed run of Notebook 02's
# selection section.
# No notebook and no test may write it. Editing it changes the configuration every model in the
# battery is fitted under, so a run made before an edit and one made after are not comparable —
# which is what the `settings_digest` on every generated artefact exists to detect.
LOCAL_MODEL_SETTINGS = PROJECT_ROOT / "spec" / "local_model_settings.csv"
