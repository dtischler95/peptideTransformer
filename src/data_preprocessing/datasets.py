"""Single source of truth for the project's datasets and how to find their files.

Three call sites used to each hard-code the same per-task globs: the random split
writer (``data_splitter.py``), the cluster split writer (``cluster_data_splitter.py``),
and the classical-ML auto-discovery (``machine_learning/train_models.py``). That
duplication is exactly what let one copy drift and re-split its own generated outputs.
Keep the "what is a base file for task X" knowledge here and import it everywhere.

A *base file* is an original, de-duplicated dataset file, never a generated
``*_train.csv`` / ``*_val.csv`` / ``*_test.csv`` split.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = _REPO_ROOT / "data"

# Canonical task -> dataset spec.
#   dir        : default data sub-directory (under data/)
#   target     : target column name
#   base_globs : glob(s) matching ONLY base files, never generated splits
DATASETS: dict[str, dict] = {
    "cls": {
        "dir": "hemo_train",
        "target": "label",
        "base_globs": ["*unvoted.csv", "*data.csv"],
    },
    "regression": {
        "dir": "regression_data",
        "target": "mic_log10",
        "base_globs": ["*regression.csv"],
    },
    "gram": {
        "dir": "gram",
        "target": "label",
        "base_globs": ["*dataset.csv"],
    },
}

# Substring of a data directory name -> task. Lets a caller infer the task from a
# directory alone, and works for the parallel "*_cluster" directories too
# (e.g. "regression_data_cluster" still contains "regression").
_DIR_HINTS: list[tuple[str, str]] = [
    ("hemo", "cls"),
    ("regression", "regression"),
    ("gram", "gram"),
]


def _spec(task: str) -> dict:
    try:
        return DATASETS[task]
    except KeyError:
        raise ValueError(f"unknown task {task!r}, expected one of {list(DATASETS)}") from None


def default_dir(task: str) -> Path:
    """Default data directory for a task (absolute, repo-rooted)."""
    return _DATA_ROOT / _spec(task)["dir"]


def target_col(task: str) -> str:
    """Target column name for a task."""
    return _spec(task)["target"]


def base_files(task: str, data_dir: str | Path | None = None) -> list[Path]:
    """Sorted base dataset files for a task, never the generated _train/_val/_test splits.

    Uses ``data_dir`` if given (e.g. a "*_cluster" directory), else the task default.
    """
    directory = Path(data_dir) if data_dir else default_dir(task)
    found: list[Path] = []
    for pattern in _spec(task)["base_globs"]:
        found.extend(directory.glob(pattern))
    return sorted(found)


def organism_slug(name: str | Path) -> str:
    """Short organism/dataset key from a filename stem (first two underscore-separated parts)."""
    return "_".join(Path(name).stem.split("_")[:2])


def task_for_dir(data_dir: str | Path) -> str:
    """Infer the canonical task from a data directory name (handles "*_cluster" dirs)."""
    name = Path(data_dir).name
    for hint, task in _DIR_HINTS:
        if hint in name:
            return task
    raise ValueError(f"cannot infer task from directory name {name!r}")


def split_basepaths_in(data_dir: str | Path) -> list[Path]:
    """Discover dataset base paths from ``*_train.csv`` files in a directory."""
    directory = Path(data_dir)
    suffix = "_train.csv"
    stems = sorted(p.name[: -len(suffix)] for p in directory.glob(f"*{suffix}"))
    return [directory / f"{stem}.csv" for stem in stems]
