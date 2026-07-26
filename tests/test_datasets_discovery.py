"""Dataset base-file discovery. This is the single source of truth that keeps the
splitters and the ML auto-discovery from re-splitting generated splits, so its
invariants are worth locking down.
"""

import pytest

from src.data_preprocessing import datasets


def test_organism_slug_takes_first_two_parts():
    assert datasets.organism_slug("enterobacter_for_regression") == "enterobacter_for"
    assert datasets.organism_slug("mini_hemo_train") == "mini_hemo"


@pytest.mark.parametrize("dirname,task", [
    ("regression_data", "regression"),
    ("regression_data_cluster", "regression"),
    ("hemo_train", "cls"),
    ("hemo_train_cluster", "cls"),
    ("gram", "gram"),
])
def test_task_for_dir(dirname, task):
    assert datasets.task_for_dir(dirname) == task


def test_task_for_dir_unknown_raises():
    with pytest.raises(ValueError):
        datasets.task_for_dir("something_else")


def test_split_basepaths_discovers_only_train_files(tmp_path):
    for stem in ("a_train", "b_train"):
        (tmp_path / f"{stem}.csv").write_text("sequence;label\nAA;0\n")
    # Noise that must be ignored.
    (tmp_path / "a_test.csv").write_text("x")
    (tmp_path / "notes.txt").write_text("x")

    bases = datasets.split_basepaths_in(tmp_path)
    assert [p.name for p in bases] == ["a.csv", "b.csv"]


def test_base_files_excludes_generated_splits(tmp_path):
    # A base file plus its generated splits; only the base must be returned.
    (tmp_path / "foo_regression.csv").write_text("x")
    (tmp_path / "foo_regression_train.csv").write_text("x")
    (tmp_path / "foo_regression_test.csv").write_text("x")

    bases = datasets.base_files("regression", data_dir=tmp_path)
    assert [p.name for p in bases] == ["foo_regression.csv"]
