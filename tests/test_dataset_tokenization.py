"""PeptideDataset residue handling, item shapes, label-column selection and the
train/val/test leakage guard. Fully offline via a fake tokenizer.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.modeling.PeptideBERTClasses.PeptideDataset import PeptideDataset
from src.modeling.fine_tune_utils import _get_labels_and_features, check_data_loader_for_leakage

_LOGGER = logging.getLogger("test")


def _dataset(peptides, tokenizer, backbone, labels=None):
    labels = labels if labels is not None else [0.0] * len(peptides)
    return PeptideDataset(peptides=peptides, features=None, tokenizer=tokenizer,
                          model_class="regression", labels=labels, max_length=16, backbone=backbone)


def test_bert_space_separates_residues(fake_tokenizer):
    ds = _dataset(["MKLV", "ACDE"], fake_tokenizer, backbone="bert")
    assert ds.peptides == ["M K L V", "A C D E"]


def test_esm_keeps_raw_residues(fake_tokenizer):
    ds = _dataset(["MKLV", "ACDE"], fake_tokenizer, backbone="esm")
    assert ds.peptides == ["MKLV", "ACDE"]


def test_getitem_returns_padded_tensors(fake_tokenizer):
    ds = _dataset(["MKLV"], fake_tokenizer, backbone="esm", labels=[1.5])
    item = ds[0]
    assert item["input_ids"].shape == (16,)
    assert item["attention_mask"].shape == (16,)
    assert float(item["labels"]) == pytest.approx(1.5)


def test_label_column_selection():
    df = pd.DataFrame({"sequence": ["AA"], "mic_log10": [2.0], "label": [1]})
    reg_labels, _ = _get_labels_and_features(df, "regression", use_features=False)
    cls_labels, _ = _get_labels_and_features(df, "binary_dense", use_features=False)
    assert reg_labels[0] == 2.0
    assert cls_labels[0] == 1


def test_feature_columns_selected_when_requested():
    df = pd.DataFrame({"sequence": ["AA"], "mic_log10": [2.0],
                       "desc__a": [0.1], "desc__b": [0.2], "other": [9]})
    _, feats = _get_labels_and_features(df, "regression", use_features=True)
    assert list(feats.columns) == ["desc__a", "desc__b"]


class _Bag:
    """Minimal object exposing the .peptides attribute the leakage check reads."""
    def __init__(self, peptides):
        self.peptides = peptides


def test_leakage_guard_raises_on_overlap():
    train = _Bag(["AAA", "BBB", "CCC"])
    val = _Bag(["CCC", "DDD"])  # CCC overlaps train
    test = _Bag(["EEE"])
    with pytest.raises(ValueError):
        check_data_loader_for_leakage(train, val, test, logger=_LOGGER)


def test_leakage_guard_passes_on_disjoint_splits():
    train = _Bag(["AAA", "BBB"])
    val = _Bag(["CCC"])
    test = _Bag(["DDD"])
    # Must not raise.
    check_data_loader_for_leakage(train, val, test, logger=_LOGGER)


def test_leakage_guard_can_be_suppressed():
    train = _Bag(["AAA"])
    test = _Bag(["AAA"])
    check_data_loader_for_leakage(train, None, test, logger=_LOGGER, ignore_leakage=True)
