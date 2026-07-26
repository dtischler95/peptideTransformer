"""CLI override application, config-shortname resolution and argument parsing."""

import sys

import pytest

from src.modeling.fine_tune_utils import _apply_overrides
import src.cli as cli


def _output_config():
    return {
        "model_save_path": "models/x",
        "plot_path": "bert_plots/x",
        "output_dir": "results/x",
        "logging_dir": "logs/x",
    }


def test_overrides_apply_and_suffix_outputs():
    config = _output_config()
    _apply_overrides(config, {"backbone": "esm",
                              "model_path": "facebook/esm2_t6_8M_UR50D",
                              "model_name": "esm8m"})
    assert config["backbone"] == "esm"
    assert config["model_path"] == "facebook/esm2_t6_8M_UR50D"
    # Output paths get suffixed with the effective model_name so ESM runs never
    # overwrite the ProtBERT outputs.
    assert config["model_save_path"] == "models/x_esm8m"
    assert config["output_dir"] == "results/x_esm8m"


def test_empty_overrides_are_a_noop():
    config = _output_config()
    _apply_overrides(config, {})
    assert config == _output_config()


def test_model_path_only_override_does_not_suffix():
    config = _output_config()
    _apply_overrides(config, {"model_path": "foo"})
    assert config["model_path"] == "foo"
    assert config["model_save_path"] == "models/x"  # unchanged, identity not touched


def test_backbone_only_override_suffixes_with_backbone():
    config = _output_config()
    _apply_overrides(config, {"backbone": "esm"})
    assert config["model_save_path"] == "models/x_esm"


def _make_cfg_tree(root):
    base = root / "modeling" / "configs"
    (base / "random_split").mkdir(parents=True)
    (base / "cluster_split").mkdir(parents=True)
    return base


def test_shortname_resolves_when_unique(tmp_path, monkeypatch):
    base = _make_cfg_tree(tmp_path)
    (base / "random_split" / "only.yaml").write_text("x: 1")
    monkeypatch.setattr(cli, "_SRC_DIR", tmp_path)
    resolved = cli._resolve_config_shortname("only.yaml")
    assert resolved.endswith("only.yaml")
    assert "random_split" in resolved


def test_shortname_ambiguous_raises(tmp_path, monkeypatch):
    base = _make_cfg_tree(tmp_path)
    (base / "random_split" / "dup.yaml").write_text("x: 1")
    (base / "cluster_split" / "dup.yaml").write_text("x: 1")
    monkeypatch.setattr(cli, "_SRC_DIR", tmp_path)
    with pytest.raises(SystemExit):
        cli._resolve_config_shortname("dup.yaml")


def test_shortname_missing_returns_flat_path(tmp_path, monkeypatch):
    _make_cfg_tree(tmp_path)
    monkeypatch.setattr(cli, "_SRC_DIR", tmp_path)
    # Not found: returns a path (downstream open() raises the clear error), no crash here.
    resolved = cli._resolve_config_shortname("missing.yaml")
    assert resolved.endswith("missing.yaml")


def test_bert_model_requires_a_config_source(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "bert_model"])
    with pytest.raises(SystemExit):  # mutually exclusive group is required
        cli.parse_inputs()


def test_bert_model_rejects_both_config_sources(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "bert_model", "--config_path", "a.yaml", "--pipe_configs", "d"])
    with pytest.raises(SystemExit):
        cli.parse_inputs()


def test_bert_model_dispatches_to_its_handler(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "bert_model", "--config_path", "a.yaml"])
    args, _ = cli.parse_inputs()
    assert args.config_path == "a.yaml"
    assert args.func is cli.cmd_bert_model
