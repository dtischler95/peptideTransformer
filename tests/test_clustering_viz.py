"""Exploratory clustering visualisation. These are viz smoke tests (no correctness
claim on the embedding), they guard that the t-SNE plotting path and the
subsampling logic keep running.
"""

import numpy as np
import pandas as pd
import pytest

# hemo_clustering imports umap at module top; skip cleanly if it is unavailable.
pytest.importorskip("umap")
from src.data_analysis.hemo_clustering import perform_clustering, reduce_data_points_for_clustering


def test_perform_clustering_writes_tsne_pdf(tmp_path):
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(20, 8))
    labels = [0, 1] * 10
    perform_clustering(embedded_sequences=embeddings, labels=labels,
                       plot_path=str(tmp_path) + "/", tag="run")
    assert (tmp_path / "run_tsne_plot.pdf").exists()


def test_reduce_data_points_passthrough_below_threshold(tmp_path):
    df = pd.DataFrame({"sequence": ["AAAA"] * 50, "label": ([0] * 25) + ([1] * 25)})
    out = reduce_data_points_for_clustering(df, plot_path=str(tmp_path / "a_"),
                                            label_0_data=10, label_1_data=10)
    # Under 1000 rows the frame is returned unchanged and nothing is written.
    assert len(out) == 50
    assert not (tmp_path / "a_data_used_for_clustering.csv").exists()


def test_reduce_data_points_subsamples_above_threshold(tmp_path):
    df = pd.DataFrame({"sequence": ["AAAA"] * 1200, "label": ([0] * 600) + ([1] * 600)})
    out = reduce_data_points_for_clustering(df, plot_path=str(tmp_path / "b_"),
                                            label_0_data=100, label_1_data=50)
    counts = out["label"].value_counts()
    assert counts[0] == 100 and counts[1] == 50
    assert (tmp_path / "b_data_used_for_clustering.csv").exists()
