"""The core guarantee of the cluster split: whole clusters stay in one split, so
no near-duplicate peptide can span train and test. This is the thesis's main
contribution, tested on a synthetic cluster assignment (no MMseqs2 binary needed).
"""

import pandas as pd
import pytest

from src.data_preprocessing.cluster_data_splitter import assign_clusters_to_splits


def _synthetic_clusters(n_clusters=20, per_cluster=2):
    """20 clusters of 2 sequences, half in each stratum, whole clusters share a stratum."""
    rows, cluster_of = [], {}
    for c in range(n_clusters):
        strat = "A" if c < n_clusters // 2 else "B"
        for j in range(per_cluster):
            seq = f"C{c}_S{j}"
            rows.append({"sequence": seq, "strat": strat})
            cluster_of[seq] = c
    return pd.DataFrame(rows), cluster_of


def test_no_cluster_spans_two_splits():
    strat_df, cluster_of = _synthetic_clusters()
    train, val, test = assign_clusters_to_splits(strat_df, cluster_of, random_state=42)

    split_of = {}
    for name, seqs in (("train", train), ("val", val), ("test", test)):
        for s in seqs:
            split_of[s] = name

    # Each cluster's sequences must all land in exactly one split.
    from collections import defaultdict
    clusters_splits = defaultdict(set)
    for seq, cluster in cluster_of.items():
        clusters_splits[cluster].add(split_of[seq])
    for cluster, splits in clusters_splits.items():
        assert len(splits) == 1, f"cluster {cluster} spans splits {splits}"


def test_splits_partition_all_sequences_without_overlap():
    strat_df, cluster_of = _synthetic_clusters()
    train, val, test = assign_clusters_to_splits(strat_df, cluster_of, random_state=42)

    assert train | val | test == set(strat_df["sequence"])
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()


def test_all_splits_non_empty_and_train_is_largest():
    strat_df, cluster_of = _synthetic_clusters()
    train, val, test = assign_clusters_to_splits(strat_df, cluster_of, random_state=42)

    assert len(test) > 0 and len(val) > 0 and len(train) > 0
    assert len(train) > len(test) and len(train) > len(val)


def test_assignment_is_deterministic_for_fixed_seed():
    strat_df, cluster_of = _synthetic_clusters()
    a = assign_clusters_to_splits(strat_df, cluster_of, random_state=42)
    b = assign_clusters_to_splits(strat_df, cluster_of, random_state=42)
    assert a == b
