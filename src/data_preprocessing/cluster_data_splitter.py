"""Similarity-aware (cluster-based) train/val/test splitting for the peptide datasets.

The default ``data_splitter.py`` splits at the sequence level: each unique peptide
lands in exactly one split, but near-duplicate peptides (e.g. a single point mutation)
can still end up in different splits. This module clusters peptides by sequence
similarity (via MMseqs2 ``easy-cluster``) and keeps whole clusters together in one
split, in addition to the existing per-sequence stratification.

Output files use the same layout as the random splitter (``*_train.csv`` /
``*_val.csv`` / ``*_test.csv``, ``;``-separated, same columns), written into a
parallel ``*_cluster`` data directory under the base file's name. BERT configs point
``train_file`` at this (non-existent) base path; only the path stem is used to locate
the ``_train``/``_val``/``_test`` files, so the base CSV itself is not duplicated here.

MMseqs2 (https://github.com/soedinglab/MMseqs2) is a standalone binary, not a Python
package, and must be on PATH. The clustering step is CPU-only.

Short peptides (3 to 36 residues here) shorter than the k-mer size ``k`` cannot be
matched by MMseqs2's prefilter and end up as singleton clusters. The split report
printed by this module includes the singleton fraction.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

try:
    from src.data_preprocessing.data_splitter import make_seq_strat_labels
except ImportError:  # allow running the file directly
    from data_splitter import make_seq_strat_labels  # type: ignore

_REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------- clustering
def run_mmseqs_cluster(sequences,
                       *,
                       min_seq_id: float = 0.5,
                       coverage: float = 0.8,
                       cov_mode: int = 1,
                       sensitivity: float = 7.5,
                       kmer: int = 5,
                       threads: int = 4,
                       workdir: str | Path | None = None) -> dict:
    """Cluster `sequences` with MMseqs2 easy-cluster.

    Returns a dict mapping each sequence to its cluster id (the representative's id).
    Raises FileNotFoundError with an install hint if mmseqs is not on PATH.
    """
    if shutil.which("mmseqs") is None:
        raise FileNotFoundError(
            "mmseqs not found on PATH. Install it with "
            "'conda install -c bioconda mmseqs2' (Linux/macOS/WSL), or use the static "
            "Windows build / a Linux notebook. See this module's docstring."
        )

    uniq = list(dict.fromkeys(str(s).strip() for s in sequences))  # de-dup, keep order
    keep_tmp = workdir is not None
    tmp = Path(workdir) if keep_tmp else Path(tempfile.mkdtemp(prefix="mmseqs_"))
    tmp.mkdir(parents=True, exist_ok=True)

    try:
        fasta = tmp / "input.fasta"
        with open(fasta, "w") as fh:
            for i, seq in enumerate(uniq):
                fh.write(f">{i}\n{seq}\n")

        prefix = tmp / "clust"
        cmd = [
            "mmseqs", "easy-cluster", str(fasta), str(prefix), str(tmp / "mmseqs_tmp"),
            "--min-seq-id", str(min_seq_id),
            "-c", str(coverage),
            "--cov-mode", str(cov_mode),
            "-s", str(sensitivity),
            "-k", str(kmer),
            "--threads", str(threads),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)

        # easy-cluster writes <prefix>_cluster.tsv with columns: representative <TAB> member
        tsv = Path(f"{prefix}_cluster.tsv")
        mapping = {}
        with open(tsv) as fh:
            for line in fh:
                rep_id, member_id = line.rstrip("\n").split("\t")
                mapping[uniq[int(member_id)]] = rep_id
        if len(mapping) != len(uniq):
            missing = set(uniq) - set(mapping)
            raise RuntimeError(f"mmseqs did not assign {len(missing)} sequences to a cluster")
        return mapping
    finally:
        if not keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------- split logic
def _group_holdout(y: np.ndarray, groups: np.ndarray, frac: float,
                   random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out about `frac` of rows, keeping whole groups together and stratifying
    by `y` as far as the grouping allows. Returns (rest_positions, holdout_positions)."""
    n_groups = len(set(groups))
    n_splits = min(max(2, round(1 / frac)), n_groups)
    if n_splits < 2:
        # too few clusters to carve out a holdout, leave everything in 'rest'
        return np.arange(len(y)), np.array([], dtype=int)

    while n_splits >= 2:
        try:
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=random_state)
            rest_idx, hold_idx = next(sgkf.split(np.zeros(len(y)), y, groups))
            return rest_idx, hold_idx
        except ValueError:
            # a stratum has fewer groups than folds, relax and retry
            n_splits -= 1
    return np.arange(len(y)), np.array([], dtype=int)


def assign_clusters_to_splits(strat_df: pd.DataFrame, cluster_of: dict, *,
                              test_size: float = 0.20, val_size: float = 0.16,
                              random_state: int = 42) -> tuple[set, set, set]:
    """Assign whole clusters to train/val/test.

    `strat_df` has columns 'sequence' and 'strat' (one row per unique sequence).
    `cluster_of` maps sequence -> cluster id. Returns (train_seqs, val_seqs, test_seqs)
    as sets of sequences, with no cluster spanning more than one split.
    """
    seqs = strat_df["sequence"].to_numpy()
    y = pd.factorize(strat_df["strat"].astype(str))[0]
    groups = np.array([cluster_of[s] for s in seqs])

    rest_idx, test_idx = _group_holdout(y, groups, test_size, random_state)

    val_rel = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
    if len(rest_idx) and val_rel > 0:
        sub_rest, sub_val = _group_holdout(y[rest_idx], groups[rest_idx],
                                           val_rel, random_state)
        train_idx = rest_idx[sub_rest]
        val_idx = rest_idx[sub_val]
    else:
        train_idx, val_idx = rest_idx, np.array([], dtype=int)

    return set(seqs[train_idx]), set(seqs[val_idx]), set(seqs[test_idx])


def cluster_split_with_val(tmp_df: pd.DataFrame, *, task: str, target_col: str,
                           cluster_of: dict, test_size: float = 0.20,
                           val_size: float = 0.16, random_state: int = 42):
    """Cluster-aware counterpart to data_splitter.split_with_val.

    Uses the same stratification definition (make_seq_strat_labels) so the only
    difference from the random split is that whole clusters stay in one split.
    Returns train_df, val_df, test_df with the same columns as the random splitter.
    """
    strat_df = make_seq_strat_labels(tmp_df, task=task, target_col=target_col)
    train_seqs, val_seqs, test_seqs = assign_clusters_to_splits(
        strat_df, cluster_of, test_size=test_size, val_size=val_size,
        random_state=random_state,
    )

    if task == "cls":
        source = strat_df.rename(columns={"strat": target_col})
    elif task == "gram":
        source = tmp_df[["sequence", "mic_log10", target_col]]
    else:
        source = tmp_df[["sequence", target_col]]

    # Filter by sequence membership (robust to row order, unlike index alignment).
    def _select(seqset):
        return source[source["sequence"].isin(seqset)].reset_index(drop=True)

    return _select(train_seqs), _select(val_seqs), _select(test_seqs)


# ------------------------------------------------------------------------ reporting
def _report_split(file_name: str, task: str, target_col: str, cluster_of: dict,
                  train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Print a sanity report and assert there is no leakage between splits."""
    sizes = pd.Series(cluster_of).value_counts()
    n_clusters = sizes.size
    n_singletons = int((sizes == 1).sum())
    n_seqs = sum(sizes)

    print(f"\n=== {file_name} ({task}) ===")
    print(f"  sequences: {n_seqs} | clusters: {n_clusters} | "
          f"singletons: {n_singletons} ({n_singletons / n_clusters:.0%}) | "
          f"largest cluster: {int(sizes.max())}")

    parts = {"train": train_df, "val": val_df, "test": test_df}
    total = sum(len(d) for d in parts.values())
    for name, d in parts.items():
        line = f"  {name:5s}: {len(d):5d} ({len(d) / total:.0%})"
        if task in ("cls",):
            bal = d[target_col].value_counts(normalize=True).sort_index()
            line += "   labels " + ", ".join(f"{k}:{v:.0%}" for k, v in bal.items())
        else:
            line += f"   {target_col} mean/std {d[target_col].mean():.2f}/{d[target_col].std():.2f}"
        print(line)

    # leakage: no sequence in more than one split
    s_tr, s_va, s_te = (set(d["sequence"]) for d in parts.values())
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), \
        "sequence leakage between splits"

    # integrity: no cluster spans more than one split
    split_of = {}
    for name, s in [("train", s_tr), ("val", s_va), ("test", s_te)]:
        for seq in s:
            split_of[seq] = name
    spanning = set()
    seen = {}
    for seq, cl in cluster_of.items():
        if seq not in split_of:
            continue
        if cl in seen and seen[cl] != split_of[seq]:
            spanning.add(cl)
        seen[cl] = split_of[seq]
    assert not spanning, f"{len(spanning)} clusters span multiple splits"
    print("  leakage check: OK (no sequence and no cluster spans splits)")


# ------------------------------------------------------------------------- pipeline
def cluster_data_splitter(task: str,
                          data_dir: str | Path | None = None,
                          out_dir: str | Path | None = None,
                          *,
                          min_seq_id: float = 0.5,
                          coverage: float = 0.8,
                          cov_mode: int = 1,
                          sensitivity: float = 7.5,
                          kmer: int = 5,
                          test_size: float = 0.20,
                          val_size: float = 0.16,
                          random_state: int = 42):
    """Write cluster-based train/val/test splits for every base file in `data_dir`.

    Mirrors data_splitter.data_splitter but groups by MMseqs2 similarity clusters and
    writes into a parallel '<data_dir>_cluster' directory (same filenames), so the
    existing ML auto-discovery and BERT loaders pick them up unchanged.
    """
    if task == "cls":
        # Mirror the ML auto-discovery globs so we only grab the base files,
        # never the generated *_train/_val/_test.csv outputs.
        in_dir = Path(data_dir) if data_dir else _REPO_ROOT / "data" / "hemo_train"
        base_files = sorted(list(in_dir.glob("*unvoted.csv")) + list(in_dir.glob("*data.csv")))
        target_col = "label"
    elif task == "regression":
        in_dir = Path(data_dir) if data_dir else _REPO_ROOT / "data" / "regression_data"
        base_files = sorted(in_dir.glob("*regression.csv"))
        target_col = "mic_log10"
    else:
        in_dir = Path(data_dir) if data_dir else _REPO_ROOT / "data" / "gram"
        base_files = sorted(in_dir.glob("*dataset.csv"))
        target_col = "label"

    out = Path(out_dir) if out_dir else in_dir.with_name(in_dir.name + "_cluster")
    out.mkdir(parents=True, exist_ok=True)

    for base_file in base_files:
        tmp_df = pd.read_csv(base_file, sep=";")
        if task == "cls":
            tmp_df = tmp_df.drop(columns=["hemo_concentration", "hemo_percent"],
                                 errors="ignore")

        unique_seqs = tmp_df["sequence"].astype(str).str.strip().unique().tolist()
        cluster_of = run_mmseqs_cluster(
            unique_seqs, min_seq_id=min_seq_id, coverage=coverage,
            cov_mode=cov_mode, sensitivity=sensitivity, kmer=kmer,
        )

        train_df, val_df, test_df = cluster_split_with_val(
            tmp_df, task=task, target_col=target_col, cluster_of=cluster_of,
            test_size=test_size, val_size=val_size, random_state=random_state,
        )

        _report_split(base_file.name, task, target_col, cluster_of,
                      train_df, val_df, test_df)

        stem = out / base_file.stem
        train_df.to_csv(stem.with_name(stem.name + "_train.csv"), sep=";", index=False)
        val_df.to_csv(stem.with_name(stem.name + "_val.csv"), sep=";", index=False)
        test_df.to_csv(stem.with_name(stem.name + "_test.csv"), sep=";", index=False)

    print(f"\nCluster splits written to: {out}")


if __name__ == "__main__":
    cluster_data_splitter(task="regression")
