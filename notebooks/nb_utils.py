"""Helper functions for the data-quality notebooks.

Keeps the notebook cells readable for a non-programmer audience: the cells hold
variables and a few named calls, the implementation lives here. Used by
``data_quality_mic.ipynb`` and ``data_quality_hemo.ipynb``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 110

_AA = "ACDEFGHIKLMNPQRSTVWY"  # the 20 standard amino acids

__all__ = [
    "find_repo_root",
    # MIC
    "all_organism_files", "load_organism", "summarize_dataset",
    "plot_length_grid", "plot_mic_grid",
    "load_mic_replicates", "replicate_summary", "plot_replicate_counts",
    "plot_sigma_per_sequence", "measurement_noise_floor",
    # hemolysis
    "load_hemo_datasets", "majority_baseline", "plot_label_and_length",
    "plot_thresholding", "plot_replicate_counts_hemo",
    "label_consistency", "plot_label_consistency",
]


# --------------------------------------------------------------------------- shared
def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: cwd) until a folder with a data/ tree is found.
    Lets the notebooks run from the repo root or from notebooks/ alike."""
    start = Path.cwd() if start is None else Path(start)
    for p in [start, *start.parents]:
        if (p / "data" / "regression_data").exists() or (p / "data" / "hemo_train").exists():
            return p
    raise FileNotFoundError("repo root (with a data/ folder) not found")


def _replicate_count_bars(ax, counts, title):
    """Shared bar chart: how many sequences were measured 1, 2, ... 10+ times."""
    binned = counts.clip(upper=10).value_counts().reindex(range(1, 11), fill_value=0)
    labels = [str(i) for i in range(1, 10)] + ["10+"]
    ax.bar(labels, binned.values, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("measurements per sequence")
    ax.set_ylabel("number of sequences")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)


# ------------------------------------------------------------------------------ MIC
def all_organism_files(repo: Path) -> list:
    """The 12 per-organism, de-duplicated MIC regression files."""
    return sorted((repo / "data" / "regression_data").glob("*_for_regression.csv"))


def _pretty_name(path: Path) -> str:
    parts = path.stem.replace("_for_regression", "").split("_")
    name = " ".join(parts[:2]).capitalize()
    return "Enterobacter sp." if name.startswith("Enterobacter") else name


def load_organism(file: Path) -> pd.DataFrame:
    """Load one de-duplicated organism file, adding a sequence-length column."""
    df = pd.read_csv(file, sep=";")
    df["length"] = df["sequence"].str.len()
    return df


def summarize_dataset(df: pd.DataFrame, organism: str) -> None:
    print(f"Organism:            {organism}")
    print(f"Unique sequences:    {len(df)}")
    print(f"Sequence length:     {df['length'].min()} - {df['length'].max()} residues")
    print(f"log10(MIC) range:    {df['mic_log10'].min():.2f} .. {df['mic_log10'].max():.2f}")
    print(f"log10(MIC) mean/std: {df['mic_log10'].mean():.2f} / {df['mic_log10'].std():.2f}")


def _organism_grid(files, value_of, xlabel, suptitle, bins, color):
    ncols = 3
    nrows = -(-len(files) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.1 * nrows))
    for ax, f in zip(axes.flat, files):
        d = pd.read_csv(f, sep=";")
        ax.hist(value_of(d), bins=bins, color=color, edgecolor="white", linewidth=0.3)
        ax.set_title(_pretty_name(f), fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.grid(True, axis="y", alpha=0.3)
    for ax in axes.flat[len(files):]:
        ax.set_visible(False)
    fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout()
    plt.show()


def plot_length_grid(files) -> None:
    _organism_grid(files, lambda d: d["sequence"].str.len(),
                   "sequence length (residues)",
                   "Peptide length distribution per organism", 30, "steelblue")


def plot_mic_grid(files) -> None:
    _organism_grid(files, lambda d: d["mic_log10"],
                   "log10(MIC)  [µM]",
                   "log10(MIC) distribution per organism", 40, "seagreen")


def load_mic_replicates(repo: Path, strain: str = "Acinetobacter") -> pd.DataFrame:
    """Raw per-measurement MIC rows for a strain, cleaned to the model's sequence rule
    (20 standard amino acids, length 3..36). Replicates are kept (one row per
    measurement). Returns columns: sequence, mic_log10."""
    raw = pd.read_csv(repo / "data" / "data_from_database" / "DBAASP_from_CalcAMP.csv")
    keep = (raw["Used_Strain"].astype(str).str.contains(strain, case=False, na=False)
            & raw["Activity_Type"].astype(str).str.fullmatch("MIC", case=False, na=False))
    mic = raw[keep].copy()
    seq = mic["Sequence"].astype(str).str.strip().str.upper()
    mic["sequence"] = seq
    mic = mic[seq.str.fullmatch(f"[{_AA}]{{3,36}}")]
    mic["value_uM"] = pd.to_numeric(mic["Transform_Value"], errors="coerce")
    mic = mic[mic["value_uM"] > 0]
    mic["mic_log10"] = np.log10(mic["value_uM"])
    return mic[["sequence", "mic_log10"]].reset_index(drop=True)


def replicate_summary(reps: pd.DataFrame) -> None:
    counts = reps.groupby("sequence")["mic_log10"].count()
    print(f"raw MIC measurements (cleaned): {len(reps)}")
    print(f"unique sequences:               {counts.size}")
    print(f"measured >= 2 times:            {int((counts >= 2).sum())}")
    print(f"max measurements on one seq:    {int(counts.max())}")


def plot_replicate_counts(reps: pd.DataFrame, organism: str) -> None:
    counts = reps.groupby("sequence")["mic_log10"].count()
    fig, ax = plt.subplots(figsize=(7, 4))
    _replicate_count_bars(ax, counts, f"Replicate counts, {organism} (MIC, DBAASP)")
    plt.show()


def plot_sigma_per_sequence(reps: pd.DataFrame, organism: str) -> None:
    g = reps.groupby("sequence")["mic_log10"]
    counts = g.count()
    multi = counts[counts >= 2].index
    sigma = g.std(ddof=1).loc[multi].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sigma, bins=30, color="seagreen", edgecolor="white", linewidth=0.3)
    ax.axvline(sigma.median(), color="black", ls="--", lw=1.5,
               label=f"median = {sigma.median():.2f}")
    ax.set_xlabel("within-sequence std of log10(MIC)")
    ax.set_ylabel("number of sequences")
    ax.set_title(f"Measurement spread per sequence, {organism} (n={len(sigma)})")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.show()


def measurement_noise_floor(reps: pd.DataFrame, model_test_mse: float) -> dict:
    """The irreducible test error from noisy targets: a perfect model still pays the
    mean within-sequence variance. Compared against the deployed model's test MSE."""
    g = reps.groupby("sequence")["mic_log10"]
    counts = g.count()
    multi = counts[counts >= 2].index
    var_floor = float(g.var(ddof=1).loc[multi].mean())
    sigma_floor = float(np.sqrt(var_floor))
    return {
        "n_replicated": len(multi),
        "sigma_floor": sigma_floor,
        "var_floor": var_floor,
        "fold": float(10 ** sigma_floor),
        "model_rmse": float(np.sqrt(model_test_mse)),
        "noise_share": var_floor / model_test_mse,
    }


# ------------------------------------------------------------------------ hemolysis
def load_hemo_datasets(repo: Path) -> dict:
    """The two hemolysis datasets, with per-measurement labels (replicates kept)."""
    d = repo / "data" / "hemo_train"
    return {
        "WhiteLab": pd.read_csv(d / "whitelab_hemo_data.csv", sep=";"),
        "Threshold": pd.read_csv(d / "threshold_style_unvoted.csv", sep=";"),
    }


def majority_baseline(df: pd.DataFrame) -> float:
    """Accuracy of a trivial model that always predicts the majority class."""
    return float(df["label"].value_counts(normalize=True).max())


def plot_label_and_length(datasets: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for j, (name, df) in enumerate(datasets.items()):
        vc = df["label"].value_counts(normalize=True).reindex([0, 1], fill_value=0)
        axes[0, j].bar(["non-hemolytic (0)", "hemolytic (1)"], vc.values,
                       color=["steelblue", "indianred"])
        axes[0, j].set_ylim(0, 1)
        axes[0, j].set_ylabel("fraction")
        axes[0, j].set_title(f"{name}: labels (majority baseline = {majority_baseline(df):.0%} acc.)")
        for i, v in enumerate(vc.values):
            axes[0, j].text(i, v + 0.02, f"{v:.0%}", ha="center")
        lengths = df.drop_duplicates("sequence")["sequence"].str.len()
        axes[1, j].hist(lengths, bins=30, color="slategray")
        axes[1, j].set_title(f"{name}: peptide length")
        axes[1, j].set_xlabel("sequence length (residues)")
        axes[1, j].set_ylabel("count")
        axes[1, j].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_thresholding(threshold_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for lab, color, name in [(0, "steelblue", "non-hemolytic"), (1, "indianred", "hemolytic")]:
        sub = threshold_df[threshold_df["label"] == lab]
        ax.scatter(sub["hemo_concentration"], sub["hemo_percent"],
                   s=8, alpha=0.3, color=color, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("concentration tested (µM, log scale)")
    ax.set_ylabel("% hemolysis")
    ax.set_title("Threshold dataset: continuous measurements -> binary label")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_replicate_counts_hemo(datasets: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, df) in zip(axes, datasets.items()):
        counts = df.groupby("sequence")["label"].count()
        _replicate_count_bars(ax, counts, f"{name}: replicate counts")
    fig.tight_layout()
    plt.show()


def label_consistency(df: pd.DataFrame) -> dict:
    """Among sequences measured >= 2 times, how many carry both labels (0 and 1)?"""
    g = df.groupby("sequence")["label"]
    counts = g.count()
    multi = counts[counts >= 2].index
    inconsistent = g.nunique().loc[multi] > 1
    n_total = len(multi)
    return {
        "n_replicated": n_total,
        "n_inconsistent": int(inconsistent.sum()),
        "frac_inconsistent": float(inconsistent.mean()) if n_total else 0.0,
    }


def plot_label_consistency(datasets: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, df) in zip(axes, datasets.items()):
        c = label_consistency(df)
        ax.bar(["consistent", "inconsistent"],
               [c["n_replicated"] - c["n_inconsistent"], c["n_inconsistent"]],
               color=["seagreen", "indianred"])
        ax.set_title(f"{name}: {c['frac_inconsistent']:.0%} of replicated sequences contradict themselves")
        ax.set_ylabel("sequences (>= 2 measurements)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()
