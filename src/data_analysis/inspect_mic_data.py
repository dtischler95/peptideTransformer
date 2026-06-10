from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""
This file is inteded for creation of images for my thesis.
This should only contain functions to create systematic views of the data
"""

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLOTS_ROOT = _REPO_ROOT / "final_plots"
_PLOTS_ROOT.mkdir(exist_ok=True)

threshold_style_train_data = str(_REPO_ROOT / "data" / "train_data" / "threshold_style_unvoted.csv")
white_lab_train_data = str(_REPO_ROOT / "data" / "train_data" / "whitelab_hemo_data.csv")
white_lab_train_data_clean = str(_REPO_ROOT / "data" / "train_data" / "whitelab_filtered_clean.csv")

def get_pretty_organism_name(file_path: str) -> str:
    name = '_'.join(file_path.split('/')[-1].split('_')[:2]).replace('_', ' ').capitalize()
    if name.startswith('Enterobacter'):
        name = 'Enterobacter sp.'
    return name

def get_mic_files():
    mic_dir = _REPO_ROOT / "data" / "regression_data"
    return [str(f) for f in mic_dir.iterdir() if f.suffix == ".csv"]

def mic_distribution_histogram(mic_files: list[str]):
    """
    Generates two hist plots with one plot showing the value distribution of the original value and one plot showing the log10 transformed value distribution.

    """
    for file in mic_files:
        df = pd.read_csv(file, sep=';')
        organism = get_pretty_organism_name(file)

        hist_plot_mic(df, organism)


def hist_plot_mic(df, organism):
    plt.figure(figsize=(10, 5))
    plt.suptitle(f"MIC Value Distribution for {organism}")
    plt.subplot(1, 2, 1)
    plt.hist(df['value'], bins=50)
    plt.title("MIC Value Distribution")
    plt.xlabel('MIC (µM)')
    plt.ylabel("Count")
    plt.grid(True, axis='y')
    plt.subplot(1, 2, 2)
    plt.hist(df['mic_log10'], bins=30)
    plt.title("MIC Value Distribution (log10 transformed)")
    plt.xlabel('MIC (log10(µM))')
    plt.ylabel("Count")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{_PLOTS_ROOT}/{organism}_mic_distribution.png")
    plt.close()


def violin_mic_distribution(mic_files: list[str]):
    """
    Shows a violin plot of the log10 mic value for each organism in one plot
    """

    # Init for violin plots
    n_rows = 4
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    axes = axes.flatten()

    for i, file in enumerate(mic_files):
        df = pd.read_csv(file, sep=';')
        organism = get_pretty_organism_name(file)
        if organism.startswith('Enterobacter'):
            organism = 'Enterobacter sp.'

        sns.violinplot(y='mic_log10', data=df, ax=axes[i], width=0.5)
        axes[i].set_title(f"{organism}", size=18)
        axes[i].set_ylabel('MIC (log10(µM))', size=12)
        axes[i].grid(True, axis='y')
    fig.suptitle("Violin Plot of MIC Values (log10 transformed)\n", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{_PLOTS_ROOT}/violin_mic_distribution.png")
    plt.close()


def mic_seq_length_distribution(mic_files: list[str]):
    """
    Shows a histogram of the sequence length for each organism in a subplot grid.
    """
    n_rows = 4
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    axes = axes.flatten()

    for i, file in enumerate(mic_files):
        df = pd.read_csv(file, sep=';')
        organism = get_pretty_organism_name(file)
        if organism.startswith('Enterobacter'):
            organism = 'Enterobacter sp.'

        axes[i].hist(df['sequence'].str.len(), bins=30)
        axes[i].set_title(f"{organism}", size=18)
        axes[i].set_xlabel('Sequence Length', size=12)
        axes[i].set_ylabel('Count', size=12)
        axes[i].grid(True, axis='y')

    fig.suptitle("Sequence Length Distribution\n", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{_PLOTS_ROOT}/hist_sequ_length_distribution.png")
    plt.close()

def hist_mic_distribution(mic_files: list[str]):
    """
    Shows a histogram of the log10 MIC values for each organism in a subplot grid.
    """
    n_rows = 4
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    axes = axes.flatten()

    for i, file in enumerate(mic_files):
        df = pd.read_csv(file, sep=';')
        organism = get_pretty_organism_name(file)
        if organism.startswith('Enterobacter'):
            organism = 'Enterobacter sp.'

        axes[i].hist(df['mic_log10'], bins=50)
        axes[i].set_title(f"{organism}", size=18)
        axes[i].set_xlabel('MIC (log10(µM))', size=12)
        axes[i].set_ylabel('Count', size=12)
        axes[i].grid(True, axis='y')

    fig.suptitle("MIC Value Distribution (log10 transformed)\n", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{_PLOTS_ROOT}/hist_mic_distribution.png")
    plt.close()

def hist_mic_log_comparison(mic_files: list[str]):
    """
    Shows a histogram of the log10 MIC values for each organism in a subplot grid.
    """
    n_rows = 6
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    axes = axes.flatten()


    for i, file in enumerate(mic_files):
        df = pd.read_csv(file, sep=';')
        organism = get_pretty_organism_name(file)
        if organism.startswith('Enterobacter'):
            organism = 'Enterobacter sp.'
        ax_left = axes[2*i]
        ax_right = axes[2*i + 1]

        ax_left.hist(df['value'], bins=50)
        ax_left.set_title(f"{organism}", size=18)
        ax_left.set_xlabel('MIC (µM)', size=12)
        ax_left.set_ylabel('Count', size=12)
        ax_left.grid(True, axis='y')
        ax_right.hist(df['mic_log10'], bins=50)
        ax_right.set_title(f"{organism}", size=18)
        ax_right.set_xlabel('MIC (log10(µM))', size=12)
        ax_right.set_ylabel('Count', size=12)
        ax_right.grid(True, axis='y')

    fig.suptitle("MIC Value Distribution (log10 transformed)\n", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{_PLOTS_ROOT}/hist_mic_distribution.png")
    plt.close()

def get_all_data_combined(mic_files: list[str]) -> pd.DataFrame:
    combined_df = pd.DataFrame()
    for file in mic_files:
        df = pd.read_csv(file, sep=';')
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

def combined_hist_plot(mic_files: list[str]):
    combined_df = get_all_data_combined(mic_files)
    #combined_df = combined_df[combined_df['value'] <= 300.0]
    hist_plot_mic(combined_df, "ESKAPE Organisms")

def hist_mic_log_nested(mic_files: list[str]):
    """
    12 subplots (one per organism), each containing 2 inner subplots:
    left = raw MIC values, right = log10-transformed MIC values
    """
    n_rows, n_cols = 6, 2  # 12 Organismen = 4x3 Grid
    fig = plt.figure(figsize=(10, 11))

    outer = fig.add_gridspec(n_rows, n_cols, wspace=0.35, hspace=0.8)

    for i, file in enumerate(mic_files):
        df = pd.read_csv(file, sep=';')
        organism = get_pretty_organism_name(file)
        if organism.startswith('Enterobacter'):
            organism = 'Enterobacter sp.'

        row, col = divmod(i, n_cols)
        inner = outer[row, col].subgridspec(1, 2, wspace=0.5, hspace=0.5)

        ax_left = fig.add_subplot(inner[0])
        ax_right = fig.add_subplot(inner[1])

        # raw MIC values
        ax_left.hist(df['value'], bins=50, color="skyblue", edgecolor="black")

        ax_left.set_xlabel("µM", fontsize=10)
        ax_left.set_ylabel("Count", fontsize=10)
        ax_left.tick_params(labelsize=10)

        # log10-transformed MIC values
        ax_right.hist(df['mic_log10'], bins=30, color="lightgreen", edgecolor="black")

        ax_right.set_xlabel("log10(µM)", fontsize=10)
        ax_right.set_ylabel("Count", fontsize=10)
        ax_right.tick_params(labelsize=10)

        # shared title per organism (spanning both subplots)
        fig.text((ax_left.get_position().x0 + ax_right.get_position().x1) / 2,
                 ax_left.get_position().y1 + 0.01,
                 organism, ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("MIC Value Histograms per Organism", fontsize=14)
    plt.savefig(f"{_PLOTS_ROOT}/hist_mic_nested.png", dpi=300, bbox_inches="tight")
    plt.close()





def mic_amino_frequency_comparison(
        mic_list
):

    def _tokenize(sequences) -> dict:
        counts = Counter()
        for seq in sequences:
            counts += Counter(seq)
        return counts
    n_rows = 6
    n_cols = 2

    fig = plt.figure(figsize=(18,20))
    outer = fig.add_gridspec(n_rows, n_cols, wspace=0.25, hspace=0.5)

    for i, file in enumerate(mic_list):
        df = pd.read_csv(file, sep=';')

        inner = outer[i].subgridspec(1, 2, wspace=0.4, hspace=0.5)
        organism = get_pretty_organism_name(file)
        ax_left = fig.add_subplot(inner[0])
        ax_right = fig.add_subplot(inner[1])


        sequences_0 = df[df['value'] <= 50.0]['sequence'].tolist()
        sequences_1 = df[df['value'] > 50.0]['sequence'].tolist()

        counts_0 = _tokenize(sequences_0)
        counts_1 = _tokenize(sequences_1)

        tick_labels = sorted(set(counts_0.keys()))  # | set(counts_1.keys()))
        vec_0 = np.array([counts_0[k] for k in tick_labels])
        vec_0 = vec_0 / vec_0.sum()

        vec_1 = np.array([counts_1[k] for k in tick_labels])
        vec_1 = vec_1 / vec_1.sum()

        x = np.arange(len(tick_labels))

        #fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        fig.suptitle("Amino Acid Frequency Comparison", fontsize=12)
        ax_left.bar(x - 0.2, vec_0, 0.4, label="Inactive")
        ax_left.bar(x + 0.2, vec_1, 0.4, label="Active")
        plt.xticks(x, tick_labels)
        ax_left.set_ylabel("AA Frequency", size=8)
        ax_left.legend(fontsize=5)
        ax_left.set_xticks(x, tick_labels)
        ax_left.set_xlabel("Amino Acids")

        # Frequency difference
        ax_right.bar(x, vec_0 - vec_1, alpha=0.75)
        ax_right.set_ylabel("Frequency Difference", size=8)
        ax_right.set_xticks(x, tick_labels)
        ax_right.set_xlabel("Amino Acids")
        fig.text((ax_left.get_position().x0 + ax_right.get_position().x1) / 2,
                 ax_left.get_position().y1 + 0.01,
                 organism, ha="center", va="bottom", fontsize=12, fontweight="bold")

    fig.suptitle("Amino Acid Frequency Comparison", fontsize=20)

    plt.savefig(f"{_PLOTS_ROOT}/all_aa_frequency_comparison.png")
    plt.close()






def main():
    mic_files = get_mic_files()
    #combined_hist_plot(mic_files=mic_files)
    #hist_mic_distribution(mic_files=mic_files)
    #mic_distribution_histogram(mic_files=mic_files)
    #violin_mic_distribution(mic_files=mic_files)
    #mic_seq_length_distribution(mic_files)
    #hist_mic_log_comparison(mic_files=mic_files)
    #hist_mic_log_nested(mic_files)
    #mic_amino_frequency_comparison(mic_files)




if __name__ == "__main__":
    main()