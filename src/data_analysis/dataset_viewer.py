from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
This script is thought to be used to visualize the dataset used for training the model.
It contains functions to plot the distribution of labels and the frequency of amino acids in the sequences.

Add new functions here.

TODO:
- Add more plots to visualize the dataset
"""


def plot_binary_label_distribution(df, plot_path=""):
    """
    Plots the distribution of binary labels in the dataset.

    Args:
        df (pd.DataFrame): The dataframe containing the dataset.
        plot_path (str): The path to save the plot. If empty, the plot will be shown.
    """
    label_counts = df["label"].value_counts()
    plt.bar(label_counts.index, label_counts.values / df.shape[0] * 100)
    plt.xlabel("Label")
    plt.ylabel("Count [%]")
    plt.title("Label Distribution")
    plt.xticks([0, 1], ["0", "1"])
    if plot_path != "":
        plt.savefig(f"/{plot_path}_label_distribution.png")
        plt.clf()
    else:
        plt.show()


def amino_acid_frequency_comparison(
    sequences_0, sequences_1, plot_path=""
):
    """
    Function from lucas bayerle snippet codes
    """
    def _tokenize(sequences) -> dict:
        counts = Counter()
        for seq in sequences:
            counts += Counter(seq)
        return counts

    counts_0 = _tokenize(sequences_0)
    counts_1 = _tokenize(sequences_1)

    tick_labels = sorted(set(counts_0.keys())) #| set(counts_1.keys()))
    vec_0 = np.array([counts_0[k] for k in tick_labels])
    vec_0 = vec_0 / vec_0.sum()

    vec_1 = np.array([counts_1[k] for k in tick_labels])
    vec_1 = vec_1 / vec_1.sum()

    x = np.arange(len(tick_labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    fig.suptitle("Amino acid frequency comparison", fontsize=12)
    ax1.bar(x - 0.2, vec_0, 0.4, label="Negativ")
    ax1.bar(x + 0.2, vec_1, 0.4, label="Positiv")
    plt.xticks(x, tick_labels)
    ax1.set_ylabel("AA Frequency")
    ax1.legend()
    ax1.set_xticks(x, tick_labels)
    ax1.set_xlabel("Amino Acids")

    # Train - gen
    ax2.bar(x, vec_0 - vec_1, alpha=0.75)
    ax2.set_ylabel("Difference in Relative Frequency")
    ax2.set_xticks(x, tick_labels)
    ax2.set_xlabel("Amino Acids")
    ax2.set_title("Negativ - Positiv")

    if plot_path != "":
        plt.savefig(f"/{plot_path}_amino_acid_frequency.png")
        plt.clf()
    else:
        plt.show()


def generate_dataset_plots(df, plot_path=""):
    """
    Generates and saves plots for the dataset.

    Args:
        df (pd.DataFrame): The dataframe containing the dataset.
        plot_path (str): The path to save the plots. If empty, the plots will be shown.
    """
    plot_binary_label_distribution(df, plot_path=plot_path)
    sequences_0 = df[df["label"] == 0]["sequence"].tolist()
    sequences_1 = df[df["label"] == 1]["sequence"].tolist()
    amino_acid_frequency_comparison(sequences_0, sequences_1, plot_path=plot_path)

if __name__ == "__main__":

    df = pd.read_csv("../../data/train_data/our_hemo_labeled.csv", sep=';')
    generate_dataset_plots(df=df,
                           plot_path="../../plots")