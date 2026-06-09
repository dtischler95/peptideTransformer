from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PLOTS_ROOT = _REPO_ROOT / "final_plots"
_PLOTS_ROOT.mkdir(exist_ok=True)

happenn_style_train_data = str(_REPO_ROOT / "data" / "train_data" / "happen_style_unvoted.csv")
white_lab_train_data = str(_REPO_ROOT / "data" / "train_data" / "whitelab_hemo_data.csv")


def length_per_label_comparison():
    df_happen = pd.read_csv(happenn_style_train_data, sep=';')
    df_happen['length'] = df_happen['sequence'].apply(len)
    df_happen_inactive = df_happen[df_happen['label'] == 0]
    df_happen_active = df_happen[df_happen['label'] == 1]


    df_whitelab = pd.read_csv(white_lab_train_data, sep=';')
    df_whitelab['length'] = df_whitelab['sequence'].apply(len)
    df_whitelab = df_whitelab[df_whitelab['length'] <= 36]
    df_whitelab_inactive = df_whitelab[df_whitelab['label'] == 0]
    df_whitelab_active = df_whitelab[df_whitelab['label'] == 1]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Längenverteilung für Hämotoxische Datensätze")

    # Top left: Whitelab Inaktiv
    sns.histplot(data=df_whitelab_inactive, x='length', hue='label', multiple='stack', bins=30, ax=axes[0, 0])
    axes[0, 0].set_title("Längenverteilung Inaktiv (Whitelab)")
    axes[0, 0].set_xlabel("Länge der Sequenz")
    axes[0, 0].set_ylabel("Häufigkeit")

    # Top right: Whitelab Aktiv
    sns.histplot(data=df_whitelab_active, x='length', hue='label', multiple='stack', bins=30, ax=axes[0, 1])
    axes[0, 1].set_title("Längenverteilung Aktiv (Whitelab)")
    axes[0, 1].set_xlabel("Länge der Sequenz")
    axes[0, 1].set_ylabel("Häufigkeit")

    # Bottom left: HAPPENN Inaktiv
    sns.histplot(data=df_happen_inactive, x='length', hue='label', multiple='stack', bins=30, ax=axes[1, 0])
    axes[1, 0].set_title("Längenverteilung Inaktiv (HAPPENN)")
    axes[1, 0].set_xlabel("Länge der Sequenz")
    axes[1, 0].set_ylabel("Häufigkeit")

    # Bottom right: HAPPENN Aktiv
    sns.histplot(data=df_happen_active, x='length', hue='label', multiple='stack', bins=30, ax=axes[1, 1])
    axes[1, 1].set_title("Längenverteilung Aktiv (HAPPENN)")
    axes[1, 1].set_xlabel("Länge der Sequenz")
    axes[1, 1].set_ylabel("Häufigkeit")

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    plt.savefig(f"{_PLOTS_ROOT}/hemo_length_comparison.png")
    plt.close()

def mic_amino_frequency_comparison_multi(
        mic_list
):

    def _tokenize(sequences) -> dict:
        counts = Counter()
        for seq in sequences:
            counts += Counter(seq)
        return counts
    n_rows = 2
    n_cols = 1

    fig = plt.figure(figsize=(15,12))
    outer = fig.add_gridspec(n_rows, n_cols, wspace=0.25, hspace=0.5)

    for i, file in enumerate(mic_list):
        df = pd.read_csv(file, sep=';')


        file_name = file.split('/')[-1].split('_')[0]
        if file_name.startswith('happen'):
            file_name = 'HAPPENN STYLE'
        elif file_name.startswith('white'):
            file_name = 'WHITELAB'

        inner = outer[i].subgridspec(1, 2, wspace=0.2, hspace=0.5)
        ax_left = fig.add_subplot(inner[0])
        ax_right = fig.add_subplot(inner[1])

        sequences_0 = df[df["label"] == 0]["sequence"].tolist()
        sequences_1 = df[df["label"] == 1]["sequence"].tolist()

        counts_0 = _tokenize(sequences_0)
        counts_1 = _tokenize(sequences_1)

        tick_labels = sorted(set(counts_0.keys()))  # | set(counts_1.keys()))
        vec_0 = np.array([counts_0[k] for k in tick_labels])
        vec_0 = vec_0 / vec_0.sum()

        vec_1 = np.array([counts_1[k] for k in tick_labels])
        vec_1 = vec_1 / vec_1.sum()

        x = np.arange(len(tick_labels))

        #fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        fig.suptitle("Aminosäuren Häufigkeits Vergleich", fontsize=12)
        ax_left.bar(x - 0.2, vec_0, 0.4, label="Inaktiv")
        ax_left.bar(x + 0.2, vec_1, 0.4, label="Aktiv")
        plt.xticks(x, tick_labels)
        ax_left.set_ylabel("AS Häufigkeit", size=12)
        ax_left.legend(fontsize=5)
        ax_left.set_xticks(x, tick_labels)
        ax_left.set_xlabel("Aminosäuren")

        # Train - gen
        ax_right.bar(x, vec_0 - vec_1, alpha=0.75)
        ax_right.set_ylabel("Differenz in der Häufigkeit", size=12)
        ax_right.set_xticks(x, tick_labels)
        ax_right.set_xlabel("Aminosäuren")
        fig.text((ax_left.get_position().x0 + ax_right.get_position().x1) / 2,
                 ax_left.get_position().y1 + 0.01,
                 file_name, ha="center", va="bottom", fontsize=12, fontweight="bold")

    fig.suptitle("Aminosäuren Häufigkeits Vergleich", fontsize=20)

    plt.savefig(f"{_PLOTS_ROOT}/hemo_aa_frequency_comparison.png")
    plt.close()



def main():
    mic_amino_frequency_comparison_multi([white_lab_train_data, happenn_style_train_data])



if __name__ == '__main__':
    length_per_label_comparison()
