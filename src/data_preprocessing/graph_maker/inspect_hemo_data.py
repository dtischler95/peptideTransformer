import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from src.data_analysis.dataset_viewer import amino_acid_frequency_comparison

happenn_style_train_data = '../../../data/train_data/happen_style_unvoted.csv'

white_lab_train_data = '../../../data/train_data/whitelab_hemo_data.csv'


def label_comparison():


    df_happen = pd.read_csv(happenn_style_train_data, sep=';')
    label_counts_happen = df_happen["label"].value_counts()


    df_whitelab = pd.read_csv(white_lab_train_data, sep=';')
    label_counts_whitelab = df_whitelab["label"].value_counts()


    plt.figure(figsize=(10, 5))
    plt.suptitle(f"Label Verteilung für Hämotoxische Datensätze")
    plt.subplot(1, 2, 1)
    plt.bar(label_counts_whitelab.index, label_counts_whitelab.values / df_whitelab.shape[0] * 100, color=['blue', 'orange'])

    plt.title(f"Verteilung der Label Klassen (Whitelab)")
    plt.xlabel('Label')
    plt.ylabel("Häufigkeit [%")
    plt.subplot(1, 2, 2)
    plt.bar(label_counts_happen.index, label_counts_happen.values / df_happen.shape[0] * 100, color=['blue', 'orange'])
    plt.title(f"Verteilung der Label Klassen (HAPPENN)")
    plt.xlabel('Label')
    plt.ylabel("Häufigkeit [%]")
    plt.tight_layout()
    plt.savefig(f"../../../final_plots/hemo_label_comparison.png")
    plt.close()

def main():
    df = pd.read_csv(happenn_style_train_data, sep=';')

    sequences_0 = df[df["label"] == 0]["sequence"].tolist()
    sequences_1 = df[df["label"] == 1]["sequence"].tolist()
    amino_acid_frequency_comparison(sequences_0=sequences_0,
                                    sequences_1=sequences_1,
                                    plot_path=f"../../../final_plots/happen")



if __name__ == '__main__':
    main()
