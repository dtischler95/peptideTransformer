import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

"""
This file is inteded for creation of images for my thesis.
This should only contain functions to create systematic views of the data
"""

happenn_style_train_data = '../../../data/train_data/happen_style_unvoted.csv'

white_lab_train_data = '../../../data/train_data/whitelab_hemo_data.csv'

white_lab_train_data_clean = '../../../data/train_data/whitelab_filtered_clean.csv'

def get_pretty_organism_name(file_path: str) -> str:
    name = '_'.join(file_path.split('/')[-1].split('_')[:2]).replace('_', ' ').capitalize()
    if name.startswith('Enterobacter'):
        name = 'Enterobacter sp.'
    return name

def get_mic_files():
    file_list = []
    mic_file_path = "../../../data/regression_data/"

    for file in os.listdir(mic_file_path):
        if file.endswith(".csv"):
            file_list.append(mic_file_path + file)

    return file_list

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
    plt.suptitle(f"Verteilung der MIC Werte für {organism}")
    plt.subplot(1, 2, 1)
    plt.hist(df['value'], bins=50)
    plt.title(f"Verteilung der MIC Werte")
    plt.xlabel('MIC (µM)')
    plt.ylabel("Häufigkeit")
    plt.grid(True, axis='y')
    plt.subplot(1, 2, 2)
    plt.hist(df['mic_log10'], bins=30)
    plt.title(f"Verteilung der MIC Werte (log10 transformiert)")
    plt.xlabel('MIC (log10(µM))')
    plt.ylabel("Häufigkeit")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"../../../final_plots/{organism}_mic_distribution.png")
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
    fig.suptitle("Violin Plot der MIC Werte (log10 transformiert)\n", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"../../../final_plots/violin_mic_distribution.png")
    plt.close()


def hist_mic_distribution(mic_files: list[str]):
    """
    Shows a histogram of the log10 MIC values for each organism in a subplot grid.
    """
    # Layout für Subplots
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
        axes[i].set_ylabel('Häufigkeit', size=12)
        axes[i].grid(True, axis='y')

    fig.suptitle("Histogramm der MIC Werte (log10 transformiert)\n", fontsize=20)
    plt.tight_layout()
    plt.savefig("../../../final_plots/hist_mic_distribution.png")
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
    hist_plot_mic(combined_df, "ESKAPE Organismen")


def main():
    mic_files = get_mic_files()
    combined_hist_plot(mic_files=mic_files)
    #hist_mic_distribution(mic_files=mic_files)
    #mic_distribution_histogram(mic_files=mic_files)
    # violin_mic_distribution(mic_files=mic_files)



if __name__ == "__main__":
    main()