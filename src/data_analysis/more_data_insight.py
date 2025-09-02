import pandas as pd
from src.data_preprocessing.binary.prepare_binary_train_file import process_units
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import seaborn as sns

eskape_organisms = [
    "Acinetobacter baumannii",
    "Bacillus subtilis",
    "Candida albicans",
    "Enterococcus faecalis",
    "Micrococcus luteus",
    "Salmonella enterica",
    "Klebsiella pneumoniae",
    "Pseudomonas aeruginosa",
    "Escherichia coli",
    "Enterobacter cloacae",
    "Staphylococcus aureus",
    "Staphylococcus epidermidis",
]

def inspet_hemolytik_sequences():
    hemolytik_path = "../../data/train_data/happen_style_unvoted.csv" # File for hemolytik sequences, with target human
    df = pd.read_csv(hemolytik_path, sep=';')



    for sequence, sequence_df in df.groupby('sequence'):
        if sequence_df.shape[0] > 1:
            percent_difference = sequence_df['hemo_percent'].max() - sequence_df['hemo_percent'].min()
            conc_difference = sequence_df['hemo_concentration'].max() - sequence_df['hemo_concentration'].min()
            if percent_difference > 0.1:
                print(1)



    print(1)

def inspect_mic_sequences():
    #complete_amp_data_path = "../../data/data_for_data_viewer/complete_amp_inspection_data.csv"
    complete_amp_data_path = "../../data/data_for_data_viewer/complete_amp_inspection_data_filtered_value_below_1000.csv"
    #complete_amp_data_path = "../../data/data_from_database/complete_amp_data.csv"



    df = pd.read_csv(complete_amp_data_path, sep=';')

    # Uncomment if u working with the base data from the database. Filter seq length and calculate units
    # df = df[df['organism'].isin(eskape_organisms)]
    # df = df[df['value'].notna()]
    # df = process_units(df, verbose=True, hemo_or_mic="mic")
    # df = df[df['sequence'].str.len() >= 5]
    # df = df[df['sequence'].str.len() <= 36]
    # df = df[df['value'] <= 1000.0]  # Filter out non-positive values
    # df.to_csv("../../data/data_from_database/complete_amp_inspection_data_filtered_value_below_1000.csv", index=False, sep=';')

    # First, I want to look at the measurements inside one sequence for each organism



    no_dif_sequences = []



    # Split for all Organisms and then for each sequence
    for organism, organism_df in df.groupby('organism'):

        # Sequences with no difference in measurement values
        no_dif_sequences_organism = []

        differences_for_plot = []

        summary_data = []

        print(f"Organism: {organism}")
        for sequence, sequence_df in organism_df.groupby('sequence'):

            difference = sequence_df['value'].max() - sequence_df['value'].min()
            if difference > 0:
                differences_for_plot.append((sequence ,difference))
                summary_data.append({
                    'sequence': sequence,
                    'mean': sequence_df['value'].mean(),
                    'std': sequence_df['value'].std(),
                    'min': sequence_df['value'].min(),
                    'max': sequence_df['value'].max()
                })

            else:
                no_dif_sequences_organism.append(sequence)
        no_dif_sequences.append(no_dif_sequences_organism)
        summary_df = pd.DataFrame(summary_data)
        #summary_df.to_csv(f"../../data/data_for_data_viewer/mic_organisms/{organism}_summary_below_1000.csv", index=False, sep=';')

        plot_box_aggregated(df=summary_df, organism=organism)
        plot_scatter(df=summary_df, organism=organism)
        plot_error_bars_numeric(df=summary_df, organism=organism)



def plot_box_aggregated(df, organism):
    # Group sequences into bins based on mean values
    df['mean_bin'] = pd.cut(df['mean'], bins=np.linspace(df['mean'].min(), df['mean'].max(), 10))
    grouped = df.groupby('mean_bin', observed=True)

    # Prepare data for box plot
    box_data = [group['mean'].values for _, group in grouped]

    plt.figure(figsize=(12, 8))
    plt.boxplot(box_data, tick_labels=[str(bin) for bin in grouped.groups.keys()], vert=True)
    plt.xlabel('Mean Value Bins')
    plt.ylabel('Value')
    plt.title(f'Box Plot: Aggregated Distribution of Values ({organism})')
    plt.tight_layout()
    plt.savefig(f"../../data/data_for_data_viewer/mic_organisms/plots/{organism}_box_plot.png")
    plt.close()
    plt.clf()

def plot_scatter(df, organism):
    plt.figure(figsize=(12, 8))
    plt.scatter(df['mean'], df['std'], c='blue', alpha=0.7, label='Data Points')

    # Calculate regression line
    slope, intercept, r_value, _, _ = linregress(df['mean'], df['std'])
    regression_line = slope * df['mean'] + intercept

    # Plot regression line
    plt.plot(df['mean'], regression_line, color='red', label=f'Regression Line (R²={r_value**2:.2f})')

    # Labels and title
    plt.xlabel('Mean')
    plt.ylabel('Standard Deviation')
    plt.title(f'Scatter Plot with Regression Line: Mean vs Standard Deviation ({organism})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../../data/data_for_data_viewer/mic_organisms/plots/{organism}_scatter_plot.png")
    plt.close()
    plt.clf()

def plot_error_bars_numeric(df, organism):
    # Group sequences into bins based on mean values
    df['mean_bin'] = pd.cut(df['mean'], bins=np.linspace(df['mean'].min(), df['mean'].max(), 20))
    grouped = df.groupby('mean_bin', observed=True)

    # Calculate aggregated metrics
    bin_means = grouped['mean'].mean()
    bin_stds = grouped['std'].mean()

    # Plot aggregated error bars
    plt.figure(figsize=(12, 8))
    plt.errorbar(range(len(bin_means)), bin_means, yerr=bin_stds, fmt='o', ecolor='red', capsize=5, label='Mean ± Std')
    plt.xlabel('Mean Value Bins')
    plt.ylabel('MIC')
    plt.title(f'Error Bars: Aggregated Mean and Standard Deviation ({organism})')
    plt.xticks(range(len(bin_means)), [str(bin) for bin in grouped.groups.keys()], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../../data/data_for_data_viewer/mic_organisms/plots/{organism}_error_plot.png")
    plt.close()
    plt.clf()

def search_mic_discrepancies():
    """
    Search for discrepancies in MIC values for the same sequence across different organisms.
    """
    df = pd.read_csv('../../data/data_for_data_viewer/complete_amp_inspection_data_filtered_value_below_1000.csv', sep=';')
    df = df[df['measure_type'] == 'MIC']
    df = df[['source', 'peptide_name','measure_type' , 'sequence', 'organism', 'strain', 'value', 'unit', 'PMID/Uniprot']]
    difference_counter = 0
    no_dif_counter = 0
    for organism, organism_df in df.groupby('organism'):
        for sequence, sequence_df in organism_df.groupby('sequence'):
            if sequence_df.shape[0] > 1:

                difference = sequence_df['value'].max() - sequence_df['value'].min()
                if difference > 100.0:
                    difference_counter += 1

                elif difference == 0.0:
                    no_dif_counter += 1
            else:
                print(0)


    total_sequences = difference_counter + no_dif_counter
    differences_in_percentage = (difference_counter / total_sequences) * 100
    no_dif_in_percentage = (no_dif_counter / total_sequences) * 100


    print(f"Number of sequences with a difference in MIC: {round(differences_in_percentage, 2)}")
    print(f"Number of sequences with no difference in MIC: {round(no_dif_in_percentage, 2)}")
    print(f"Total number of sequences: {difference_counter + no_dif_counter}")
if __name__ == "__main__":


    hemolytik_path = "../../data/data_from_database/Hemolytik_scraped.csv"

    inspect_mic_sequences()