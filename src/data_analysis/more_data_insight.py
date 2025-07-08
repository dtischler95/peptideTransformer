import pandas as pd
from src.data_preprocessing.binary.prepare_binary_train_file import process_units


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

def inspect_mic_sequences():
    complete_amp_data_path = "../../data/data_from_database/complete_amp_data.csv"

    df = pd.read_csv(complete_amp_data_path, sep=';')
    df = df[df['organism'].isin(eskape_organisms)]
    df = df[df['value'].notna()]
    df = process_units(df, verbose=True, hemo_or_mic="mic")
    df = df[df['sequence'].str.len() >= 5]
    df = df[df['sequence'].str.len() <= 36]

    # First, I want to look at the measurements inside one sequence for each organism


    no_dif_counter = 0
    no_dif_sequences = []
    difference_sequences = []

    for organism, organism_df in df.groupby('organism'):
        no_dif_sequences_organism = []
        difference_sequences_organism = []
        print(f"Organism: {organism}")
        for sequence, sequence_df in organism_df.groupby('sequence'):
            difference = sequence_df['value'].max() - sequence_df['value'].min()
            if difference > 0:
                difference_sequences_organism.append(sequence)
                print(f"  Sequence: {sequence}")
                print(f"    Measurements: {sequence_df['value'].tolist()}")
                print(f"    Mean: {sequence_df['value'].mean()}")
                print(f"    Std: {sequence_df['value'].std()}")
                print(f"    Min: {sequence_df['value'].min()}")
                print(f"    Max: {sequence_df['value'].max()}")
            else:
                no_dif_counter += 1
                no_dif_sequences_organism.append(sequence)
        no_dif_sequences.append(no_dif_sequences_organism)
        difference_sequences.append(difference_sequences_organism)
    print(f"Number of sequences with no difference in measurements: {no_dif_counter}")





    print(1)





if __name__ == "__main__":


    hemolytik_path = "../../data/data_from_database/Hemolytik_scraped.csv"

    inspect_mic_sequences()