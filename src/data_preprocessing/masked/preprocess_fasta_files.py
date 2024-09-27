import pandas as pd
from src.data_preprocessing.preprocess_utils import load_and_filter_data


def extract_sequences_from_fasta(fasta_path: str, out_path: str, to_file: bool = True) -> pd.DataFrame:
    """
    Function to extract sequences from given fasta file.
    We also apply our filter steps to the data.
    Idea is to pretrain a BERT in the same way as ProteinBERT was pretrained so we can use this model
    for peptide related Tasks.

    StarPEP Data accessed via: https://mobiosd-hub.com/starpep/
    Database tool needs JDK8 to run! You can download chosen sequences to a fasta file from there

    Peptide Atlas data accessed via: https://peptideatlas.org/builds/human/
    Fasta directly downloadable from there

    :param fasta_path: Path to the fasta file containing the StarPEP data
    :param out_path: Path to save the formatted data to. Not used if to_file is False
    :param to_file: Flag to save the data to a file
    """

    # Load data from file
    with open(fasta_path) as f:
        data = f.readlines()

    # Remove Header lines from Fasta file
    # We only need the Sequences for later MLM training
    starpep_data = []
    for line in data:
        if not line.startswith('>'):
            starpep_data.append(line.strip())

    # Create DataFrame and filter data
    df = load_and_filter_data(pd.DataFrame(data=starpep_data, columns=['sequence']))

    # Save data to csv file
    if to_file:
        df.to_csv(out_path, index=False)
        return df
    else:
        return df


if __name__ == '__main__':
    # extract_starpep_data(starpep_fasta_path='../../../data/data_from_database/starPEP_peptides.fasta',
    #                      out_path='../../../data/train_data/starpep_sequences.csv')

    extract_sequences_from_fasta(fasta_path='../../../data/data_from_database/APD_Hs_all.fasta',
                                 out_path='../../../data/train_data/apd_sequences.csv')
