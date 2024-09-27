import pandas as pd
from src.data_preprocessing.preprocess_utils import load_and_filter_data


def extract_starpep_data(starpep_fasta_path: str, out_path: str):
    """
    Filter the data from StarPEP database and save it to a csv file.
    This data contains a set of bioactive Peptides and is used here for pretrain the BERT model.
    Idea is to pretrain a BERT in the same way as ProteinBERT was pretrained so we can use this model
    for peptide related Tasks.

    Data accessed via: https://mobiosd-hub.com/starpep/
    Database tool needs JDK8 to run! You can download chosen sequences to a fasta file from there

    :param starpep_fasta_path: Path to the fasta file containing the StarPEP data
    :param out_path: Path to save the formatted data to
    """

    # Load data from file
    with open(starpep_fasta_path) as f:
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
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    extract_starpep_data(starpep_fasta_path='../../../data/data_from_database/general_peptides.fasta',
                         out_path='../../../data/train_data/starpep_sequences.csv')
