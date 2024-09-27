import pandas as pd
from src.data_preprocessing.preprocess_utils import load_and_filter_data


def extract_starpep_data(file_path: str = '../../../data/data_from_database/general_peptides.fasta'):
    """
    Filter the data from StarPEP database and save it to a csv file.
    This data contains a set of bioactive Peptides and is used here for pretrain the BERT model.
    Idea is to pretrain a BERT in the same way as ProteinBERT was pretrained so we can use this model
    for peptide related Tasks.

    Data accessed via: https://mobiosd-hub.com/starpep/
    Database tool needs JDK8 to run! You can download chosen sequences to a fasta file from there
    """

    # Load data from file
    with open(file_path) as f:
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
    df.to_csv("../../../data/train_data/starpep_sequences.csv", index=False)


if __name__ == '__main__':
    extract_starpep_data()
