import numpy as np
from transformers import BertTokenizer
import pandas as pd
import urllib.request


def decoded_sequences_to_file(outfile_name: str,
                              create_formatted_whitelab_csv: bool = False,
                              verbose: bool = False):
    """
    Decodes and processes sequences from Whitelab data files, optionally creating a formatted CSV.

    :param outfile_name: Base name for output files
    :param create_formatted_whitelab_csv: Flag to create formatted Whitelab .csv file with a custom schema
    :param verbose: Flag for verbose output
    """

    # Define special tokens used in BERT tokenization
    special_tokens = ["[CLS]", "[SEP]", "[MASK]", "[UNK]", "[PAD]"]

    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd')

    # Load the .npz files containing the data
    neg_data = np.load('../data/whitelab_data/hemo-negative.npz')
    pos_data = np.load('../data/whitelab_data/hemo-positive.npz')

    # Process sequences and write to files if needed
    df_negative = write_sequences_to_list(f"../data/data_for_data_viewer/{outfile_name}_raw_negativ.txt", neg_data['arr_0'], tokenizer)
    df_positive = write_sequences_to_list(f"../data/data_for_data_viewer/{outfile_name}_raw_positive.txt", pos_data['arr_0'], tokenizer)

    if create_formatted_whitelab_csv:
        produce_whitelab_csv_file(df_positive, df_negative, special_tokens)

    format_whitelab_sequences(df_negative, f"{outfile_name}_negative", special_tokens, verbose=verbose)
    format_whitelab_sequences(df_positive, f"{outfile_name}_positive", special_tokens, verbose=verbose)


def write_sequences_to_list(file_name: str, sequences: np.ndarray, tokenizer: BertTokenizer) -> pd.DataFrame:
    """
    Decodes sequences using a tokenizer, writes to a file if required, and returns a DataFrame.

    :param file_name: Name of the output file
    :param sequences: Array of encoded sequences
    :param tokenizer: Tokenizer for decoding the sequences
    :param to_file: Flag to write the decoded sequences to a file
    :return: DataFrame containing decoded sequences
    """

    decoded_sequences = [tokenizer.decode(seq) for seq in sequences]
    special_tokens = ["[CLS]", "[SEP]", "[MASK]", "[UNK]", "[PAD]"]
    formatted_sequences = [''.join([token for token in seq.split() if token not in special_tokens])
                           for seq in decoded_sequences]
    df = pd.DataFrame(formatted_sequences, columns=['sequence'])
    df.to_csv(file_name, sep=';', index=False)

    return df


def format_whitelab_sequences(df: pd.DataFrame, outfile_name: str, special_tokens: list[str], verbose: bool = False):
    """
    Formats Whitelab sequences by removing special tokens and saving to a formatted CSV file.

    :param df: DataFrame containing sequences
    :param outfile_name: Output file name for formatted sequences
    :param special_tokens: List of special tokens to be removed from sequences
    :param verbose: Flag for verbose output
    """

    unique_sequences = {}

    for sequence in df['sequence']:
        filtered_sequence = ''.join([token for token in sequence.split() if token not in special_tokens])
        if filtered_sequence in unique_sequences:
            unique_sequences[filtered_sequence] += 1
        else:
            unique_sequences[filtered_sequence] = 1

    # Convert to DataFrame and save to CSV
    result_df = pd.DataFrame(unique_sequences.items(), columns=['sequence', 'count'])
    result_df.to_csv(f"../data/whitelab_data/{outfile_name}_formatted.csv", sep=';', index=False)

    if verbose:
        for seq, count in unique_sequences.items():
            print(f"Sequence: {seq} | Count: {count}")




def produce_whitelab_csv_file(positive_data: pd.DataFrame, negative_data: pd.DataFrame, special_tokens: list[str]):
    """
    Formats positive and negative Whitelab data into a single CSV file.

    :param positive_data: DataFrame with positive sequences
    :param negative_data: DataFrame with negative sequences
    :param special_tokens: List of special tokens to be removed from sequences
    """

    def format_whitelab_file(df: pd.DataFrame, label: int) -> pd.DataFrame:
        df['sequence'] = df['sequence'].apply(
            lambda x: ''.join([token for token in x.split() if token not in special_tokens]))
        df['label'] = label
        return df

    negative_df = format_whitelab_file(negative_data, 0)
    positive_df = format_whitelab_file(positive_data, 1)

    result_df = pd.concat([negative_df, positive_df])
    result_df.to_csv('../data/train_data/whitelab_hemo_data.csv', sep=';', index=False)


def download_and_prepare_whitelab_data(create_whitelab_csv: bool = False, verbose: bool = False):
    """
    From PeptideBERT repository
    https://github.com/ChakradharG/PeptideBERT/blob/master/data/download_data.py
    """

    m1 = [
        '[PAD]', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
    ]
    m2 = dict(zip(
        ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'L',
         'A', 'G', 'V', 'E', 'S', 'I', 'K', 'R', 'D', 'T', 'P', 'N',
         'Q', 'F', 'Y', 'M', 'H', 'C', 'W', 'X', 'U', 'B', 'Z', 'O'],
        range(30)
    ))

    def download_hemolysis():
        urllib.request.urlretrieve(
            'https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/hemo-positive.npz',
            '../data/whitelab_data/hemo-positive.npz',
        )
        urllib.request.urlretrieve(
            'https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/hemo-negative.npz',
            '../data/whitelab_data/hemo-negative.npz',
        )

    def func1(file):
        f = np.load(file)
        arr = np.array(
            list(map(
                lambda x: m2[m1[int(x)]],
                f['arr_0'].flat
            ))
        ).reshape(f['arr_0'].shape)

        np.savez(
            file,
            arr_0=arr
        )

    def func2(task):
        func1(f'../data/whitelab_data/{task}-positive.npz')
        func1(f'../data/whitelab_data/{task}-negative.npz')

    def prepare_downloaded_data():
        func2('hemo')

    # First Download the Whitelab Data
    download_hemolysis()

    # Whitelab has a different encoding. PeptideBERT uses a different encoding for the sequences
    prepare_downloaded_data()

    # outfile_name should stay whitelab, since this probably only works for the whitelab data
    decoded_sequences_to_file(outfile_name='whitelab_data', create_formatted_whitelab_csv=create_whitelab_csv,
                              verbose=verbose)


if __name__ == '__main__':
    # decoded_sequences_to_file('whitelab_data', create_intermediate_files=True, verbose=False)
    download_and_prepare_whitelab_data(create_whitelab_csv=True, verbose=False)
