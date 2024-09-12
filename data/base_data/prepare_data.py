import numpy as np
from transformers import BertTokenizer
import pandas as pd
import urllib.request


def download_and_prepare_whitelab_data(create_intermediate_files: bool = False, verbose: bool = False):
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
            './hemo-positive.npz',
        )
        urllib.request.urlretrieve(
            'https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/hemo-negative.npz',
            './hemo-negative.npz',
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
        func1(f'./{task}-positive.npz')
        func1(f'./{task}-negative.npz')

    def main():
        func2('hemo')

    download_hemolysis()
    main()
    # outfile_name should stay whitelab, since this probably only works for the whitelab data
    decoded_sequences_to_file(outfile_name='whitelab_data', create_intermediate_files=create_intermediate_files,
                              verbose=verbose)


def decoded_sequences_to_file(outfile_name: str,
                              create_intermediate_files: bool = False,
                              verbose: bool = False):
    """
    Use for whitelab data, so we can see the sequences in the file
    Need convert_encodings from peptideBERT to convert the sequences so the tokenizer can decode them correctly

    :param outfile_name:
    :param create_intermediate_files:
    :param verbose:
    :return:
    """

    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd')

    """
    *.npz files are downloaded from here:
    https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/hemo-positive.npz
    https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/hemo-negative.npz
    """
    neg_data = np.load('./hemo-negative.npz')
    pos_data = np.load('./hemo-positive.npz')

    df_negativ = write_sequences_to_list(f"./{outfile_name}_raw_negativ.txt", neg_data['arr_0'], tokenizer,
                                         to_file=create_intermediate_files)
    df_positiv = write_sequences_to_list(f"./{outfile_name}_raw_positive.txt", pos_data['arr_0'], tokenizer,
                                         to_file=create_intermediate_files)

    format_whitelab_sequences(df=df_negativ, outfile_name=f"{outfile_name}_negative", verbose=verbose)
    format_whitelab_sequences(df=df_positiv, outfile_name=f"{outfile_name}_positive", verbose=verbose)


def write_sequences_to_list(file_name: str, sequences: list, tokenizer: BertTokenizer,
                            to_file: bool = False) -> pd.DataFrame:
    """
    Writes the sequences to a list and optionally to a file

    :param file_name: name of the file to write to
    :param sequences: list of sequences
    :param tokenizer: tokenizer to decode the sequences
    :param to_file: if the sequences should be written to a file

    :return: dataframe of the sequences
    """

    sequence_list = []
    for seq in sequences:
        decoded_seq = tokenizer.decode(seq)
        sequence_list.append(decoded_seq)

    tmp_df = pd.DataFrame(sequence_list, columns=['sequence'])
    if to_file:
        tmp_df.to_csv(file_name, sep=';', index=False)

    return tmp_df


def format_whitelab_sequences(df: pd.DataFrame, outfile_name: str, verbose: bool = False):
    """
    Formats the whitelab data to a csv file with the unique sequences and their counts

    :param df: dataframe with the sequences
    :param outfile_name: name of the output file
    """
    special_tokens = [
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "[UNK]",
        "[PAD]"
    ]
    unique_sequences = {}

    for line in df.iloc[:, 0]:
        sequence = []
        splitted_line = line.strip().split(' ')
        for token in splitted_line:
            if token not in special_tokens:
                sequence.append(token)
        joined_sequence = ''.join(sequence)
        if joined_sequence not in unique_sequences:
            unique_sequences[joined_sequence] = 1
        else:
            unique_sequences[joined_sequence] += 1

    result_df = pd.DataFrame(unique_sequences.items(), columns=['sequence', 'count'])
    result_df.to_csv(f"./{outfile_name}_formatted.csv", sep=';', index=False)
    if verbose:
        for sequence, count in unique_sequences.items():
            print(f"Seq {sequence} with count: {count}")


def split_positive_and_negativ(file_path: str):
    """
    Splits our data into positive and negative data for further analysis
    """
    df = pd.read_csv(file_path, sep=';')

    positive_df = df[df['label'] == 1].groupby('sequence').size().reset_index(name='count')
    positive_df.to_csv('our_positive.csv', sep=';', index=False)
    negative_df = df[df['label'] == 0].groupby('sequence').size().reset_index(name='count')
    negative_df.to_csv('our_negative.csv', sep=';', index=False)


if __name__ == '__main__':
    download_and_prepare_whitelab_data(create_intermediate_files=True,
                                       verbose=False)
    split_positive_and_negativ('./splitted_hemo_labeled.csv')
