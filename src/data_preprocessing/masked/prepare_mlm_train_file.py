import pandas as pd
from src.data_preprocessing.preprocess_utils import load_and_filter_data
from src.data_preprocessing.masked.preprocess_fasta_files import extract_sequences_from_fasta


def concat_all_files(sequence_data: list[str] or list[tuple[str, pd.DataFrame]],
                     down_sample_fraction: float,
                     out_path: str,
                     load_from_path: bool = True):
    """
    Basic logic for concatenating sequence data for MLM training.
    This Function can take any number of csv files that contain a column named 'sequence'.
    We load and apply our filter to the data and concatenate it to a single DataFrame.
    In the end we drop duplicates and save the data to a new csv file.
    Create more if cases if new data need some special treatment

    :param sequence_data: Path to csv file or as a tuple of (name, pd.DataFrame) for the data to be concatenated.
    :param down_sample_fraction: Fraction of the data to down sample. Only used for the APD data.
    :param out_path: Path to save the concatenated data to
    :param load_from_path: Flag to load data from path or from DataFrame

    """
    dataframes = []

    # use Paths for single function use or easier debugging
    if load_from_path:
        for file_path in sequence_data:
            df = load_and_filter_data(file_path, sep=';', filter_on_column='sequence')['sequence']
            dataframes.append((file_path.split('/')[-1], df))
    # more convenient use in pipeline
    else:
        for df in sequence_data:
            dataframes.append((df[0], df[1]))

    tmp_df = pd.DataFrame()
    for df in dataframes:
        if df[0] == "APD_Hs_all.fasta":  # Add more if conditions if new data needs special treatment

            tmp_df = pd.concat([tmp_df, df[1].sample(frac=down_sample_fraction, random_state=42)])
            print(
                f"\n --> Data from {df[0]} added {df[1].sample(frac=down_sample_fraction).shape[0]} to the Training Data from original {df[1].shape[0]} Datapoints")
            print(
                f"\033[31m\n\t --> {df[0]} contains many not necessarily Bioactive Peptides.\n\t\t We will down sample this data to ensure a more balanced training data.\n\t\t Since we try to generalize the language of Peptides but also keep a strong representation of Bioactive Peptides in later embeddings.\033[0m")

        else:
            tmp_df = pd.concat([tmp_df, df[1]])
            print(f"\n --> Data from {df[0]} added {df[1].shape[0]} to the Training Data")

    print(f"\033[31m\n\t --> Number of duplicates removed: {tmp_df.shape[0] - tmp_df.drop_duplicates().shape[0]}\033[0m")
    tmp_df = tmp_df.drop_duplicates()
    print(f"\033[32m\n\t --> New Number of data: {tmp_df.shape[0]}\033[0m")
    tmp_df.to_csv(out_path, sep=';', index=False)


def gather_data_and_generate_mlm_train_file(*input_file_paths: str,
                                            out_path: str,
                                            down_sample_if_to_big: float):
    """
    This function is used to gather all data needed for the generate_mlm_train_file function.
    We use this function to keep the generate_mlm_train_file function clean and easy to read.

    Made for use in notebook.

    :param input_file_paths: Paths to the fasta files or csv files containing the data
    :param out_path: Path to save the concatenated data to
    :param down_sample_if_to_big: Fraction of the data to down sample. Only used for the APD data.
    """
    dataframes = []
    for file_path in input_file_paths:
        if file_path.endswith('.fasta'):
            tmp_df = extract_sequences_from_fasta(fasta_path=file_path,
                                                  out_path="No effect when to_file is False",
                                                  to_file=False)
            dataframes.append((file_path.split('/')[-1], tmp_df))
        elif file_path.endswith('.csv'):
            tmp_df = load_and_filter_data(file_path, sep=';', filter_on_column='sequence')['sequence']
            dataframes.append((file_path.split('/')[-1], tmp_df))

    concat_all_files(sequence_data=dataframes,
                     down_sample_fraction=down_sample_if_to_big,
                     load_from_path=False,
                     out_path=out_path)


if __name__ == '__main__':
    gather_data_and_generate_mlm_train_file('../../../data/data_from_database/starPEP_peptides.fasta',
                                            '../../../data/data_from_database/APD_Hs_all.fasta',
                                            '../../../data/data_from_database/complete_amp_data.csv',
                                            out_path='../../../data/train_data/mlm/mlm_train_data.csv',
                                            down_sample_if_to_big=0.1
                                            )
