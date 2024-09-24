import pandas as pd


def load_and_filter_data(path: str, sep: str = ';', filter_on_column: str = 'sequence') -> pd.DataFrame:
    """
    Load AMP related data from a csv file and filter out sequences with our Filter Schema.

    :param path: Path to the csv file.
    :param sep: Separator used in the csv file.
    :param filter_on_column: Column to filter on. should be always sequence. But sometimes the column is named differently.

    :return: Filtered DataFrame.
    """
    df = pd.read_csv(filepath_or_buffer=path, sep=sep)
    df = df[df[filter_on_column].str.len() <= 36]
    df = df[df[filter_on_column].str.len() >= 3]

    df[filter_on_column] = df[filter_on_column].str.upper()
    df = df[~df[filter_on_column].str.contains('[^ACDEFGHIKLMNPQRSTVWY]')]
    df = df[~df[filter_on_column].str.contains('X')]
    df = df[~df[filter_on_column].str.contains('Z')]
    df = df[~df[filter_on_column].str.contains('B')]
    df = df[~df[filter_on_column].str.contains('J')]
    df = df[~df[filter_on_column].str.contains('U')]
    df = df[~df[filter_on_column].str.contains('O')]
    df = df[~df[filter_on_column].str.contains('-')]
    return df


def split_positive_and_negative(data: str or pd.DataFrame, to_file: bool = False):
    """
    Splits data into positive and negative sequences based on labels.
    Used for our train data only so far

    :param data: Path to the CSV file or just a DataFrame
    :param to_file: Flag to save the split data to files
    :return: DataFrames for positive and negative data if to_file is False
    """
    if type(data) == str:
        df = pd.read_csv(data, sep=';')
    elif type(data) == pd.DataFrame:
        df = data
    else:
        raise ValueError('Invalid data type. Please provide a path to a CSV file or a DataFrame.')

    positive_df = df[df['label'] == 1].groupby('sequence').size().reset_index(name='count')
    negative_df = df[df['label'] == 0].groupby('sequence').size().reset_index(name='count')

    if to_file:
        positive_df.to_csv('../data/data_for_data_viewer/our_positive.csv', sep=';', index=False)
        negative_df.to_csv('../data/data_for_data_viewer/our_negative.csv', sep=';', index=False)
    else:
        return positive_df, negative_df

def filter_and_evaluate_ambiguous_sequences(labeled_df: pd.DataFrame, out_path: str = '../data/train_data/our_hemo_filtered_labeled.csv'):
    """
    Filter ambiguous sequences and sort them into positive or negative based on the majority label.

    :param labeled_df: DataFrame containing labeled sequences.
    """

    result_df = pd.DataFrame()

    # Group by sequence and check if there are multiple labels for the same sequence
    for seq_df in labeled_df.groupby(by=['sequence']):

        # If there are multiple labels for the same sequence, sort them into positive or negative based on the majority label
        if seq_df[1]['label'].unique().shape[0] > 1:
            positive_compare = pd.DataFrame()
            negative_compare = pd.DataFrame()
            for label_df in seq_df[1].groupby(by=['label']):

                # sort the data into positive or negative
                if label_df[0][0] == 0:
                    negative_compare = label_df[1]
                elif label_df[0][0] == 1:
                    positive_compare = label_df[1]

            # Choosing Majority label is done here
            if positive_compare.shape[0] > negative_compare.shape[0]:
                result_df = pd.concat([result_df, positive_compare])
            elif positive_compare.shape[0] < negative_compare.shape[0]:
                result_df = pd.concat([result_df, negative_compare])

            # If the number of positive and negative labels is the same, choose the positive label
            else:
                result_df = pd.concat([result_df, positive_compare])

        # If there is only one label for the sequence, add it to the result DataFrame
        else:
            result_df = pd.concat([result_df, seq_df[1]])

    result_df.to_csv(out_path, sep=';', index=False)

