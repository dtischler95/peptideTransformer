import torch
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cpu')


def get_overall_stats():
    """
    Basic algorithm to compare a column of four different datasets and the occurrences of entries between each other
    Used here to get insights of the used sequences in our and whitelabs positive and negative data for
    hemolytic peptides.

    File Paths are fixed and should be changed if the data is moved
    """

    # Load our and whitelabs positive and negative data
    our_positive_df = pd.read_csv('../data/base_data/our_positive.csv', sep=';')
    our_negative_df = pd.read_csv('../data/base_data/our_negative.csv', sep=';')
    whitelab_negative_df = pd.read_csv('../data/base_data/whitelab_data_negative_formatted.csv', sep=';')
    whitelab_positive_df = pd.read_csv('data/base_data/whitelab_data_positive_formatted.csv', sep=';')

    # Gets basic statistics for the data and returns the percentage of positive and negative data for later plotting
    our_positive_percent, our_negative_percent = print_basic_info(negative_df=our_negative_df,
                                                                  positive_df=our_positive_df, info_for='Our Data')

    whitelab_positive_percent, whitelab_negative_percent = print_basic_info(negative_df=whitelab_negative_df,
                                                                            positive_df=whitelab_positive_df,
                                                                            info_for='Whitelab Data')

    # Plot the relation between the positive and negative data
    plot_data_relation(x_data=['Positive', 'Negative'],
                       y_data1=[our_positive_percent, our_negative_percent],
                       y_data2=[whitelab_positive_percent, whitelab_negative_percent],
                       dataset_info1='Our Data',
                       dataset_info2='Whitelab Data')

    # Get the common and uncommon sequences between our data and whitelab data and stats about the specific datafiles
    our_percent_uncommon_positive, whitelab_percent_uncommon_positive, our_percent_common_positive, whitelab_percent_common_positive = get_sequences_common_in_data(
        our_data=our_positive_df, whitelab_data=whitelab_positive_df, dataset_info='Positive Data')
    our_percent_uncommon_negative, whitelab_percent_uncommon_negative, our_percent_common_negative, whitelab_percent_common_negative = get_sequences_common_in_data(
        our_data=our_negative_df, whitelab_data=whitelab_negative_df, dataset_info='Negative Data')

    # Plot the relation between the common and uncommon sequences
    plot_data_relation(x_data=['Our_common', 'Whitelab_common', 'Our_uncommon', 'Whitelab_uncommon'],
                       y_data1=[our_percent_common_positive, whitelab_percent_common_positive,
                                our_percent_uncommon_positive, whitelab_percent_uncommon_positive],
                       y_data2=[our_percent_common_negative, whitelab_percent_common_negative,
                                our_percent_uncommon_negative, whitelab_percent_uncommon_negative],
                       dataset_info1='Positive Data',
                       dataset_info2='Negative Data')


def print_basic_info(negative_df: pd.DataFrame, positive_df: pd.DataFrame, info_for: str) -> (float, float):
    """
    Count all data points in one dataset and calculate number and percent of unique sequences
    for positive and negative data

    :param negative_df: dataframe with negative data
    :param positive_df: dataframe with positive data
    :param info_for: name of the dataset

    :return: percentage of positive and negative data
    """
    data_sum = positive_df['count'].sum() + negative_df['count'].sum()
    positive_percent = (positive_df['count'].sum() / data_sum) * 100
    negative_percent = (negative_df['count'].sum() / data_sum) * 100
    positive_sequences = set(positive_df['sequence'])
    negative_sequences = set(negative_df['sequence'])
    common_sequences = positive_sequences.intersection(negative_sequences)
    sequence_df = pd.concat([positive_df, negative_df])
    sequence_df = sequence_df[sequence_df['sequence'].isin(common_sequences)]

    print(f"\nInfo for: {info_for}")
    print(f"\tNumber of all our data_points: {data_sum}")
    print(f"\tPercent of positive Data {round(positive_percent, 2)}%")
    print(f"\tPercent of negative Data {round(negative_percent, 2)}%")
    print(f"\tNumber of unique sequences: {len(positive_df) + len(negative_df)}")
    print(f"\tNumber of unique Positives: {len(positive_df)}")
    print(f"\tNumber of unique Negatives: {len(negative_df)}")
    print(f"\tNumber of unique sequences appearing in positive and negative data: {len(common_sequences)}")
    print(f"\tNumber of individual data points in positive and negative data: {sequence_df['count'].sum()}")
    print(
        f"\tPercent of unique sequences appearing in both positive and negative data: {round((len(common_sequences) / (len(positive_df) + len(negative_df))) * 100, 2)}%")
    print(
        f"\tPercent of individual data points in positive and negative data: {round((sequence_df['count'].sum() / data_sum) * 100, 2)}%")

    return positive_percent, negative_percent


def get_sequences_common_in_data(our_data: pd.DataFrame, whitelab_data: pd.DataFrame, dataset_info: str) -> (
        float, float, float, float):
    """
    Get the common sequences between our data and whitelab data, and calculate the percentage of common sequences

    :param our_data: dataframe with our data
    :param whitelab_data: dataframe with whitelab data
    :param dataset_info: name of the dataset

    :return: percentage of unique sequences in our and whitelab data
    """
    our_data_set = set(our_data['sequence'])
    whitelab_data_set = set(whitelab_data['sequence'])

    common_sequences = our_data_set.intersection(whitelab_data_set)
    our_not_common_sequences = our_data_set.difference(whitelab_data_set)
    whitelab_not_common_sequences = whitelab_data_set.difference(our_data_set)

    our_percent_common = (len(common_sequences) / len(our_data_set)) * 100
    whitelab_percent_common = (len(common_sequences) / len(whitelab_data_set)) * 100
    our_percent_uncommon = (len(our_not_common_sequences) / len(our_data_set)) * 100
    whitelab_percent_uncommon = (len(whitelab_not_common_sequences) / len(whitelab_data_set)) * 100

    print(f"\n\nInfo for: {dataset_info}")
    print(f"\tNumber of sequences in our data: {len(our_data_set)}")
    print(f"\tNumber of sequences in whitelab data: {len(whitelab_data_set)}")
    print(f"\tNumber of common sequences: {len(common_sequences)}")
    print(f"\tPercent of common sequences for our data: {round(our_percent_common, 2)}%")
    print(f"\tPercent of common sequences for whitelab data: {round(whitelab_percent_common, 2)}%")
    print(f"\tNumber of sequences in our data which are not in whitelab: {len(our_not_common_sequences)}")
    print(f"\tPercent of sequences in our data which are not in whitelab: {round(our_percent_uncommon, 2)}%")
    print(f"\tNumber of sequences in whitelab data which are not in our: {len(whitelab_not_common_sequences)}")
    print(f"\tPercent of sequences in whitelab data which are not in our: {round(whitelab_percent_uncommon, 2)}%")

    # print sequences in common sequences to file
    print_set_to_file(sequences=common_sequences, dataset_info=dataset_info, file_tag='common_sequences')
    print_set_to_file(sequences=our_not_common_sequences, dataset_info=dataset_info,
                      file_tag='our_not_common_sequences')
    print_set_to_file(sequences=whitelab_not_common_sequences, dataset_info=dataset_info,
                      file_tag='whitelab_not_common_sequences')

    return our_percent_uncommon, whitelab_percent_uncommon, our_percent_common, whitelab_percent_common


def plot_data_relation(x_data, y_data1, y_data2, dataset_info1: str, dataset_info2: str):
    """
    plots the relation between two datasets

    :param x_data: x axis data
    :param y_data1: y axis data for dataset 1
    :param y_data2: y axis data for dataset 2
    :param dataset_info1: name of the first dataset
    :param dataset_info2: name of the second dataset
    """

    if len(y_data1) == 4:
        y_data1.insert(2, 0.0)
        y_data2.insert(2, 0.0)
        x_data.insert(2, ' ')
    plt.figure(figsize=(10, 5))

    # First bar plot
    plt.subplot(1, 2, 1)
    bars1 = plt.bar(x_data, y_data1)
    plt.xlabel('Positive Data')
    plt.ylabel('Percent of Sequences')
    plt.title(f"Percent of Sequences in {dataset_info1}")
    plt.xticks(rotation=45)
    write_value_on_bars(bars1)

    # Second bar plot
    plt.subplot(1, 2, 2)
    bars2 = plt.bar(x_data, y_data2)
    plt.xlabel('Negative Data')
    plt.ylabel('Percent of Sequences')
    plt.title(f"Percent of Sequences in {dataset_info2}")
    plt.xticks(rotation=45)
    write_value_on_bars(bars2)

    plt.tight_layout()
    plt.savefig(f'../plots/{dataset_info1}_{dataset_info2}_relation.png')


def write_value_on_bars(bars):
    """
    Writes the value on top of the bars for a matplotlib bar plot

    :param bars: bars to write the values on
    """
    for bar in bars:
        yval = bar.get_height()
        if yval != 0:
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')


def print_set_to_file(sequences: set, dataset_info: str, file_tag: str):
    """
    Prints a set of sequences to a file

    :param sequences: set of sequences
    :param dataset_info: name of the dataset
    :param file_tag: tag for the file
    """
    with open(f'../data/sequence_analysis/{dataset_info}_{file_tag}.txt', 'w') as f:
        for seq in sequences:
            f.write(seq + '\n')


if __name__ == '__main__':
    get_overall_stats()
