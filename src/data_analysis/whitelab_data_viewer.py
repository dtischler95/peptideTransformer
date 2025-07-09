import pandas as pd
import matplotlib.pyplot as plt

"""
This script is thought to be used soley on the whitelab dataset.
Functions here provide some inside into the specific problems i had with the data provided by them and the
way the other labs handled them.

"""
# TODO COMPLETE REFACTOR! First Markdown makes sense, but for secend i need to remove the ambigous data first

def get_overall_stats(our_positive_path: str,
                      our_negative_path: str,
                      positive_to_compare_path: str,
                      negative_to_compare_path: str,
                      plot_path: str,
                      out_path: str):
    """
    Basic algorithm to compare a column of four different datasets and the occurrences of entries between each other
    Used here to get insights of the used sequences in our and whitelabs positive and negative data for
    hemolytic peptides.

    File Paths are fixed and should be changed if the data is moved

    :param our_positive_path: path to our positive data
    :param our_negative_path: path to our negative data
    :param positive_to_compare_path: path to whitelab positive data
    :param negative_to_compare_path: path to whitelab negative data
    :param plot_path: path to save the plots
    :param out_path: path to save the output files.
    """

    # Load our and whitelabs positive and negative data
    our_positive_df = pd.read_csv(our_positive_path, sep=';')
    our_negative_df = pd.read_csv(our_negative_path, sep=';')
    whitelab_negative_df = pd.read_csv(negative_to_compare_path, sep=';')
    whitelab_positive_df = pd.read_csv(positive_to_compare_path, sep=';')

    our_positive_percent, our_negative_percent, whitelab_positive_percent, whitelab_negative_percent = analyse_ambiguous_labeled_sequences_tabular(
        negative_df_our=our_negative_df,
        positive_df_our=our_positive_df,
        negative_df_whitelab=whitelab_negative_df,
        positive_df_whitelab=whitelab_positive_df)

    # Plot the relation between the positive and negative data
    plot_data_relation(x_data=['Positive', 'Negative'], y_data1=[our_positive_percent, our_negative_percent],
                       y_data2=[whitelab_positive_percent, whitelab_negative_percent], data1_name='Our_Data',
                       data2_name='Whitelab_Data', plot_path=plot_path, tag="Pos_Neg_distribution")

    # Get the common and uncommon sequences between our data and whitelab data and stats about the specific datafiles
    our_percent_uncommon_positive, whitelab_percent_uncommon_positive, our_percent_common_positive, whitelab_percent_common_positive = compare_sequence_occurrences_between_datasets_tabular(
        our_data=our_positive_df, whitelab_data=whitelab_positive_df, dataset_info='Positive_Data', out_path=out_path)
    our_percent_uncommon_negative, whitelab_percent_uncommon_negative, our_percent_common_negative, whitelab_percent_common_negative = compare_sequence_occurrences_between_datasets_tabular(
        our_data=our_negative_df, whitelab_data=whitelab_negative_df, dataset_info='Negative_Data', out_path=out_path)

    # Plot the relation between the common and uncommon sequences
    plot_data_relation(x_data=['Our_common', 'Whitelab_common', 'Our_uncommon', 'Whitelab_uncommon'],
                       y_data1=[our_percent_common_positive, whitelab_percent_common_positive,
                                our_percent_uncommon_positive, whitelab_percent_uncommon_positive],
                       y_data2=[our_percent_common_negative, whitelab_percent_common_negative,
                                our_percent_uncommon_negative, whitelab_percent_uncommon_negative],
                       data1_name='Positive Data', data2_name='Negative Data', plot_path=plot_path,
                       tag="Uncommon_Common_distribution")


def analyse_ambiguous_labeled_sequences_tabular(negative_df_our: pd.DataFrame, positive_df_our: pd.DataFrame,
                                                negative_df_whitelab: pd.DataFrame,
                                                positive_df_whitelab: pd.DataFrame) -> (float, float):
    """
    Count all data points in one dataset and calculate number and percent of unique sequences
    for positive and negative data

    :param negative_df_our: dataframe with negative data from our data
    :param positive_df_our: dataframe with positive data from our data
    :param negative_df_whitelab: dataframe with negative data from whitelab data
    :param positive_df_whitelab: dataframe with positive data from whitelab data

    :return: percentage of positive and negative data

    # TODO Track how many ambiguous labels we have with my labeling algorithm
    """
    ambiguous_sequences_df_our, data_sum_our, negative_percent_our, positive_percent_our, positives_that_are_in_negatives_our = _calc_statistics(
        negative_df_our, positive_df_our)

    ambiguous_sequences_df_whitelab, data_sum_whitelab, negative_percent_whitelab, positive_percent_whitelab, positives_that_are_in_negatives_whitelab = _calc_statistics(
        negative_df_whitelab, positive_df_whitelab)

    df_dict = {
        "Sum Data Points": [data_sum_our, data_sum_whitelab],
        "Positive Data [%]": [positive_percent_our, positive_percent_whitelab],
        "Negative Data [%]": [negative_percent_our, negative_percent_whitelab],
        "Unique Sequences": [len(positive_df_our) + len(negative_df_our),
                             len(positive_df_whitelab) + len(negative_df_whitelab)],
        "Unique Positives": [len(positive_df_our), len(positive_df_whitelab)],
        "Individual Positive data points": [positive_df_our['count'].sum(), positive_df_whitelab['count'].sum()],
        "Unique Negatives": [len(negative_df_our), len(negative_df_whitelab)],
        "Individual Negative data points": [negative_df_our['count'].sum(), negative_df_whitelab['count'].sum()],
        "Ambiguous unique sequences": [len(positives_that_are_in_negatives_our),
                                       len(positives_that_are_in_negatives_whitelab)],
        "Individual data points in both datasets": [ambiguous_sequences_df_our['count'].sum(),
                                                    ambiguous_sequences_df_whitelab['count'].sum()],
        "Percent of data points in both datasets [%]": [
            round((ambiguous_sequences_df_our['count'].sum() / (
                        positive_df_our['count'].sum() + negative_df_our['count'].sum())) * 100, 2),
            round((ambiguous_sequences_df_whitelab['count'].sum() / (
                    positive_df_whitelab['count'].sum() + negative_df_whitelab['count'].sum())) * 100, 2)]
    }

    df = pd.DataFrame(df_dict).T
    df = df.rename(columns={0: "Our Data", 1: "Whitelab Data"})

    print(df.to_markdown() + '\n')

    return positive_percent_our, negative_percent_our, positive_percent_whitelab, negative_percent_whitelab


def _calc_statistics(negative_df, positive_df):
    sequence_df = pd.concat([positive_df, negative_df])
    data_sum = sequence_df['count'].sum()
    positive_percent = (positive_df['count'].sum() / data_sum) * 100
    negative_percent = (negative_df['count'].sum() / data_sum) * 100
    positive_sequences = set(positive_df['sequence'])
    negative_sequences = set(negative_df['sequence'])
    # get unique sequences of positive sequences that are in negative sequences
    positives_that_are_in_negatives = positive_sequences.intersection(negative_sequences)
    ambiguous_sequences_df = sequence_df[sequence_df['sequence'].isin(positives_that_are_in_negatives)]
    return ambiguous_sequences_df, data_sum, negative_percent, positive_percent, positives_that_are_in_negatives


def compare_sequence_occurrences_between_datasets_tabular(our_data: pd.DataFrame,
                                                          whitelab_data: pd.DataFrame,
                                                          dataset_info: str,
                                                          out_path: str) -> (
        float, float, float, float):
    """
    Get the common sequences between our data and whitelab data, and calculate the percentage of common sequences

    :param our_data: dataframe with our data
    :param whitelab_data: dataframe with whitelab data
    :param dataset_info: name of the dataset
    :param out_path: path to save the files

    :return: percentage of unique sequences in our and whitelab data
    """
    # TODO ADD ABSOLUTE NUMBERS
    our_data_set = set(our_data['sequence'])
    whitelab_data_set = set(whitelab_data['sequence'])

    common_sequences = our_data_set.intersection(whitelab_data_set)
    our_not_common_sequences = our_data_set.difference(whitelab_data_set)
    whitelab_not_common_sequences = whitelab_data_set.difference(our_data_set)

    our_percent_common = (len(common_sequences) / len(our_data_set)) * 100
    whitelab_percent_common = (len(common_sequences) / len(whitelab_data_set)) * 100
    our_percent_uncommon = (len(our_not_common_sequences) / len(our_data_set)) * 100
    whitelab_percent_uncommon = (len(whitelab_not_common_sequences) / len(whitelab_data_set)) * 100

    stats_dict = {
        "Seqs in our data": [len(our_data_set)],
        "Seqs in whitelab data": [len(whitelab_data_set)],
        "Common seqs": [len(common_sequences)],
        "Common seqs % (our)": [round(our_percent_common, 2)],
        "Common seqs % (whitelab)": [round(whitelab_percent_common, 2)],
        "Seqs in our not in whitelab": [len(our_not_common_sequences)],
        "Seqs % (our not in whitelab)": [round(our_percent_uncommon, 2)],
        "Seqs in whitelab not in our": [len(whitelab_not_common_sequences)],
        "Seqs % (whitelab not in our)": [round(whitelab_percent_uncommon, 2)]
    }

    df = pd.DataFrame(stats_dict).T
    df = df.rename(columns={0: dataset_info})

    print(df.to_markdown() + '\n')

    # print sequences in common sequences to file
    print_set_to_file(sequences=common_sequences, dataset_info=dataset_info, file_tag='common_sequences', out_path=out_path)
    print_set_to_file(sequences=our_not_common_sequences, dataset_info=dataset_info,
                      file_tag='our_not_common_sequences', out_path=out_path)
    print_set_to_file(sequences=whitelab_not_common_sequences, dataset_info=dataset_info,
                      file_tag='whitelab_not_common_sequences', out_path=out_path)

    return our_percent_uncommon, whitelab_percent_uncommon, our_percent_common, whitelab_percent_common


def plot_data_relation(x_data,
                       y_data1,
                       y_data2,
                       data1_name: str,
                       data2_name: str,
                       plot_path: str,
                       tag: str):
    """
    plots the relation between two datasets

    :param x_data: x_axis data
    :param y_data1: y_axis data for dataset 1
    :param y_data2: y_axis data for dataset 2
    :param data1_name: name of the first dataset
    :param data2_name: name of the second dataset
    :param plot_path: path to save the plot
    :param tag: tag for the plot
    """
    # TODO REFACTOR GRAPHS LAYOUT AND STUFF
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
    plt.title(f"Percent of Sequences in {data1_name}")
    plt.xticks(rotation=45)
    write_value_on_bars(bars1)

    # Second bar plot
    plt.subplot(1, 2, 2)
    bars2 = plt.bar(x_data, y_data2)
    plt.xlabel('Negative Data')
    plt.ylabel('Percent of Sequences')
    plt.title(f"Percent of Sequences in {data2_name}")
    plt.xticks(rotation=45)
    write_value_on_bars(bars2)

    plt.tight_layout()
    plt.savefig(f'{plot_path}{tag}_{data1_name}_{data2_name}_relation.png')
    plt.show()
    plt.close()


def write_value_on_bars(bars):
    """
    Writes the value on top of the bars for a matplotlib bar plot

    :param bars: bars to write the values on
    """
    for bar in bars:
        yval = bar.get_height()
        if yval != 0:
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')


def print_set_to_file(sequences: set, dataset_info: str, file_tag: str, out_path: str):
    """
    Prints a set of sequences to a file

    :param sequences: set of sequences
    :param dataset_info: name of the dataset
    :param file_tag: tag for the file
    :param out_path: path to save the
    """
    with open(f'{out_path}{dataset_info}_{file_tag}.txt', 'w') as f:
        for seq in sequences:
            f.write(seq + '\n')


if __name__ == '__main__':
    get_overall_stats(our_negative_path='../../data/data_for_data_viewer/first_looks/our_negative.csv',
                      our_positive_path='../../data/data_for_data_viewer/first_looks/our_positive.csv',
                      negative_to_compare_path='../../data/data_for_data_viewer/first_looks/whitelab_data_negative_formatted.csv',
                      positive_to_compare_path='../../data/data_for_data_viewer/first_looks/whitelab_data_positive_formatted.csv',
                      plot_path='../../plots/',
                      out_path='../../data/sequence_analysis/')
