"""
This file is meaned for script like access to data for viewing and retrieving metadata.
"""
import pandas as pd


def compare_rows_of_two_dataframes(df_1: pd.DataFrame, df_2: pd.DataFrame):
    """
    Check for overlapping data from two dataframes and prints the count of overlapping and non-overlapping data.
    """
    print('Checking for overlapping data...')
    print(f"Dataframe 1: {df_1['sequence'].drop_duplicates().shape[0]} rows")
    print(f"Dataframe 2: {df_2['sequence'].drop_duplicates().shape[0]} rows")

    # Check for overlapping data
    df_1_sequences = set(df_1['sequence'])
    df_2_sequences = set(df_2['sequence'])
    all_sequences = df_1_sequences.union(df_2_sequences)
    overlapping_sequences = df_1_sequences.intersection(df_2_sequences)
    non_overlapping_sequences = all_sequences - overlapping_sequences

    print(f"All sequences: {len(all_sequences)}")
    print(f"Overlapping sequences: {len(overlapping_sequences)}")
    print(f"Non-overlapping sequences: {len(non_overlapping_sequences)}")

    print(1)


def prepare_happenn_data(file_path: str):

    sequence_label_dict = {'sequence': [], 'label': []}
    sequence_counter = 0
    with open(file_path, 'r') as file:
        for line in file.readlines():
            if line.startswith('>'):

                if line.split('|')[7].strip() == 'hemolytic':
                    sequence_label_dict.update({'label': sequence_label_dict['label'] + [1]})
                elif line.split('|')[7].strip() == 'non-hemolytic':
                    sequence_label_dict.update({'label': sequence_label_dict['label'] + [0]})

            else:
                sequence_label_dict.update({'sequence': sequence_label_dict['sequence'] + [line.strip()]})
                sequence_counter += 1

    df = pd.DataFrame(sequence_label_dict)
    df.to_csv('../../data/data_from_database/happen_data.csv', sep=';', index=False)
    print(sequence_counter)
    print(len(sequence_label_dict['sequence']))



if __name__ == '__main__':

    df_1 = pd.read_csv('../../data/train_data/our_hemo_labeled.csv', sep=';')
    df_2 = pd.read_csv('../../data/data_from_database/happen_data.csv', sep=';')
    df_3 = pd.read_csv('../../data/train_data/whitelab_hemo_data.csv', sep=';')
    df_tmp = pd.concat([df_1, df_2, df_3])
    compare_rows_of_two_dataframes(df_1, df_2)

    # file_path = "../../data/data_from_database/happen_data.fasta"
    #
    # prepare_happenn_data(file_path=file_path)