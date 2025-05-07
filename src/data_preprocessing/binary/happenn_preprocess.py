import pandas as pd

"""
This script is used to prepare the data from the HAPENN database for training a binary classification model.
The data is in FASTA format and contains sequences and their corresponding labels (hemolytic or non-hemolytic).
The script reads the data, extracts the sequences and labels, and saves them in a CSV file.
"""


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
    df.to_csv('../../data/train_data/happen_data.csv', sep=';', index=False)
    print(sequence_counter)
    print(len(sequence_label_dict['sequence']))


if __name__ == "__main__":
    file_path = "../../data/data_from_database/happen_data.fasta"
    prepare_happenn_data(file_path=file_path)
