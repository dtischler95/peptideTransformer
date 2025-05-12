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

def label_after_happen_style(df: pd.DataFrame) -> pd.DataFrame:

    row_already_flagged = False

    if df['hemo_percent'] >= 50 and df['hemo_concentration'] <= 300 and not row_already_flagged:
        df['label'] = 1
        row_already_flagged = True

    if df['hemo_percent'] >= 55 and df['hemo_concentration'] <= 330 and not row_already_flagged:
        df['label'] = 1
        row_already_flagged = True

    if df['hemo_percent'] >= 60 and df['hemo_concentration'] <= 360 and not row_already_flagged:
        df['label'] = 1
        row_already_flagged = True

    if df['hemo_percent'] >= 70 and df['hemo_concentration'] <= 420 and not row_already_flagged:
        df['label'] = 1
        row_already_flagged = True

    if df['hemo_percent'] >= 75 and df['hemo_concentration'] <= 450 and not row_already_flagged:
        df['label'] = 1
        row_already_flagged = True

    if df['hemo_percent'] >= 80 and df['hemo_concentration'] <= 480 and not row_already_flagged:
        df['label'] = 1
        row_already_flagged = True

    if df['hemo_percent'] >= 85 and df['hemo_concentration'] <= 510 and not row_already_flagged:
        df['label'] = 1
        row_already_flagged = True

    if df['hemo_percent'] >= 90 and df['hemo_concentration'] <= 540 and not row_already_flagged:
        df['label'] = 1
        row_already_flagged = True

    if df['hemo_percent'] >= 95 and df['hemo_concentration'] <= 570 and not row_already_flagged:
        df['label'] = 1
        row_already_flagged = True

    if df['hemo_percent'] >= 100 and df['hemo_concentration'] <= 600 and not row_already_flagged:
        df['label'] = 1
        row_already_flagged = True
    return df

def label_after_happenn(df: pd.DataFrame) -> pd.DataFrame:
    # Define thresholds as sorted lists of tuples (activity, conc_threshold)
    hemolytic_thresholds = sorted([
        (50, 300), (55, 330), (60, 360), (65, 390), (70, 420),
        (75, 450), (80, 480), (85, 510), (90, 540), (95, 570), (100, 600)
    ])

    non_hemolytic_thresholds = sorted([
        (0, 30), (5, 30), (10, 60), (15, 90), (20, 120),
        (25, 150), (30, 180), (35, 210), (40, 240), (45, 270)
    ])

    def classify(row):
        conc = row['hemo_concentration']
        activity = row['hemo_percent']

        # Try hemolytic label
        for act_thresh, max_conc in hemolytic_thresholds:
            if activity >= act_thresh and conc <= max_conc:
                return 1

        # Try non-hemolytic label
        for act_thresh, min_conc in non_hemolytic_thresholds:
            if activity <= act_thresh and conc > min_conc:
                return 0

        # If no label is assigned, ask for manual input
        print(f"Unclassified: Activity {activity}, Concentration {conc}")

        input_save = input("Please adjust Label Manually: ")
        return input_save

    df['label'] = df.apply(classify, axis=1)
    return df



if __name__ == "__main__":
    file_path = "../../data/data_from_database/happen_data.fasta"
    prepare_happenn_data(file_path=file_path)
