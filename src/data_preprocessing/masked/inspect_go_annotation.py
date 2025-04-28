import pandas as pd


def look_into_df(file_path: str):
    df = pd.read_csv(file_path, sep='\t', header=None)
    df_test = pd.read_csv('../../../data/train_data/mlm_starpep_data.csv', sep='\t')
    df = df[[0, 13]]
    counter = 0
    for group in df.groupby(0):
        if group[1][13].shape[0] > 1:

            entry_list = group[1][13].unique()[group[1][13].unique() != '-']
            if entry_list.shape[0] > 1:
                counter += 1
                print(group[0])




    print(f"Sequences with differing GO annotations: {counter}")


if __name__ == '__main__':
    # Example usage
    file_path = './mlm_starpep_data_go_annotated.tsv'
    look_into_df(file_path)