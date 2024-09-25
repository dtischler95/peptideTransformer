import pandas as pd
from preprocess_utils import load_and_filter_data


def expand_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data contains multiple values for different organisms in one row. This function expands the rows to have one for
    each organism.

    :param df: DataFrame with data.
    """

    result_df = pd.DataFrame()
    df = df.drop_duplicates(subset='Sequence')
    for row in df.iterrows():
        splitted_values = row[1]['Toxicity'].split(';')
        tmp_row_df = pd.DataFrame()
        for data in splitted_values:
            tmp_row_df = pd.concat([tmp_row_df, pd.DataFrame({'Sequence': [row[1]['Sequence']],
                                                              'Toxicity': [data],
                                                              'Group_Target': [row[1]['Group_Target']]
                                                              # 'Nter_Modif': [row[1]['Nter_Modif']],
                                                              # 'Cter_Modif': [row[1]['Cter_Modif']],
                                                              # 'Original_Strain': [row[1]['Original_Strain']],
                                                              # 'Activity_Type': [row[1]['Activity_Type']],
                                                              # 'Group_Target': [row[1]['Group_Target']]
                                                              })])
        result_df = pd.concat([result_df, tmp_row_df])

    return result_df


def create_dbaasp_hemo_file(in_path: str,
                            out_path: str):
    """
    This function creates a csv file with the hemolysis data from DBAASP. The data is filtered on human hemolysis data.
    """
    df = load_and_filter_data(data=in_path,
                              sep=',',
                              filter_on_column='Sequence')

    df = df[['Sequence', 'Toxicity', 'Group_Target']]
    df = df[df['Toxicity'].notna()]
    df = df[df['Toxicity'].str.contains('Hemolysis')]

    # Making sure its human hemo data!!
    df = df[df['Toxicity'].str.contains('Human')]
    df = expand_rows(df=df)

    # Renaming columns for further downstream processing
    df = df.rename(columns={'Toxicity': 'measure_type',
                            'Group_Target': 'activity',
                            'Sequence': 'sequence'})

    df.to_csv(path_or_buf=out_path,
              sep=';',
              index=False)


if __name__ == '__main__':

    create_dbaasp_hemo_file(in_path = "../data/data_from_database/DBAASP_from_CalcAMP.csv",
                            out_path = "../data/data_from_database/dbaasp_scraped.csv")
