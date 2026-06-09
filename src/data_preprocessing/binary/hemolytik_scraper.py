import pandas as pd
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

def data_scraper(file_path: str):
    df = pd.read_csv(file_path, sep=';')
    #df = df[['sequence', 'measure_type', 'value', 'unit']]
    df = df.dropna(subset=['measure_type', 'value', 'unit'])

    # keep the rows where the measure_type is not null

    def combine_columns(row):
        return f"{row['measure_type']} {round(row['value'], 2)}{row['unit']}"

    hemo_df = df[df['measure_type'].str.contains('Hemo')]
    hc_df = df[df['measure_type'].str.contains('HC')]
    result_df = pd.concat([hemo_df, hc_df])
    result_df.loc[:, 'measure_type'] = result_df.apply(combine_columns, axis=1)
    result_df.to_csv(str(_REPO_ROOT / "data" / "data_from_database" / "Hemolytik_scraped.csv"), sep=';', index=False)

if __name__ == '__main__':
    file_path = str(_REPO_ROOT / "data" / "data_from_database" / "complete_amp_data.csv")
    data_scraper(file_path)