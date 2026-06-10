import pandas as pd
import numpy as np
import re
from pathlib import Path
from peptides import Peptide as Pep

_REPO_ROOT = Path(__file__).resolve().parents[3]
from src.data_preprocessing.preprocess_utils import load_and_filter_data, split_positive_and_negative, \
    filter_and_evaluate_ambiguous_sequences
from src.data_preprocessing.binary.happenn_preprocess import label_by_threshold


"""
Script for our data preprocessing. It contains the main logic for creating 
the training files for the Hemolytic Activity Prediction by setting labels and combining our different references.
"""


def get_filtered_and_combined_dataframe(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Used Data:
    - "../data/data_from_database/dbaasp_scraped.csv"
    - "../data/data_from_database/Hemolytik_scraped.csv"

    If more databases need to be parsed, add the corresponding file as a DataFrame here
    NO GUARANTEE THAT DOWNSTREAM PROCESSING WILL WORK WITH NEW DATA. YOU NEED TO MAKE SURE NEW DATA IS FETCHED CORRECTLY

    :param dataframes: DataFrames containing the data from the databases

    :return: DataFrame containing the combined and filtered data
    """
    # Load the data first. If new data is added, add it here.

    if not dataframes:
        raise ValueError(
            'No dataframes provided. Please provide the dataframes containing the data from the databases.')
    if len(dataframes) == 1:
        return dataframes[0]
    df_to_concat = []
    for df in dataframes:
        # Filter Sequences by seq_length and Ambiguous Amino Acids
        df = load_and_filter_data(df)
        df = df[['sequence', 'measure_type', 'activity']]
        df_to_concat.append(df)

    # Concatenate the dataframes and select only the relevant columns
    df = pd.concat(df_to_concat)
    df = df[['sequence', 'measure_type']]
    return df


def split_measure_type(x):
    """
    Split the measure_type column into percent, concentration, and unit.
    There are still datapoints not fetched with all this regex here. This is a first step to get the data.
    If more data is needed, you can add more regex patterns here or find them in the data.
    So far we get with this regex pattern roundabout 7000 datapoints. This is a good start for the training data.

    Basic Schema is:

    Typically, the measure_type column contains information about the hemolytic activity of a peptide in form of
    x% Hemo at y *unit* concentration. This function extracts the x, y by splitting at the string between the values.
    The unit is extracted with regex pattern matching by searching the specific unit in the string.
    Out Data seem to contain different Unicode characters for the same unit. This is handled in the function by
    duplicating the unit with the same value. This is not the best way to handle this, but it works for now.
    Raw Data could be cleaner...

    1% Hemolysis at 500µg/ml (Human erythrocytes) pmid: 11352918

    :param x: The measure_type column.
    """
    # Check for 'hemolysis at' pattern
    if 'Poor hemolytic' in x:
        percent = "0"
        concentration = "0"
        unit = "unknown"
    elif 'hemolysis at' in x:
        percent = x.split('hemolysis at')[0]
        concentration = x.split('hemolysis at')[1]
        unit = re.search(r'(μM|μg/ml|mg/ml|mM|nM|pM|μg|µM|µg/ml*)', concentration).group() if re.search(
            r'(μM|μg/ml|mg/ml|mM|nM|pM|μg|µM|µg/ml*)', concentration) else "why"
    elif 'Hemolysis at' in x:
        percent = x.split('Hemolysis at')[0]
        concentration = x.split('Hemolysis at')[1]
        unit = re.search(r'(μM|μg/ml|mg/ml|mM|nM|pM|μg|µM|µg/ml*)', concentration).group() if re.search(
            r'(μM|μg/ml|mg/ml|mM|nM|pM|μg|µM|µg/ml*)', concentration) else "why2"
    # Check for other patterns (e.g., relative % score and concentration)
    elif 'hemolysis in' in x:
        percent = x.split('hemolysis in')[0]
        concentration = x.split('hemolysis in')[1]
        unit = re.search(r'(μM|μg/ml|mg/ml|mM|nM|pM|μg|µM|µg/ml*)', concentration).group() if re.search(
            r'(μM|μg/ml|mg/ml|mM|nM|pM|μg|µM|µg/ml*)', concentration) else "why3"
    # Check for other patterns (e.g., relative % score and concentration)
    elif 'hemolysisat' in x:
        percent = x.split('hemolysisat')[0]
        concentration = x.split('hemolysisat')[1]
        unit = re.search(r'(μM|μg/ml|mg/ml|mM|nM|pM|μg|µM|µg/ml*)', concentration).group() if re.search(
            r'(μM|μg/ml|mg/ml|mM|nM|pM|μg|µM|µg/ml*)', concentration) else "why5"
    elif 'hemolsyis upto' in x:
        percent = x.split('hemolsyis upto')[0]
        concentration = x.split('hemolsyis upto')[1]
        unit = re.search(r'(μM|μg/ml|mg/ml|mM|nM|pM|μg|µM|µg/ml*)', concentration).group() if re.search(
            r'(μM|μg/ml|mg/ml|mM|nM|pM|μg|µM|µg/ml*)', concentration) else "why6"
    elif re.search(r'\d+\.?\d*%.*\d+\.?\d*', x):
        match = re.search(r'(\d+\.?\d*)%.*?(\d+\.?\d*)(μM|μg/ml|mg/ml|mM|nM|pM|μg|µg/ml*)', x)
        if match:
            percent = match.group(1)
            concentration = match.group(2)
            unit = match.group(3) if match.group(3) else "wtf"
        else:
            percent = "X"
            concentration = "X"
            unit = "unknown"
    else:
        percent = "X"
        concentration = "X"
        unit = "unknown"

    # Handle ranges by taking the first number
    percent = re.search(r'\d+\.?\d*', percent).group() if re.search(r'\d+\.?\d*', percent) else "X"
    concentration = re.search(r'\d+\.?\d*', concentration).group() if re.search(r'\d+\.?\d*', concentration) else "X"

    return percent, concentration, unit


# Normalize the various unit spellings/typos found in the raw data to a
# small set of canonical unit strings.
_UNIT_ALIASES = {
    "µg/mL": "µg/ml",
    "mu_mol": "µM",
    "mu_m/mL": "µM/ml",
    "mug_ml": "µg/ml",
    "mu_mol/L": "µM/l",
    "µm/ml": "µM/ml",
    "mu_m]": "µM",
    "uM": "µM",
    "mu_m": "µM",
    "μmol": "µM",
    "ng/mL": "ng/ml",
    "μM": "µM",
    "mm": "mM",
    "µM/ml": "µM/ml",
    "μg/ml": "µg/ml",
    "mu_m/l": "µM/l",
    "mg/L": "mg/l",
    "mu_m/ml": "µM/ml",
    "CFU/ml": "cfu/ml",
    "mmol/l": "mM/l",
    "nmol/ml": "nM/ml",
    "pmol": "pM",
    "mol/l": "M/l",
    "pmol/ml": "pM/ml",
    "μg": "µg",
    "nmol/g": "nM/g",
    "mg/mL": "mg/ml",
    "g/mL": "g/ml",
}

# Conversion factor to µM for each unit, and whether the value additionally
# needs to be divided by the peptide's molecular weight (for mass-based units).
_UNIT_TO_MICROMOLAR = {
    "µg/ml": (1e3, True),
    "mg/ml": (1e6, True),
    "g/ml": (1e9, True),
    "ng/ml": (1, True),
    "µg/µl": (1e6, True),
    "µg/nl": (1e9, True),
    "g/l": (1e6, True),
    "mg/l": (1e3, True),
    "µg/l": (1, True),
    "nM": (1e-3, False),
    "nM/ml": (1e12, False),
    "mM": (1e3, False),
    "mM/l": (1e3, False),
    "M": (1e6, False),
    "M/l": (1e6, False),
    "pM": (1e6, False),
    "pM/ml": (1e15, False),
}


def _log_unit_stats(df: pd.DataFrame, value_column: str, label: str):
    print(label)
    print(df.groupby("unit").agg(
        count=('unit', 'size'),
        min=(value_column, 'min'),
        max=(value_column, 'max'),
        mean=(value_column, 'mean'),
        std=(value_column, 'std'),
        median=(value_column, lambda x: np.median(x)),
        units=('unit', 'unique'),
    ).sort_values(by='count', ascending=False))


def process_units(final: pd.DataFrame, verbose: bool = False, hemo_or_mic: str = "hemo") -> pd.DataFrame:
    """
    Convert the concentration/value column to a uniform µM unit, based on
    the (often inconsistently spelled) unit strings in the raw data.

    Mass-based units (e.g. µg/ml) are converted to µM using each peptide's
    molecular weight; molar units (e.g. nM, mM) are converted with a fixed
    factor.

    :param final: DataFrame with a "unit" column and the value column to convert
    :param verbose: if True, print unit statistics before and after conversion
    :param hemo_or_mic: "hemo" to convert "hemo_concentration", anything else to convert "value"
    :return: the DataFrame with the value column converted to µM and "unit" set to "µM" where possible
    """
    print("Processing units")

    value_column = "hemo_concentration" if hemo_or_mic == "hemo" else "value"

    if verbose:
        units_before = final["unit"].nunique()
        _log_unit_stats(final, value_column, "VOR Umrechnung der Einheiten")

    final["unit"] = final["unit"].replace(_UNIT_ALIASES)
    final["mol_weight"] = final["sequence"].apply(lambda p: Pep(p).molecular_weight())

    for unit, (factor, needs_mol_weight) in _UNIT_TO_MICROMOLAR.items():
        mask = final["unit"] == unit
        if not mask.any():
            continue
        final.loc[mask, value_column] *= factor
        if needs_mol_weight:
            final.loc[mask, value_column] /= final.loc[mask, "mol_weight"]
        final.loc[mask, "unit"] = "µM"

    final = final.drop(columns=["mol_weight"])

    if verbose:
        units_after = final["unit"].nunique()
        _log_unit_stats(final, value_column, "NACH Umrechnung der Einheiten")
        print(f"\nUNITS before: {units_before} -> after: {units_after}\n")

    return final


def parse_and_label_hemolytic_data(*data_paths: str,
                                   out_path: str,
                                   dataset_tag: str,
                                   vote_label: bool = False,
                                   label_threshold: float or None):
    """
    Main logic for creating the Training Files for the Hemolytic Activity Prediction.
    This function is specific for our data and should be refactored if new data is added.
    I tried to be as general as possible, but some of our data needs to be handled specifically.

    :param data_paths: Paths to the data files containing the hemolytic activity data.
    :param out_path: Path to the data directory. should contain a train_data and data_for_data_viewer directory.
    :param dataset_tag: Tag for the dataset. Used for naming the output files.
    :param vote_label: Flag to filter ambiguous sequences based on the majority label. Should be True
    :param label_threshold: Threshold for the label. If the relation between hemo_percent and hemo_concentration is greater or equal to this threshold, the label is set to 1, otherwise to 0.
    """

    out_path_train_file = f"{out_path}train_data/{dataset_tag}"
    out_path_splitted_file = f"{out_path}data_for_data_viewer/{dataset_tag}"

    data_df_list = []
    for file_path in data_paths:
        data_df = pd.read_csv(file_path, sep=';')
        data_df_list.append(data_df)

    base_df = get_filtered_and_combined_dataframe(dataframes=data_df_list)
    base_df = base_df.dropna()

    # DataFrame containing HC50 annotations. This Data inside here is not used in the current train data
    df_hc = base_df[base_df['measure_type'].str.contains('HC5')]

    # Filter all Datapoints that are hemolytik active! HERE NO CHECK FOR HUMAN RELATED DATA This should be done before using this Main Function!
    df_raw_hemo = base_df[base_df['measure_type'].str.contains('emo')]

    # Copy the DataFrame to avoid SettingWithCopyWarning. REFACTOR THIS LOOKS BAD THIS WAY
    df_raw_hemo = df_raw_hemo.copy()

    # Now we can extract the relevant information from the measure_type column
    df_raw_hemo.loc[:, 'hemo_percent'], df_raw_hemo.loc[:, 'hemo_concentration'], df_raw_hemo.loc[:, 'unit'] = zip(
        *df_raw_hemo['measure_type'].apply(split_measure_type))

    # Separate rows with 'X' in hemo_percent or hemo_concentration
    # can be viewed if more data needed, since this dataframe should contain all datapoints that our regex could not parse and which we are able to capture
    df_to_watch = df_raw_hemo[(df_raw_hemo['hemo_percent'] == 'X') | (df_raw_hemo['hemo_concentration'] == 'X')]

    # Remove wrongly parsed data. If more Data needed later, you can try to find them here
    # those units cant be converted to µM
    df_raw_hemo = df_raw_hemo[df_raw_hemo['unit'] != "μg"]
    df_raw_hemo = df_raw_hemo[df_raw_hemo['unit'] != "µg/m"]
    df_raw_hemo = df_raw_hemo[df_raw_hemo['unit'] != "unknown"]

    # Remove rows with 'X' in hemo_percent or hemo_concentration
    df_raw_hemo = df_raw_hemo[df_raw_hemo['hemo_percent'] != 'X']
    df_raw_hemo = df_raw_hemo[df_raw_hemo['hemo_concentration'] != 'X']


    df_raw_hemo['hemo_percent'] = df_raw_hemo['hemo_percent'].str.extract(r'(\d+\.?\d*)').astype(float)
    df_raw_hemo['hemo_concentration'] = df_raw_hemo['hemo_concentration'].str.extract(r'(\d+\.?\d*)').astype(
        float)

    df_raw_hemo = df_raw_hemo[df_raw_hemo['hemo_concentration'] != 0.0]
    df_raw_hemo = df_raw_hemo[df_raw_hemo['hemo_percent'] <= 100.0]

    # needed for this specific usecase. Found no better way to filter dynamically for wrongly parsed data in time
    df_raw_hemo = df_raw_hemo[df_raw_hemo['unit'] != "why"]
    df_raw_hemo = df_raw_hemo[df_raw_hemo['unit'] != "why5"]
    df_raw_hemo = df_raw_hemo[df_raw_hemo['unit'] != "why2"]

    # Process units
    df_raw_hemo = process_units(df_raw_hemo, verbose=True)

    if label_threshold is not None:
        df_raw_hemo = label_via_relation(df=df_raw_hemo, label_threshold=label_threshold)
    else:
        df_raw_hemo = label_by_threshold(df=df_raw_hemo)



    # result_df should contain cleaned data useable for training and further analysis
    result_df = df_raw_hemo[['sequence', 'hemo_concentration', 'hemo_percent', 'label']]

    # Split the data into positive and negative sequences for later analysis
    split_positive_and_negative(data=result_df, to_file=True, out_path=out_path_splitted_file)


    print(result_df.shape)



    if vote_label:
        new_file_name = f"{out_path_train_file}_voted.csv"
        # Finally filter ambiguous labeled sequences and sort them into positive or negative based on the majority label
        filter_and_evaluate_ambiguous_sequences(labeled_df=result_df,
                                                out_path=new_file_name)
    else:
        # If you dont want to filter ambiguous sequences, just save the data
        print(f"\033[31mSkipping Vote of ambiguous sequences.\033[0m")
        result_df.to_csv(f"{out_path_train_file}_unvoted.csv", sep=';', index=False)





def label_via_relation(df: pd.DataFrame, label_threshold: float=0.8) -> pd.DataFrame:
    # Calculate the relation between hemo_percent and hemo_concentration
    # Based on my thoughts on how to quantify the hemolytic activity of a peptide
    df['relation'] = df['hemo_percent'] / df['hemo_concentration']
    # Label the data based on a threshold on the relation. ADJUST THIS THRESHOLD IF NEEDED INSIDE THE FUNCTION.
    df['label'] = df['relation'].apply(lambda x: 1 if x >= label_threshold else 0)

    return df


if __name__ == '__main__':
    # dbaasp_preprocessor() need to be called to create the dbaasp_scraped.csv file needed for parse_and_label_hemolytic_data()
    # This part is hardcoded since this algorithm is specific for our data
    # This is meant for the preprocessing step

    hemolytik_db = str(_REPO_ROOT / "data" / "data_from_database" / "Hemolytik_scraped.csv")
    dbaasp_db = str(_REPO_ROOT / "data" / "data_from_database" / "dbaasp_scraped.csv")
    all_db = str(_REPO_ROOT / "data" / "data_from_database" / "complete_amp_data.csv")

    parse_and_label_hemolytic_data(hemolytik_db,
                                   dbaasp_db,
                                   out_path=str(_REPO_ROOT / "data") + "/",
                                   dataset_tag='threshold_style',
                                   vote_label=False,
                                   label_threshold=None)
