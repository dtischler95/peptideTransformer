import pandas as pd
import numpy as np
import re
from peptides import Peptide as Pep
from preprocess_utils import load_and_filter_data, split_positive_and_negative, filter_and_evaluate_ambiguous_sequences



def get_filtered_and_combined_dataframe(*dataframes) -> pd.DataFrame:
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
        raise ValueError('No dataframes provided. Please provide the dataframes containing the data from the databases.')

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
    # Check for other patterns (e.g., relative % score and concentration)
    # elif 'hemolysis' in x:
    #     percent = x.split('hemolysis')[0]
    #     concentration = x.split('hemolysis')[1]
    #     unit = re.search(r'(μM|μg/ml|mg/ml|mM|nM|pM|μg*)', concentration).group() if re.search(r'(μM|μg/ml|mg/ml|mM|nM|pM|μg*)', concentration) else "why1"
    # Check for other patterns (e.g., relative % score and concentration)
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


def _process_units(final, verbose: bool = False):
    """
    Function from Lukas Beyerle
    Calculates µM value for each datapoint based on the given unit and concentration.
    """
    print("Processing units")

    # Replace dictionary for initial unit replacements
    replace_units = {
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

    # Replacement units after calculation
    replace_units_after_calculation = {
        "µg/ml": "µM",
        "mg/ml": "µM",
        "ng/ml": "µM",
        "pM/ml": "µM",
        "mg/l": "µM",
        "mM/l": "µM",
        "nM/ml": "µM",
        "µM/l": "µM",
        "µM/ml": "µM",
        "g/l": "µM",
        "µg/l": "µM",
        "µg/µl": "µM",
        "µg/nl": "µM",
        "g/ml": "µM",
        "M": "µM",
        "nM": "µM",
        "mM": "µM",
        "pM": "µM",
        "M/l": "µM",
    }
    units_before = "Something went Wrong with Unit conversion"
    # If verbose, show units before processing
    if verbose:
        units_before = final["unit"].nunique()
        print("VOR Umrechnung der Einheiten")
        print(final.groupby("unit").agg(
            count=('unit', 'size'),
            min=('hemo_concentration', 'min'),
            max=('hemo_concentration', 'max'),
            mean=('hemo_concentration', 'mean'),
            std=('hemo_concentration', 'std'),
            median=('hemo_concentration', lambda x: np.median(x)),
            units=('unit', 'unique'),
        ).sort_values(by='count', ascending=False))

    # Replace units before calculation
    final['unit'] = final['unit'].replace(replace_units)

    # Calculate molecular weight using lambda
    final['mol_weight'] = final['sequence'].apply(lambda p: Pep(p).molecular_weight())

    # Apply conversions based on units
    def convert_values(row):
        if row['unit'] == 'µg/ml':
            return row['hemo_concentration'] * 10 ** 3 / row['mol_weight']
        elif row['unit'] == 'mg/ml':
            return row['hemo_concentration'] * 10 ** 6 / row['mol_weight']
        elif row['unit'] == 'nM':
            return row['hemo_concentration'] / 10 ** 3
        elif row['unit'] == 'nM/ml':
            return row['hemo_concentration'] * 10 ** 12
        elif row['unit'] == 'mM':
            return row['hemo_concentration'] * 10 ** 3
        elif row['unit'] == 'mM/l':
            return row['hemo_concentration'] * 10 ** 3
        elif row['unit'] == 'M':
            return row['hemo_concentration'] * 10 ** 6
        elif row['unit'] == 'pM':
            return row['hemo_concentration'] * 10 ** 6
        elif row['unit'] == 'pM/ml':
            return row['hemo_concentration'] * 10 ** 15
        elif row['unit'] == 'µg/µl':
            return row['hemo_concentration'] * 10 ** 6 / row['mol_weight']
        elif row['unit'] == 'M/l':
            return row['hemo_concentration'] * 10 ** 6
        elif row['unit'] == 'g/l':
            return row['hemo_concentration'] * 10 ** 6 / row['mol_weight']
        elif row['unit'] == 'g/ml':
            return row['hemo_concentration'] * 10 ** 9 / row['mol_weight']
        elif row['unit'] == 'mg/l':
            return row['hemo_concentration'] * 10 ** 3 / row['mol_weight']
        elif row['unit'] == 'µg/l':
            return row['hemo_concentration'] / row['mol_weight']
        elif row['unit'] == 'ng/ml':
            return row['hemo_concentration'] / row['mol_weight']
        elif row['unit'] == 'µg/nl':
            return row['hemo_concentration'] * 10 ** 9 / row['mol_weight']
        else:
            return row['hemo_concentration']

    final['hemo_concentration'] = final.apply(convert_values, axis=1)

    # Drop the temporary molecular weight column
    final = final.drop(columns=["mol_weight"])

    # Replace units after calculation
    final['unit'] = final['unit'].replace(replace_units_after_calculation)

    if verbose:
        units_after = final["unit"].nunique()
        print("NACH Umrechnung der Einheiten")
        print(final.groupby("unit").agg(
            count=('unit', 'size'),
            min=('hemo_concentration', 'min'),
            max=('hemo_concentration', 'max'),
            mean=('hemo_concentration', 'mean'),
            std=('hemo_concentration', 'std'),
            median=('hemo_concentration', lambda x: np.median(x)),
            units=('unit', 'unique'),
        ).sort_values(by='count', ascending=False))
        print(f"\nUNITS before: {units_before} -> after: {units_after}\n")

    return final


def give_label_by_threshold(x):
    """
    Set the Label via threshold for a given sequences.
    Used as a DataFrame.apply() function.

    :param x: The relation between hemo_percent and hemo_concentration
    """
    if x >= 0.8:
        return 1
    else:
        return 0


def parse_and_label_hemolytic_data(our_data: pd.DataFrame, whitelab_data: pd.DataFrame, *dataframes, filter_sequences: bool = False):
    """
    Main logic for creating the Training Files for the Hemolytic Activity Prediction.
    This function is specific for our data and should be refactored if new data is added.
    I tried to be as general as possible, but some of our data needs to be handled specifically.

    :param our_data: DataFrame containing our labeled data
    :param whitelab_data: DataFrame containing the Whitelab data

    :param dataframes: Additional Dataframes that should be included in the training data. NO GUARANTEE OF WORKING HERE!
                            - Bring new Data in our Format and name Columns correctly!
                            - You may add new cases for column parsing in the split_measure_type function
    """
    base_df = get_filtered_and_combined_dataframe(our_data, whitelab_data)

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
    # TODO Collect removed datapoints into seperat DataFrame for later inspection
    df_raw_hemo = df_raw_hemo[df_raw_hemo['unit'] != "μg"]
    df_raw_hemo = df_raw_hemo[df_raw_hemo['unit'] != "µg/m"]
    df_raw_hemo = df_raw_hemo[df_raw_hemo['unit'] != "unknown"]

    # Remove rows with 'X' in hemo_percent or hemo_concentration
    df_raw_hemo = df_raw_hemo[df_raw_hemo['hemo_percent'] != 'X']
    df_raw_hemo = df_raw_hemo[df_raw_hemo['hemo_concentration'] != 'X']

    df_raw_hemo['hemo_percent'] = df_raw_hemo['hemo_percent'].str.extract(r'(\d+\.?\d*)').astype(float)
    df_raw_hemo['hemo_concentration'] = df_raw_hemo['hemo_concentration'].str.extract(r'(\d+\.?\d*)').astype(
        float)

    # TODO MAY SET 0.0 % HEMO ACTIVITY TO 0.001% SO THAT DATA WONT GET LOST
    df_raw_hemo = df_raw_hemo[df_raw_hemo['hemo_concentration'] != 0.0]

    # needed for this specific usecase. Found no better way to filter dynamically for wrongly parsed data in time
    df_raw_hemo = df_raw_hemo[df_raw_hemo['unit'] != "why"]
    df_raw_hemo = df_raw_hemo[df_raw_hemo['unit'] != "why5"]
    df_raw_hemo = df_raw_hemo[df_raw_hemo['unit'] != "why2"]

    # Process units
    df_raw_hemo = _process_units(df_raw_hemo, verbose=True)

    # Calculate the relation between hemo_percent and hemo_concentration
    # Based on my thoughts on how to quantify the hemolytic activity of a peptide
    df_raw_hemo['relation'] = df_raw_hemo['hemo_percent'] / df_raw_hemo['hemo_concentration']

    # Label the data based on a threshold on the relation. ADJUST THIS THRESHOLD IF NEEDED INSIDE THE FUNCTION.
    # TODO IS THERE A WAY TO SET THIS PARAMETER VIA THE APPLY FUNCTION?
    df_raw_hemo['label'] = df_raw_hemo['relation'].apply(give_label_by_threshold)

    # result_df should contain cleaned data useable for training and further analysis
    result_df = df_raw_hemo[['sequence', 'label']]

    # Split the data into positive and negative sequences for later analysis
    split_positive_and_negative(data=result_df, to_file=True)

    if filter_sequences:
        # Finally filter ambiguous labeled sequences and sort them into positive or negative based on the majority label
        filter_and_evaluate_ambiguous_sequences(labeled_df=result_df)
    else:
        # If you dont want to filter ambiguous sequences, just save the data
        print("Skipping filtering of ambiguous sequences.")
        result_df.to_csv('../data/train_data/our_hemo_labeled.csv', sep=';', index=False)



if __name__ == '__main__':
    # dbaasp_preprocessor() need to be called to create the dbaasp_scraped.csv file needed for parse_and_label_hemolytic_data()
    # This part is hardcoded since this algorithm is specific for our data
    # This is meant for the preprocessing step
    our_df = pd.read_csv('../data/train_data/our_hemo_labeled.csv', sep=';')
    whitelab_df = pd.read_csv('../data/train_data/whitelab_hemo_data.csv', sep=';')

    parse_and_label_hemolytic_data(our_data=our_df, whitelab_data=whitelab_df)


