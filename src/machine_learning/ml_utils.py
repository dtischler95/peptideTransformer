import sys

import pandas as pd
import json
import os
import pickle
from time import sleep
import seaborn as sns
from matplotlib import pyplot as plt
import peptides as pep
import plot_utils
import numpy as np
from sklearn.metrics import (r2_score,
                             mean_absolute_error,
                             explained_variance_score,
                             mean_squared_error)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
alphabet = {
    "-": 0,
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "N": 13,
    "P": 14,
    "Q": 15,
    "R": 16,
    "S": 17,
    "T": 18,
    "V": 19,
    "W": 20,
    "Y": 21,
}


def save_params(logdir, filename, params):
    with open(os.path.join(logdir, filename + "_parameters_.json"), 'w', encoding='utf8') as f:
        json.dump(params, f, indent=2)


def save_model(model, path="./model.pkl"):
    with open(path, mode="wb") as outfile:
        pickle.dump(model, outfile)


def load_model(path):
    with open(path, mode="rb") as infile:
        model = pickle.load(infile)
    return model


def load_data(path: str, decimal: str) -> pd.DataFrame:
    sleep(0.75)
    print(f"Loading data: {path}")

    # Automatically determines seperator between ; and \t in file
    # to-do: should i add more seperators?
    # POSSIBLE BUG WITH THIS EXPRESSION! Since for example .gff files can contain multiple line seperators with their
    # attribute fields. This could lead to a wrong seperator detection since those files contain \t and ; as seperators
    seperator = ';' if len(open(path, 'r').readline().split(';')) > 1 else '\t'
    df = pd.read_csv(filepath_or_buffer=path,
                     header=0,
                     sep=seperator,
                     verbose=False,
                     decimal=decimal
                     )

    return df

def check_label_col(label_col, data: pd.DataFrame):
    # Check single target col
    # Check multiple target cols
    # Some did you mean xy output?

    # print(f"-> Checking for label column: {label_col}")
    sleep(0.75)
    print(f"[green]Checking for label column: {label_col}")
    all_columns = data.columns

    if label_col not in all_columns:

        # print(f"-> Column: {label_col} not found in Dataframe")
        # sleep(0.75)
        print(f"[red]Column: {label_col} not found in Dataframe")
    else:
        print(f"[green]Select label column: {label_col}")

    print(f"[green]Removing labels from original data")
    target = data[label_col]
    data_df = data.drop(columns=label_col)  # Labels aus Daten entfernen
    features = data_df.columns

    if not len(all_columns) > len(features):
        print(f"[red]Removing Labelcolumn didnt work properly!")

    return data_df, target, features


def print_regression_metrics(y_true, y_pred):
    sys.stdout(f"Regression metrics: \n"
                f"    -> R2:  {r2_score(y_true=y_true, y_pred=y_pred):.5f}\n"
                f"    -> MAE: {mean_absolute_error(y_true=y_true, y_pred=y_pred):.5f}\n"
                f"    -> MSE: {mean_squared_error(y_true=y_true, y_pred=y_pred):.5f}\n"
                f"    -> VAR: {explained_variance_score(y_true=y_true, y_pred=y_pred):.5f}\n")
    return r2_score(y_true=y_true, y_pred=y_pred), mean_squared_error(y_true=y_true, y_pred=y_pred)

def get_model_stats(model,
                    plot_dir: str,
                    feature_data,
                    target_data,
                    tag: str):
    pred_train = model.predict(feature_data)

    r2, mse = print_regression_metrics(y_true=target_data, y_pred=pred_train)

    # plot regression train
    plot_utils.plot_with_seaborn(y_true=target_data, y_pred=pred_train, path=plot_dir + f"/{tag}_regression.pdf",
                                 tag=f"{tag}")

    return r2, mse


def grid_search_setup(model, model_dir, model_name, param_grid, x_train, y_train):
    """
    Setup for the Gridsearch in machine learning logic

    """

    if model_name == 'svr':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', model)
        ])
        # if svr is used, rename model to the appropriate pipeline model
        model = pipeline
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, return_train_score=True, refit=True,
                               n_jobs=-1, verbose=3, cv=5).fit(x_train, y_train)
    best_estimator = grid_search.best_estimator_
    save_model(model=best_estimator, path=f"{model_dir}{model.__class__.__name__}.keras")
    return best_estimator, grid_search, model


def log_encoded_sequences(sequence_encoder):
    """
    Logs encoded Sequences for the main train logic

    """
    for category in sequence_encoder.categories_[0]:
        # Pad for better logging visualisation
        cat_formated = category + ((36 - len(category)) * ' ')
        # Transform using a DataFrame
        encoded_value = sequence_encoder.transform(
            pd.DataFrame([[category]], columns=sequence_encoder.feature_names_in_)
        )
        print(f"Sequence: {cat_formated}\tEncoded: {encoded_value}")

def enocde_onehot_without_features(sequences: pd.DataFrame, pad_len: int):
    padded = sequences['sequence'].str.pad(width=pad_len, side='right', fillchar='-').tolist()
    df = add_pos_columns(sequences, L=MAX_LEN)
    pos_cols = [f"pos{i + 1}" for i in range(MAX_LEN)]
    ohe = OneHotEncoder(categories=[CATEGORIES] * 36, handle_unknown="ignore", sparse_output=False)
    onehot = ohe.fit_transform(df[pos_cols])
    label = sequences['mic_log10'].tolist()
    return onehot, label

def overall_stats(best_estimator, x_test, y_test, save_path):

    # Make predictions on the test set
    predictions = best_estimator.predict(x_test)

    # 1. Plotting the distribution of the target feature (y_test)
    plt.figure(figsize=(10, 6))
    sns.histplot(y_test, kde=True)
    plt.title('Verteilung der MIC-Werte (Test Set)')
    plt.xlabel('MIC (log10)')
    plt.ylabel('Häufigkeit')
    plt.savefig(save_path + '/target_distribution.pdf')

    # 2. Calculate variance of the target feature in the test set
    target_variance = np.var(y_test)
    print(f"Variance of the target feature (value) in test set: {target_variance}")

    # 3. Plotting Predictions vs Actuals for the test set
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title('Vorhersagen vs Tatsächlich (Test Set)')
    plt.xlabel('Tatsächlicher Wert')
    plt.ylabel('Vorhergesagter Wert')
    plt.savefig(save_path + '/predictions_vs_actuals.pdf')
    plt.close()
    plt.clf()

    # 4. Plotting Residuals in the test set
    residuals = y_test - predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Verteilung der Residuen (Test Set)')
    plt.xlabel('Residuen')
    plt.ylabel('Häufigkeit')
    plt.savefig(save_path + '/residuals_distribution.pdf')
    plt.close()
    plt.clf()

    # 5. Print MSE for comparison on the test set
    mse = np.mean((y_test - predictions) ** 2)
    print(f"Mean Squared Error (MSE) on Test Set: {mse}")

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the features for a given sequence inside the DataFrame
    """


    peptide_df = pd.DataFrame()



    for sequence in df['sequence']:

        peptide_object = pep.Peptide(sequence)



        tmp_peptide_df = pd.DataFrame([list(peptide_object.descriptors().values())], columns=list(peptide_object.descriptors().keys()))
        tmp_peptide_df['sequence_checker_2'] = sequence



        peptide_df = pd.concat([peptide_df, tmp_peptide_df])

        # aa_dipeptide_composition_df = pd.concat([aa_dipeptide_composition_df, tmp_aa_dipeptide_composition_df])


    peptide_df.reset_index(drop=True, inplace=True)



    df = pd.concat([df, peptide_df], axis=1)

    df = df.drop(columns=['sequence_checker_2'])


    return df


PAD = "-"
MAX_LEN = 36
AA = list("ACDEFGHIKLMNPQRSTVWY")         # Standard-20
CATEGORIES = AA + [PAD]


def pad_seq(seq, L=MAX_LEN, pad=PAD):
    seq = str(seq)
    return seq[:L] + pad * max(0, L - len(seq))

def add_pos_columns(df, L=MAX_LEN, pad=PAD):
    # keine Inplace-Änderung am Slice; wir bauen neue Spalten und geben ein neues DF zurück
    seq_pad = (
        df["sequence"].astype(str)
        .str.slice(0, L)            # trunkieren
        .str.ljust(L, fillchar=pad) # rechts padden
    )
    # Vektorisierter Bau der Positionsspalten
    pos_df = pd.DataFrame(
        {f"pos{i+1}": seq_pad.str[i] for i in range(L)},
        index=df.index
    )
    # neues DF zurückgeben
    return pd.concat([df.copy(), pos_df], axis=1)

def compute_desc_row(sequence):
    p = pep.Peptide(sequence)
    d = p.descriptors()                     # dict -> nur Zahlen
    return {f"desc__{k}": float(v) for k, v in d.items()}

def add_descriptors(df):
    desc_rows = [compute_desc_row(s) for s in df["sequence"]]
    desc_df = pd.DataFrame(desc_rows).reset_index(drop=True)
    df = df.reset_index(drop=True).join(desc_df)
    # sauber halten:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

# --- Build design matrix (OHE || Deskriptoren) + seq-basierter Split ---
def encode_onehot_with_features(df, target_col="value"):
    df = df.copy()
    df = add_pos_columns(df, L=MAX_LEN)
    df = add_descriptors(df)

    pos_cols  = [f"pos{i+1}" for i in range(MAX_LEN)]
    desc_cols = [c for c in df.columns if c.startswith("desc__")]

    # Split ohne Leckage: nach einzigartigen Sequenzen
    uniq = df["sequence"].unique()
    tr_seqs, va_seqs = train_test_split(uniq, test_size=0.2, random_state=42, shuffle=True)
    train_df = df[df["sequence"].isin(tr_seqs)].copy()
    val_df   = df[df["sequence"].isin(va_seqs)].copy()

    y_train = train_df[target_col].astype(float).to_numpy()
    y_val   = val_df[target_col].astype(float).to_numpy()

    # One-Hot nur auf TRAIN fitten, Kategorien fix (inkl. PAD)
    ohe = OneHotEncoder(categories=[CATEGORIES]*len(pos_cols), handle_unknown="ignore", sparse_output=False)
    X_pos_train = ohe.fit_transform(train_df[pos_cols])
    X_pos_val   = ohe.transform(val_df[pos_cols])

    # Numerische Features, robust gegen NaNs
    imp = SimpleImputer(strategy="median")
    X_desc_train = imp.fit_transform(train_df[desc_cols])
    X_desc_val   = imp.transform(val_df[desc_cols])

    # Concatenate: [OHE | Deskriptoren]
    X_train = np.hstack([X_pos_train, X_desc_train])
    X_val   = np.hstack([X_pos_val,   X_desc_val])

    return X_train, y_train, X_val, y_val

def print_results_tabular(results: list[dict]):
    import sys
    for results in results:
        sys.stdout(f"{results['name']}\tR2 Train: {results['r2_train']:.3f}\tR2 Test: {results['r2_val']:.3f}\t"
              f"MSE Train: {results['mse_train']:.3f}\tMSE Test: {results['mse_val']:.3f}")

if __name__ == '__main__':
    ...
