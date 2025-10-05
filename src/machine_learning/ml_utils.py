import logging
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from sklearn.metrics import (r2_score,
                             mean_absolute_error,
                             explained_variance_score,
                             mean_squared_error, precision_recall_curve, roc_auc_score, average_precision_score,
                             roc_curve, classification_report, ConfusionMatrixDisplay, PrecisionRecallDisplay, f1_score,
                             precision_score, recall_score, matthews_corrcoef)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
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


def print_regression_metrics(y_true, y_pred, logger: logging.Logger):
    logger.info(f"Regression metrics: \n"
                f"    -> R2:  {r2_score(y_true=y_true, y_pred=y_pred):.5f}\n"
                f"    -> MAE: {mean_absolute_error(y_true=y_true, y_pred=y_pred):.5f}\n"
                f"    -> MSE: {mean_squared_error(y_true=y_true, y_pred=y_pred):.5f}\n"
                f"    -> VAR: {explained_variance_score(y_true=y_true, y_pred=y_pred):.5f}\n")
    return r2_score(y_true=y_true, y_pred=y_pred), mean_squared_error(y_true=y_true, y_pred=y_pred)


def get_model_stats(model,
                    plot_dir: str,
                    feature_data,
                    target_data,
                    logger: logging.Logger,
                    file_name,
                    tag: str):
    pred_train = model.predict(feature_data)

    r2, mse = print_regression_metrics(y_true=target_data, y_pred=pred_train, logger=logger)

    # plot regression train
    plot_utils.make_regression(y_true=target_data, y_pred=pred_train, path=plot_dir + f"/{tag}_regression.pdf",
                               file_name=file_name)

    return r2, mse


def grid_search_setup(model, model_dir, model_name, param_grid, x_train, y_train, task):
    """
    Setup for the Gridsearch in machine learning logic

    """

    if model_name == 'svr':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', model)
        ])
        model = pipeline
    if model_name == 'svc':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', model)
        ])
        model = pipeline

    if task == 'hemo':
        scoring = {
            'roc_auc': 'roc_auc',
            'ap': 'average_precision',
            'f1': 'f1',
            'bal_acc': 'balanced_accuracy',
        }
        refit = 'roc_auc'
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    elif task == 'mic':
        scoring = {
            'r2': 'r2',
            'neg_mse': 'neg_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error'
        }
        refit = 'r2'
        cv = 5
    else:
        raise NotImplementedError

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, return_train_score=True, refit=refit,
                               n_jobs=-1, verbose=0, cv=cv, scoring=scoring).fit(x_train, y_train)
    best_estimator = grid_search.best_estimator_
    save_model(model=best_estimator, path=f"{model_dir}{model.__class__.__name__}.keras")
    return best_estimator, grid_search, model


def enocde_onehot_without_features(sequences: pd.DataFrame, target_col: str):
    df = add_pos_columns(sequences, L=MAX_LEN)
    pos_cols = [f"pos{i + 1}" for i in range(MAX_LEN)]
    ohe = OneHotEncoder(categories=[CATEGORIES] * 36, handle_unknown="ignore", sparse_output=False)
    onehot = ohe.fit_transform(df[pos_cols])
    label = sequences[target_col].tolist()
    return onehot, label


def overall_stats(best_estimator, x_test, y_test, save_path, tag):
    # Make predictions on the test set
    predictions = best_estimator.predict(x_test)

    # 1. Plotting the distribution of the target feature (y_test)
    plt.figure(figsize=(10, 6))
    sns.histplot(y_test, kde=True)
    plt.title(f'Verteilung der MIC-Werte ({tag})')
    plt.xlabel('MIC (log10)')
    plt.ylabel('Häufigkeit')
    plt.savefig(save_path + f'/{tag}_target_distribution.pdf')
    plt.close()
    plt.clf()

    # 2. Calculate variance of the target feature in the test set
    target_variance = np.var(y_test)
    print(f"Variance of the target feature (value) in test set: {target_variance}")

    # 4. Plotting Residuals in the test set
    residuals = y_test - predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title(f'Verteilung der Residuen ({tag})')
    plt.xlabel('Residuen MIC/log(µM)')
    plt.ylabel('Häufigkeit')
    plt.savefig(save_path + f'/{tag}residuals_distribution.pdf')
    plt.close()
    plt.clf()

    df = pd.DataFrame({
        "MIC (log$_{10}$ µM)": y_test,
        "Residuen (log$_{10}$ µM)": residuals
    })

    # Einteilung in 4 Quantile – du kannst q=5 oder q=[0,.25,.5,.75,1.] nehmen
    df["MIC-Quantil"] = pd.qcut(df["MIC (log$_{10}$ µM)"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

    plt.figure(figsize=(8, 4))
    sns.boxplot(x="MIC-Quantil", y="Residuen (log$_{10}$ µM)", data=df, color="skyblue")

    plt.xlabel("Quantile des tatsächlichen MIC-Werts (log$_{10}$ µM)")
    plt.ylabel("Residuen (tatsächlich – vorhergesagt) (log$_{10}$ µM)")
    plt.title("Residuenverteilung nach Quantilen des tatsächlichen MIC-Werts")
    plt.tight_layout()
    plt.savefig(save_path + f'/{tag}residuals_quantils.pdf')
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

        tmp_peptide_df = pd.DataFrame([list(peptide_object.descriptors().values())],
                                      columns=list(peptide_object.descriptors().keys()))
        tmp_peptide_df['sequence_checker_2'] = sequence

        peptide_df = pd.concat([peptide_df, tmp_peptide_df])

        # aa_dipeptide_composition_df = pd.concat([aa_dipeptide_composition_df, tmp_aa_dipeptide_composition_df])

    peptide_df.reset_index(drop=True, inplace=True)

    df = pd.concat([df, peptide_df], axis=1)

    df = df.drop(columns=['sequence_checker_2'])

    return df


PAD = "-"
MAX_LEN = 36
AA = list("ACDEFGHIKLMNPQRSTVWY")  # Standard-20
CATEGORIES = AA + [PAD]


def pad_seq(seq, L=MAX_LEN, pad=PAD):
    seq = str(seq)
    return seq[:L] + pad * max(0, L - len(seq))


def add_pos_columns(df, L=MAX_LEN, pad=PAD):
    # keine Inplace-Änderung am Slice; wir bauen neue Spalten und geben ein neues DF zurück
    seq_pad = (
        df["sequence"].astype(str)
        .str.slice(0, L)  # trunkieren
        .str.ljust(L, fillchar=pad)  # rechts padden
    )
    # Vektorisierter Bau der Positionsspalten
    pos_df = pd.DataFrame(
        {f"pos{i + 1}": seq_pad.str[i] for i in range(L)},
        index=df.index
    )
    # neues DF zurückgeben
    return pd.concat([df.copy(), pos_df], axis=1)


def compute_desc_row(sequence):
    p = pep.Peptide(sequence)
    d = p.descriptors()  # dict -> nur Zahlen
    return {f"desc__{k}": float(v) for k, v in d.items()}


def add_descriptors(df):
    desc_rows = [compute_desc_row(s) for s in df["sequence"]]
    desc_df = pd.DataFrame(desc_rows).reset_index(drop=True)
    df = df.reset_index(drop=True).join(desc_df)
    # sauber halten:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def print_results_tabular(results: list[dict], logger: logging.Logger):
    for results in results:
        logger.info(
            f"{results['name']}\t{results['model_tag']}\tR2 Train: {results['r2_train']:.3f}\tR2 Test: {results['r2_val']:.3f}\t"
            f"MSE Train: {results['mse_train']:.3f}\tMSE Test: {results['mse_val']:.3f}")


def evaluate_mic_models(best_estimator, plot_path, x_train, x_val, y_train, y_val, logger, file_name):
    train_r2, train_mse = get_model_stats(model=best_estimator,
                                          plot_dir=plot_path,
                                          feature_data=x_train,
                                          target_data=y_train,
                                          logger=logger,
                                          file_name=file_name,
                                          tag="Training")
    val_r2, val_mse = get_model_stats(model=best_estimator,
                                      plot_dir=plot_path,
                                      feature_data=x_val,
                                      target_data=y_val,
                                      logger=logger,
                                      file_name=file_name,
                                      tag="Validierung")

    overall_stats(best_estimator=best_estimator, x_test=x_train, y_test=y_train, save_path=plot_path, tag='Training')
    overall_stats(best_estimator=best_estimator, x_test=x_val, y_test=y_val, save_path=plot_path, tag='Validierung')
    return train_mse, train_r2, val_mse, val_r2


def prepare_df(file_path, task):
    df = pd.read_csv(file_path, sep=';')
    if task == 'mic':
        target_col = 'mic_log10'
    elif task == 'hemo':
        try:
            df = df.drop(columns=['hemo_percent', 'hemo_concentration'])
        except KeyError:
            pass
        target_col = 'label'
    else:
        raise ValueError("Invalid task. Please choose 'mic' or 'hemo'.")
    return df, target_col


def _get_scores(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, 'decision_function'):
        s = model.decision_function(X)
        # Map to 0..1 for plotting consistency (ranking is what matters)
        smin, smax = np.min(s), np.max(s)
        return (s - smin) / (smax - smin + 1e-12)
    # last resort – not ideal for curves
    return model.predict(X).astype(float)


def _best_f1_threshold(y_true, y_score, plot_path):
    p, r, thr = precision_recall_curve(y_true, y_score)

    display = PrecisionRecallDisplay.from_predictions(y_true, y_score, plot_chance_level=True, pos_label=1)
    _ = display.ax_.set_title("2-Klassen Precision-Recall Kurve")
    display.plot()
    plt.xlabel("Recall (Positive Klasse: 1)")
    plt.ylabel("Precision (Positive Klasse: 1)")
    plt.savefig(plot_path + '/precision_recall_curve.png')
    plt.close()
    plt.clf()

    f1 = 2 * p * r / (p + r + 1e-12)
    i = np.nanargmax(f1)
    # thresholds has length = len(p)-1; clamp index
    use_i = min(i, len(thr) - 1) if len(thr) > 0 else 0
    return (thr[use_i] if len(thr) else 0.5), f1[i], p[i], r[i]


def evaluate_hemo_model(best_estimator, model_name, plot_path, x_data, y_true, tag, logger):
    y_score = _get_scores(best_estimator, x_data)

    auc = roc_auc_score(y_true, y_score)

    ap = average_precision_score(y_true, y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score)

    thresholds = np.linspace(0.0, 1.0, 101)
    f1s, precisions, recalls, mccs = [], [], [], []

    for thr in thresholds:
        preds_bin = (y_score >= thr).astype(int)
        f1s.append(f1_score(y_true, preds_bin))
        precisions.append(precision_score(y_true, preds_bin))
        recalls.append(recall_score(y_true, preds_bin))
        mccs.append(matthews_corrcoef(y_true, preds_bin))

    # Plot F1 vs threshold
    plt.plot(thresholds, f1s, label="F1")
    plt.plot(thresholds, precisions, label="Präzision")
    plt.plot(thresholds, recalls, label="Sensitivität")
    plt.plot(thresholds, mccs, label="MCC")
    plt.xlabel("Schwellenwert")
    plt.ylabel("Wert")
    plt.legend()
    thres_curve = f'{plot_path}/{tag}_{model_name}_threshold_curves.png'
    plt.savefig(thres_curve)
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUROC={auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Falsch-Positiven-Rate')
    plt.ylabel('Richtig-Positiven-Rate (Sensitivität)')
    plt.title(f'ROC-Kurve ({tag})')
    plt.legend(loc='lower right')
    plt.grid(True)
    roc_path = f'{plot_path}/{tag}_{model_name}_roc.png'
    plt.savefig(roc_path)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label=f'AP={ap:.3f}')
    plt.hlines(np.mean(y_true), 0, 1, linestyles='--')
    plt.xlabel('Sensitivität')
    plt.ylabel('Präzision')
    plt.title(f'Präzisions-Sensivitäts-Kurve ({tag})')
    plt.legend(loc='lower left')
    plt.grid(True)
    pr_path = f'{plot_path}/{tag}_{model_name}_pr.png'
    plt.savefig(pr_path)
    plt.close()

    # Threshold tuning for F1
    thr, f1_best, p_best, r_best = _best_f1_threshold(y_true, y_score, plot_path)
    logger.info(f'Best F1 on val by thresholding: F1={f1_best:.4f} at thr={thr:.4f} '
                f'(P={p_best:.4f}, R={r_best:.4f})')

    # Generating predictions based on calculated threshold
    y_pred = (y_score >= thr).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred)
    logger.info(f"MCC: {mcc:.4f}")
    logger.info(f'AUROC: {auc:.4f}')
    logger.info(f'Average Precision (PR-AUC): {ap:.4f}')
    logger.info(f"Cls report here for {tag}")
    logger.info(classification_report(y_true, y_pred, digits=3))
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    with np.errstate(all='ignore'):
        confusion_matrix_normalized = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
    titles_options = [
        (f"{tag} Konfusionsmatrix, ohne Normalisierung", confusion_matrix),
        (f"{tag} Konfusionsmatrix, mit Normalisierung", confusion_matrix_normalized),
    ]

    for title, matrix in titles_options:
        cm_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0, 1])
        cm_display.plot()
        plt.title(title)
        plt.xlabel('Vorhergesagte Klasse')
        plt.ylabel('Tatsächliche Klasse')
        plt.savefig(f"{plot_path}/{tag}_{title}.png")
        plt.close()
        plt.clf()


def prepare_train_val_data(calculate_features, df, target_col, out_dir, feature_selection=None):
    if calculate_features:
        # x_train, y_train, x_val, y_val = encode_onehot_with_features(df, target_col=target_col)
        x_train, y_train, x_val, y_val, feature_names = encode_kmer_with_features(df, target_col=target_col,
                                                                                  feature_selection=feature_selection)

        if out_dir is not None:
            x_val_df = pd.DataFrame(x_val, columns=feature_names)

            # y_val als DataFrame oder Series
            y_val_df = pd.Series(y_val, name=target_col)  # oder DataFrame: pd.DataFrame(y_val, columns=[target_col])

            # Kombinieren (falls du Features + Label in einem DF willst)
            val_df = pd.concat([x_val_df, y_val_df.reset_index(drop=True)], axis=1)

            val_df.to_csv(f"{out_dir}/val_data_features.csv", sep=';', index=False)

        return x_train, x_val, y_train, y_val, feature_names
    else:
        # df_to_split = df['sequence'].unique()
        #
        # train, test = train_test_split(df_to_split,
        #                                train_size=0.8,
        #                                test_size=0.2,
        #                                shuffle=True,
        #                                random_state=42)
        #
        # train_df = df[df['sequence'].isin(train)]
        # val_df = df[df['sequence'].isin(test)]
        #
        # x_train, y_train = enocde_onehot_without_features(train_df, target_col=target_col)
        # x_val, y_val = enocde_onehot_without_features(val_df, target_col=target_col)

        x_train, y_train, x_val, y_val = encode_kmer_no_features(df, target_col=target_col)

        if out_dir is not None:
            x_val_df = pd.DataFrame(x_val)

            # y_val als DataFrame oder Series
            y_val_df = pd.Series(y_val, name=target_col)  # oder DataFrame: pd.DataFrame(y_val, columns=[target_col])

            # Kombinieren (falls du Features + Label in einem DF willst)
            val_df = pd.concat([x_val_df, y_val_df.reset_index(drop=True)], axis=1)

            val_df.to_csv(f"{out_dir}/val_data.csv", sep=';', index=False)

        return x_train, x_val, y_train, y_val, None


def encode_kmer_with_features(df, target_col, n_components=128, feature_selection=None):
    df = add_descriptors(df)  # <- deine Funktion für desc__*
    df = df[feature_selection + ['sequence', target_col]] if feature_selection else df

    desc_cols = [c for c in df.columns if c.startswith('desc__')]

    # Split ohne Leckage: nach einzigartigen Sequenzen
    uniq = df['sequence'].unique()
    tr_seqs, va_seqs = train_test_split(uniq, test_size=0.2,
                                        random_state=42, shuffle=True)
    train_df = df[df['sequence'].isin(tr_seqs)].copy()
    val_df = df[df['sequence'].isin(va_seqs)].copy()

    y_train = train_df[target_col].astype(float).to_numpy()
    y_val = val_df[target_col].astype(float).to_numpy()

    # --- k-mer TF-IDF ---
    tfidf = TfidfVectorizer(analyzer='char',
                            ngram_range=(3, 4),  # 3- und 4-mer
                            min_df=2)  # ignoriert seltene
    Xk_tr = tfidf.fit_transform(train_df['sequence'])
    Xk_va = tfidf.transform(val_df['sequence'])

    # --- Dimensionalität reduzieren ---
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Z_tr = svd.fit_transform(Xk_tr)
    Z_va = svd.transform(Xk_va)

    # --- Deskriptoren ---
    imp = SimpleImputer(strategy='median')
    D_tr = imp.fit_transform(train_df[desc_cols])
    D_va = imp.transform(val_df[desc_cols])

    # --- Concatenate ---
    X_train = np.hstack([Z_tr, D_tr])
    X_val = np.hstack([Z_va, D_va])

    # Feature-Namen: SVD-Komp. + Deskriptoren
    svd_names = [f'kmer_svd{i + 1}' for i in range(n_components)]
    feature_names = svd_names + desc_cols

    return X_train, y_train, X_val, y_val, feature_names


def encode_kmer_no_features(df, target_col, n_components=128, feature_selection=None):
    # Split ohne Leckage: nach einzigartigen Sequenzen
    uniq = df['sequence'].unique()
    tr_seqs, va_seqs = train_test_split(uniq, test_size=0.2,
                                        random_state=42, shuffle=True)
    train_df = df[df['sequence'].isin(tr_seqs)].copy()
    val_df = df[df['sequence'].isin(va_seqs)].copy()

    y_train = train_df[target_col].astype(float).to_numpy()
    y_val = val_df[target_col].astype(float).to_numpy()

    # --- k-mer TF-IDF ---
    tfidf = TfidfVectorizer(analyzer='char',
                            ngram_range=(3, 4),  # 3- und 4-mer
                            min_df=2)  # ignoriert seltene
    x_train = tfidf.fit_transform(train_df['sequence'])
    x_val = tfidf.transform(val_df['sequence'])

    # --- Dimensionalität reduzieren ---
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    x_train = svd.fit_transform(x_train)
    x_val = svd.transform(x_val)

    return x_train, y_train, x_val, y_val


def get_feature_importance(file_path: str,
                           task,
                           plot_path):
    """
    After
    https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py

    """
    if task == 'mic':
        model = RandomForestRegressor(n_estimators=800,
                                      max_depth=None,
                                      max_features='sqrt',
                                      min_samples_split=2,
                                      random_state=42,
                                      n_jobs=1)

        model_for_pi = RandomForestRegressor(n_estimators=800,
                                             max_depth=None,
                                             max_features='sqrt',
                                             min_samples_split=2,
                                             random_state=42,
                                             n_jobs=1)
        scoring = 'r2'

    else:
        model = RandomForestClassifier(n_estimators=800,
                                       max_depth=None,
                                       max_features='sqrt',
                                       min_samples_split=2,
                                       random_state=42,
                                       n_jobs=1)

        model_for_pi = RandomForestClassifier(n_estimators=800,
                                              max_depth=None,
                                              max_features='sqrt',
                                              min_samples_split=2,
                                              random_state=42,
                                              n_jobs=1)
        scoring = 'roc_auc'

    df, target_col = prepare_df(file_path, task)

    x_train, x_val, y_train, y_val, feature_names = prepare_train_val_data(calculate_features, df, target_col, None)

    desc_idx = [i for i, f in enumerate(feature_names) if f.startswith('desc__')]
    desc_names = [desc for desc in feature_names if desc.startswith('desc__')]
    x_desc = x_train[:, desc_idx]
    x_val_desc = x_val[:, desc_idx]

    print("Starting PI")
    model_for_pi = model_for_pi.fit(x_desc, y_train)
    plot_utils.plot_permutation_importance_from_est(estimator=model_for_pi,
                                                    x_data=x_val_desc,
                                                    y_data=y_val,
                                                    plot_path=plot_path + '/feature_importance_est.png',
                                                    summary_path=plot_path + '/feature_importance_summary.txt',
                                                    feature_names=desc_names,
                                                    )

    print("PI done")

    rfecv = RFECV(estimator=model,
                  cv=5,
                  scoring=scoring,
                  n_jobs=-1)
    rfecv.fit(x_desc, y_train)

    data = {
        key: value
        for key, value in rfecv.cv_results_.items()
        if key in ["n_features", "mean_test_score", "std_test_score"]
    }
    cv_results = pd.DataFrame(data)
    plt.figure()
    plt.xlabel("Anzahl der gewählten Features")
    plt.ylabel("Mittlere Testgenauigkeit")
    plt.errorbar(
        x=cv_results["n_features"],
        y=cv_results["mean_test_score"],
        yerr=cv_results["std_test_score"],
    )
    plt.title("Rekursive Feature Elimination")
    plt.savefig(plot_path + '/Feature_elimination.png')
    plt.close()
    plt.clf()

    # Auswahl der desc__ Features
    selected_desc_idx = [i for i, keep in zip(desc_idx, rfecv.support_) if keep]
    selected_names = [feature_names[i] for i in selected_desc_idx]

    with open(f"{plot_path}/feature_selection.txt", 'w', encoding='utf-8') as file:
        for line in selected_names:
            file.write(line + '\n')

    return selected_names


if __name__ == '__main__':
    ...
