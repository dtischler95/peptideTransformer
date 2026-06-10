import logging
from pathlib import Path
import pandas as pd
import json
import os
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import peptides as pep
try:
    from src.evaluation import eval_utils
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from evaluation import eval_utils
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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



print_regression_metrics = eval_utils.print_regression_metrics


def get_model_stats(model, plot_dir: str, feature_data, target_data, logger: logging.Logger,
                    file_name, tag: str, model_name):
    pred = model.predict(feature_data)
    r2, mse = eval_utils.print_regression_metrics(y_true=target_data, y_pred=pred, logger=logger, tag=tag)
    eval_utils.make_regression_plot(y_true=target_data, y_pred=pred,
                                    path=f"{plot_dir}/{tag}_regression.pdf",
                                    file_name=file_name, tag=tag)
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
            'f1': 'f1'
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
                               n_jobs=-1, verbose=3, cv=cv, scoring=scoring).fit(x_train, y_train)
    best_estimator = grid_search.best_estimator_
    save_model(model=best_estimator, path=f"{model_dir}{model.__class__.__name__}.keras")
    return best_estimator, grid_search, model



def overall_stats(best_estimator, x_test, y_test, save_path, tag, file_name, model_name):
    predictions = best_estimator.predict(x_test)
    eval_utils.overall_stats(y_true=y_test, predictions=predictions, save_path=save_path,
                             tag=tag, file_name=file_name, model_name=model_name)



PAD = "-"
MAX_LEN = 36
AA = list("ACDEFGHIKLMNPQRSTVWY")  # standard 20 amino acids
CATEGORIES = AA + [PAD]



def compute_desc_row(sequence):
    p = pep.Peptide(sequence)
    d = p.descriptors()
    return {f"desc__{k}": float(v) for k, v in d.items()}


def add_descriptors(df):
    desc_rows = [compute_desc_row(s) for s in df["sequence"]]
    desc_df = pd.DataFrame(desc_rows).reset_index(drop=True)
    df = df.reset_index(drop=True).join(desc_df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def print_results_tabular(results: list[dict], logger: logging.Logger):
    for results in results:
        logger.info(
            f"{results['name']}\t{results['model_tag']}\tR2 Train: {results['r2_train']:.3f}\tR2 Test: {results['r2_val']:.3f}\t"
            f"MSE Train: {results['mse_train']:.3f}\tMSE Test: {results['mse_val']:.3f}")


def evaluate_mic_models(best_estimator, plot_path, x_train, x_val, y_train, y_val, logger, file_name, model_name):
    train_r2, train_mse = get_model_stats(model=best_estimator,
                                          plot_dir=plot_path,
                                          feature_data=x_train,
                                          target_data=y_train,
                                          logger=logger,
                                          file_name=file_name,
                                          tag="Training",
                                          model_name=model_name)
    val_r2, val_mse = get_model_stats(model=best_estimator,
                                      plot_dir=plot_path,
                                      feature_data=x_val,
                                      target_data=y_val,
                                      logger=logger,
                                      file_name=file_name,
                                      tag="Test",
                                      model_name=model_name)

    overall_stats(best_estimator=best_estimator,
                  x_test=x_train,
                  y_test=y_train,
                  save_path=plot_path,
                  tag='Training',
                  file_name=file_name,
                  model_name=model_name)
    overall_stats(best_estimator=best_estimator,
                  x_test=x_val,
                  y_test=y_val,
                  save_path=plot_path,
                  tag='Test',
                  file_name=file_name,
                  model_name=model_name)
    return train_mse, train_r2, val_mse, val_r2


def prepare_df(file_path, task):
    base = Path(file_path).with_suffix('')
    train_df = pd.read_csv(f"{base}_train.csv", sep=';')
    test_df = pd.read_csv(f"{base}_test.csv", sep=';')
    if task == 'mic':
        target_col = 'mic_log10'
    elif task == 'hemo':
        try:
            train_df = train_df.drop(columns=['hemo_percent', 'hemo_concentration'])
            test_df = test_df.drop(columns=['hemo_percent', 'hemo_concentration'])
        except KeyError:
            pass
        target_col = 'label'
    else:
        raise ValueError("Invalid task. Please choose 'mic' or 'hemo'.")
    return train_df, test_df, target_col


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


def plot_residuals_vs_length_from_df(model,
                      df,
                      target_col,
                      plot_path,
                      calculate_features,
                      title="Residual Density Plot"):
    df = df.copy()
    df["label"] = df["label"].map({0: "Gram–", 1: "Gram+"})

    y_true = df[target_col].astype(float)

    _, x_test, _, y_test, _ = prepare_train_val_data(
        calculate_features=calculate_features,
        train_df=df,
        test_df=df,
        target_col=target_col,
        out_dir=''
    )

    y_pred = model.predict(x_test)
    resid = y_true - y_pred

    plot_df = df.assign(resid=resid)

    # Figure
    plt.figure(figsize=(7, 5))

    sns.kdeplot(
        data=plot_df,
        x="resid",
        hue="label",
        fill=True,
        common_norm=False,
        alpha=0.35,
        linewidth=1.8
    )

    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.title(title)
    plt.tight_layout()

    plt.savefig(plot_path + "/residual_kde.png")
    plt.close()

    return resid


def evaluate_hemo_model(best_estimator, model_name, plot_path, x_data, y_true, tag, logger, data_name):
    y_score = _get_scores(best_estimator, x_data)
    y_pred = (y_score >= 0.5).astype(int)
    eval_utils.evaluate_hemo(y_true=y_true, y_score=y_score, y_pred=y_pred, plot_path=plot_path,
                             tag=tag, file_name=data_name, model_name=model_name, logger=logger)


def prepare_train_val_data(calculate_features, train_df, test_df, target_col, out_dir):
    if calculate_features:

        x_train, y_train, x_val, y_val, feature_names = encode_kmer_with_features(train_df=train_df, val_df=test_df, target_col=target_col)

        if out_dir is not None:
            x_val_df = pd.DataFrame(x_val, columns=feature_names)


            y_val_df = pd.Series(y_val, name=target_col)

            pd.concat([x_val_df, y_val_df.reset_index(drop=True)], axis=1)



        return x_train, x_val, y_train, y_val, feature_names
    else:


        x_train, y_train, x_val, y_val = encode_kmer_no_features(train_df=train_df, val_df=test_df, target_col=target_col)


        return x_train, x_val, y_train, y_val, None


def encode_kmer_with_features(train_df, val_df, target_col, n_components=128):
    train_df = add_descriptors(train_df)
    val_df = add_descriptors(val_df)


    desc_cols = [c for c in train_df.columns if c.startswith('desc__')]


    y_train = train_df[target_col].astype(float).to_numpy()
    y_val = val_df[target_col].astype(float).to_numpy()

    # --- k-mer TF-IDF ---
    tfidf = TfidfVectorizer(analyzer='char',
                            ngram_range=(3, 4),  # 3- and 4-mers
                            min_df=2)  # ignores rare k-mers
    Xk_tr = tfidf.fit_transform(train_df['sequence'])
    Xk_va = tfidf.transform(val_df['sequence'])

    # --- reduce dimensionality ---
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Z_tr = svd.fit_transform(Xk_tr)
    Z_va = svd.transform(Xk_va)

    # --- descriptors ---
    imp = SimpleImputer(strategy='median')
    D_tr = imp.fit_transform(train_df[desc_cols])
    D_va = imp.transform(val_df[desc_cols])

    # --- Concatenate ---
    X_train = np.hstack([Z_tr, D_tr])
    X_val = np.hstack([Z_va, D_va])

    # feature names: SVD components + descriptors
    svd_names = [f'kmer_svd{i + 1}' for i in range(n_components)]
    feature_names = svd_names + desc_cols

    return X_train, y_train, X_val, y_val, feature_names


def encode_kmer_no_features(train_df, val_df, target_col, n_components=128, feature_selection=None):


    y_train = train_df[target_col].astype(float).to_numpy()
    y_val = val_df[target_col].astype(float).to_numpy()

    # --- k-mer TF-IDF ---
    tfidf = TfidfVectorizer(analyzer='char',
                            ngram_range=(3, 4),  # 3- and 4-mers
                            min_df=2)  # ignores rare k-mers
    x_train = tfidf.fit_transform(train_df['sequence'])
    x_val = tfidf.transform(val_df['sequence'])

    # --- reduce dimensionality ---
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    x_train = svd.fit_transform(x_train)
    x_val = svd.transform(x_val)

    return x_train, y_train, x_val, y_val


if __name__ == '__main__':
    ...
