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
from sklearn.compose import ColumnTransformer
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


def _remap_grid_to_model(param_grid):
    """
    Normalise param-grid keys to the estimator step of the pipeline.

    Grids in config.py mix bare keys (tree models) and estimator-prefixed keys
    (``svr__C``, ``svc__C``). Since the estimator now always lives under the
    ``model`` step of the full featurising pipeline, every key is rewritten to
    ``model__<param>`` so the grids themselves stay untouched.
    """
    return {f"model__{k.split('__')[-1]}": v for k, v in param_grid.items()}


def grid_search_setup(pipeline, model_dir, model_name, param_grid, x_train, y_train, task):
    """
    Setup for the Gridsearch in machine learning logic.

    ``pipeline`` is the full featurising pipeline (k-mer TF-IDF -> SVD, optional
    descriptors and scaler, then the estimator under the ``model`` step). Because
    the featuriser is part of the pipeline, GridSearchCV refits it inside every
    CV fold, so no preprocessing is fit on a fold's held-out data.
    """
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

    grid = _remap_grid_to_model(param_grid)
    grid_search = GridSearchCV(estimator=pipeline, param_grid=grid, return_train_score=True, refit=refit,
                               n_jobs=-1, verbose=3, cv=cv, scoring=scoring).fit(x_train, y_train)
    best_estimator = grid_search.best_estimator_
    save_model(model=best_estimator, path=f"{model_dir}{model_name}.keras")
    return best_estimator, grid_search, pipeline



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

    val_path = Path(f"{base}_val.csv")
    val_df = pd.read_csv(val_path, sep=';') if val_path.exists() else None

    if task == 'mic':
        target_col = 'mic_log10'
    elif task == 'hemo':
        drop_cols = ['hemo_percent', 'hemo_concentration']
        train_df = train_df.drop(columns=drop_cols, errors='ignore')
        test_df = test_df.drop(columns=drop_cols, errors='ignore')
        if val_df is not None:
            val_df = val_df.drop(columns=drop_cols, errors='ignore')
        target_col = 'label'
    else:
        raise ValueError("Invalid task. Please choose 'mic' or 'hemo'.")

    # Fold the validation split into the development pool, so hyperparameters are
    # chosen by 5-fold CV over train+val. This matches the data budget BERT uses
    # (train for fitting, val for early stopping). The test split stays untouched
    # and is only used for the final reported metrics.
    if val_df is not None:
        train_df = pd.concat([train_df, val_df], ignore_index=True)

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

    _, _, x_test, _, _ = build_xy(
        train_df=df,
        test_df=df,
        target_col=target_col,
        calculate_features=calculate_features,
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


class SafeTruncatedSVD(TruncatedSVD):
    """
    TruncatedSVD that clamps ``n_components`` to the available feature count.

    Inside a CV fold the k-mer vocabulary can occasionally be smaller than the
    requested number of components (small or low-diversity folds). TruncatedSVD
    requires ``n_components < n_features``, so we clamp at fit time. GridSearchCV
    clones the estimator before each fit, so the original 128 is always restored
    for the next fold.
    """
    def fit(self, X, y=None):
        n_features = X.shape[1]
        if self.n_components >= n_features:
            self.n_components = max(1, n_features - 1)
        return super().fit(X, y)

    def fit_transform(self, X, y=None):
        n_features = X.shape[1]
        if self.n_components >= n_features:
            self.n_components = max(1, n_features - 1)
        return super().fit_transform(X, y)


def build_xy(train_df, test_df, target_col, calculate_features):
    """
    Build the *raw* model inputs (a DataFrame with a ``sequence`` column and,
    optionally, deterministic per-sequence descriptors) plus the target arrays.

    No fitted transformation happens here: descriptors are a pure function of the
    sequence, so computing them outside the CV loop does not leak. Everything that
    is *fit* (TF-IDF, SVD, imputer, scaler) lives in the pipeline from
    ``build_feature_pipeline`` and is therefore refit per CV fold.
    """
    y_train = train_df[target_col].astype(float).to_numpy()
    y_test = test_df[target_col].astype(float).to_numpy()

    if calculate_features:
        tr = add_descriptors(train_df[['sequence']].copy())
        te = add_descriptors(test_df[['sequence']].copy())
        desc_cols = [c for c in tr.columns if c.startswith('desc__')]
        x_train = tr[['sequence'] + desc_cols]
        x_test = te[['sequence'] + desc_cols]
    else:
        desc_cols = []
        x_train = train_df[['sequence']].copy()
        x_test = test_df[['sequence']].copy()

    return x_train, y_train, x_test, y_test, desc_cols


def build_feature_pipeline(estimator, model_name, calculate_features, desc_cols, n_components=128):
    """
    Full featurising pipeline ending in the estimator under the ``model`` step.

    Sequence branch: char k-mer TF-IDF (3-4mers) -> SVD. Descriptor branch (only
    when ``calculate_features``): median imputation. SVR/SVC additionally get a
    StandardScaler on the combined feature matrix. Because the whole thing is one
    pipeline, GridSearchCV refits every fitted step inside each CV fold.
    """
    seq_branch = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char',
                                  ngram_range=(3, 4),  # 3- and 4-mers
                                  min_df=2)),           # ignores rare k-mers
        ('svd', SafeTruncatedSVD(n_components=n_components, random_state=42)),
    ])

    transformers = [('seq', seq_branch, 'sequence')]
    if calculate_features and desc_cols:
        transformers.append(('desc', SimpleImputer(strategy='median'), desc_cols))

    features = ColumnTransformer(transformers, remainder='drop')

    steps = [('features', features)]
    if model_name in ('svr', 'svc'):
        steps.append(('scaler', StandardScaler()))
    steps.append(('model', estimator))

    return Pipeline(steps)


if __name__ == '__main__':
    ...
