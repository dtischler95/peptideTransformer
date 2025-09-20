from sklearnex import patch_sklearn

patch_sklearn(verbose=False)
from sklearn import metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report,
                             ConfusionMatrixDisplay, average_precision_score, roc_curve, precision_recall_curve)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import sys
import ml_utils
import pandas as pd
import config
import logging

from pathlib import Path
from typing import Iterable, Any

logger = logging.getLogger(__name__)

# --------------------- Setup logging and configs ---------------------
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)



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

def _best_f1_threshold(y_true, y_score):
    p, r, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * p * r / (p + r + 1e-12)
    i = np.nanargmax(f1)
    # thresholds has length = len(p)-1; clamp index
    use_i = min(i, len(thr)-1) if len(thr) > 0 else 0
    return (thr[use_i] if len(thr) else 0.5), f1[i], p[i], r[i]

def train_regressors(file_path: str,
                     regressor,
                     param_grid,
                     plot_path: str,
                     model_name: str,
                     task: str = 'mic',
                     calculate_features: bool = True
                     ):
    df, target_col = ml_utils.prepare_df(file_path, task)

    if calculate_features:
        x_train, y_train, x_val, y_val = ml_utils.encode_onehot_with_features(df, target_col=target_col)
    else:
        df_to_split = df['sequence'].unique()

        train, test = train_test_split(df_to_split,
                                       train_size=0.8,
                                       test_size=0.2,
                                       shuffle=True,
                                       random_state=42)

        train_df = df[df['sequence'].isin(train)]
        val_df = df[df['sequence'].isin(test)]

        x_train, y_train = ml_utils.enocde_onehot_without_features(train_df, target_col=target_col)
        x_val, y_val = ml_utils.enocde_onehot_without_features(val_df, target_col=target_col)

    logger.info(f"Train shape: {len(x_train)}\tTest shape: {len(x_val)}"
                f"\nTrain target shape: {len(y_train)}\tTest target shape: {len(y_val)}")

    best_estimator, grid_search, model = ml_utils.grid_search_setup(regressor, plot_path + '/', model_name, param_grid,
                                                                    x_train,
                                                                    y_train,
                                                                    task)

    test_score = best_estimator.score(x_val, y_val)
    logger.info(f"Test score of the best model: {test_score}")
    logger.info(f"Best Params: {grid_search.best_params_}")

    if task == 'mic':
        train_mse, train_r2, val_mse, val_r2 = ml_utils.evaluate_mic_models(best_estimator, plot_path, x_train, x_val, y_train,
                                                                   y_val, logger)

        return train_r2, train_mse, val_r2, val_mse

    elif task == 'hemo':
        #y_pred = best_estimator.predict(x_val)

        y_score = _get_scores(best_estimator, x_val)



        try:
            auc = roc_auc_score(y_val, y_score)
        except Exception:
            auc = float('nan')
        ap = average_precision_score(y_val, y_score)


        # ROC curve
        try:
            fpr, tpr, _ = roc_curve(y_val, y_score)
            plt.figure()
            plt.plot(fpr, tpr, label=f'AUROC={auc:.3f}')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel('Falsch-Positiven-Rate')
            plt.ylabel('Richtig-Positiven-Rate (Recall)')
            plt.title('ROC-Kurve (Validierung)')
            plt.legend(loc='lower right')
            plt.grid(True)
            roc_path = f'{plot_path}/{model_name}_roc.png'
            plt.savefig(roc_path)
            plt.close()
        except Exception as e:
            logger.warning(f'ROC plotting skipped: {e}')


        # PR curve
        try:
            precision, recall, _ = precision_recall_curve(y_val, y_score)
            plt.figure()
            plt.plot(recall, precision, label=f'AP={ap:.3f}')
            plt.hlines(np.mean(y_val), 0, 1, linestyles='--')
            plt.xlabel('Sensitivität')
            plt.ylabel('Präzision')
            plt.title('Präzisions-Sensivitäts-Kurve (Validierung)')
            plt.legend(loc='lower left')
            plt.grid(True)
            pr_path = f'{plot_path}/{model_name}_pr.png'
            plt.savefig(pr_path)
            plt.close()
        except Exception as e:
            logger.warning(f'PR plotting skipped: {e}')

        # Threshold tuning for F1
        thr, f1_best, p_best, r_best = _best_f1_threshold(y_val, y_score)
        logger.info(f'Best F1 on val by thresholding: F1={f1_best:.4f} at thr={thr:.4f} '
                    f'(P={p_best:.4f}, R={r_best:.4f})')

        y_pred = (y_score >= thr).astype(int)

        logger.info(f'AUROC: {auc:.4f}')
        logger.info(f'Average Precision (PR-AUC): {ap:.4f}')

        logger.info("Cls report here")
        logger.info(classification_report(y_val, y_pred, digits=3))

        confusion_matrix = metrics.confusion_matrix(y_val, y_pred)

        with np.errstate(all='ignore'):
            confusion_matrix_normalized = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

        titles_options = [
            ("Konfusionsmatrix, ohne Normalisierung", confusion_matrix),
            ("Konfusionsmatrix, mit Normalisierung", confusion_matrix_normalized),
        ]

        for title, matrix in titles_options:

            cm_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0, 1])
            cm_display.plot()
            plt.title(title)
            plt.xlabel('Vorhergesagte Klasse')
            plt.ylabel('Tatsächliche Klasse')
            plt.savefig(f"{plot_path}/{title}.png")
            plt.close()
            plt.clf()



    return None


def run_all(
        data_dir: str | Path = "../../data/regression_data",
        output_root: str | Path = "./final_plots/",
        task: str = 'mic',
        models: Iterable[tuple[str, Any]] = None,  # e.g. regressor_list
        grids: Iterable[dict] = None,  # e.g. param_grids
        calculate_features: bool = True,
):
    """
    Iterate over all CSVs in data_dir and train each model/grid combo.
    Creates output folders like: {output_root}/{model_tag}/{file_slug}/
    Returns a list of RunResult with basic status info per run.
    """


    # Prevent silent truncation if lengths differ
    models = list(models)
    grids = list(grids)

    data_dir = Path(data_dir)
    output_root = Path(output_root)

    results = []
    logger.info(f"Running all models on {data_dir} and saving plots to {output_root}")

    csv_files = sorted(data_dir.glob("*.csv"))
    for csv_path in csv_files:
        # Safer way to derive a short slug from filename, OS-independent
        parts = csv_path.stem.split("_")
        file_name = "_".join(parts[:2])  # if len(parts) >= 2 else csv_path.stem

        for (model_tag, estimator), grid in zip(models, grids):
            out_dir = output_root / model_tag / file_name
            out_dir.mkdir(parents=True, exist_ok=True)

            if task == 'mic':
                # Pass the estimator/grid down so train_regressors is self-contained
                train_r2, train_mse, val_r2, val_mse = train_regressors(
                    file_path=str(csv_path),
                    regressor=estimator,
                    param_grid=grid,
                    model_name=model_tag,
                    calculate_features=calculate_features,
                    task=task,
                    plot_path=str(out_dir)
                )
                results.append({
                    "name": file_name,
                    "r2_train": train_r2,
                    "r2_val": val_r2,
                    "mse_train": train_mse,
                    "mse_val": val_mse,
                })
            elif task == 'hemo':
                train_regressors(
                    file_path=str(csv_path),
                    regressor=estimator,
                    param_grid=grid,
                    model_name=model_tag,
                    calculate_features=calculate_features,
                    task=task,
                    plot_path=str(out_dir)
                )

    ml_utils.print_results_tabular(results, logger=logger)


if __name__ == "__main__":
    # train_regressors('../../data/regression_data/acinetobacter_baumannii_for_regression.csv',
    #                  calculate_features=False)
    # TODO USE CONFIG FOR ALL THIS
    task = 'mic' # 'hemo' or 'mic'

    if task == 'mic':
        data_dir = './data/regression_data/'
        model_list = [  # ('gb', GradientBoostingRegressor()),
            ('xtra', ExtraTreesRegressor()),
            ('xgb', XGBRegressor()),
            ('rf', RandomForestRegressor()),
            ('svr', SVR())
        ]

        param_grids = [  # config.gb_param_grid,
            config.xtra_param_grid,
            config.xgb_param_grid,
            config.rf_param_grid,
            config.svr_param_grid
        ]
    elif task =='hemo':
        data_dir = './data/hemo_train/'
        model_list = [#('gb',   GradientBoostingClassifier()),
            ('xtra', ExtraTreesClassifier()),
            ('xgb', XGBClassifier()),
            ('rf', RandomForestClassifier()),
            ('svc', SVC(probability=True)),

        ]

        param_grids = [  # config.gb_param_grid,
            config.xtra_cls_param_grid,
            config.xgb_cls_param_grid,
            config.rf_cls_param_grid,
            config.svc_cls_param_grid
        ]
    else:
        raise ValueError("Invalid task. Please choose 'mic' or 'hemo'.")


    run_all(calculate_features=False,
            task=task,
            data_dir=data_dir,
            models=model_list,
            grids=param_grids)
