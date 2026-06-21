import sys
import logging
from pathlib import Path
from typing import Iterable, Any

import matplotlib as mpl
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

try:
    from src.machine_learning import ml_utils, config
    from src.data_preprocessing import datasets
except ImportError:
    import ml_utils  # type: ignore
    import config    # type: ignore
    import datasets  # type: ignore

_REPO_ROOT = Path(__file__).resolve().parents[2]

mpl.rcParams.update({
    'font.size': 15,        # base font size
    'axes.titlesize': 16,       # axis title size
    'axes.labelsize': 15,       # axis label size (xlabel, ylabel)
    'xtick.labelsize': 14,      # x-axis tick label size
    'ytick.labelsize': 14,      # y-axis tick label size
})

logger = logging.getLogger(__name__)

# --------------------- Setup logging and configs ---------------------
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)

# O(n^2) kernel matrices blow up memory when CV folds run in parallel.
_SERIAL_GRID_MODELS = {"svr", "svc"}


def train_classificators(file_path: str,
                         regressor,
                         param_grid,
                         plot_path: str,
                         model_name: str,
                         data_name: str,
                         calculate_features: bool = True,
                         split: str = "random",
                         seed: int = 42,
                         n_jobs: int = -1
                         ):
    train_df, test_df, target_col = ml_utils.prepare_df(file_path, 'hemo')

    x_train, y_train, x_test, y_test, desc_cols = ml_utils.build_xy(train_df,
                                                                    test_df,
                                                                    target_col,
                                                                    calculate_features)

    logger.info(f"Train shape: {len(x_train)}\tTest shape: {len(x_test)}"
                f"\nTrain target shape: {len(y_train)}\tTest target shape: {len(y_test)}\n"
                f"For File {file_path}")

    pipeline = ml_utils.build_feature_pipeline(regressor, model_name, calculate_features, desc_cols)

    best_estimator, grid_search, _ = ml_utils.grid_search_setup(pipeline, plot_path + '/', model_name, param_grid,
                                                                x_train,
                                                                y_train,
                                                                'hemo',
                                                                seed=seed,
                                                                n_jobs=n_jobs)

    test_auc = roc_auc_score(y_test, ml_utils._get_scores(best_estimator, x_test))
    logger.info(f"CV best AUROC: {grid_search.best_score_:.3f} | Test AUROC: {test_auc:.3f}")
    logger.info(f"Best Params: {grid_search.best_params_}")

    ml_utils.evaluate_hemo_model(best_estimator=best_estimator,
                                 model_name=model_name,
                                 plot_path=plot_path,
                                 x_data=x_train,
                                 y_true=y_train,
                                 tag='Training',
                                 logger=logger,
                                 data_name=data_name)

    ml_utils.evaluate_hemo_model(best_estimator=best_estimator,
                                 model_name=model_name,
                                 plot_path=plot_path,
                                 x_data=x_test,
                                 y_true=y_test,
                                 tag='Test',
                                 logger=logger,
                                 data_name=data_name)

    y_score_test = ml_utils._get_scores(best_estimator, x_test)
    y_pred_test = (y_score_test >= 0.5).astype(int)
    ml_utils.write_run_artifacts(
        out_dir=plot_path,
        data_name=data_name,
        model_name=model_name,
        task="hemo",
        split=split,
        seed=seed,
        features=calculate_features,
        y_true=y_test,
        y_pred=y_pred_test,
        y_score=y_score_test,
        sequences=x_test["sequence"].to_numpy() if "sequence" in x_test.columns else None,
    )


def train_regressors(file_path: str,
                     regressor,
                     param_grid,
                     plot_path: str,
                     model_name: str,
                     file_name: str,
                     calculate_features: bool = True,
                     gram_mode: bool = False,
                     split: str = "random",
                     seed: int = 42,
                     n_jobs: int = -1
                     ):
    train_df, test_df, target_col = ml_utils.prepare_df(file_path, 'mic')  # mic for gram and mic

    x_train, y_train, x_test, y_test, desc_cols = ml_utils.build_xy(train_df,
                                                                    test_df,
                                                                    target_col,
                                                                    calculate_features)

    logger.info(f"Train shape: {len(x_train)}\tTest shape: {len(x_test)}"
                f"\nTrain target shape: {len(y_train)}\tTest target shape: {len(y_test)}")

    logger.info(f"File Name: {file_path} and Model: {model_name}")

    pipeline = ml_utils.build_feature_pipeline(regressor, model_name, calculate_features, desc_cols)

    best_estimator, grid_search, _ = ml_utils.grid_search_setup(pipeline, plot_path + '/', model_name, param_grid,
                                                                x_train,
                                                                y_train,
                                                                'mic',
                                                                seed=seed,
                                                                n_jobs=n_jobs)

    test_score = best_estimator.score(x_test, y_test)
    logger.info(f"CV best R2: {grid_search.best_score_:.3f} | Test R2: {test_score:.3f}")
    logger.info(f"Best Params: {grid_search.best_params_}")

    if gram_mode:
        ml_utils.plot_residuals_vs_length_from_df(
            model=best_estimator,
            df=test_df,
            target_col=target_col,
            plot_path=plot_path,
            calculate_features=calculate_features,
        )

    train_mse, train_r2, val_mse, val_r2 = ml_utils.evaluate_mic_models(best_estimator,
                                                                        plot_path,
                                                                        x_train,
                                                                        x_test,
                                                                        y_train,
                                                                        y_test,
                                                                        logger,
                                                                        file_name,
                                                                        model_name)

    y_pred_test = best_estimator.predict(x_test)
    ml_utils.write_run_artifacts(
        out_dir=plot_path,
        data_name=file_name,
        model_name=model_name,
        task="mic",
        split=split,
        seed=seed,
        features=calculate_features,
        y_true=y_test,
        y_pred=y_pred_test,
        sequences=x_test["sequence"].to_numpy() if "sequence" in x_test.columns else None,
    )

    return train_r2, train_mse, val_r2, val_mse


def run_classification(
        data_dir: str | Path = _REPO_ROOT / "data" / "hemo_train",
        output_root: str | Path = _REPO_ROOT / "ml_plots",
        models: Iterable[tuple[str, Any]] = None,  # e.g. classifier_list
        grids: Iterable[dict] = None,  # e.g. param_grids
        calculate_features: bool = True,
        seed: int = 42,
        organism: str | None = None,
        n_jobs: int = -1,
):
    """Train all classification models on each dataset in data_dir."""

    # Prevent silent truncation if lengths differ
    models = list(models)
    grids = list(grids)
    data_dir = Path(data_dir)
    output_root = Path(output_root)
    split = "cluster" if "cluster" in data_dir.name else "random"
    csv_files = datasets.split_basepaths_in(data_dir)
    if organism:
        csv_files = [p for p in csv_files if organism.lower() in p.stem.lower()]
        if not csv_files:
            raise ValueError(f"no base file in {data_dir} matches organism {organism!r}")

    for csv_path in csv_files:
        file_name = datasets.organism_slug(csv_path.stem)

        logger.info(f"Running all models on {data_dir} and saving plots to {output_root}")

        for (model_tag, estimator), grid in zip(models, grids):

            out_dir = output_root / file_name / model_tag
            logger.info(f"Running {model_tag} on {file_name} and saving plots to {out_dir}")
            out_dir.mkdir(parents=True, exist_ok=True)

            train_classificators(
                file_path=str(csv_path),
                regressor=estimator,
                param_grid=grid,
                model_name=model_tag,
                calculate_features=calculate_features,
                data_name=file_name,
                plot_path=str(out_dir),
                split=split,
                seed=seed,
                n_jobs=1 if model_tag in _SERIAL_GRID_MODELS else n_jobs,
            )


def run_regression(
        data_dir: str | Path = _REPO_ROOT / "data" / "regression_data",
        output_root: str | Path = _REPO_ROOT / "ml_plots",
        models: Iterable[tuple[str, Any]] = None,  # e.g. regressor_list
        grids: Iterable[dict] = None,  # e.g. param_grids
        calculate_features: bool = True,
        seed: int = 42,
        organism: str | None = None,
        n_jobs: int = -1,
):
    """Train all regression models on each dataset in data_dir."""

    # Prevent silent truncation if lengths differ
    models = list(models)
    grids = list(grids)

    output_root = Path(output_root)

    results = []

    data_dir = Path(data_dir)
    split = "cluster" if "cluster" in data_dir.name else "random"
    gram_mode = datasets.task_for_dir(data_dir) == "gram"
    csv_files = datasets.split_basepaths_in(data_dir)
    if organism:
        csv_files = [p for p in csv_files if organism.lower() in p.stem.lower()]
        if not csv_files:
            raise ValueError(f"no base file in {data_dir} matches organism {organism!r}")

    for csv_path in csv_files:
        file_name = datasets.organism_slug(csv_path.stem)
        logger.info(f"Running {file_name}")

        logger.info(f"Running all models on {data_dir} and saving plots to {output_root}")

        for (model_tag, estimator), grid in zip(models, grids):
            out_dir = output_root / file_name / model_tag
            out_dir.mkdir(parents=True, exist_ok=True)

            train_r2, train_mse, val_r2, val_mse = train_regressors(
                file_path=str(csv_path),
                regressor=estimator,
                param_grid=grid,
                model_name=model_tag,
                calculate_features=calculate_features,
                plot_path=str(out_dir),
                file_name=file_name,
                gram_mode=gram_mode,
                split=split,
                seed=seed,
                n_jobs=1 if model_tag in _SERIAL_GRID_MODELS else n_jobs,
            )
            results.append({
                "name": file_name,
                "model_tag": model_tag,
                "r2_train": train_r2,
                "r2_val": val_r2,
                "mse_train": train_mse,
                "mse_val": val_mse,
            })

    ml_utils.print_results_tabular(results, logger=logger)


if __name__ == "__main__":

    data_dir = _REPO_ROOT / "data" / "hemo_train"

    classifier_list = [
        ('dummy', DummyClassifier(strategy='prior')),
        ('logreg', LogisticRegression(max_iter=1000)),
        ('xtra', ExtraTreesClassifier()),
        ('xgb', XGBClassifier()),
        ('svc', SVC(probability=True)),
    ]
    classifier_grids = [
        config.dummy_param_grid,
        config.logreg_param_grid,
        config.xtra_cls_param_grid,
        config.xgb_cls_param_grid,
        config.svc_cls_param_grid,
    ]
    run_classification(calculate_features=False,
                       data_dir=data_dir,
                       models=classifier_list,
                       grids=classifier_grids)

    regressor_list = [
        ('dummy', DummyRegressor(strategy='mean')),
        ('ridge', Ridge()),
        ('xtra', ExtraTreesRegressor(n_jobs=1)),
        ('xgb', XGBRegressor(n_jobs=1, tree_method="hist")),
        ('svr', SVR())
    ]
    regressor_grids = [
        config.dummy_param_grid,
        config.ridge_param_grid,
        config.xtra_param_grid,
        config.xgb_param_grid,
        config.svr_param_grid
    ]
    run_regression(calculate_features=False,
                   data_dir=data_dir,
                   models=regressor_list,
                   grids=regressor_grids)
