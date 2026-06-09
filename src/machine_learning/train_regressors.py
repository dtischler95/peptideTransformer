from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
import sys
import logging
import matplotlib as mpl

from pathlib import Path

try:
    from src.machine_learning import ml_utils, config
except ImportError:
    import ml_utils  # type: ignore
    import config    # type: ignore

_REPO_ROOT = Path(__file__).resolve().parents[2]
from typing import Iterable, Any

logger = logging.getLogger(__name__)
mpl.rcParams.update({
    'font.size': 16,  # Basisgröße für alles,
    'axes.titlesize': 18,  # Titel der Achsen
    'axes.labelsize': 15,  # Achsenbeschriftungen (xlabel, ylabel)
    'xtick.labelsize': 14,  # Tick-Beschriftungen X-Achse
    'ytick.labelsize': 14,  # Tick-Beschriftungen Y-Achse
})

# --------------------- Setup logging and configs ---------------------
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


def train_regressors(file_path: str,
                     regressor,
                     param_grid,
                     plot_path: str,
                     model_name: str,
                     file_name: str,
                     calculate_features: bool = True,
                     gram_mode: bool = False
                     ):
    train_df, test_df, target_col = ml_utils.prepare_df(file_path, 'mic')

    x_train, x_val, y_train, y_val, _ = ml_utils.prepare_train_val_data(calculate_features,
                                                                        train_df[['sequence', target_col]],
                                                                        test_df[['sequence', target_col]],
                                                                        target_col,
                                                                        plot_path)

    logger.info(f"Train shape: {len(x_train)}\tTest shape: {len(x_val)}"
                f"\nTrain target shape: {len(y_train)}\tTest target shape: {len(y_val)}")

    logger.info(f"File Name: {file_path} and Model: {model_name}")

    best_estimator, grid_search, model = ml_utils.grid_search_setup(regressor, plot_path + '/', model_name, param_grid,
                                                                    x_train,
                                                                    y_train,
                                                                    'mic')

    test_score = best_estimator.score(x_val, y_val)
    logger.info(f"Test score of the best model: {test_score}")
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
                                                                        x_val,
                                                                        y_train,
                                                                        y_val,
                                                                        logger,
                                                                        file_name,
                                                                        model_name)

    return train_r2, train_mse, val_r2, val_mse


def run_all(
        data_dir: str | Path = _REPO_ROOT / "data" / "regression_data",
        output_root: str | Path = _REPO_ROOT / "final_plots",
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

    output_root = Path(output_root)

    results = []

    data_dir = Path(data_dir)
    gram_mode = False
    if 'regression' in data_dir.name:
        csv_files = sorted(data_dir.glob("*regression.csv"))
    else:
        csv_files = sorted(data_dir.glob("*dataset.csv"))
        gram_mode = True

    for csv_path in csv_files:
        # Safer way to derive a short slug from filename, OS-independent
        parts = csv_path.stem.split("_")
        file_name = "_".join(parts[:2])  # if len(parts) >= 2 else csv_path.stem
        logger.info(f"Running {file_name}")

        logger.info(f"Running all models on {data_dir} and saving plots to {output_root}")

        for (model_tag, estimator), grid in zip(models, grids):
            out_dir = output_root / file_name / model_tag
            out_dir.mkdir(parents=True, exist_ok=True)

            # Pass the estimator/grid down so train_regressors is self-contained
            train_r2, train_mse, val_r2, val_mse = train_regressors(
                file_path=str(csv_path),
                regressor=estimator,
                param_grid=grid,
                model_name=model_tag,
                calculate_features=calculate_features,
                plot_path=str(out_dir),
                file_name=file_name,
                gram_mode=gram_mode
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
    data_dir = _REPO_ROOT / "data" / "regression"
    model_list = [  # ('gb', GradientBoostingRegressor()),
        #('xtra', ExtraTreesRegressor(n_jobs=1)),
        #('xgb', XGBRegressor(n_jobs=1, tree_method="hist")),
        ('rf', RandomForestRegressor(n_jobs=1)),
        #('svr', SVR(n_jobs=1))
    ]

    param_grids = [  # config.gb_param_grid,
        #config.xtra_gram,
        #config.xgb_param_grid,
        config.rf_test_param_grid,
        #config.svr_param_grid
    ]

    run_all(calculate_features=False,
            data_dir=data_dir,
            models=model_list,
            grids=param_grids)
