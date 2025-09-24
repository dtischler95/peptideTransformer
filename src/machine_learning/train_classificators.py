from sklearnex import patch_sklearn
from xgboost import XGBClassifier
from sklearn.svm import SVC
patch_sklearn(verbose=False)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import sys
import ml_utils
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


def train_classificators(file_path: str,
                         regressor,
                         param_grid,
                         plot_path: str,
                         model_name: str,
                         feature_selection,
                         calculate_features: bool = True
                         ):
    df, target_col = ml_utils.prepare_df(file_path, 'hemo')

    x_train, x_val, y_train, y_val, _ = ml_utils.prepare_train_val_data(calculate_features,
                                                                        df,
                                                                        target_col,
                                                                        plot_path,
                                                                        feature_selection)

    logger.info(f"Train shape: {len(x_train)}\tTest shape: {len(x_val)}"
                f"\nTrain target shape: {len(y_train)}\tTest target shape: {len(y_val)}\n"
                f"For File {file_path}")

    # TODO SAFE EVAL DATA TO FILE!!
    best_estimator, grid_search, model = ml_utils.grid_search_setup(regressor, plot_path + '/', model_name, param_grid,
                                                                    x_train,
                                                                    y_train,
                                                                    'hemo')

    test_score = best_estimator.score(x_val, y_val)
    logger.info(f"Test score of the best model: {test_score}")
    logger.info(f"Best Params: {grid_search.best_params_}")

    ml_utils.evaluate_hemo_model(best_estimator, model_name, plot_path, x_val, y_val, logger)


def run_all(
        data_dir: str | Path = "../../data/regression_data",
        output_root: str | Path = "./final_plots/",
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

    csv_files = sorted(data_dir.glob("*.csv"))
    for csv_path in csv_files:
        # Safer way to derive a short slug from filename, OS-independent
        parts = csv_path.stem.split("_")
        file_name = "_".join(parts[:2])  # if len(parts) >= 2 else csv_path.stem
        if calculate_features:
            tmp_file_path = output_root / file_name
            tmp_file_path.mkdir(parents=True, exist_ok=True)
            features = ml_utils.get_feature_importance(file_path=str(csv_path),
                                                       task='hemo',
                                                       plot_path=str(tmp_file_path))
            logger.info(f"Anzahl verwenderter Features für {str(tmp_file_path)} ist : {len(features)}")
        else:
            features = None

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
                feature_selection=features,
                plot_path=str(out_dir)
            )


if __name__ == "__main__":
    data_dir = '../../data/hemo_train/'
    model_list = [  # ('gb',   GradientBoostingClassifier()),
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

    run_all(calculate_features=True,
            data_dir=data_dir,
            models=model_list,
            grids=param_grids)
