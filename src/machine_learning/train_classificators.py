from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import sys
import logging
from pathlib import Path

try:
    from src.machine_learning import ml_utils, config
except ImportError:
    import ml_utils  # type: ignore
    import config    # type: ignore

_REPO_ROOT = Path(__file__).resolve().parents[2]
from typing import Iterable, Any
import matplotlib as mpl

mpl.rcParams.update({
    'font.size': 15,        # Basisgröße für alles,
    'axes.titlesize': 16,       # Titel der Achsen
    'axes.labelsize': 15,       # Achsenbeschriftungen (xlabel, ylabel)
    'xtick.labelsize': 14,      # Tick-Beschriftungen X-Achse
    'ytick.labelsize': 14,      # Tick-Beschriftungen Y-Achse
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


def train_classificators(file_path: str,
                         regressor,
                         param_grid,
                         plot_path: str,
                         model_name: str,
                         data_name: str,
                         calculate_features: bool = True
                         ):
    train_df, test_df, target_col = ml_utils.prepare_df(file_path, 'hemo')

    x_train, x_val, y_train, y_val, _ = ml_utils.prepare_train_val_data(calculate_features,
                                                                        train_df[['sequence', target_col]],
                                                                        test_df[['sequence', target_col]],
                                                                        target_col,
                                                                        plot_path)

    logger.info(f"Train shape: {len(x_train)}\tTest shape: {len(x_val)}"
                f"\nTrain target shape: {len(y_train)}\tTest target shape: {len(y_val)}\n"
                f"For File {file_path}")

    best_estimator, grid_search, model = ml_utils.grid_search_setup(regressor, plot_path + '/', model_name, param_grid,
                                                                    x_train,
                                                                    y_train,
                                                                    'hemo')

    test_score = best_estimator.score(x_val, y_val)
    logger.info(f"Test score of the best model: {test_score}")
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
                                 x_data=x_val,
                                 y_true=y_val,
                                 tag='Test',
                                 logger=logger,
                                 data_name=data_name)


def run_all(
        data_dir: str | Path = _REPO_ROOT / "data" / "hemo_train",
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
    data_dir = Path(data_dir)
    output_root = Path(output_root)
    if 'hemo' in data_dir.name:
        csv_files = sorted(list(data_dir.glob("*unvoted.csv")) + list(data_dir.glob("*data.csv")))
    else:
        csv_files = sorted(data_dir.glob("*dataset.csv"))

    for csv_path in csv_files:
        # Safer way to derive a short slug from filename, OS-independent
        parts = csv_path.stem.split("_")
        file_name = "_".join(parts[:2])  # if len(parts) >= 2 else csv_path.stem


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
                plot_path=str(out_dir)
            )


if __name__ == "__main__":
    data_dir = _REPO_ROOT / "data" / "gram"
    model_list = [  # ('gb',   GradientBoostingClassifier()),
        ('xtra', ExtraTreesClassifier()),
        ('xgb', XGBClassifier()),
        ('rf', RandomForestClassifier()),
        ('svc', SVC(probability=True)),

    ]

    param_grids = [  # config.gb_param_grid,
        config.xtra_gram_param_grid,
        config.xgb_cls_param_grid,
        config.rf_cls_param_grid,
        config.svc_cls_param_grid
    ]

    run_all(calculate_features=False,
            data_dir=data_dir,
            models=model_list,
            grids=param_grids)
