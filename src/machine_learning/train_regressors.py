# from sklearnex import patch_sklearn
#
# patch_sklearn(verbose=False)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

import ml_utils
import pandas as pd
import config

from pathlib import Path
from typing import Iterable, Any


regressor_list = [('gb', GradientBoostingRegressor()),
                  ('xtra', ExtraTreesRegressor()),
                  ('xgb', XGBRegressor()),
                  ('rf', RandomForestRegressor()),
                  ('svr', SVR())
                  ]

param_grids = [config.gb_param_grid,
               config.xtra_param_grid,
               config.xgb_param_grid,
               config.rf_param_grid,
               config.svr_param_grid
               ]


def train_regressors(file_path: str,
                     regressor,
                     param_grid,
                     plot_path: str,
                     calculate_features: bool = True
                     ):
    df = pd.read_csv(file_path, sep=';')
    df = df.drop(columns='value')


    if calculate_features:
        x_train, y_train, x_val, y_val = ml_utils.encode_onehot_with_features(df, target_col="mic_log10")
    else:
        df_to_split = df['sequence'].unique()

        train, test = train_test_split(df_to_split,
                                       train_size=0.8,
                                       test_size=0.2,
                                       shuffle=True,
                                       random_state=42)

        train_df = df[df['sequence'].isin(train)]
        val_df = df[df['sequence'].isin(test)]

        x_train, y_train = ml_utils.enocde_onehot_without_features(train_df, pad_len=36)
        x_val, y_val = ml_utils.enocde_onehot_without_features(val_df, pad_len=36)

    print(f"Train shape: {len(x_train)}\tTest shape: {len(x_val)}"
          f"\nTrain target shape: {len(y_train)}\tTest target shape: {len(y_val)}")

    best_estimator, grid_search, model = ml_utils.grid_search_setup(regressor, './', 'test', param_grid,
                                                                    x_train,
                                                                    y_train)

    test_score = best_estimator.score(x_val, y_val)
    print(f"Test score of the best model: {test_score}")
    print(f"Best Params: {grid_search.best_params_}")

    train_r2, train_mse = ml_utils.get_model_stats(model=best_estimator,
                             plot_dir=plot_path,
                             feature_data=x_train,
                             target_data=y_train,
                             tag="Train")

    val_r2, val_mse = ml_utils.get_model_stats(model=best_estimator,
                             plot_dir=plot_path,
                             feature_data=x_val,
                             target_data=y_val,
                             tag="Test")

    ml_utils.overall_stats(best_estimator=best_estimator, x_test=x_val, y_test=y_val, save_path=plot_path)

    return train_r2, train_mse, val_r2, val_mse



def run_all(
    data_dir: str | Path = "../../data/regression_data",
    output_root: str | Path = "../../final_plots/",
    models: Iterable[tuple[str, Any]] = None,        # e.g. regressor_list
    grids:  Iterable[dict] = None,                    # e.g. param_grids
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


    csv_files = sorted(data_dir.glob("*.csv"))
    for csv_path in csv_files:
        if csv_path.stem.startswith("candida"):
            break
        # Safer way to derive a short slug from filename, OS-independent
        parts = csv_path.stem.split("_")
        file_name = "_".join(parts[:2])# if len(parts) >= 2 else csv_path.stem

        for (model_tag, estimator), grid in zip(models, grids):
            out_dir = output_root / model_tag / file_name
            out_dir.mkdir(parents=True, exist_ok=True)

            # Pass the estimator/grid down so train_regressors is self-contained
            train_r2, train_mse, val_r2, val_mse = train_regressors(
                file_path=str(csv_path),
                regressor=estimator,
                param_grid=grid,
                calculate_features=calculate_features,
                plot_path=str(out_dir)
            )
            results.append({
                "name": file_name,
                "r2_train": train_r2,
                "r2_val": val_r2,
                "mse_train": train_mse,
                "mse_val": val_mse,
            })



    ml_utils.print_results_tabular(results)





if __name__ == "__main__":
    # train_regressors('../../data/regression_data/acinetobacter_baumannii_for_regression.csv',
    #                  calculate_features=False)

    run_all(calculate_features=False,
            models=regressor_list,
            grids=param_grids)
