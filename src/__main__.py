import argparse
import os
from pathlib import Path

from src.bert_model.fine_tune_protBERT import fine_tune
from src.data_preprocessing.data_splitter import data_splitter

_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent



def main():
    args, parser = parse_inputs()

    # Check if bert_model flags are proper set
    if args.config_path and args.pipe_configs:
        parser.error("Please provide either --config_path or --pipe_configs, not both.")
    if not args.config_path and not args.pipe_configs:
        parser.error("Please provide either --config_path or --pipe_configs.")

    if args.command == 'bert_model':
        # If we have configs in our config dir we can just pass the config name.
        if args.pipe_configs:
            # If no path provides use a default path
            path = str(_SRC_DIR / 'bert_model' / 'peptideBERT_configs' / 'config_pipe_dir') if args.pipe_configs == 'default' else args.pipe_configs

            for file in os.listdir(path):
                if file.endswith('.yaml'):
                    tmp_config = os.path.join(path, file)
                    fine_tune(config_path=tmp_config)

        if args.config_path:
            if '/' not in args.config_path:
                args.config_path = str(_SRC_DIR / 'bert_model' / 'peptideBERT_configs' / args.config_path)
            # Main function Wrapper for the Training Pipeline. Any additional settings are done via the config.yaml inside peptideBERT_configs directory
            fine_tune(
                config_path=args.config_path
            )


    elif args.command == 'data_init':
        data_splitter(task=args.task, data_dir=args.data_dir if args.data_dir else None)

    elif args.command == 'ml_classify':
        from src.machine_learning.train_classificators import run_all
        from src.machine_learning import config
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        from sklearn.svm import SVC
        from xgboost import XGBClassifier
        model_list = [
            ('xtra', ExtraTreesClassifier()),
            ('xgb', XGBClassifier()),
            ('rf', RandomForestClassifier()),
            ('svc', SVC(probability=True)),
        ]
        param_grids = [
            config.xtra_gram_param_grid,
            config.xgb_cls_param_grid,
            config.rf_cls_param_grid,
            config.svc_cls_param_grid,
        ]
        run_all(
            data_dir=args.data_dir or (_REPO_ROOT / "data" / "hemo_train"),
            output_root=args.output_dir or (_REPO_ROOT / "final_plots"),
            models=model_list,
            grids=param_grids,
            calculate_features=not args.no_features,
        )

    elif args.command == 'ml_regress':
        from src.machine_learning.train_regressors import run_all
        from src.machine_learning import config
        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
        from sklearn.svm import SVR
        from xgboost import XGBRegressor
        model_list = [
            ('xtra', ExtraTreesRegressor(n_jobs=1)),
            ('xgb', XGBRegressor(n_jobs=1, tree_method="hist")),
            ('rf', RandomForestRegressor(n_jobs=1)),
            ('svr', SVR()),
        ]
        param_grids = [
            config.xtra_gram,
            config.xgb_param_grid,
            config.rf_param_grid,
            config.svr_param_grid,
        ]
        run_all(
            data_dir=args.data_dir or (_REPO_ROOT / "data" / "regression_data"),
            output_root=args.output_dir or (_REPO_ROOT / "final_plots"),
            models=model_list,
            grids=param_grids,
            calculate_features=not args.no_features,
        )

    elif args.command == 'generate_bert_model_config':
        # Call your config generation function here
        from src.bert_model.fine_tune_utils import generate_custom_yaml_file
        generate_custom_yaml_file(config_name='',
                                  file_path='')
    else:
        parser.print_help()


def parse_inputs():
    parser = argparse.ArgumentParser(description='Peptide Transformers CLI')
    subparsers = parser.add_subparsers(dest='command')
    # Subparser for bert_model
    fine_tune_parser = subparsers.add_parser('bert_model', help='Fine-tune the model')
    fine_tune_parser.add_argument('--config_path', type=str, required=False, help='Path to the config file')
    fine_tune_parser.add_argument('--pipe_configs', type=str, required=False, help='Path to the Directory containing config files. Will use every config inside this dir.')
    # Subparser for data_preprocess
    data_preprocess_parser = subparsers.add_parser('data_init', help='Preprocess the data')
    data_preprocess_parser.add_argument('--task', type=str, required=True, help='task of the train data ["classification", "regression", "gram"]')
    data_preprocess_parser.add_argument('--data_dir', type=str, default=None, help='Path to data directory (uses repo default if omitted)')
    # Subparser for ml_classify
    ml_classify_parser = subparsers.add_parser('ml_classify', help='Train classical ML classification models')
    ml_classify_parser.add_argument('--data_dir', type=str, default=None, help='Path to data directory')
    ml_classify_parser.add_argument('--output_dir', type=str, default=None, help='Path to output directory')
    ml_classify_parser.add_argument('--no_features', action='store_true', help='Disable biochemical feature engineering')
    # Subparser for ml_regress
    ml_regress_parser = subparsers.add_parser('ml_regress', help='Train classical ML regression models')
    ml_regress_parser.add_argument('--data_dir', type=str, default=None, help='Path to data directory')
    ml_regress_parser.add_argument('--output_dir', type=str, default=None, help='Path to output directory')
    ml_regress_parser.add_argument('--no_features', action='store_true', help='Disable biochemical feature engineering')
    # Subparser for reproduce
    reproduce_parser = subparsers.add_parser('reproduce', help='Reproduce our results')
    # Add arguments for reproduce here
    # Subparser for autogernerating a config file
    generate_config_parser = subparsers.add_parser('generate_bert_model_config',
                                                   help='Generate a default set bert config yaml file')
    generate_config_parser.add_argument('--config_name', type=str, required=True, help='Name for the config file')
    generate_config_parser.add_argument('--file_path', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    return args, parser


if __name__ == '__main__':
    main()
