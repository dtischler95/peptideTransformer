import argparse
import os
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent


def _resolve_config_shortname(name: str) -> str:
    """Resolve a bare config filename to a full path, searching subfolders of
    configs so configs organised into random_split/ and cluster_split/
    still work as `--config_path name.yaml`."""
    cfg_root = _SRC_DIR / 'modeling' / 'configs'
    flat = cfg_root / name
    if flat.exists():
        return str(flat)
    matches = list(cfg_root.rglob(name))
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) > 1:
        opts = ', '.join(str(m.relative_to(cfg_root)) for m in matches)
        raise SystemExit(f"Config name '{name}' is ambiguous ({opts}). "
                         f"Pass a full path or use --pipe_configs with the folder.")
    return str(flat)  # not found: let the downstream open() raise a clear error


# --------------------------------------------------------------------------- #
# Command handlers. One per subcommand; each does its own lazy imports so that
# invoking a lightweight command (e.g. data_init) never pulls in torch or sklearn.
# parse_inputs() wires each to its subparser via set_defaults(func=...).
# --------------------------------------------------------------------------- #

def cmd_bert_model(args):
    from src.modeling.fine_tune import fine_tune
    # Optional backbone overrides let the existing ProtBERT configs drive an ESM run
    # unchanged (same hyperparameters/train_file, different encoder).
    overrides = {k: v for k, v in (('backbone', args.backbone),
                                   ('model_path', args.model_path),
                                   ('model_name', args.model_name)) if v is not None}
    # Exactly one of config_path / pipe_configs is guaranteed by the mutually
    # exclusive group in parse_inputs().
    if args.pipe_configs:
        path = (str(_SRC_DIR / 'modeling' / 'configs' / 'config_pipe_dir')
                if args.pipe_configs == 'default' else args.pipe_configs)
        for file in os.listdir(path):
            if file.endswith('.yaml'):
                fine_tune(config_path=os.path.join(path, file), overrides=overrides)
    else:
        config_path = args.config_path
        if '/' not in config_path:
            config_path = _resolve_config_shortname(config_path)
        fine_tune(config_path=config_path, overrides=overrides)


def cmd_data_init(args):
    from src.data_preprocessing.data_splitter import data_splitter
    data_splitter(task=args.task,
                  data_dir=args.data_dir or None,
                  out_dir=args.out_dir or None,
                  random_state=args.seed)


def cmd_cluster_init(args):
    from src.data_preprocessing.cluster_data_splitter import cluster_data_splitter
    cluster_data_splitter(task=args.task,
                          data_dir=args.data_dir or None,
                          out_dir=args.out_dir or None,
                          min_seq_id=args.min_seq_id,
                          coverage=args.coverage,
                          kmer=args.kmer,
                          random_state=args.seed)


def cmd_ml_classify(args):
    from src.machine_learning.train_models import run_classification, default_classifiers
    models, grids = default_classifiers()
    run_classification(data_dir=args.data_dir or (_REPO_ROOT / "data" / "hemo_train"),
                       output_root=args.output_dir or (_REPO_ROOT / "ml_plots"),
                       models=models, grids=grids,
                       calculate_features=args.features, seed=args.seed, organism=args.organism)


def cmd_ml_regress(args):
    from src.machine_learning.train_models import run_regression, default_regressors
    models, grids = default_regressors()
    run_regression(data_dir=args.data_dir or (_REPO_ROOT / "data" / "regression_data"),
                   output_root=args.output_dir or (_REPO_ROOT / "ml_plots"),
                   models=models, grids=grids,
                   calculate_features=args.features, seed=args.seed, organism=args.organism)


def cmd_generate_config(args):
    from src.modeling.fine_tune_utils import generate_custom_yaml_file
    generate_custom_yaml_file(config_name=args.config_name,
                              file_path=args.file_path,
                              model_class=args.model_class)


def main():
    args, parser = parse_inputs()
    handler = getattr(args, 'func', None)
    if handler is None:
        parser.print_help()
        return
    handler(args)


def parse_inputs():
    parser = argparse.ArgumentParser(description='Peptide Transformers CLI')
    subparsers = parser.add_subparsers(dest='command')

    # bert_model: fine-tune a transformer (ProtBERT or ESM-2)
    fine_tune_parser = subparsers.add_parser('bert_model', help='Fine-tune the model')
    # Exactly one config source is required; argparse enforces the XOR.
    config_source = fine_tune_parser.add_mutually_exclusive_group(required=True)
    config_source.add_argument('--config_path', type=str, help='Path to the config file')
    config_source.add_argument('--pipe_configs', type=str,
                               help='Directory of config files. Runs every yaml inside. Use "default" for the built-in config_pipe_dir.')
    fine_tune_parser.add_argument('--backbone', type=str, choices=['bert', 'esm'], default=None,
                                  help='Override the encoder backbone, reusing the config with a different model family (e.g. esm)')
    fine_tune_parser.add_argument('--model_path', type=str, default=None,
                                  help='Override model_path (HF id or local path), e.g. facebook/esm2_t33_650M_UR50D')
    fine_tune_parser.add_argument('--model_name', type=str, default=None,
                                  help='Override model_name: grouping key in metrics.json and output-path suffix, e.g. esm650m')
    fine_tune_parser.set_defaults(func=cmd_bert_model)

    # data_init: random stratified splits
    data_init_parser = subparsers.add_parser('data_init', help='Preprocess the data')
    data_init_parser.add_argument('--task', type=str, required=True, help='task of the train data ["cls", "regression", "gram"]')
    data_init_parser.add_argument('--data_dir', type=str, default=None, help='Path to data directory (uses repo default if omitted)')
    data_init_parser.add_argument('--out_dir', type=str, default=None, help='Write splits here instead of next to the base files')
    data_init_parser.add_argument('--seed', type=int, default=42, help='Random seed for the split (default 42)')
    data_init_parser.set_defaults(func=cmd_data_init)

    # cluster_init: similarity-aware (MMseqs2 cluster) splits
    cluster_init_parser = subparsers.add_parser('cluster_init', help='Create similarity-aware (MMseqs2 cluster) train/val/test splits')
    cluster_init_parser.add_argument('--task', type=str, required=True, help='task of the train data ["cls", "regression", "gram"]')
    cluster_init_parser.add_argument('--data_dir', type=str, default=None, help='Input data directory (repo default if omitted)')
    cluster_init_parser.add_argument('--out_dir', type=str, default=None, help='Write cluster splits here instead of <dir>_cluster')
    cluster_init_parser.add_argument('--min_seq_id', type=float, default=0.5, help='MMseqs2 minimum sequence identity for clustering (default 0.5)')
    cluster_init_parser.add_argument('--coverage', type=float, default=0.8, help='MMseqs2 coverage threshold -c (default 0.8)')
    cluster_init_parser.add_argument('--kmer', type=int, default=5, help='MMseqs2 k-mer size -k (default 5)')
    cluster_init_parser.add_argument('--seed', type=int, default=42, help='Random seed for the cluster-to-split assignment (default 42)')
    cluster_init_parser.set_defaults(func=cmd_cluster_init)

    # ml_classify: classical ML classifiers
    ml_classify_parser = subparsers.add_parser('ml_classify', help='Train classical ML classification models')
    ml_classify_parser.add_argument('--data_dir', type=str, default=None, help='Path to data directory')
    ml_classify_parser.add_argument('--output_dir', type=str, default=None, help='Path to output directory')
    ml_classify_parser.add_argument('--features', action='store_true', help='Enable biochemical feature engineering')
    ml_classify_parser.add_argument('--seed', type=int, default=42, help='Seed for CV folds (default 42)')
    ml_classify_parser.add_argument('--organism', type=str, default=None, help='Run only base files matching this substring')
    ml_classify_parser.set_defaults(func=cmd_ml_classify)

    # ml_regress: classical ML regressors
    ml_regress_parser = subparsers.add_parser('ml_regress', help='Train classical ML regression models')
    ml_regress_parser.add_argument('--data_dir', type=str, default=None, help='Path to data directory')
    ml_regress_parser.add_argument('--output_dir', type=str, default=None, help='Path to output directory')
    ml_regress_parser.add_argument('--features', action='store_true', help='Enable biochemical feature engineering')
    ml_regress_parser.add_argument('--seed', type=int, default=42, help='Seed for CV folds (default 42)')
    ml_regress_parser.add_argument('--organism', type=str, default=None, help='Run only base files matching this substring')
    ml_regress_parser.set_defaults(func=cmd_ml_regress)

    # generate_bert_model_config: scaffold a default config yaml
    generate_config_parser = subparsers.add_parser('generate_bert_model_config',
                                                   help='Generate a default set bert config yaml file')
    generate_config_parser.add_argument('--config_name', type=str, required=True, help='Name for the config file')
    generate_config_parser.add_argument('--file_path', type=str, required=True, help='Path to the config file')
    generate_config_parser.add_argument('--model_class', type=str, required=True, help='Task for model init')
    generate_config_parser.set_defaults(func=cmd_generate_config)

    args = parser.parse_args()
    return args, parser
