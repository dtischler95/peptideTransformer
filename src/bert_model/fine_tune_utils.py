import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import peptides as pep
import yaml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from transformers import DefaultDataCollator, BertConfig
from transformers import BertTokenizer

from src.bert_model.PeptideBERTClasses.PeptideBertForBinaryClassification import PeptideBertForBinaryClassification
from src.bert_model.PeptideBERTClasses.PeptideBertForRegression import PeptideBertForRegression
from src.bert_model.PeptideBERTClasses.PeptideDataset import PeptideDataset
from src.bert_model.PeptideBERTClasses.PeptideTrainingArguments import PeptideTrainingArguments
from src.bert_model.transformer_metrics import binary_metrics, regression_metrics
from src.evaluation.eval_utils import overall_stats, get_model_stats, evaluate_hemo

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FILE_PATH_KEYS = frozenset({'train_file', 'model_save_path', 'plot_path', 'output_dir', 'logging_dir'})


def _compute_desc_row(sequence: str) -> dict:
    p = pep.Peptide(sequence)
    return {f"desc__{k}": float(v) for k, v in p.descriptors().items()}


def _add_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    desc_df = pd.DataFrame([_compute_desc_row(s) for s in df["sequence"]]).reset_index(drop=True)
    df = df.reset_index(drop=True).join(desc_df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def _load_split_files(base_file, debug_sample: bool = False) -> tuple:
    """Loads pre-split train/val/test CSVs derived from a base file path."""
    base = Path(base_file).with_suffix('')
    train_df = pd.read_csv(f"{base}_train.csv", sep=';')
    val_df = pd.read_csv(f"{base}_val.csv", sep=';')
    test_df = pd.read_csv(f"{base}_test.csv", sep=';')
    if debug_sample:
        train_df = train_df.sample(frac=1)[:100]
    return train_df, val_df, test_df


def _get_labels_and_features(data: pd.DataFrame, model_class: str, use_features: bool):
    if model_class.startswith('regression'):
        target_column = 'mic_log10'
        drop_cols = ['sequence', target_column]
    else:
        target_column = 'label'
        drop_cols = ['sequence', target_column]
        if 'hemo_percent' in data.columns:
            drop_cols += ['hemo_percent', 'hemo_concentration']
    labels = data[target_column].values
    features = data.drop(columns=drop_cols) if use_features else None
    return labels, features


def prepare_datasets(model_class: str,
                     model_path: str,
                     train_file: str,
                     logger: logging.Logger,
                     show_encoding: bool = False,
                     ignore_leakage: bool = False,
                     max_length: int = 36,
                     cut_df_for_faster_debug: bool = False,
                     add_features: bool = False
                     ) -> tuple[BertTokenizer, PeptideDataset, PeptideDataset, PeptideDataset, int]:
    """
    Loads pre-split train/val/test CSVs and builds PeptideDatasets for the given model class.

    :param model_class: One of 'binary_dense' or 'regression'
    :param model_path: HuggingFace repo or local path to the pretrained model
    :param train_file: Path to the base CSV (expects _train/_val/_test variants alongside it)
    :param logger: Logger instance
    :param show_encoding: Print the tokenizer vocabulary encoding
    :param ignore_leakage: Suppress the ValueError on data leakage (keeps a warning)
    :param max_length: Padding/truncation length for the tokenizer
    :param cut_df_for_faster_debug: Slice train set to 100 rows for quick iteration
    :param add_features: Append peptide descriptors as additional input features
    :return: (tokenizer, train_dataset, val_dataset, test_dataset, n_extra_features)
    """
    df_train, df_val, df_test = _load_split_files(train_file, debug_sample=cut_df_for_faster_debug)

    if add_features:
        df_train = _add_descriptors(df_train)
        df_val = _add_descriptors(df_val)
        df_test = _add_descriptors(df_test)

    desc_cols = [c for c in df_train.columns if c.startswith('desc__')]
    feature_size = len(desc_cols)

    label_train, feat_train = _get_labels_and_features(df_train, model_class, add_features)
    label_val, feat_val = _get_labels_and_features(df_val, model_class, add_features)
    label_test, feat_test = _get_labels_and_features(df_test, model_class, add_features)

    tokenizer = BertTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=True, do_lower_case=False)

    if add_features:
        imp = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        feat_train = scaler.fit_transform(imp.fit_transform(feat_train.values)).astype(np.float32)
        feat_val = scaler.transform(imp.transform(feat_val.values)).astype(np.float32)
        feat_test = scaler.transform(imp.transform(feat_test.values)).astype(np.float32)

    def _make_dataset(df, labels, features):
        return PeptideDataset(peptides=df['sequence'], features=features, tokenizer=tokenizer,
                              labels=labels, max_length=max_length, model_class=model_class)

    train_dataset = _make_dataset(df_train, label_train, feat_train)
    val_dataset = _make_dataset(df_val, label_val, feat_val)
    test_dataset = _make_dataset(df_test, label_test, feat_test)

    if show_encoding:
        get_encoding(tokenizer)

    check_data_loader_for_leakage(train_data_loader=train_dataset, val_data_loader=val_dataset,
                                  test_data_loader=test_dataset, ignore_leakage=ignore_leakage, logger=logger)

    return tokenizer, train_dataset, val_dataset, test_dataset, feature_size


def get_encoding(tokenizer: BertTokenizer):
    """
    Prints out the encoding of the vocabulary used by the tokenizer

    :param tokenizer: The tokenizer used for encoding
    :return:
    """

    for i in range(tokenizer.vocab_size):
        character = tokenizer.decode(i)
        print(f"{i}: {character}")


def check_data_loader_for_leakage(train_data_loader: PeptideDataset or None,
                                  val_data_loader: PeptideDataset or None,
                                  test_data_loader: PeptideDataset or None,
                                  logger: logging.Logger,
                                  decoded_input: bool = False,
                                  ignore_leakage: bool = False):
    """
    Function that can be imported in other scripts to check for data leakage between the data loaders.
    Raises a ValueError if data leakage is detected.

    PeptideDataset is a class that inherits from the transformers useful Dataset class.

    Use decoded_input=True if the input is decoded like in whitelab's data case.

    :param train_data_loader: Training data loader
    :param val_data_loader: Validation data loader
    :param test_data_loader: Test data loader
    :param logger: Logger for logging
    :param decoded_input: If the input is decoded or not. Default is False.
    :param ignore_leakage: If the leakage should be ignored. Default is False.
    """
    if decoded_input:

        tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
        train_sequences = [tokenizer.decode(row, skip_special_tokens=True) for row in
                           train_data_loader.input_ids] if train_data_loader is not None else []
        val_sequences = [tokenizer.decode(row, skip_special_tokens=True) for row in
                         val_data_loader.input_ids] if val_data_loader is not None else []
        test_sequences = [tokenizer.decode(row, skip_special_tokens=True) for row in
                          test_data_loader.input_ids] if test_data_loader is not None else []

    else:
        train_sequences = train_data_loader.peptides if train_data_loader is not None else []
        val_sequences = val_data_loader.peptides if val_data_loader is not None else []
        test_sequences = test_data_loader.peptides if test_data_loader is not None else []

    train_val_leakage = set(train_sequences) & set(val_sequences)
    train_test_leakage = set(train_sequences) & set(test_sequences)
    val_test_leakage = set(val_sequences) & set(test_sequences)

    logger.info(f"Number of Training Data Points: {len(train_sequences)}")
    logger.info(f"Number of Validation Data Points: {len(val_sequences)}")
    logger.info(f"Number of Test Data Points: {len(test_sequences)}\n")

    if len(train_test_leakage) or len(train_val_leakage) or len(val_test_leakage) > 0:
        logger.warning("\n-------------- Data Leakage Detected --------------\n")

        if len(train_val_leakage) > 0:
            logger.warning(f"Number of data Points leaked in Training and Validation Data: {len(train_val_leakage)}\n")
        if len(train_test_leakage) > 0:
            logger.warning(f"Number of data Points leaked in Training and Test Data: {len(train_test_leakage)}\n")
        if len(val_test_leakage) > 0:
            logger.warning(f"Number of data Points leaked in Validation and Test Data: {len(val_test_leakage)}\n")

        logger.warning("\n-------------- Data Leakage Detected --------------\n")
        if not ignore_leakage:
            raise ValueError("Data Leakage Detected! Please recheck your data preprocessing steps.")
    else:
        logger.info("\n-------------- No Data Leakage Detected --------------\n")


def _resolve_paths(config: dict) -> None:
    """Resolve relative file paths in the config dict against the repo root in-place.

    This allows configs to be launched from any working directory. Paths in
    _FILE_PATH_KEYS are treated as repo-root-relative if not absolute.
    model_path is left untouched unless it starts with '.' (HuggingFace model
    IDs such as 'Rostlab/prot_bert_bfd' are passed through unchanged).
    """
    for key in _FILE_PATH_KEYS:
        value = config.get(key)
        if value and not Path(str(value)).is_absolute():
            config[key] = str(_REPO_ROOT / value)
    mp = config.get('model_path', '')
    if mp and not Path(mp).is_absolute() and mp.startswith('.'):
        config['model_path'] = str(_REPO_ROOT / mp)


def load_training_arguments(config_file: str, logger: logging.Logger) -> PeptideTrainingArguments:
    """Load training arguments from a YAML config file.

    Relative file paths are resolved against the repository root so the script
    can be launched from any working directory.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    _resolve_paths(config)

    print("Set Parameters for this training run:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    return PeptideTrainingArguments(**config)


def format_logit_to_label(logits):
    """
    Format the given logit tensor to a list of labels.
    0.5 is a typical threshold
    """
    return (logits > 0.5).astype(int).flatten()


def format_one_hot_to_label(one_hot_tensor):
    """
    Format the given one-hot encoded tensor to a list of labels.
    """
    return np.argmax(one_hot_tensor, axis=1).flatten()

# TODO Update this function for all new parameters and formats
_CLS_YAML_TEMPLATE = """\
# Training and Evaluation Settings
model_class: 'binary_dense'                            # 'binary_dense' for hemolysis classification
do_train: true                                         # Train the model
do_eval: true                                          # Evaluate the model
num_train_epochs: 50                                   # Number of epochs to train the model
per_device_train_batch_size: 64                        # Batch size for training
per_device_eval_batch_size: 64                         # Batch size for evaluation
early_stopping_patience: 5                             # Patience for early stopping
early_stop_metric: 'eval_loss'                         # Metric for early stopping
early_stop_mode: 'min'                                 # Mode for early stopping
early_stop_warm_up: 8                                  # Warm-up period for early stopping
dataloader_num_workers: 1                              # Number of dataloader workers (higher can affect performance)
load_best_model_at_end: true                           # Load the best model at the end of training

# Model and Optimizer Settings
learning_rate: 0.00001                                 # Learning rate for optimizer
weight_decay: 0.01                                     # Weight decay for optimizer
lr_scheduler_type: 'reduce_lr_on_plateau'              # Learning rate scheduler type
lr_scheduler_kwargs:                                   # Additional scheduler arguments
  patience: 4                                          # Patience for ReduceLROnPlateau scheduler
max_length: 36                                         # Maximum input sequence length

# Binary Classification Settings
label_0_cluster_data: 500                              # Number of data points for label 0 used in downstream clustering
label_1_cluster_data: 500                              # Number of data points for label 1 used in downstream clustering
loss_function: 'bce_logit_loss'                        # Possible Choices ['bce', 'bce_logit_loss']
add_features: false                                    # Use concentration as extra feature [WARNING] U need a prepared train datafile for this


# Dataset and File Paths
train_file: 'data/PATH_TO_TRAIN_FILE'                  # Repo-root-relative path to training data
model_path: 'Rostlab/prot_bert_bfd'                    # HuggingFace model ID or local path
model_save_path: 'models/MY_RUN_NAME'                  # Where to save the trained model
plot_path: 'model_plots/MY_RUN_NAME'                   # Where to save evaluation plots

# Logging Settings
output_dir: 'results/MY_RUN_NAME'                      # Path to checkpoints
logging_dir: 'logs/MY_RUN_NAME'                        # Path to logging directory
logging_strategy: 'epoch'                              # Logging strategy (set to 'epoch' for this logic)
log_level: 'info'                                      # Log level
eval_strategy: 'epoch'                                 # Evaluation strategy (set to 'epoch' for this logic)
save_strategy: 'epoch'                                 # Save strategy (set to 'epoch' for this logic)

# Hardware Settings
use_cpu: false                                         # Use CPU for training

# Miscellaneous Settings
ignore_leakage: false                                  # Ignore leakage in training data (only if certain)
fast_debug_mode: false                                 # Developer mode for pipeline testing
"""

_REG_YAML_TEMPLATE = """\
# Training and Evaluation Settings
model_class: 'regression'                              # 'regression' for MIC regression
do_train: true                                         # Train the model
do_eval: true                                          # Evaluate the model
num_train_epochs: 50                                   # Number of epochs to train the model
per_device_train_batch_size: 64                        # Batch size for training
per_device_eval_batch_size: 64                         # Batch size for evaluation
early_stopping_patience: 5                             # Patience for early stopping
early_stop_metric: 'eval_loss'                         # Metric for early stopping
early_stop_mode: 'min'                                 # Mode for early stopping
early_stop_warm_up: 8                                  # Warm-up period for early stopping
dataloader_num_workers: 1                              # Number of dataloader workers (higher can affect performance)
load_best_model_at_end: true                           # Load the best model at the end of training
add_features: false                                    # Use concentration as extra feature [WARNING] U need a prepared train datafile for this

# Model and Optimizer Settings
learning_rate: 0.00001                                 # Learning rate for optimizer
weight_decay: 0.01                                     # Weight decay for optimizer
lr_scheduler_type: 'reduce_lr_on_plateau'              # Learning rate scheduler type
lr_scheduler_kwargs:                                   # Additional scheduler arguments
  patience: 4                                          # Patience for ReduceLROnPlateau scheduler
max_length: 36                                         # Maximum input sequence length


# Dataset and File Paths
train_file: 'data/PATH_TO_TRAIN_FILE'                  # Repo-root-relative path to training data
model_path: 'Rostlab/prot_bert_bfd'                    # HuggingFace model ID or local path
model_save_path: 'models/MY_RUN_NAME'                  # Where to save the trained model
plot_path: 'model_plots/MY_RUN_NAME'                   # Where to save evaluation plots

# Logging Settings
output_dir: 'results/MY_RUN_NAME'                      # Path to checkpoints
logging_dir: 'logs/MY_RUN_NAME'                        # Path to logging directory
logging_strategy: 'epoch'                              # Logging strategy (set to 'epoch' for this logic)
log_level: 'info'                                      # Log level
eval_strategy: 'epoch'                                 # Evaluation strategy (set to 'epoch' for this logic)
save_strategy: 'epoch'                                 # Save strategy (set to 'epoch' for this logic)

# Hardware Settings
use_cpu: false                                         # Use CPU for training

# Miscellaneous Settings
ignore_leakage: false                                  # Ignore leakage in training data (only if certain)
fast_debug_mode: false                                 # Developer mode for pipeline testing
"""


def generate_custom_yaml_file(config_name: str,
                              model_class: str = 'binary_dense',
                              file_path: str = './bert_model/peptideBERT_configs/'):
    """
    Generates a YAML config file template for a new training run.
    The template mirrors the current production config structure exactly.

    Parameters:
    - config_name (str): Filename of the YAML to create (e.g. 'my_organism.yaml').
    - model_class (str): Task type — 'binary_dense' for hemolysis classification,
                         'regression' for MIC regression.
    - file_path (str): Directory where the YAML will be saved. Should be the peptideBERT_configs folder.
    """
    if model_class == 'binary_dense':
        yaml_content = _CLS_YAML_TEMPLATE
    elif model_class == 'regression':
        yaml_content = _REG_YAML_TEMPLATE
    else:
        raise ValueError(f"model_class must be 'binary_dense' or 'regression', got: {model_class!r}")

    out_path = Path(file_path) / config_name
    try:
        out_path.write_text(yaml_content, encoding='utf-8')
        print(f"YAML template created at: {out_path}\n"
              f"Set train_file, model_save_path, output_dir, and logging_dir before use.")
    except Exception as e:
        print(f"Error while creating YAML file: {e}")


def prepare_hemo_eval(test_dataset: PeptideDataset,
                      trainer,
                      plot_path: str,
                      tag: str,
                      file_name: str):
    y_true = test_dataset.labels
    logits = trainer.predict(test_dataset).predictions
    y_score = logits.flatten()
    y_pred = format_logit_to_label(logits=logits)
    evaluate_hemo(y_true=y_true, y_score=y_score, y_pred=y_pred,
                  plot_path=plot_path, tag=tag, file_name=file_name)


def get_bce_label_weight(labels):
    """
    Get the label weights for binary cross-entropy loss.
    """
    import torch
    label_0 = np.count_nonzero(labels == 0)
    label_1 = np.count_nonzero(labels == 1)
    return torch.tensor([label_0 / label_1])


def init_model(train_dataset, training_args, n_features):
    if training_args.model_class == 'binary_dense':

        # ('GrimSqueaker/proteinBERT')
        config = BertConfig.from_pretrained(training_args.model_path)
        model = PeptideBertForBinaryClassification(config,
                                                   model_path=training_args.model_path,
                                                   loss_function=training_args.loss_function,
                                                   bce_logit_weight=get_bce_label_weight(
                                                       labels=train_dataset.labels).to(training_args.device),
                                                   n_features=n_features)
        data_collator = DefaultDataCollator()
        run_metric = binary_metrics


    elif training_args.model_class == 'regression':
        # raise NotImplementedError("Custom task not implemented yet")
        config = BertConfig.from_pretrained(training_args.model_path)
        config.hidden_size = 1024
        config.num_labels = 1
        # model = BertForSequenceClassification.from_pretrained(training_args.model_path, config=config)
        model = PeptideBertForRegression(config,
                                         model_path=training_args.model_path,
                                         n_features=n_features)
        data_collator = DefaultDataCollator()
        run_metric = regression_metrics

    else:
        raise ValueError(
            f"model_class must be either 'binary_dense' or 'regression'. You provided: '{training_args.model_class}'")
    return data_collator, model, run_metric




if __name__ == '__main__':
    # data_leakage_wrapper()
    generate_custom_yaml_file(config_name="custom_config.yaml",
                              file_path='peptideBERT_configs/')
