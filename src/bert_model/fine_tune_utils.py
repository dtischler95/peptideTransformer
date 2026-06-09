import logging
import os

import numpy as np
import pandas as pd
import peptides as pep
import yaml
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import DefaultDataCollator, BertConfig
from transformers import BertTokenizer

from src.bert_model.PeptideBERTClasses.PeptideBertForBinaryClassification import PeptideBertForBinaryClassification
from src.bert_model.PeptideBERTClasses.PeptideBertForRegression import PeptideBertForRegression
from src.bert_model.PeptideBERTClasses.PeptideDataset import PeptideDataset
from src.bert_model.PeptideBERTClasses.PeptideTrainingArguments import PeptideTrainingArguments
from src.bert_model.transformer_metrics import binary_metrics, regression_metrics
from src.machine_learning.eval_utils import (
    get_pretty_name, print_regression_metrics, make_regression_plot,
    overall_stats, get_model_stats, evaluate_hemo,
)


def prepare_datasets(model_class: str,
                     model_path: str,
                     train_file: str,
                     logger: logging.Logger,
                     show_encoding: bool = False,
                     ignore_leakage: bool = False,
                     max_length: int = 36,
                     cut_df_for_faster_debug: bool = False,
                     add_features: bool = False
                     ) -> tuple[
    BertTokenizer, PeptideDataset, PeptideDataset, PeptideDataset, int]:
    """
    Creates the datasets for training, validation and testing. For the given transformers Dataset class
    Differentiates between binary classification and masked language modeling in this case.
    Update the Dataset class if you want to use a different model or a different task so the Dataset class fits the
    data and the task.

    :param model_class: Model class to use, either 'binary_dense' or 'regression'
    :param model_path: Path to the pretrained model
    :param show_encoding: If the encoding of the vocabulary should be shown
    :param train_file: Path to the training data
    :param val_file: Path to the validation data
    :param logger: Logger for logging
    :param ignore_leakage: If data leakage should be ignored or cause an error to stop training
    :param max_length: Maximum length for padding/truncation.
    :param cut_df_for_faster_debug: If the dataframe should be cut for faster debugging. Default is False.
    :param validation_data_size: Size of the validation data. Default is 0.2.
    :param test_data_size: Size of the test data. Default is 0.5.
    :param random_data_shuffle: If the data should be shuffled randomly or data previewed via like CD-Hit
    :param add_features: If the concentration data should be used for training. Default is False.

    :return: Tokenizer, train_dataset, val_dataset, test_dataset
    """


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

    def load_and_sample_data(file, sample=False):
        train_df = pd.read_csv(file.replace('.csv', "_train.csv"), sep=';')
        test_df = pd.read_csv(file.replace('.csv', "_test.csv"), sep=';')
        val_df = pd.read_csv(file.replace('.csv', "_val.csv"), sep=';')
        return train_df.sample(frac=1)[:100] if sample else train_df, test_df, val_df


    def get_labels_and_features(data, concentration):

        if model_class.startswith('regression'):
            target_column = 'mic_log10'
            columns_to_drop = ['sequence', target_column]
        elif model_class.startswith('binary'):
            target_column = 'label'
            columns_to_drop = ['sequence', target_column]
            if 'hemo_percent' in data.keys():
                columns_to_drop += ['hemo_percent', 'hemo_concentration']
        labels = data[target_column].values
        concentrations = data.drop(columns=columns_to_drop) if concentration else None
        return labels, concentrations

    df_train, df_test, df_val = load_and_sample_data(train_file, cut_df_for_faster_debug)

    no_feature_size = df_train.shape[1]
    if add_features:
        df_train = add_descriptors(df_train)
        df_test = add_descriptors(df_test)
        df_val = add_descriptors(df_val)

    feature_size = df_train.shape[1] - no_feature_size


    label_data_train, feature_data_train = get_labels_and_features(df_train, add_features)
    label_data_val, feature_data_val = get_labels_and_features(df_val, add_features)
    label_data_test, feature_data_test = get_labels_and_features(df_test, add_features)

    tokenizer = BertTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=True, do_lower_case=False)

    if add_features:
        imp = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        feature_data_train = scaler.fit_transform(imp.fit_transform(feature_data_train.values)).astype(np.float32)
        feature_data_val = scaler.transform(imp.transform(feature_data_val.values)).astype(np.float32)
        feature_data_test = scaler.transform(imp.transform(feature_data_test.values)).astype(np.float32)

    train_dataset = PeptideDataset(peptides=df_train['sequence'],
                                   features=feature_data_train,
                                   tokenizer=tokenizer,
                                   labels=label_data_train,
                                   max_length=max_length,
                                   model_class=model_class)
    val_dataset = PeptideDataset(peptides=df_val['sequence'],
                                 features=feature_data_val,
                                 tokenizer=tokenizer,
                                 labels=label_data_val,
                                 max_length=max_length,
                                 model_class=model_class)
    test_dataset = PeptideDataset(peptides=df_test['sequence'],
                                  features=feature_data_test,
                                  tokenizer=tokenizer,
                                  labels=label_data_test,
                                  max_length=max_length,
                                  model_class=model_class)


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


def load_training_arguments(config_file: str, logger: logging.Logger) -> PeptideTrainingArguments:
    """
    Function to load the training arguments from a config file for the given training type.
    You can implement more config checks here if needed.
    Training Configuration will be logged.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    logger.info("Logger wont log this anymore :(")
    # more logging, we all love logging

    print("Set Parameters for this training run:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    if config['plot_path'] is None:
        config.plot_path = './plots'
        logger.info(f"Plot path not set. Using default path: {config.plot_path}")

    if not os.path.exists(config['plot_path']):
        os.makedirs(config['plot_path'])
        logger.info(f"Created directory: {config['plot_path']}")

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
def generate_custom_yaml_file(config_name: str,
                              file_path: str = './bert_model/peptideBERT_configs/'):
    """
    Generates a custom-formatted YAML file with grouped settings and comments.
    This is for much easier configuration and readability of the YAML file for new runs.

    Parameters:
    - config_name (str): Name of the current setup for easier identification.
    - file_path (str): The path where the YAML file will be saved. !Should be the peptideBERT_config folder.!
    """

    yaml_content = """
# Training and Evaluation Settings
model_class: 'ENTER MODELTYPE HERE'                    # Model class to use, either 'binary_dense' or 'regression'
do_train: true                                         # Train the model
do_eval: true                                          # Evaluate the model
do_predict: true                                       # Predict with the model
num_train_epochs: 50                                   # Number of epochs to train the model
per_device_train_batch_size: 256                       # Batch size for training
per_device_eval_batch_size: 64                         # Batch size for evaluation
early_stopping_patience: 7                             # Patience for early stopping
early_stop_metric: 'eval_loss'                         # Metric for early stopping
early_stop_mode: 'min'                                 # Mode for early stopping
early_stop_warm_up: 30                                 # Warm-up period for early stopping
dataloader_drop_last: false                            # Drop last batch if smaller than batch size
dataloader_num_workers: 1                              # Number of dataloader workers (higher can affect performance)
show_encoding: False                                   # Show the encoding of the sequences from the tokenizer
load_best_model_at_end: true                           # Load the best model at the end of training
add_features: true                                     # Use concentration as extra feature [WARNING] U need a prepared train datafile for this


# Model and Optimizer Settings
learning_rate: 0.00005                                 # Learning rate for optimizer
weight_decay: 0.01                                     # Weight decay for optimizer
lr_scheduler_type: 'reduce_lr_on_plateau'              # Learning rate scheduler type
lr_scheduler_kwargs:                                   # Additional scheduler arguments
  patience: 4                                          # Patience for ReduceLROnPlateau scheduler
max_length: 36                                         # Maximum input sequence length

# Binary Classification Settings
label_0_cluster_data: 500                              # Number of data points for label 0 used in downstream clustering
label_1_cluster_data: 500                              # Number of data points for label 1 used in downstream clustering


# Dataset and File Paths
train_file: SET TRAIN DATA PATH HERE                   # Path to training data
model_path: SET MODEL PATH HERE                        # Path to pretrained model
model_save_path: SET MODEL SAVE PATH HERE              # Path to save the model
plot_path: './plots'                                   # Path to save the plots

# Validation and Test Settings
validation_data_size: 0.2                              # Validation dataset size (fraction)
test_data_size: 0.5                                    # Test dataset size (fraction)
metric_for_best_model: 'loss'                          # Metric for selecting best model ['loss', 'accuracy']

# Logging Settings
output_dir: './results'                                # Path to checkpoints
logging_dir: './logs'                                  # Path to logging directory
logging_strategy: 'epoch'                              # Logging strategy (set to 'epoch' for this logic)
log_level: 'info'                                      # Log level
eval_strategy: 'epoch'                                 # Evaluation strategy (set to 'epoch' for this logic)

# Reproducibility
seed: 42                                               # Random seed for reproducibility

# Hardware Settings
use_cpu: false                                         # Use CPU for training

# Miscellaneous Settings
classification_weighted_labels: false                  # Use weighted labels for classification
ignore_leakage: false                                  # Ignore leakage in training data (only if certain)
fast_debug: false                                      # Use fast debug mode (only if certain)
"""

    try:
        # Write the YAML content to a file
        with open(file_path + config_name, 'w') as file:
            file.write(yaml_content.strip())
        print(f"YAML file successfully created at: {file_path + config_name}. "
              f"Please set all needed data file Paths and review the settings of this file for proper usage."
              f"Parameters provided here are just used as an example.")
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
