import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.bert_model.PeptideBERTClasses.PeptideTrainingArguments import PeptideTrainingArguments
from src.bert_model.PeptideBERTClasses.PeptideDataset import PeptideDataset
#from src.bert_model.PeptideBERTClasses.PeptideTrainer import PeptideTrainer # TODO FIX CIRCULAR IMPORT FOR FISHER EXACT


def prepare_datasets(binary_or_mlm: str,
                     model_path: str,
                     show_encoding: bool,
                     train_file: str,
                     val_file: str,
                     logger: logging.Logger,
                     ignore_leakage: bool = False,
                     max_length: int = 36,
                     cut_df_for_faster_debug: bool = False,
                     validation_data_size: float = 0.2,
                     test_data_size: float = 0.5,
                     random_data_shuffle: bool = False) -> tuple[
    BertTokenizer, PeptideDataset, PeptideDataset, PeptideDataset]:
    """
    Creates the datasets for training, validation and testing. For the given transformers Dataset class
    Differentiates between binary classification and masked language modeling in this case.
    Update the Dataset class if you want to use a different model or a different task so the Dataset class fits the
    data and the task.

    :param binary_or_mlm: If the model should be fine-tuned for binary classification or masked language modeling
                          Is used for your task, update this flag if you want to use a different model or a different task
    :param model_path: path to the pretrained model
    :param show_encoding: if the encoding of the vocabulary should be shown
    :param train_file: path to the training data
    :param val_file: path to the validation data
    :param logger: logger for logging
    :param ignore_leakage: if data leakage should be ignored or cause an error to stop training
    :param max_length: Maximum length for padding/truncation.
    :param cut_df_for_faster_debug: If the dataframe should be cut for faster debugging. Default is False.
    :param validation_data_size: Size of the validation data. Default is 0.2.
    :param test_data_size: Size of the test data. Default is 0.5.
    :param random_data_shuffle: If the data should be shuffled randomly or data previewed via like CD-Hit

    :return: tokenizer, train_dataset, val_dataset, test_dataset
    """
    # Load the data
    df = pd.read_csv(train_file, sep=';')
    df_val = pd.read_csv(val_file, sep=';')
    df = df.sample(frac=1)[:50] if cut_df_for_faster_debug else df
    # Get unique sequence id for train/test split. We create our split data with the IDs to avoid data Leakage
    if ignore_leakage:
        df_to_split = df
    else:
        df_to_split = df['sequence'].unique()

    # Cut the dataframe for faster debugging if enabled. shuffle the df to ensure labels are mixed
    # TODO add stratified args for train_test_split

    # Split the data into training, validation and test sets

    if random_data_shuffle:
        train_sequences, df_val_handler = train_test_split(df_to_split, test_size=validation_data_size, shuffle=True)
        val_sequences, test_sequences = train_test_split(df_val_handler, test_size=test_data_size, shuffle=True)
    else:
        train_sequences = df
        val_sequences, test_sequences = train_test_split(df_val, test_size=test_data_size, shuffle=True)

    if not ignore_leakage:
        # Assigning the given train/val/test task to a given sequence id ensuring there's no Leakage
        train_sequences = df[df['sequence'].isin(train_sequences)]
        val_sequences = df[df['sequence'].isin(val_sequences)]
        test_sequences = df[df['sequence'].isin(test_sequences)]

    label_data_train = train_sequences['label'].values if binary_or_mlm == 'binary' else None
    label_data_val = val_sequences['label'].values if binary_or_mlm == 'binary' else None
    label_data_test = test_sequences['label'].values if binary_or_mlm == 'binary' else None

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=True)

    train_dataset = PeptideDataset(peptides=train_sequences['sequence'], tokenizer=tokenizer, labels=label_data_train,
                                   max_length=max_length)
    val_dataset = PeptideDataset(peptides=val_sequences['sequence'], tokenizer=tokenizer, labels=label_data_val,
                                 max_length=max_length)
    test_dataset = PeptideDataset(peptides=test_sequences['sequence'], tokenizer=tokenizer, labels=label_data_test,
                                  max_length=max_length)

    # print out the encoding of the vocabulary used by the tokenizer if wanted
    get_encoding(tokenizer=tokenizer) if show_encoding else None

    # Important Data Leakage Check
    check_data_loader_for_leakage(train_data_loader=train_dataset,
                                  val_data_loader=val_dataset,
                                  test_data_loader=test_dataset,
                                  ignore_leakage=ignore_leakage,
                                  logger=logger)

    return tokenizer, train_dataset, val_dataset, test_dataset


def get_encoding(tokenizer: BertTokenizer):
    """
    Prints out the encoding of the vocabulary used by the tokenizer

    :param tokenizer: The tokenizer used for encoding
    :return:
    """

    for i in range(tokenizer.vocab_size):
        character = tokenizer.decode(i)
        print(f"{i}: {character}")


def dummy_data_loader(train_file: str, number_of_data_to_use: int = 1000) -> tuple[
    PeptideDataset, PeptideDataset, PeptideDataset]:
    """
    Creates a dummy data loader for testing purposes.

    :param train_file: Path to the training data
    :param number_of_data_to_use: Number of data to use.
                                  Needs to be reasonably high for provoking data leakage and low enough
                                  for better handling for data viewing.

    """
    # Load the data
    # Extract to method if I want to pipe binary and mlm fine-tuning
    df = pd.read_csv(train_file, sep=';')[:number_of_data_to_use]

    # Split the data into training, validation and test sets
    df_train, df_val_handler = train_test_split(df, test_size=0.2)
    df_val, df_test = train_test_split(df_val_handler, test_size=0.5)

    sequence_data_train = df_train['sequence'].values
    sequence_data_val = df_val['sequence'].values
    sequence_data_test = df_test['sequence'].values

    # None bc of Testing Usage here. We only need sequences for testing for leakage
    label_data_train = None
    label_data_val = None
    label_data_test = None

    tokenizer = None

    train_dataset = PeptideDataset(peptides=sequence_data_train, tokenizer=tokenizer, labels=label_data_train)
    val_dataset = PeptideDataset(peptides=sequence_data_val, tokenizer=tokenizer, labels=label_data_val)
    test_dataset = PeptideDataset(peptides=sequence_data_test, tokenizer=tokenizer, labels=label_data_test)

    return train_dataset, val_dataset, test_dataset


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


def data_leakage_wrapper():
    """
    cleaner dummy data use for testing data leakage.
    Just a wrapper for testing the data leakage check.
    """

    train_data_loader, val_data_loader, test_data_loader = dummy_data_loader(
        train_file='../../data/train_data/our_hemo_labeled.csv',
        number_of_data_to_use=1000)

    check_data_loader_for_leakage(train_data_loader=None,
                                  val_data_loader=val_data_loader,
                                  test_data_loader=test_data_loader,
                                  logger=logging.Logger(name="debug_logger"))


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
    if config['run_verbose']:
        print("Set Parameters for this training run:")
        for k, v in config.items():
            print(f"  {k}: {v}")

    if config['plot_path'] is None:
        config.plot_path = './plots'
        logger.info(f"Plot path not set. Using default path: {config.plot_path}")

    import os
    if not os.path.exists(config['plot_path']):
        os.makedirs(config['plot_path'])
        logger.info(f"Created directory: {config['plot_path']}")

    return PeptideTrainingArguments(**config)


def prepare_label_debug_datasets(tokenizer, training_args):
    """
    Load and prepare the negative and positive datasets for predictions.
    """
    # Load datasets
    negative_df = pd.read_csv("./data/train_data/mlm_train_data.csv", sep=';')[:10000]
    bioactive_df = pd.read_csv("./data/train_data/our_hemo_labeled_filtered.csv", sep=';')

    # Filter out sequences in positive_df from negative_df
    negative_df = negative_df[~negative_df["sequence"].isin(bioactive_df['sequence'])]

    # Prepare positive dataset
    positive_df = bioactive_df[bioactive_df["label"] == 1]

    # df for containing non-hemolytic but bioactive peptides
    positive_negative_df = bioactive_df[bioactive_df["label"] == 0]

    # Create PeptideDataset instances
    negative_dataset = PeptideDataset(
        peptides=negative_df['sequence'],
        tokenizer=tokenizer,
        labels=[0] * len(negative_df),
        max_length=training_args.max_length
    )

    positive_dataset = PeptideDataset(
        peptides=positive_df['sequence'].values,
        tokenizer=tokenizer,
        labels=positive_df['label'].values,
        max_length=training_args.max_length
    )

    positive_negative_dataset = PeptideDataset(
        peptides=positive_negative_df['sequence'].values,
        tokenizer=tokenizer,
        labels=positive_negative_df['label'].values,
        max_length=training_args.max_length
    )

    return negative_dataset, positive_dataset, positive_negative_dataset


def format_logit_to_label(logits):
    """
    Format the given logit tensor to a list of labels.
    0.5 is a typical threshold
    """
    return (logits > 0.5).astype(int).flatten()


def calculate_abundance(predictions):
    """
    Calculate the abundance of 0s and 1s in the predictions.
    """
    abundance = {
        0: np.count_nonzero(predictions == 0),
        1: np.count_nonzero(predictions == 1)
    }
    return abundance


def plot_abundance(abundance_dict, title, ax):
    """
    Plot the abundance of predicted labels.
    """
    ax.bar(abundance_dict.keys(), abundance_dict.values(), color=['blue', 'orange'])
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Abundance')
    ax.set_xticks(list(abundance_dict.keys()))
    ax.set_ylim(0, max(abundance_dict.values()) + 1)  # Add some space above the bars
    ax.grid(axis='y', linestyle='--', alpha=0.7)



def plot_label_abundance(label_0_counter,
                         label_1_counter,
                         task_name: str,
                         plot_path: str):
    """
    Plot the abundance of label classes in the predictions per Epoch or Batch wise

    TODO Better looking graphs. Maybe make it more dynamic to given task name

    :param label_0_counter: Counter for label 0
    :param label_1_counter: Counter for label 1
    :param task_name: Name of the task
    :param plot_path: Path to save the plot
    """

    max_labels = label_0_counter[0] + label_1_counter[0]

    if task_name == "train_batch_wise_label_prediction":
        x_label = "Batches"
    else:
        x_label = "Epochs"

    epochs = range(1, len(label_0_counter) + 1)
    plt.figure(figsize=(10, 5))

    plt.plot(epochs, label_0_counter, label='Label 0', color='blue')
    plt.xlabel(f"{x_label}")
    plt.ylabel('Label 0 [%]', color='green')
    plt.tick_params(axis='y', labelcolor='green')
    plt.ylim(0, max_labels)
    plt.xticks(epochs)
    plt.xlim(1, len(label_0_counter))
    label_0_counter_np = np.array(label_0_counter)
    plt.fill_between(epochs, label_0_counter, max_labels, where=(label_0_counter_np <= max_labels),
                     color='red', alpha=0.3)

    plt.fill_between(epochs, label_0_counter, 0, where=(label_0_counter_np >= 0), color='green', alpha=0.3)

    plt.savefig(f"{plot_path}/{task_name}_label_abundance.png")
    plt.close()


# TODO Update this function for all new parameters and formats
def generate_custom_yaml_file(config_name: str,
                              file_path: str = './bert_model/peptideBERT_configs/'):
    """
    Generates a custom-formatted YAML file with grouped settings and comments.
    This is for more easier configuration and readability of the YAML file for new runs.

    Parameters:
    - config_name (str): Name of the current setup for easier identification.
    - file_path (str): The path where the YAML file will be saved. !Should be the peptideBERT_config folder.!
    """

    yaml_content = """
# Training and Evaluation Settings
model_class: 'ENTER MODELTYPE HERE'         # Model class to use, either 'binary' or 'mlm' 
do_train: true                # Train the model
do_eval: true                 # Evaluate the model
do_predict: true              # Predict with the model
num_train_epochs: 50          # Number of epochs to train the model
per_device_train_batch_size: 256  # Batch size for training
per_device_eval_batch_size: 64    # Batch size for evaluation
early_stopping_patience: 7        # Patience for early stopping
early_stop_metric: 'eval_loss'    # Metric for early stopping
early_stop_mode: 'min'            # Mode for early stopping
early_stop_warm_up: 30            # Warm-up period for early stopping
dataloader_drop_last: false       # Drop last batch if smaller than batch size
dataloader_num_workers: 2         # Number of dataloader workers (higher can affect performance)
show_encoding: False              # Show the encoding of the sequences from the tokenizer

# Model and Optimizer Settings
learning_rate: 0.00005            # Learning rate for optimizer
weight_decay: 0.01                # Weight decay for optimizer
lr_scheduler_type: 'reduce_lr_on_plateau'  # Learning rate scheduler type
lr_scheduler_kwargs:              # Additional scheduler arguments
  patience: 4                     # Patience for ReduceLROnPlateau scheduler
max_length: 36                    # Maximum input sequence length

# Binary Classification Settings
label_0_cluster_data: 500         # Number of data points for label 0 used in downstream clustering
label_1_cluster_data: 500         # Number of data points for label 1 used in downstream clustering
loss_function: 'bce'              # Possible Choices ['bce', 'bce_logit_loss']

# MLM Settings
mlm_probability: 0.15                       # Masking probability for MLM
mlm_curriculum_learning: false              # Enable curriculum learning for MLM
mlm_curriculum_increase_step: 0.0000001     # Step size for curriculum learning
mlm_curriculum_max_prob: 0.15               # Maximum masking probability for MLM

# Dataset and File Paths
train_file: SET TRAIN DATA PATH HERE                   # Path to training data
model_path: SET MODEL PATH HERE                        # Path to pretrained model
model_save_path: SET MODEL SAVE PATH HERE              # Path to save the model
plot_path: './plots'                                   # Path to save the plots

# Validation and Test Settings
validation_data_size: 0.2       # Validation dataset size (fraction)
test_data_size: 0.5             # Test dataset size (fraction)
metric_for_best_model: 'loss'   # Metric for selecting best model ['loss', 'accuracy']

# Logging Settings
output_dir: './results'         # Path to checkpoints
logging_dir: './logs'           # Path to logging directory
logging_strategy: 'epoch'       # Logging strategy (set to 'epoch' for this logic)
log_level: 'info'               # Log level
eval_strategy: 'epoch'          # Evaluation strategy (set to 'epoch' for this logic)

# Reproducibility
seed: 42                        # Random seed for reproducibility

# Hardware Settings
use_cpu: false                  # Use CPU for training

# Miscellaneous Settings
classification_weighted_labels: false   # Use weighted labels for classification
ignore_leakage: false                   # Ignore leakage in training data (only if certain)
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


def prepare_fisher_exact(test_dataset: PeptideDataset,
                         trainer,
                         plot_path: str):
    from sklearn import metrics
    actual = test_dataset.labels
    logits = trainer.predict(test_dataset).predictions
    confusion_matrix = metrics.confusion_matrix(actual, format_logit_to_label(logits=logits))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
    cm_display.plot()
    plt.savefig(f"{plot_path}/confusion_matrix.png")
    plt.close()
    plt.clf()


def get_bce_label_weight(labels):
    """
    Get the label weights for binary cross-entropy loss.
    """
    import torch
    label_0 = np.count_nonzero(labels == 0)
    label_1 = np.count_nonzero(labels == 1)
    return torch.tensor([label_0 / label_1])

if __name__ == '__main__':
    # data_leakage_wrapper()
    generate_custom_yaml_file(config_name="custom_config.yaml",
                              file_path='peptideBERT_configs/')
