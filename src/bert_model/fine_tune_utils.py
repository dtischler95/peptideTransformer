import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.bert_model.PeptideBERTClasses.PeptideTrainingArguments import PeptideTrainingArguments
from src.bert_model.PeptideBERTClasses.PeptideDataset import PeptideDataset


def prepare_datasets(binary_or_mlm: str,
                     show_encoding: bool,
                     train_file: str,
                     logger: logging.Logger,
                     ignore_leakage: bool = False,
                     max_length: int = 36,
                     cut_df_for_faster_debug: bool = False,
                     validation_data_size: float = 0.2,
                     test_data_size: float = 0.5) -> tuple[
    BertTokenizer, PeptideDataset, PeptideDataset, PeptideDataset]:
    """
    Creates the datasets for training, validation and testing. for the given transformers Dataset class
    Differentiates between binary classification and masked language modeling in this case.
    Update the Dataset class if you want to use a different model or a different task so the Dataset class fits the
    data and the task.

    :param binary_or_mlm: if the model should be fine-tuned for binary classification or masked language modeling
                          Is used for your task, update this flag if you want to use a different model or a different task
    :param show_encoding: if the encoding of the vocabulary should be shown
    :param train_file: path to the training data
    :param logger: logger for logging
    :param ignore_leakage: if data leakage should be ignored or cause an error to stop training
    :param max_length: Maximum length for padding/truncation.
    :param cut_df_for_faster_debug: If the dataframe should be cut for faster debugging. Default is False.
    :param validation_data_size: Size of the validation data. Default is 0.2.
    :param test_data_size: Size of the test data. Default is 0.5.

    :return: tokenizer, train_dataset, val_dataset, test_dataset
    """
    # Load the data
    df = pd.read_csv(train_file, sep=';')
    df = df.sample(frac=1)[:200] if cut_df_for_faster_debug else df
    # Get unique sequence id for train/test split. We create our split data with the IDs to avoid data Leakage
    if ignore_leakage:
        df_to_split = df
    else:
        df_to_split = df['sequence'].unique()

    # Cut the dataframe for faster debugging if enabled. shuffle the df to ensure labels are mixed
    # TODO add stratified args for train_test_split

    # Split the data into training, validation and test sets
    train_sequences, df_val_handler = train_test_split(df_to_split, test_size=validation_data_size, shuffle=True)
    val_sequences, test_sequences = train_test_split(df_val_handler, test_size=test_data_size, shuffle=True)

    if not ignore_leakage:
        # Assigning the given train/val/test task to a given sequence id ensuring there's no Leakage
        train_sequences = df[df['sequence'].isin(train_sequences)]
        val_sequences = df[df['sequence'].isin(val_sequences)]
        test_sequences = df[df['sequence'].isin(test_sequences)]

    label_data_train = train_sequences['label'].values if binary_or_mlm == 'binary' else None
    label_data_val = val_sequences['label'].values if binary_or_mlm == 'binary' else None
    label_data_test = test_sequences['label'].values if binary_or_mlm == 'binary' else None

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', clean_up_tokenization_spaces=True)

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


def load_training_arguments(config_file: str, training_type: str, logger: logging.Logger) -> PeptideTrainingArguments:
    """
    Function to load the training arguments from a config file for the given training type.
    The training type is used to select the correct training arguments from the config file.
    Training Configuration will be logged.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    logger.info("Logger wont log this anymore :(")
    # more logging, we all love logging
    print("Set Parameters for this training run:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    return PeptideTrainingArguments(**config)


def load_and_prepare_datasets(tokenizer, training_args):
    """
    Load and prepare the negative and positive datasets for predictions.
    """
    # Load datasets
    negative_df = pd.read_csv("./data/train_data/mlm_train_data.csv", sep=';')
    positive_df = pd.read_csv("./data/train_data/our_hemo_labeled_filtered.csv", sep=';')

    # Filter out sequences in positive_df from negative_df
    negative_df = negative_df[~negative_df["sequence"].isin(positive_df['sequence'])]

    # Prepare positive dataset
    positive_df = positive_df[positive_df["label"] == 1]

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

    return negative_dataset, positive_dataset


def get_predictions(trainer, dataset):
    """
    Get predictions for a given dataset using the trainer.
    """
    predictions = trainer.predict(dataset)
    return (predictions.predictions > 0.5).astype(int)


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


def test_binary_label_bias(tokenizer, trainer, training_args):
    """
    Function to test the label bias in the predictions of the model.
    """
    # Load datasets
    negative_dataset, positive_dataset = load_and_prepare_datasets(tokenizer, training_args)

    # Get predictions
    negative_predictions = get_predictions(trainer, negative_dataset)
    positive_predictions = get_predictions(trainer, positive_dataset)

    # Calculate abundances
    abundance_negative = calculate_abundance(negative_predictions)
    abundance_positive = calculate_abundance(positive_predictions)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot for negative dataset
    plot_abundance(abundance_negative, 'Predicted Labels of Negative Dataset', axs[0])

    # Plot for positive dataset
    plot_abundance(abundance_positive, 'Predicted Labels of Positive Dataset', axs[1])

    # Adjust layout
    plt.tight_layout()
    plt.savefig("./plots/label_prediction_bias.png")

    # TODO: Box plots for label on dataset


if __name__ == '__main__':
    data_leakage_wrapper()
