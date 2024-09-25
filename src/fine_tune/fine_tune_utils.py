from src.fine_tune.PeptideDataset import PeptideDataset
import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import yaml
import os

def prepare_datasets(binary_or_mlm: str,
                     drop_duplicates: bool,
                     show_encoding: bool,
                     train_file: str,
                     ignore_leakage: bool = False) -> tuple[
    BertTokenizer, PeptideDataset, PeptideDataset, PeptideDataset]:
    """
    Creates the datasets for training, validation and testing. for the given transformers Dataset class
    Differentiates between binary classification and masked language modeling in this case.
    Update the Dataset class if you want to use a different model or a different task so the Dataset class fits the
    data and the task.

    :param binary_or_mlm: if the model should be fine-tuned for binary classification or masked language modeling
                          Is used for your task, update this flag if you want to use a different model or a different task
    :param drop_duplicates: if duplicated sequences should be dropped
    :param show_encoding: if the encoding of the vocabulary should be shown
    :param train_file: path to the training data
    :param ignore_leakage: if data leakage should be ignored or cause an error to stop training

    :return: tokenizer, train_dataset, val_dataset, test_dataset
    """
    # Load the data
    # Extract to method if I want to pipe binary and mlm fine-tuning
    df = pd.read_csv(train_file, sep=';')
    # Drop duplicates only if you explicitly want to
    df = df.drop_duplicates(subset=['sequence']) if drop_duplicates else df
    # Split the data into training, validation and test sets
    # TODO Create Accuracy Validation for Testdata
    df_train, df_val_handler = train_test_split(df, test_size=0.2)
    df_val, df_test = train_test_split(df_val_handler, test_size=0.5)

    sequence_data_train = df_train['sequence'].values
    sequence_data_val = df_val['sequence'].values
    sequence_data_test = df_test['sequence'].values

    label_data_train = df_train['label'].values if binary_or_mlm == 'binary' else None
    label_data_val = df_val['label'].values if binary_or_mlm == 'binary' else None
    label_data_test = df_test['label'].values if binary_or_mlm == 'binary' else None

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', clean_up_tokenization_spaces=True)

    train_dataset = PeptideDataset(peptides=sequence_data_train, tokenizer=tokenizer, labels=label_data_train)
    val_dataset = PeptideDataset(peptides=sequence_data_val, tokenizer=tokenizer, labels=label_data_val)
    test_dataset = PeptideDataset(peptides=sequence_data_test, tokenizer=tokenizer, labels=label_data_test)

    # print out the encoding of the vocabulary used by the tokenizer if wanted
    get_encoding(tokenizer=tokenizer) if show_encoding else None

    check_data_loader_for_leakage(train_data_loader=train_dataset,
                                  val_data_loader=val_dataset,
                                  test_data_loader=test_dataset,
                                  ignore_leakage=ignore_leakage)

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
                                  decoded_input: bool = False,
                                  ignore_leakage: bool = False):
    """
    Function that can be imported in other scripts to check for data leakage between the data loaders.
    Raises a ValueError if data leakage is detected.

    Use decoded_input=True if the input is decoded like in whitelab's data case.

    :param train_data_loader: Training data loader
    :param val_data_loader: Validation data loader
    :param test_data_loader: Test data loader
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

    if len(train_test_leakage) or len(train_val_leakage) or len(val_test_leakage) > 0:
        print("\n-------------- Data Leakage Detected --------------\n")

        print(f"Number of Training Data Points: {len(train_sequences)}")
        print(f"Number of Validation Data Points: {len(val_sequences)}")
        print(f"Number of Test Data Points: {len(test_sequences)}\n")


        if len(train_val_leakage) > 0:
            print(f"Number of data Points leaked in Training and Validation Data: {len(train_val_leakage)}\n")
        if len(train_test_leakage) > 0:
            print(f"Number of data Points leaked in Training and Test Data: {len(train_test_leakage)}\n")
        if len(val_test_leakage) > 0:
            print(f"Number of data Points leaked in Validation and Test Data: {len(val_test_leakage)}\n")

        print("\n-------------- Data Leakage Detected --------------\n")
        if not ignore_leakage:
            raise ValueError("Data Leakage Detected! Please recheck your data preprocessing steps.")
    else:
        print("\n-------------- No Data Leakage Detected --------------\n")


def data_leakage_wrapper():
    """
    cleaner dummy data use for testing data leakage.
    Just a wrapper for the data leakage check.
    """

    train_data_loader, val_data_loader, test_data_loader = dummy_data_loader(
        train_file='../../data/train_data/our_hemo_labeled.csv',
        number_of_data_to_use=1000)

    check_data_loader_for_leakage(None, val_data_loader, test_data_loader)


def load_training_arguments(config_file: str, training_type: str, use_cpu: bool = False):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    training_args_dict = config['training_arguments'][training_type]
    training_args_dict['use_cpu'] = use_cpu

    from transformers import TrainingArguments

    return TrainingArguments(**training_args_dict)

def check_directory(paths: list[str] or str):
    paths_to_check = [paths] if type(paths) == str else paths
    for path in paths_to_check:
        try:
            os.mkdir(path)
        except FileExistsError:
            pass


if __name__ == '__main__':
    data_leakage_wrapper()
