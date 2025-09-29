import logging
import os

import numpy as np
import pandas as pd
import peptides as pep
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from scipy.stats import linregress
from sklearn.impute import SimpleImputer
from sklearn.metrics import (r2_score,
                             mean_absolute_error,
                             explained_variance_score,
                             mean_squared_error,
                             precision_recall_curve,
                             PrecisionRecallDisplay)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import BertForMaskedLM, DefaultDataCollator, BertConfig, DataCollatorForLanguageModeling
from transformers import BertTokenizer

from src.bert_model.PeptideBERTClasses.PeptideBertForBinaryClassification import PeptideBertForBinaryClassification
from src.bert_model.PeptideBERTClasses.PeptideBertForRegression import PeptideBertForRegression
from src.bert_model.PeptideBERTClasses.PeptideDataset import PeptideDataset
from src.bert_model.PeptideBERTClasses.PeptideTrainingArguments import PeptideTrainingArguments
from src.bert_model.transformer_metrics import binary_metrics, mlm_metrics, regression_metrics


def prepare_datasets(model_class: str,
                     model_path: str,
                     train_file: str,
                     val_file: str,
                     logger: logging.Logger,
                     show_encoding: bool = False,
                     ignore_leakage: bool = False,
                     max_length: int = 36,
                     cut_df_for_faster_debug: bool = False,
                     validation_data_size: float = 0.2,
                     test_data_size: float = 0.5,
                     random_data_shuffle: bool = False,
                     add_features: bool = False) -> tuple[
    BertTokenizer, PeptideDataset, PeptideDataset, PeptideDataset, int]:
    """
    Creates the datasets for training, validation and testing. For the given transformers Dataset class
    Differentiates between binary classification and masked language modeling in this case.
    Update the Dataset class if you want to use a different model or a different task so the Dataset class fits the
    data and the task.

    :param model_class: Model class to use, either 'binary_dense', 'binary_conv', 'mlm' or 'regression'
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

    # Load the data
    def compute_desc_row(sequence):
        p = pep.Peptide(sequence)
        d = p.descriptors()  # dict -> nur Zahlen
        return {f"desc__{k}": float(v) for k, v in d.items()}

    def add_descriptors(df):
        desc_rows = [compute_desc_row(s) for s in df["sequence"]]
        desc_df = pd.DataFrame(desc_rows).reset_index(drop=True)
        df = df.reset_index(drop=True).join(desc_df)
        # sauber halten:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df

    def load_and_sample_data(file, sample=False):
        df = pd.read_csv(file, sep=';')
        return df.sample(frac=1)[:100] if sample else df

    def split_sequences(data, test_size, shuffle=True):
        return train_test_split(data, test_size=test_size, shuffle=shuffle, random_state=42)

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

    # Load and preprocess data
    df = load_and_sample_data(train_file, cut_df_for_faster_debug)
    selected_features = []
    if add_features:
        df_train = add_descriptors(df)


        with open(file=f"{train_file.replace('.csv', '.txt')}") as f:
            selected_features = [line.strip() for line in f]

        df_train = pd.concat([df_train[selected_features], df.reset_index()], axis=1).drop(columns='index')

    else:
        df_train = df
    df_to_split = df_train if ignore_leakage else df_train['sequence'].unique()

    if random_data_shuffle:
        train_sequences, val_handler = split_sequences(df_to_split, validation_data_size)
        val_sequences, test_sequences = split_sequences(val_handler, test_data_size)
        all_sequence_df = df_train
    else:
        train_sequences = df_to_split
        df_val = load_and_sample_data(val_file)
        val_sequences, test_sequences = split_sequences(df_val['sequence'].unique(), test_data_size)
        all_sequence_df = pd.concat([df_train, df_val], ignore_index=True)

    if not ignore_leakage:
        train_sequences = all_sequence_df[all_sequence_df['sequence'].isin(train_sequences)]
        val_sequences = all_sequence_df[all_sequence_df['sequence'].isin(val_sequences)]
        test_sequences = all_sequence_df[all_sequence_df['sequence'].isin(test_sequences)]

    # Extract labels and concentrations
    label_data_train, feature_data_train = get_labels_and_features(train_sequences, add_features)
    label_data_val, feature_data_val = get_labels_and_features(val_sequences, add_features)
    label_data_test, feature_data_test = get_labels_and_features(test_sequences, add_features)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=True, do_lower_case=False)

    if add_features:
        imp = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        feature_data_train = scaler.fit_transform(imp.fit_transform(feature_data_train.values)).astype(np.float32)
        feature_data_val = scaler.transform(imp.transform(feature_data_val.values)).astype(np.float32)
        feature_data_test = scaler.transform(imp.transform(feature_data_test.values)).astype(np.float32)

    # Create datasets
    train_dataset = PeptideDataset(peptides=train_sequences['sequence'],
                                   features=feature_data_train,
                                   tokenizer=tokenizer,
                                   labels=label_data_train,
                                   max_length=max_length,
                                   model_class=model_class)
    val_dataset = PeptideDataset(peptides=val_sequences['sequence'],
                                 features=feature_data_val,
                                 tokenizer=tokenizer,
                                 labels=label_data_val,
                                 max_length=max_length,
                                 model_class=model_class)
    test_dataset = PeptideDataset(peptides=test_sequences['sequence'],
                                  features=feature_data_test,
                                  tokenizer=tokenizer,
                                  labels=label_data_test,
                                  max_length=max_length,
                                  model_class=model_class)

    # Optional: Show encoding
    if show_encoding:
        get_encoding(tokenizer)

    # Check for data leakage
    check_data_loader_for_leakage(train_data_loader=train_dataset, val_data_loader=val_dataset,
                                  test_data_loader=test_dataset, ignore_leakage=ignore_leakage, logger=logger)

    return tokenizer, train_dataset, val_dataset, test_dataset, len(selected_features)


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


def format_logit_to_label(logits, threshold: float = 0.5):
    """
    Format the given logit tensor to a list of labels.
    0.5 is a typical threshold
    """
    return (logits > threshold).astype(int).flatten()


def format_one_hot_to_label(one_hot_tensor):
    """
    Format the given one-hot encoded tensor to a list of labels.
    """
    return np.argmax(one_hot_tensor, axis=1).flatten()


def calculate_abundance(predictions):
    """
    Calculate the abundance of 0's and 1's in the predictions.
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
    This is for much easier configuration and readability of the YAML file for new runs.

    Parameters:
    - config_name (str): Name of the current setup for easier identification.
    - file_path (str): The path where the YAML file will be saved. !Should be the peptideBERT_config folder.!
    """

    yaml_content = """
# Training and Evaluation Settings
model_class: 'ENTER MODELTYPE HERE'                    # Model class to use, either 'binary' or 'mlm' 
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
dataloader_num_workers: 2                              # Number of dataloader workers (higher can affect performance)
show_encoding: False                                   # Show the encoding of the sequences from the tokenizer

# Model and Optimizer Settings
learning_rate: 0.00005                                 # Learning rate for optimizer
weight_decay: 0.01                                     # Weight decay for optimizer
lr_scheduler_type: 'reduce_lr_on_plateau'  # Learning rate scheduler type
lr_scheduler_kwargs:                                   # Additional scheduler arguments
  patience: 4                                          # Patience for ReduceLROnPlateau scheduler
max_length: 36                                         # Maximum input sequence length

# Binary Classification Settings
label_0_cluster_data: 500                              # Number of data points for label 0 used in downstream clustering
label_1_cluster_data: 500                              # Number of data points for label 1 used in downstream clustering
loss_function: 'bce'                                   # Possible Choices ['bce', 'bce_logit_loss']

# MLM Settings
mlm_probability: 0.15                                  # Masking probability for MLM
mlm_curriculum_learning: false                         # Enable curriculum learning for MLM
mlm_curriculum_increase_step: 0.0000001                # Step size for curriculum learning
mlm_curriculum_max_prob: 0.15                          # Maximum masking probability for MLM

# Dataset and File Paths
data_shuffle: true or false                            # [True] Randomly splits train/val/test data from one given input [false] if train and test data are already split beforehand
train_file: SET TRAIN DATA PATH HERE                   # Path to training data
val_file: SET VAL DATA PATH HERE                       # Path to validation data
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


def prepare_fisher_exact(test_dataset: PeptideDataset,
                         trainer,
                         plot_path: str):
    from sklearn import metrics
    actual = test_dataset.labels
    logits = trainer.predict(test_dataset).predictions

    thr, f1_best, p_best, r_best = best_f1_threshold(y_true=actual, y_score=logits.flatten(), plot_path=plot_path)

    print(f'Best F1 on val by thresholding: F1={f1_best:.4f} at thr={thr:.4f} '
                f'(P={p_best:.4f}, R={r_best:.4f})')

    confusion_matrix = metrics.confusion_matrix(actual, format_logit_to_label(logits=logits, threshold=thr))
    with np.errstate(all='ignore'):
        confusion_matrix_normalized = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

    titles_options = [
        ("Konfusionsmatrix, ohne Normalisierung", confusion_matrix),
        ("Konfusionsmatrix, mit Normalisierung", confusion_matrix_normalized),
    ]

    for title, matrix in titles_options:
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0, 1])
        cm_display.plot()

        # Deutsche Achsenbeschriftung
        plt.xlabel("Vorhergesagte Klasse")
        plt.ylabel("Wahre Klasse")
        plt.title("Konfusionsmatrix")
        plt.savefig(f"{plot_path}/{title}confusion_matrix.png", dpi=300, bbox_inches="tight")
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


def init_model(tokenizer, train_dataset, training_args, n_features):

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



    # Load the model, the model is a BertForMaskedLM model based on the Rostlab/prot_bert_bfd model
    # Our Idea is to 'fine tune' the ProtBERT model on MLM to further introduce the model to the peptide sequences instead
    # of the protein sequences. We hope to increase the binary classification performance by fine-tuning the model on MLM
    # first.
    elif training_args.model_class == 'mlm':
        config = BertConfig.from_pretrained(training_args.model_path)
        model = BertForMaskedLM.from_pretrained(training_args.model_path, config=config)
        # data_collator = PeptideCurriculumDataCollator(tokenizer=tokenizer,
        #                                               initial_prob=training_args.mlm_probability,
        #                                               increase_step=training_args.mlm_curriculum_increase_step,
        #                                               max_prob=training_args.mlm_curriculum_max_prob)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                                        mlm_probability=training_args.mlm_probability)

        # Add Curriculum Learning Callback if enabled
        # This callback can be adjusted if another metric for increasing/decreasing mlm_probability is needed
        # callback_list.append(CurriculumLearningCallback()) if training_args.mlm_curriculum_learning else ...
        run_metric = mlm_metrics
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
        run_metric = regression_metrics  # TODO implement regression metrics

    else:
        raise ValueError(
            f"binary_or_mlm must be either 'binary_dense', 'binary_conv' or 'mlm'. You provided: '{training_args.model_class}'")
    return data_collator, model, run_metric



def regression_plot(y_true, y_pred, path, logger, sequence_data=None):
    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))

    # Regression part
    slope, intercept, r_value, p_value, std_err = linregress(y_pred, y_true)
    # reg_equation = "y = {:.2f}x".format(slope)

    # calculate the residuals
    residuals = y_true - y_pred
    std_residuals = np.std(residuals)

    if sequence_data is not None:
        # Sort out the highest residuals and print them
        joined_info = zip(sequence_data, residuals)
        positive_extreme_peptides = sorted(joined_info, key=lambda x: x[1], reverse=True)[:30]
        joined_info = zip(sequence_data, residuals)
        negative_extreme_peptides = sorted(joined_info, key=lambda x: x[1], reverse=False)[:30]
        #highest_sequences = sorted_peptides[:50]
        # print the sequences to file
        with open(f"{path}/highest_residuals.csv", "w") as f:
            f.write("sequence;residual\n")
            for sequence in positive_extreme_peptides:
                f.write(f"{sequence[0]};{sequence[1]}\n")
            for sequence in negative_extreme_peptides:
                f.write(f"{sequence[0]};{sequence[1]}\n")


    logger.info(f"Steigung: {slope}, Standartabweichung der Residuen: {std_residuals} log(µM)")

    # generating residual plot
    sns.residplot(x=y_pred, y=residuals, ax=axs[1])
    axs[1].set_title(
        "Residuen gegen vorhergesagte Werte\nStandardabweichung der Residuen: {:.2f} log(µM)".format(std_residuals))
    axs[1].set_xlabel("Vorhergesagter Wert MHK/log(µM)")
    axs[1].set_ylabel("Residuum MHK/log(µM)")

    # Plot two red horizontal lines representing positive and negative standard deviations
    axs[1].axhline(std_residuals, color='red', linestyle='--')
    axs[1].axhline(-std_residuals, color='red', linestyle='--')

    # Plot manually added regression line with confidence interval
    y_pred_sorted = np.sort(y_pred)

    # generate scatter plot
    sns.regplot(x=y_pred, y=y_true, ax=axs[0], fit_reg=False)

    # plot the fitted line through the origin and also a line with slope 1 for comparison
    # axs[0].plot(y_pred_sorted, slope * y_pred_sorted, color='red')
    # standarf f(x) function for getting slope=1
    axs[0].plot([-1, 4], [-1, 4], linestyle='--', color='green', label='45-degree Line')

    # Calculate bounds for lines parallel to the regression line
    lower_bound = 1 * y_pred_sorted - std_residuals
    upper_bound = 1 * y_pred_sorted + std_residuals

    # reg_equation2 = f"y = {slope:.2f}x + {intercept:.2f}"
    # axs[0].text(0.05, 0.95, reg_equation2, transform=axs[0].transAxes, fontsize=12,
    #            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

    # Plot two lines parallel to the regression line representing positive and negative standard deviations
    axs[0].plot(y_pred_sorted, lower_bound, color='red', linestyle='--')
    axs[0].plot(y_pred_sorted, upper_bound, color='red', linestyle='--')

    axs[0].set_title("Tatsächliche gegen vorhergesagte Werte MHK/log(µM)")
    axs[0].set_xlabel("Vorhergesagter Wert MHK/log(µM)")
    axs[0].set_ylabel("Tatsächlicher Wert MHK/log(µM)")

    fig.suptitle("Regressions -und Residuenplot")
    plt.tight_layout()
    plt.savefig(f"{path}/regression_plot.png")
    plt.close()
    plt.clf()


def overall_stats(predictions, y_test, save_path, tag):


    # 1. Plotting the distribution of the target feature (y_test)
    plt.figure(figsize=(10, 6))
    sns.histplot(y_test, kde=True)
    plt.title('Verteilung der MIC-Werte (Test Set)')
    plt.xlabel('MIC (log10)')
    plt.ylabel('Häufigkeit')
    plt.savefig(save_path + f'/{tag}target_distribution.pdf')
    plt.close()
    plt.clf()

    # 2. Calculate variance of the target feature in the test set
    target_variance = np.var(y_test)
    print(f"Variance of the target feature (value) in test set: {target_variance}")


    # 4. Plotting Residuals in the test set
    residuals = y_test - predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Verteilung der Residuen (Test Set)')
    plt.xlabel('Residuen')
    plt.ylabel('Häufigkeit')
    plt.savefig(save_path + f'/{tag}residuals_distribution.pdf')
    plt.close()
    plt.clf()

    df = pd.DataFrame({
        "MIC (log$_{10}$ µM)": y_test,
        "Residuen (log$_{10}$ µM)": residuals
    })

    # Einteilung in 4 Quantile – du kannst q=5 oder q=[0,.25,.5,.75,1.] nehmen
    df["MIC-Quantil"] = pd.qcut(df["MIC (log$_{10}$ µM)"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

    plt.figure(figsize=(8, 4))
    sns.boxplot(x="MIC-Quantil", y="Residuen (log$_{10}$ µM)", data=df, color="skyblue")

    plt.xlabel("Quantile des tatsächlichen MIC-Werts (log$_{10}$ µM)")
    plt.ylabel("Residuen (tatsächlich – vorhergesagt) (log$_{10}$ µM)")
    plt.title("Residuenverteilung nach Quantilen des tatsächlichen MIC-Werts")
    plt.tight_layout()
    plt.savefig(save_path + f'/{tag}residuals_quantils.pdf')
    plt.close()
    plt.clf()

    # 5. Print MSE for comparison on the test set
    mse = np.mean((y_test - predictions) ** 2)
    print(f"Mean Squared Error (MSE) on Test Set: {mse}")


def best_f1_threshold(y_true, y_score, plot_path):
    p, r, thr = precision_recall_curve(y_true, y_score)

    display = PrecisionRecallDisplay.from_predictions(y_true, y_score, plot_chance_level=True, pos_label=1)
    _ = display.ax_.set_title("2-Klassen Precision-Recall Kurve")
    display.plot()
    plt.xlabel("Recall (Positive Klasse: 1)")
    plt.ylabel("Precision (Positive Klasse: 1)")
    plt.savefig(plot_path + '/precision_recall_curve.png')
    plt.close()
    plt.clf()


    f1 = 2 * p * r / (p + r + 1e-12)
    i = np.nanargmax(f1)
    # thresholds has length = len(p)-1; clamp index
    use_i = min(i, len(thr) - 1) if len(thr) > 0 else 0
    return (thr[use_i] if len(thr) else 0.5), f1[i], p[i], r[i]


def get_model_stats(model,
                    plot_dir: str,
                    predictions,
                    target_data,
                    logger: logging.Logger,
                    tag: str):
    #pred_train = model.predict(feature_data)

    r2, mse = print_regression_metrics(y_true=target_data, y_pred=predictions, logger=logger)

    # plot regression train
    plot_with_seaborn(y_true=target_data, y_pred=predictions, path=plot_dir + f"/{tag}_regression.pdf",
                      tag=f"{tag}")

    return r2, mse

def print_regression_metrics(y_true, y_pred, logger: logging.Logger):
    logger.info(f"Regression metrics: \n"
                f"    -> R2:  {r2_score(y_true=y_true, y_pred=y_pred):.5f}\n"
                f"    -> MAE: {mean_absolute_error(y_true=y_true, y_pred=y_pred):.5f}\n"
                f"    -> MSE: {mean_squared_error(y_true=y_true, y_pred=y_pred):.5f}\n"
                f"    -> VAR: {explained_variance_score(y_true=y_true, y_pred=y_pred):.5f}\n")
    return r2_score(y_true=y_true, y_pred=y_pred), mean_squared_error(y_true=y_true, y_pred=y_pred)



def plot_with_seaborn(y_true, y_pred, path, tag):
    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))

    # Regression part
    slope, intercept, r_value, p_value, std_err = linregress(y_pred, y_true)
    # reg_equation = "y = {:.2f}x".format(slope)

    # calculate the residuals
    residuals = y_true - y_pred
    std_residuals = np.std(residuals)

    print(f"Steigung: {slope}, Standartabweichung der Residuen: {std_residuals} log(µM) for {tag}")

    # generating residual plot
    sns.residplot(x=y_pred, y=residuals, ax=axs[1])
    axs[1].set_title(
        "Residuen gegen vorhergesagte Werte\nStandardabweichung der Residuen: {:.2f} log(µM)".format(std_residuals))
    axs[1].set_xlabel("Vorhergesagter Wert MIC/log(µM)")
    axs[1].set_ylabel("Residuum MHK/log(µM)")

    # Plot two red horizontal lines representing positive and negative standard deviations
    axs[1].axhline(std_residuals, color='red', linestyle='--')
    axs[1].axhline(-std_residuals, color='red', linestyle='--')

    # Plot manually added regression line with confidence interval
    y_pred_sorted = np.sort(y_pred)

    # generate scatter plot
    sns.regplot(x=y_pred, y=y_true, ax=axs[0], fit_reg=False)

    # plot the fitted line through the origin and also a line with slope 1 for comparison
    # axs[0].plot(y_pred_sorted, slope * y_pred_sorted, color='red')
    # standarf f(x) function for getting slope=1
    axs[0].plot([-1, 4], [-1, 4], linestyle='--', color='green', label='45-degree Line')

    # Calculate bounds for lines parallel to the regression line
    lower_bound = 1 * y_pred_sorted - std_residuals
    upper_bound = 1 * y_pred_sorted + std_residuals

    # reg_equation2 = f"y = {slope:.2f}x + {intercept:.2f}"
    # axs[0].text(0.05, 0.95, reg_equation2, transform=axs[0].transAxes, fontsize=12,
    #            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

    # Plot two lines parallel to the regression line representing positive and negative standard deviations
    axs[0].plot(y_pred_sorted, lower_bound, color='red', linestyle='--')
    axs[0].plot(y_pred_sorted, upper_bound, color='red', linestyle='--')

    axs[0].set_title("Tatsächliche gegen vorhergesagte Werte MIC/log(µM)")
    axs[0].set_xlabel("Vorhergesagter Wert MIC/log(µM)")
    axs[0].set_ylabel("Tatsächlicher Wert MIC/log(µM)")

    fig.suptitle("Regressions -und Residuenplot")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plt.clf()

if __name__ == '__main__':
    # data_leakage_wrapper()
    generate_custom_yaml_file(config_name="custom_config.yaml",
                              file_path='peptideBERT_configs/')
