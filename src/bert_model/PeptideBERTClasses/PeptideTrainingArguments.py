from typing import Literal

from transformers import TrainingArguments
import sys
from pathlib import Path


class PeptideTrainingArguments(TrainingArguments):
    """
    Class for adjusting TrainingArguments for PeptideBERT to our needs and collecting all necessary arguments in one place.
    I tried to stick to commonly used values for the parameters, but feel free to adjust them to your needs.
    """

    def __init__(self, *args,
                 train_file=None,
                 model_path: str = 'Rostlab/prot_bert_bfd',
                 model_save_path: str = '../default_path_BERT',
                 plot_path: str = '../../plots',
                 ignore_leakage: bool = False,
                 mlm_probability: float = 0.15,
                 max_length: int = 36,
                 fast_debug_mode: bool = False,
                 validation_data_size: float = 0.2,
                 test_data_size: float = 0.2,
                 early_stopping_patience: int = 30,
                 early_stop_metric: str = 'eval_loss',
                 early_stop_mode: Literal['min', 'max'] = 'min',
                 early_stop_warm_up: int = 0,
                 **kwargs):
        """
        Custom Init for the Training Arguments to adjust behavior to our needs.
        Added some Parameters convenient for tracking here and also added ReduceLROnPlateau Callback parameters, which
        is implemented inside our Custom PeptideTrainer class.

        :param train_file: Path to the training file
        :param model_path: Path to the model to be used. Can be huggingFace Repository or local path
        :param model_save_path: Path to save the model to
        :param plot_path: Path to save the plots to
        :param ignore_leakage: Ignore leakage in the training data
        :param mlm_probability: Probability of masking tokens in the input (only used for MLM)
        :param max_length: Maximum length of the input sequence
        :param fast_debug_mode: Cut the dataset to 500 samples for faster debugging (development only)
        :param validation_data_size: Fraction of the data to be used for validation
        :param test_data_size: Fraction of the data to be used for testing (this will take a faction of the validation data)
        :param early_stopping_patience: Number of epochs with no improvement after which training will be stopped
        :param early_stop_metric: Metric to watch for early stopping
        :param early_stop_mode: One of min, max. In min mode, training will be stopped when the metric stops decreasing; in max mode it will be stopped when the metric stops increasing
        :param early_stop_warm_up: Number of epochs to wait before starting to watch for early stopping
        :param kwargs: Additional arguments
        """
        super().__init__(*args, **kwargs)
        self.train_file = train_file
        if self.train_file is None:
            # TODO may raise an error here and make a different inference function
            print(
                f"\033[31m[warning] If you use model for Inference you can ignore this warning. Otherwise, you should provide a train_file in the config file.\033[0m")
        self.model_path = model_path
        self.model_save_path = model_save_path
        self.plot_path = plot_path
        # Create the model and plot save path if it does not exist
        Path(self.model_save_path).mkdir(parents=True, exist_ok=True)
        Path(self.plot_path).mkdir(parents=True, exist_ok=True)
        self.ignore_leakage = ignore_leakage
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        self.fast_debug_mode = fast_debug_mode
        if self.fast_debug_mode:
            print(
                f"\033[31m[warning] fast_debug_mode is set to True. This will cut the dataset to 100 samples. Set only to True if you want to test functions!\033[0m")
        self.validation_data_size = validation_data_size
        self.test_data_size = test_data_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stop_metric = early_stop_metric
        self.early_stop_mode = early_stop_mode
        self.early_stop_warm_up = early_stop_warm_up
