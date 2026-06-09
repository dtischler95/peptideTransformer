from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parents[3]

from transformers import TrainingArguments


class PeptideTrainingArguments(TrainingArguments):
    """
    Class for adjusting TrainingArguments for PeptideBERT to our needs and collecting all necessary arguments in one place.
    I tried to stick to commonly used values for the parameters, but feel free to adjust them to your needs.
    """

    def __init__(self, *args,
                 model_class: str,
                 train_file=None,
                 model_path: str = 'Rostlab/prot_bert_bfd',
                 model_save_path: str = str(_REPO_ROOT / 'default_path_BERT'),
                 plot_path: str = str(_REPO_ROOT / 'plots'),
                 ignore_leakage: bool = False,
                 max_length: int = 36,
                 fast_debug_mode: bool = False,
                 early_stopping_patience: int = 30,
                 early_stop_metric: str = 'eval_loss',
                 early_stop_mode: Literal['min', 'max'] = 'min',
                 early_stop_warm_up: int = 0,
                 label_0_cluster_data: int = 500,
                 label_1_cluster_data: int = 500,
                 loss_function: str = 'bce',
                 data_shuffle: bool = True,
                 add_features: bool = False,
                 **kwargs):
        """
        Custom Init for the Training Arguments to adjust behavior to our needs.
        Added some Parameters convenient for tracking here and also added ReduceLROnPlateau Callback parameters, which
        is implemented inside our Custom PeptideTrainer class.

        :param model_class: One of binary_dense, regression.
        :param train_file: Path to the training file
        :param model_path: Path to the model to be used. Can be huggingFace Repository or local path
        :param model_save_path: Path to save the model to
        :param plot_path: Path to save the plots to
        :param ignore_leakage: Ignore leakage in the training data
        :param max_length: Maximum length of the input sequence
        :param fast_debug_mode: Cut the dataset to 500 samples for faster debugging (development only)
        :param early_stopping_patience: Number of epochs with no improvement after which training will be stopped
        :param early_stop_metric: Metric to watch for early stopping
        :param early_stop_mode: One of min, max. In min mode, training will be stopped when the metric stops decreasing; in max mode it will be stopped when the metric stops increasing
        :param early_stop_warm_up: Number of epochs to wait before starting to watch for early stopping
        :param label_0_cluster_data: Number of samples to cluster for label 0
        :param label_1_cluster_data: Number of samples to cluster for label 1
        :param loss_function: Loss function to be used for the model
        :param add_features: Use concentration as input for the model
        :param kwargs: Additional arguments
        """
        super().__init__(*args, **kwargs)
        self.model_class = model_class
        self.train_file = train_file
        self.model_path = model_path
        self.model_save_path = model_save_path
        self.plot_path = plot_path
        # Create the model and plot_save_path if it does not exist
        Path(self.model_save_path).mkdir(parents=True, exist_ok=True)
        Path(self.plot_path).mkdir(parents=True, exist_ok=True)
        self.ignore_leakage = ignore_leakage
        self.max_length = max_length
        self.fast_debug_mode = fast_debug_mode
        self.early_stopping_patience = early_stopping_patience
        self.early_stop_metric = early_stop_metric
        self.early_stop_mode = early_stop_mode
        self.early_stop_warm_up = early_stop_warm_up
        self.label_0_cluster_data = label_0_cluster_data
        self.label_1_cluster_data = label_1_cluster_data
        self.loss_function = loss_function
        self.add_features = add_features
        if self.fast_debug_mode:
            print(
                f"\033[31m[warning] fast_debug_mode is set to True. This will cut the dataset to 100 samples. Set only to True if you want to test functions!\033[0m")
        if self.train_file is None:
            print(
                f"\033[31m[warning] If you use model for Inference you can ignore this warning. Otherwise, you should provide a train_file in the config file.\033[0m")
