from typing import Union, Optional

from torch import nn
from transformers import Trainer, PreTrainedModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.fine_tune.PeptideBERTClasses.PeptideTrainingArguments import PeptideTrainingArguments


class PeptideTrainer(Trainer):
    """
    Custom Trainer class with overridden methods.
    """
    def __init__(self, model: Union[PreTrainedModel, nn.Module]=None, args: PeptideTrainingArguments = None, **kwargs):
        """
        Needed so we don't get type hint Errors from the Trainer Class not recognizing the PeptideTrainingArguments class
        I see no further init logic so far
        """
        super().__init__(model=model, args=args, **kwargs)
        self.args = args
        # Additional initialization if needed

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        We constantly got the error message "RuntimeError: input is not contiguous" when saving the model. That's a Quickfix here

        Args:
            output_dir (Optional[str]): The directory where the model should be saved.
            state_dict: The state_dict to save. If None, the model's state_dict is saved.
        """
        # Ensure that all model parameters are contiguous before saving
        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.contiguous()
        super()._save(output_dir=output_dir, state_dict=state_dict)

    def log(self, logs):
        """
        The log function just prints the metric dictionary in a more readable format.

        Args:
            logs (Dict): The dictionary containing the metrics to be logged.

        """
        # Custom logging logic
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        output = {**logs, **{"step": self.state.global_step}}

        # Custom print format
        for key, value in output.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    print(f"{sub_key.capitalize()}: {sub_value}")
            else:
                print(f"{key.capitalize()}: {value}")

        # Call the original log method to ensure other logging mechanisms are still in place
        super().log(logs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Create an AdamW optimizer and a ReduceLROnPlateau scheduler for the Training Pipeline.
        """

        # TODO make this dynamically to switch to default by invoking super() method if i want to change stuff later example adam with weight decay
        if self.optimizer is None:
            self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)

        if self.lr_scheduler is None:
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer,
                                                  mode=self.args.lro_mode,
                                                  factor=self.args.lro_factor,
                                                  patience=self.args.lro_patience,
                                                  min_lr=self.args.lro_min_lr)
            self._created_lr_scheduler = True
