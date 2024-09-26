from transformers import Trainer
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class PeptideTrainer(Trainer):
    """
    Custom Trainer class with overridden methods.


    """

    def _save(self, output_dir: str):
        """
        We constantly got the error message "RuntimeError: input is not contiguous" when saving the model. That's a Quickfix here

        Args:
            output_dir (str): The output directory where the model should be saved.
        """
        # Ensure that all model parameters are contiguous before saving
        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.contiguous()
        super()._save(output_dir)

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
        Create the optimizer and learning rate scheduler.
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
