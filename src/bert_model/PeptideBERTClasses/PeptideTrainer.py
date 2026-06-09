from typing import Union, Optional, Dict, Any
import torch
from torch import nn
from transformers import Trainer, PreTrainedModel
from src.bert_model.PeptideBERTClasses.PeptideTrainingArguments import PeptideTrainingArguments
from src.bert_model.fine_tune_utils import format_logit_to_label



class PeptideTrainer(Trainer):
    """
    Custom Trainer class with overridden methods.
    """

    def __init__(self,
                 model: Union[PreTrainedModel, nn.Module] = None,
                 args: PeptideTrainingArguments = None,
                 **kwargs):
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

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        """

        # Custom Callback logic to fetch the batch-wise accuracy and calculate a dataset wide mean accuracy
        if 'grad_norm' in logs.keys():
            for callback in self.callback_handler.callbacks:
                if callback.__class__.__name__ == 'CollectBatchWiseTrainMetrics':
                    if self.args.model_class.startswith('regression'):
                        logs['MSE'] = round(float(callback.get_train_mse()), 4)
                        logs['MAE'] = round(float(callback.get_train_mae()), 4)
                        logs['r2'] = round(float(callback.get_train_r2()), 4)
                    if self.args.model_class.startswith('binary'):
                        logs['accuracy'] = round(callback.get_train_accuracy(), 4)
                        logs['precision'] = round(callback.get_train_precision(), 4)
                        logs['MCC'] = round(callback.get_train_mcc(), 4)
                        logs['recall'] = round(callback.get_train_recall(), 4)
                        logs['f1'] = round(callback.get_train_f1(), 4)
                        logs['ROC-AUC'] = round(float(callback.get_train_auroc()), 4)
                        logs['average_precision'] = round(float(callback.get_train_average_precision()), 4)

                    callback.clear_results_after_epoch()

        super().log(logs, start_time)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],
                      num_items_in_batch=None) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            num_items_in_batch (Optional[int]):
                The number of items in the batch. If `None`, the number of items is determined by the first input
                tensor with a `size(0)` attribute.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """

        loss = super().training_step(model, inputs, num_items_in_batch)

        inputs = self._prepare_inputs(inputs)

        train_metric_callback = None
        for callback in self.callback_handler.callbacks:
            if callback.__class__.__name__ == 'CollectBatchWiseTrainMetrics':
                train_metric_callback = callback

        if train_metric_callback is None:
            raise ValueError("CollectBatchWiseTrainMetrics Callback not found. This is a critical error.")

        if "labels" in inputs:
            if self.args.model_class.startswith('binary'):

                preds = model(**inputs)[1].detach().cpu().numpy()
                cpu_inputs = inputs["labels"].detach().cpu().numpy()
                pred_labels = format_logit_to_label(logits=preds)
                train_metric_callback.append_batch_results(predictions=pred_labels, labels=cpu_inputs, probabilities=preds.flatten())

            elif self.args.model_class == 'regression':
                # Extract the logits for regression
                preds = model(**inputs)[1].detach().cpu().numpy()
                cpu_inputs = inputs["labels"].detach().cpu().numpy()
                train_metric_callback.append_batch_results(predictions=preds.squeeze(), labels=cpu_inputs)

        # Apply gradient norm clipping in case of exploding gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        return loss
