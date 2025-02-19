from typing import Union, Optional, Dict, Any
from datasets import Dataset
import torch
import evaluate
from sklearn.metrics import matthews_corrcoef
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import Trainer, PreTrainedModel
from transformers.utils.import_utils import is_datasets_available
from transformers.trainer_utils import seed_worker
from src.bert_model.PeptideBERTClasses.PeptideTrainingArguments import PeptideTrainingArguments


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
                    logs['accuracy'] = callback.get_batch_wise_mean_accuracy()
                    if self.args.model_class == 'binary':
                        logs['mcc'] = callback.get_batch_wise_mean_mcc()
                        logs['f1'] = callback.get_batch_wise_mean_f1()
                        logs['recall'] = callback.get_batch_wise_mean_recall()
                        logs['precision'] = callback.get_batch_wise_mean_precision()
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
        # TODO is there any way to use the compute_metric function here??

        loss = super().training_step(model, inputs, num_items_in_batch)

        inputs = self._prepare_inputs(inputs)

        train_metric_callback = None
        for callback in self.callback_handler.callbacks:
            if callback.__class__.__name__ == 'CollectBatchWiseTrainMetrics':
                train_metric_callback = callback

        if train_metric_callback is None:
            raise ValueError("CollectBatchWiseTrainMetrics Callback not found. This is a critical error.")

        if "labels" in inputs:
            if self.args.model_class == 'binary':

                preds = model(**inputs)[1].detach().cpu().numpy()
                cpu_inputs = inputs["labels"].detach().cpu().numpy()
                acc = evaluate.load("accuracy").compute(predictions=preds.argmax(axis=1), references=cpu_inputs)["accuracy"]
                mcc = matthews_corrcoef(y_true=cpu_inputs, y_pred=preds.argmax(axis=1))
                f1 = evaluate.load("f1").compute(predictions=preds.argmax(axis=1), references=cpu_inputs)["f1"]
                precision = evaluate.load("precision").compute(predictions=preds.argmax(axis=1),
                                                                references=cpu_inputs,
                                                               zero_division=0)["precision"]
                recall = evaluate.load("recall").compute(predictions=preds.argmax(axis=1),
                                                          references=cpu_inputs)["recall"]
                train_metric_callback.append_batch_wise_accuracy(acc)
                train_metric_callback.append_batch_wise_mcc(mcc)
                train_metric_callback.append_batch_wise_f1(f1)
                train_metric_callback.append_batch_wise_precision(precision)
                train_metric_callback.append_batch_wise_recall(recall)
            elif self.args.model_class == 'mlm':
                # Extract the logits for the masked tokens
                preds = model(**inputs).logits.detach().cpu().numpy()

                # Get the indices of the masked tokens
                masked_indices = inputs["labels"].detach().cpu().numpy() != -100

                # Get the predictions for the masked tokens
                masked_preds = preds[masked_indices].argmax(axis=1)

                # Get the true labels for the masked tokens
                masked_labels = inputs["labels"].detach().cpu().numpy()[masked_indices]

                # Calculate the accuracy
                acc = evaluate.load("accuracy").compute(predictions=masked_preds.flatten(), references=masked_labels.flatten())["accuracy"]
                
                train_metric_callback.append_batch_wise_accuracy(acc)
            elif self.args.model_class == 'custom':
                """
                Implement Regression Metrics here
                """
                ...




        # Apply gradient norm clipping in case of exploding gradients.
        # Observed while training mlm with large train data points.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        return loss

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        if self.args.classification_weighted_labels:

            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            train_dataset = self.train_dataset
            data_collator = self.data_collator
            if is_datasets_available() and isinstance(train_dataset, Dataset):
                train_dataset = self._remove_unused_columns(train_dataset, description="training")
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

            weights = self.get_weighted_labels(train_dataset)

            # Create a weighted sampler
            sampler = WeightedRandomSampler(weights, len(weights))

            dataloader_params = {
                "batch_size": self._train_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
                "sampler": sampler,
                "drop_last": self.args.dataloader_drop_last,
                "worker_init_fn": seed_worker,
                "prefetch_factor": self.args.dataloader_prefetch_factor,
            }
            return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        else:
            return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if self.args.classification_weighted_labels:

            if eval_dataset is None and self.eval_dataset is None:
                raise ValueError("Trainer: evaluation requires an eval_dataset.")

            # If we have persistent workers, don't do a fork bomb especially as eval datasets
            # don't change during training
            dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
            if (
                    hasattr(self, "_eval_dataloaders")
                    and dataloader_key in self._eval_dataloaders
                    and self.args.dataloader_persistent_workers
            ):
                return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

            eval_dataset = (
                self.eval_dataset[eval_dataset]
                if isinstance(eval_dataset, str)
                else eval_dataset
                if eval_dataset is not None
                else self.eval_dataset
            )
            data_collator = self.data_collator

            if is_datasets_available() and isinstance(eval_dataset, Dataset):
                eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

            weights = self.get_weighted_labels(eval_dataset)

            # Create a weighted sampler
            sampler = WeightedRandomSampler(weights, len(weights))

            dataloader_params = {
                "batch_size": self.args.eval_batch_size,
                "collate_fn": data_collator,
                "sampler": sampler,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
                "drop_last": self.args.dataloader_drop_last,
                "prefetch_factor": self.args.dataloader_prefetch_factor,
            }

            # accelerator.free_memory() will destroy the references, so
            # we need to store the non-prepared version
            eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
            if self.args.dataloader_persistent_workers:
                if hasattr(self, "_eval_dataloaders"):
                    self._eval_dataloaders[dataloader_key] = eval_dataloader
                else:
                    self._eval_dataloaders = {dataloader_key: eval_dataloader}

            return self.accelerator.prepare(eval_dataloader)
        else:
            return super().get_eval_dataloader()

    @staticmethod
    def get_weighted_labels(train_dataset):
        # Calculate class weights
        labels = torch.tensor(train_dataset.labels)
        class_counts = torch.bincount(labels)
        class_weights = 1.0 / class_counts.float()
        weights = class_weights[labels]
        return weights
