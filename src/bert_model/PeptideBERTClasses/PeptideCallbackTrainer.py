import os
from copy import deepcopy

import torch
import evaluate
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
import matplotlib.pyplot as plt
from src.bert_model.PeptideBERTClasses.PeptideTrainingArguments import PeptideTrainingArguments
from src.bert_model.fine_tune_utils import plot_label_abundance, format_logit_to_label


class LearningCurveCallback(TrainerCallback):
    """
    Custom Callback class for pretty logging and creating learning curves for accuracy and loss metrics during training and evaluation.
    logging_steps and plotting steps are synced to ensure that every point is updated when plotted to avoid straight lines in curves.
    """

    def __init__(self, args: PeptideTrainingArguments):
        self.args = args # this could be done smoother
        self.eval_accuracy_metrics = []
        self.train_accuracy_metric = []
        self.eval_loss_metric = []
        self.train_loss_metric = []
        # Maybe too much giving LearningCurveCallback the args, but I don't know how to do it better if I want to create a dir on init...
        os.makedirs(args.plot_path, exist_ok=True)


    def on_epoch_end(self, args: PeptideTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Log the metrics and plot the learning curves on every logging step
        """

        logs = state.log_history

        # I dont know why, but the logs are empty on the first epoch
        if len(logs) == 0:
            return

        # scraping the metrics from the logs
        # eval metrics are always the last ones while the third last are the train metrics
        # scraping them every epoch to update the learning curves by storing the values
        self.eval_accuracy_metrics.append(logs[-1].get("eval_accuracy"))
        self.eval_loss_metric.append(logs[-1].get("eval_loss"))
        self.train_accuracy_metric.append(logs[-3].get("train_accuracy"))
        self.train_loss_metric.append(logs[-3].get("train_loss"))

        self.plot_learning_curves()



    def plot_learning_curves(self):
        """
        Plots a learning curve for the accuracy and loss metrics on every logging step
        """
        epochs = range(1, len(self.eval_accuracy_metrics) + 1)
        plt.figure(figsize=(10, 5))

        # Plot accuracy on the primary y-axis
        plt.plot(epochs, self.eval_accuracy_metrics, label='Accuracy', color='blue')
        plt.plot(epochs, self.train_accuracy_metric, label='Train Accuracy', color='yellow')
        plt.xlabel('Epochs')
        plt.ylabel('Eval Accuracy', color='blue')
        plt.tick_params(axis='y', labelcolor='blue')

        # Create a second y-axis for loss
        ax2 = plt.gca().twinx()  # Get the current axes and create a twin y-axis
        ax2.plot(epochs, self.eval_loss_metric, label='Eval Loss', color='red')
        ax2.plot(epochs, self.train_loss_metric, label='Train Loss', color='green')
        ax2.set_ylabel('Loss', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Set the title and legend
        plt.title(f'Learning Curves for {self.args.model_class} task')
        plt.legend()
        ax2.legend()

        # Save the figure
        plt.savefig(os.path.join(self.args.plot_path, f"{self.args.model_class}_learning_curves.png"))
        plt.close()


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self):
        """
        Callback Class for early stopping based on a given metric. Choose min for loss and max for accuracy.
        """
        self.best_metric = None
        self.num_bad_epochs = 0

    def on_evaluate(self, args: PeptideTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Needs to be on_evaluate since there will be the calculations of those metrics.
        BUG -> It seems that early_stop_warmup affects updates of eval_metrics somehow.... Idk
        """

        # TODO add warmup
        if state.epoch < args.early_stop_warm_up:
            return

        logs = kwargs.get("metrics", {})
        current_metric = logs.get(args.early_stop_metric)

        if current_metric is None:
            return

        if self.best_metric is None or \
                (args.early_stop_mode == "min" and current_metric < self.best_metric) or \
                (args.early_stop_mode == "max" and current_metric > self.best_metric):
            self.best_metric = current_metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= args.early_stopping_patience:
            control.should_training_stop = True
            print(
                f"Early stopping triggered. No improvement in {args.early_stop_metric} for {args.early_stopping_patience} evaluations.")


class EnableTrainMetricPrints(TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

class CurriculumLearningCallback(TrainerCallback):
    """
    Idea so far, make a callback "on_evaluate" or "on_epoch_begin" that changes the training data for the next curriculum step
    A curriculum step is not defined for me so far. It could be something like every 10 Epochs. Need to do some more research on this.
    """
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Update the masking percentage for the curriculum MLM task.
        Conditions can be added here for adjusting the update algorithm.
        """
        if state.epoch % 20 == 0:
            kwargs['train_dataloader'].base_dataloader.collate_fn.data_collator.update_probability()
