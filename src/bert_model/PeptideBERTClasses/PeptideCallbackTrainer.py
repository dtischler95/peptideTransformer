import numpy as np
from transformers import TrainerCallback, TrainerState, TrainerControl
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score

from src.bert_model.PeptideBERTClasses.PeptideTrainingArguments import PeptideTrainingArguments


class LearningCurveCallback(TrainerCallback):
    """
    Custom Callback class for pretty logging and creating learning curves for accuracy and loss metrics during training and evaluation.
    logging_steps and plotting steps are synced to ensure that every point is updated when plotted to avoid straight lines in curves.
    """

    def __init__(self, args: PeptideTrainingArguments, interval=1, task_name='no_task_name_provided'):
        self.interval = interval
        self.task_name = task_name
        self.isTrain = True
        self.eval_accuracy_metrics = []
        self.eval_loss_metric = []
        self.train_loss_metric = []
        self.train_accuracy_metric = []
        self.label_0_counter = []
        self.label_1_counter = []
        # Maybe too much giving LearningCurveCallback the args, but I don't know how to do it better if I want to create a dir on init...
        os.makedirs(args.plot_path, exist_ok=True)

    def on_train_end(self, args: PeptideTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Set the mode to 'eval' when evaluating the model so the learning curves are plotted for the evaluation phase
        """
        self.isTrain = False

    def on_log(self, args: PeptideTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Logging metrics here only works because we set 'epoch' for logging_steps in the TrainingArguments.
        If we want to log steps we would need to adjust this logging behavior in Callbacks.
        Is there a way to get the metrics for Learning Curves at a more robust step in Training?
        """

        logs = kwargs.get("logs", {})
        current_loss = logs.get("loss")
        if current_loss is None:
            return

        self.train_loss_metric.append(current_loss)



    def on_evaluate(self, args: PeptideTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Log the metrics and plot the learning curves on every logging step
        """

        logs = kwargs.get("metrics", {})
        current_loss = logs.get("eval_loss")
        current_accuracy = logs.get("eval_accuracy")['accuracy']
        label_0_count = logs.get("eval_label_0_count_on_epoch_end")
        label_1_count = logs.get("eval_label_1_count_on_epoch_end")

        if current_accuracy is None or current_loss is None:
            return

        self.eval_accuracy_metrics.append(current_accuracy)
        self.eval_loss_metric.append(current_loss)
        self.label_0_counter.append((label_0_count / (label_0_count + label_1_count)) * 100)
        self.label_1_counter.append((label_1_count / (label_0_count + label_1_count)) * 100)

        if state.epoch % self.interval == 0 and len(self.eval_accuracy_metrics) > 1 and self.isTrain:
            if len(self.eval_accuracy_metrics) != len(self.eval_loss_metric):
                # Just for prevent bugs. im understanding more and more, but I still don't trust the on_eval call [when and how is it called??]
                raise ValueError("The length of the accuracy and loss metrics must be equal BUG!")
            self.plot_learning_curves(plot_path=args.plot_path)

            if len(self.label_0_counter) > 1:
                self.plot_label_abundance(plot_path=args.plot_path)

    def plot_label_abundance(self, plot_path: str):
        """
        Plot the abundance of label classes in the predictions per Epoch.

        """

        max_labels = self.label_0_counter[0] + self.label_1_counter[0]

        epochs = range(1, len(self.label_0_counter) + 1)
        plt.figure(figsize=(10, 5))


        plt.plot(epochs, self.label_0_counter, label='Label 0', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Label 0 [%]', color='green')
        plt.tick_params(axis='y', labelcolor='green')
        plt.ylim(0, max_labels)
        plt.xticks(epochs)
        plt.xlim(1, len(self.label_0_counter))
        label_0_counter_np = np.array(self.label_0_counter)
        plt.fill_between(epochs, self.label_0_counter, max_labels, where=(label_0_counter_np <= max_labels),
                         color='red', alpha=0.3)

        plt.fill_between(epochs, self.label_0_counter, 0, where=(label_0_counter_np >= 0), color='green', alpha=0.3)

        plt.savefig(f"{plot_path}/{self.task_name}_label_abundance.png")


    def plot_learning_curves(self, plot_path: str):
        """
        Plots a learning curve for the accuracy and loss metrics on every logging step
        """
        epochs = range(1, len(self.eval_accuracy_metrics) + 1)
        plt.figure(figsize=(10, 5))

        # Plot accuracy on the primary y-axis
        plt.plot(epochs, self.eval_accuracy_metrics, label='Accuracy', color='blue')
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
        plt.title(f'Learning Curves for {self.task_name} task')
        plt.legend()
        ax2.legend()

        # Save the figure
        plt.savefig(os.path.join(plot_path, f"{self.task_name}_learning_curves.png"))
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


class CurriculumLearningCallback(TrainerCallback):
    """
    Idea so far, make a callback "on_evaluate" or "on_epoch_begin" that changes the training data for the next curriculum step
    A curriculum step is not defined for me so far. It could be something like every 10 Epochs. Need to do some more research on this.
    """
    ...
