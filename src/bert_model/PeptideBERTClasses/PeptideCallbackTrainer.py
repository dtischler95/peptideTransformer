import os

import evaluate
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, \
    average_precision_score
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

from src.bert_model.PeptideBERTClasses.PeptideTrainingArguments import PeptideTrainingArguments


class LearningCurveCallback(TrainerCallback):
    """
    Custom Callback class for pretty logging and creating learning curves for accuracy and loss metrics during training and evaluation.
    Logging_steps and plotting steps are synced to ensure that every point is updated when plotted to avoid straight lines in curves.
    This is just for combining some metrics in one graph. Every single graph will be plotted in a separate callback.
    """

    def __init__(self, plot_path: str):
        self.eval_accuracy_metrics = []
        self.eval_loss_metric = []
        self.train_loss_metric = []
        self.train_accuracy_metric = []
        # Maybe too much giving LearningCurveCallback the args, but I don't know how to do it better if I want to create a dir on init...
        os.makedirs(plot_path, exist_ok=True)

    def on_log(self, args: PeptideTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Log the metrics and plot the learning curves on every logging step
        I think I should miss the last epoch metrics either when logging on epoch begin or on epoch end
        """

        logs = state.log_history

        # scraping the metrics from the logs
        # eval metrics are always the last ones while the third last are the train metrics
        # scraping them every epoch to update the learning curves by storing the values
        metric_to_get = "r2" if args.model_class.endswith('regression') else "accuracy"
        if "eval_loss" in logs[-1]:
            self.eval_accuracy_metrics.append(logs[-1].get(f"eval_{metric_to_get}"))
            self.eval_loss_metric.append(logs[-1].get("eval_loss"))
            self.plot_learning_curves(args=args, axis_label=metric_to_get)
        else:
            self.train_accuracy_metric.append(logs[-1].get(metric_to_get))
            self.train_loss_metric.append(logs[-1].get("loss"))

    def plot_learning_curves(self, args: PeptideTrainingArguments, axis_label: str):
        """
        Plots a learning curve for the accuracy and loss metrics on every logging step
        """
        epochs = range(1, len(self.eval_accuracy_metrics) + 1)
        plt.figure(figsize=(10, 5))

        # Plot accuracy on the primary y-axis
        plt.plot(epochs, self.eval_accuracy_metrics, label=f"Eval_{axis_label}", color='blue')
        plt.plot(epochs, self.train_accuracy_metric, label=f"Train_{axis_label}", color='yellow')
        plt.xlabel('Epochs')
        plt.ylabel(f"Eval_{axis_label}", color='blue')
        plt.tick_params(axis='y', labelcolor='blue')

        # Create a second y-axis for loss
        ax2 = plt.gca().twinx()  # Get the current axes and create a twin y-axis
        ax2.plot(epochs, self.eval_loss_metric, label='Eval_loss', color='red')
        ax2.plot(epochs, self.train_loss_metric, label='Train_loss', color='green')
        ax2.set_ylabel('Eval_loss', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.xticks(epochs)

        # Set the title and legend
        plt.title(f'Learning Curves for {args.model_class} task')
        plt.legend()
        ax2.legend()

        # Save the figure
        plt.savefig(os.path.join(args.plot_path, f"{args.model_class}_learning_curves.png"))
        plt.close()


class PlotMetricsCallback(TrainerCallback):
    class MetricCatcher:
        """
        Simple class to store train and eval Metrics for a given metric_name
        """

        def __init__(self, metric_name):
            self.metric_name = metric_name
            self.train_metric = []
            self.eval_metric = []

    def __init__(self):
        self.object_list = []  # stores an object instance of MetricCatcher for every metric to plot
        self.metrics_to_plot = []  # Stores the metrics to plot based on the first train log given

    def on_log(self, args: PeptideTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logs = state.log_history

        # ---------------------------------------------------------------------------------------------------
        # This will be only executed once to initialize all data storages for the metrics
        # Initialize the metrics to plot based on the first log
        if not self.metrics_to_plot:
            self.metrics_to_plot = [metric for metric in logs[0] if metric != "epoch" and metric != "step"]

        # Initialize the object list based on the metrics to plot
        if not self.object_list:
            self.object_list = [self.MetricCatcher(metric) for metric in self.metrics_to_plot]
        # ---------------------------------------------------------------------------------------------------

        for obj in self.object_list:
            if "eval_loss" in logs[-1]:  # TODO is there a better way to check if there are eval metrics?
                obj.eval_metric.append(logs[-1].get("eval_" + obj.metric_name))
                self.plot_metric_curve(args=args,
                                       train_metrics=obj.train_metric,
                                       eval_metrics=obj.eval_metric,
                                       metric_name=obj.metric_name)
            else:
                obj.train_metric.append(logs[-1].get(obj.metric_name))

    @staticmethod
    def plot_metric_curve(args: PeptideTrainingArguments,
                          train_metrics: list,
                          eval_metrics: list,
                          metric_name: str):

        file_name = args.train_file.split('/')[-1]

        if file_name.startswith("acineto"):
            file_name = "Acinetobacter baumannii"
        elif file_name.startswith("bacillus"):
            file_name = "Bacillus subtilis"
        elif file_name.startswith("candida"):
            file_name = "Candida albicans"
        elif file_name.startswith("enterobacter"):
            file_name = "Enterobacter sp."
        elif file_name.startswith("enterococc"):
            file_name = "Enterococcus faecalis"
        elif file_name.startswith("escher"):
            file_name = "Escherichia coli"
        elif file_name.startswith("klebsie"):
            file_name = "Klebsiella pneumoniae"
        elif file_name.startswith("micro"):
            file_name = "Micrococcus luteus"
        elif file_name.startswith("pseudo"):
            file_name = "Pseudomonas aeruginosa"
        elif file_name.startswith("salmonella"):
            file_name = "Salmonella enterica"
        elif file_name.startswith("staphylococcus_aureus"):
            file_name = "Staphylococcus aureus"
        elif file_name.startswith("staphylococcus_epi"):
            file_name = "Staphylococcus epidermidis"
        elif file_name.startswith('happen_style'):
            file_name = 'Schwellenwert-Datensatz'
        else:
            file_name = 'WhiteLab-Datensatz'

        epochs = range(1, len(eval_metrics) + 1)
        plt.figure(figsize=(10, 5))

        plt.plot(epochs, eval_metrics, label=f"Validierungs {metric_name}", color='blue')
        plt.plot(epochs, train_metrics, label=f"Trainings {metric_name}", color='red')
        plt.xlabel('Epoche')
        plt.ylabel(metric_name, color='blue')
        plt.tick_params(axis='y', labelcolor='blue')

        plt.xticks(epochs)

        plt.title(f"{metric_name}\nDaten: {file_name}")
        plt.legend()

        plt.savefig(os.path.join(args.plot_path, f"{args.model_class}_{metric_name}_curves.png"))
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
        """

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


class CollectBatchWiseTrainMetrics(TrainerCallback):
    """
    Needed so the Trainer calculates the metrics on the training dataset as well.
    Solution taken from:
    https://discuss.huggingface.co/t/metrics-for-training-set-in-trainer/2461/4
    But instead of overwriting the compute_metrics function, I overwrote train_step and passed all the
    necessary metrics to this callback.
    This Callback is absolutely Needed for this script to work. Since Learning Curves and value passing
    rely on this callback.
    A batch_wise_mean method is NEEDED to reset epoch counter for mean calculation.
    This method NEEDS to be called in the log Function of the PeptideTrainer subclass.
    """

    def __init__(self) -> None:
        self.collected_predictions = []
        self.collected_labels = []
        self.collected_probabilities = []

    def append_batch_results(self, predictions, labels, probabilities=None):
        self.collected_predictions.extend(predictions)
        self.collected_labels.extend(labels)
        if probabilities is not None:
            self.collected_probabilities.extend(probabilities)

    def clear_results_after_epoch(self):
        self.collected_predictions = []
        self.collected_labels = []
        self.collected_probabilities = []

    def get_train_average_precision(self):
        return average_precision_score(y_true=self.collected_labels, y_score=self.collected_probabilities)

    def get_train_auroc(self):
        return roc_auc_score(y_true=self.collected_labels, y_score=self.collected_probabilities)

    def get_train_mcc(self):
        return matthews_corrcoef(y_true=self.collected_labels, y_pred=self.collected_predictions)

    def get_train_accuracy(self):
        return \
        evaluate.load("accuracy").compute(predictions=self.collected_predictions, references=self.collected_labels)[
            "accuracy"]

    def get_train_precision(self):
        return \
        evaluate.load("precision").compute(predictions=self.collected_predictions, references=self.collected_labels)[
            "precision"]

    def get_train_recall(self):
        return \
        evaluate.load("recall").compute(predictions=self.collected_predictions, references=self.collected_labels)[
            "recall"]

    def get_train_f1(self):
        return evaluate.load("f1").compute(predictions=self.collected_predictions, references=self.collected_labels)[
            "f1"]

    def get_train_mse(self):
        return mean_squared_error(self.collected_labels, self.collected_predictions)

    def get_train_mae(self):
        return mean_absolute_error(self.collected_labels, self.collected_predictions)

    def get_train_r2(self):
        return r2_score(self.collected_labels, self.collected_predictions)


class CurriculumLearningCallback(TrainerCallback):
    """
    Idea so far, make a callback "on_evaluate" or "on_epoch_begin" that changes the training data for the next curriculum step
    A curriculum step is not defined for me so far. It could be something like every 10 Epochs. I need to do some more research on this.
    """

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Update the masking percentage for the curriculum MLM task.
        Conditions can be added here for adjusting the update algorithm.
        """
        if state.epoch % 20 == 0:
            kwargs['train_dataloader'].base_dataloader.collate_fn.data_collator.update_probability()
