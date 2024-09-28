from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
import matplotlib.pyplot as plt
import os


class LearningCurveCallback(TrainerCallback):
    """
    Custom Callback class for pretty logging and creating learning curves for accuracy and loss metrics during training and evaluation.
    logging_steps and plotting steps are synced to ensure that every point is updated when plotted to avoid straight lines in curves.
    """
    def __init__(self, plot_dir='plot', interval=1, task_name='no_task_name_provided'):
        self.plot_dir = plot_dir
        self.interval = interval
        self.task_name = task_name
        self.mode = 'train'
        self.accuracy_metrics = []
        self.loss_metrics = []
        os.makedirs(self.plot_dir, exist_ok=True)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Set the mode to 'eval' when evaluating the model so the learning curves are plotted for the evaluation phase
        """
        self.mode = 'eval'

        # Reset the metrics for the evaluation phase
        self.accuracy_metrics = []
        self.loss_metrics = []

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Log the metrics and plot the learning curves on every logging step
        """

        logs = kwargs.get("metrics", {})
        current_loss = logs.get("eval_loss")
        current_accuracy = logs.get("eval_accuracy")['accuracy']

        if current_accuracy is None or current_loss is None:
            return

        self.accuracy_metrics.append(current_accuracy)
        self.loss_metrics.append(current_loss)

        if state.epoch % self.interval == 0 and len(self.accuracy_metrics) > 1:
            if len(self.accuracy_metrics) != len(self.loss_metrics):
                # Just for prevent bugs. im understanding more and more, but I still don't trust the on_eval call [when and how is it called??]
                raise ValueError("The length of the accuracy and loss metrics must be equal BUG!")
            self.plot_learning_curves()

    def plot_learning_curves(self):
        """
        Plots a learning curve for the accuracy and loss metrics on every logging step
        """
        epochs = range(1, len(self.accuracy_metrics) + 1)
        plt.figure(figsize=(10, 5))

        # Plot accuracy on the primary y-axis
        plt.plot(epochs, self.accuracy_metrics, label='Accuracy', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy', color='blue')
        plt.tick_params(axis='y', labelcolor='blue')

        # Create a second y-axis for loss
        ax2 = plt.gca().twinx()  # Get the current axes and create a twin y-axis
        ax2.plot(epochs, self.loss_metrics, label='Loss', color='red')
        ax2.set_ylabel('Loss', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Set the title and legend
        plt.title(f'Learning Curves for {self.mode}_{self.task_name} task')
        plt.legend(loc='upper left')  # Place legend for accuracy on the left
        ax2.legend(loc='upper right')  # Place legend for loss on the right # TODO why is it not working?

        # Save the figure
        plt.savefig(os.path.join(self.plot_dir, f"{self.task_name}_{self.mode}_learning_curves.png"))
        plt.close()

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int, metric_name: str = "eval_loss", mode: str = "min"):
        """
        Callback Class for early stopping based on a given metric. Choose min for loss and max for accuracy.
        """
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self.best_metric = None
        self.num_bad_epochs = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Needs to be on_evaluate since there will be the calculations of those metrics.


        """
        logs = kwargs.get("metrics", {})
        current_metric = logs.get(self.metric_name)

        if current_metric is None:
            return

        if self.best_metric is None or \
           (self.mode == "min" and current_metric < self.best_metric) or \
           (self.mode == "max" and current_metric > self.best_metric):
            self.best_metric = current_metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            control.should_training_stop = True
            print(f"Early stopping triggered. No improvement in {self.metric_name} for {self.patience} evaluations.")


class CurriculumLearningCallback(TrainerCallback):
    """
    Idea so far, make a callback "on_evaluate" or "on_epoch_begin" that changes the training data for the next curriculum step
    A curriculum step is not defined for me so far. It could be something like every 10 Epochs. Need to do some more research on this.
    """
    ...