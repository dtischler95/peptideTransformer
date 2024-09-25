from transformers import TrainerCallback, TrainerState, TrainerControl
import matplotlib.pyplot as plt
import os


class PeptideCallback(TrainerCallback):
    """
    Custom Callback class to individualize Callbacks for the Trainer Class
    """

    def __init__(self, plot_dir='plot', interval=100, task_name='no_task_name_provided'):
        self.plot_dir = plot_dir
        self.interval = interval
        self.task_name = task_name
        self.mode = 'train'
        self.accuracy_metrics = []
        self.loss_metrics = []
        self.hasMetricUpdated = False
        os.makedirs(self.plot_dir, exist_ok=True)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Set the mode to 'eval' when evaluating the model so the learning curves are plotted for the evaluation phase
        """
        self.mode = 'eval'

        # Reset the metrics for the evaluation phase
        self.accuracy_metrics = []
        self.loss_metrics = []

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """
        Log the metrics and plot the learning curves on every logging step
        """
        if logs is None:
            return

        if "eval_accuracy" in logs:
            self.accuracy_metrics.append(logs['eval_accuracy']['accuracy'])
            self.hasMetricUpdated = True

        if "eval_loss" in logs:
            self.loss_metrics.append(logs["eval_loss"])

        if state.global_step % self.interval == 0 and self.hasMetricUpdated:
            if len(self.accuracy_metrics) != len(self.loss_metrics):
                # I'm not too sure about when on_log is called in the trainer class and if there could be some bugs with this
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
        ax2.legend(loc='upper right')  # Place legend for loss on the right

        # Save the figure
        plt.savefig(os.path.join(self.plot_dir, f"{self.task_name}_{self.mode}_learning_curves.png"))
        plt.close()
