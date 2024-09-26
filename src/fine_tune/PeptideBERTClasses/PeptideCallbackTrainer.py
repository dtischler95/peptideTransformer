from transformers import TrainerCallback, TrainerState, TrainerControl
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


        self.accuracy_metrics.append(kwargs['metrics']['eval_accuracy']['accuracy'])
        self.loss_metrics.append(kwargs['metrics']['eval_loss'])

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


class CurriculumLearningCallback(TrainerCallback):
    """
    Idea so far, make a callback "on_evaluate" or "on_epoch_begin" that changes the training data for the next curriculum step
    A curriculum step is not defined for me so far. It could be something like every 10 Epochs. Need to do some more research on this.
    """
    ...