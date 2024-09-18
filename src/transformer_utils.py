import numpy as np
import evaluate

# Metrics
# TODO CHECK THIS

accuracy = evaluate.load("accuracy")

label_list = [0 , 1] # assuming binary classification # TODO check this!


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    # Remove ignored index (special tokens)
    true_predictions = [label_list[p] for (p, l) in zip(predictions, labels) if l != -100]


    true_labels = [label_list[l] for (p, l) in zip(predictions, labels) if l != -100]



    return {
        "precision": accuracy.compute(predictions=true_predictions, references=true_labels)
    }