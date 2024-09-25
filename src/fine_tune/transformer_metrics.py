import numpy as np
import evaluate


# Metrics
# TODO CHECK THIS


def compute_metrics(p):
    accuracy = evaluate.load("accuracy")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {
        "precision": accuracy.compute(predictions=predictions, references=labels)
    }
