import numpy as np
import evaluate

# Metrics
# TODO CHECK THIS

accuracy = evaluate.load("accuracy")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {
        "precision": accuracy.compute(predictions=predictions, references=labels)
    }
