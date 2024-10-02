import numpy as np
import evaluate


def binary_metrics(eval_preds, debug_print: bool = True) -> dict:
    """
    Taken and adapted from:
    https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py

    :param eval_preds: predictions and labels
    :param debug_print: if the predictions should be printed for debugging purposes

    :return: dictionary with the metrics
    """

    accuracy = evaluate.load("accuracy")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")
    roc_auc = evaluate.load("roc_auc")

    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=-1)

    # ----- Added for debugging purposes -----
    # Save check for Prediction Bias towards one label
    if debug_print:
        _debug_label_printer(predictions)

    print(f"Acuracy Save Print: {accuracy.compute(predictions=predictions, references=labels)}")

    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels),
        "recall": recall.compute(predictions=predictions, references=labels),
        "f1": f1.compute(predictions=predictions, references=labels),
        "roc_auc": roc_auc.compute(prediction_scores=predictions, references=labels)
    }


def _debug_label_printer(predictions):
    label_0_counter = 0
    label_1_counter = 0
    for pred in predictions:
        if pred == 0:
            label_0_counter += 1
        elif pred == 1:
            label_1_counter += 1
        else:
            print(f"Invalid Prediction: {pred}")
    print(f"\nLabel 0 Counter: {label_0_counter}")
    print(f"Label 1 Counter: {label_1_counter}\n")


def mlm_metrics(eval_preds) -> dict:
    """
    computes accuracy for mlm task. Ignore -100 labels and calculate accuracy only on predictions of masked tokens

    :param eval_preds: predictions and labels

    :return: dictionary with the metrics
    """

    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_preds

    # Get predicted labels from logits
    preds = np.argmax(predictions, axis=-1)

    # Ignore the -100 labels
    mask = labels != -100  # Create a mask for valid labels
    masked_preds = preds[mask]  # Filter predictions using the mask
    masked_labels = labels[mask]  # Filter labels using the mask

    # Calculate accuracy only on valid predictions
    return {'accuracy': accuracy.compute(predictions=masked_preds.flatten(), references=masked_labels.flatten())}
