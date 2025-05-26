import numpy as np
import evaluate
from sklearn.metrics import matthews_corrcoef


def regression_metrics(eval_preds) -> dict:
    """
    computes accuracy for regression task.

    :param eval_preds: predictions and labels

    :return: dictionary with the metrics
    """

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    predictions, labels = eval_preds
    # Get predicted labels from logits

    preds = predictions.squeeze()
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)

    return {
        "mse": round(mse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4)
    }


def binary_metrics(eval_preds, debug_print: bool = True) -> dict:
    """
    Taken and adapted from:
    https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py

    :param eval_preds: predictions and labels
    :param debug_print: if the predictions should be printed for debugging purposes

    :return: dictionary with the metrics
    """
    from src.bert_model.fine_tune_utils import format_logit_to_label
    accuracy = evaluate.load("accuracy")

    logits, labels = eval_preds
    predictions = format_logit_to_label(logits=logits)  # Apply the threshold to convert probabilities to binary predictions

    mcc = matthews_corrcoef(y_true=labels, y_pred=predictions)
    precision = evaluate.load("precision").compute(predictions=predictions, references=labels, zero_division=0)["precision"]
    f1 = evaluate.load("f1").compute(predictions=predictions, references=labels)["f1"]
    recall = evaluate.load("recall").compute(predictions=predictions, references=labels)["recall"]
    # ----- Added for debugging purposes -----
    # Save check for Prediction Bias towards one label
    if debug_print:
        label_0_counter, label_1_counter = _debug_predicted_labels(predictions)
        return {
            "accuracy": round(accuracy.compute(predictions=predictions, references=labels)["accuracy"], 4),
            "mcc": round(mcc, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "recall": round(recall, 4),
            "label_0_count_on_epoch_end": label_0_counter,
            "label_1_count_on_epoch_end": label_1_counter
        }

    return {
        "accuracy": round(accuracy.compute(predictions=predictions, references=labels)["accuracy"], 4),
        "mcc": round(mcc, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "recall": round(recall, 4)
    }


def _debug_predicted_labels(predictions):
    """
    Helper function to print the amount of predicted labels
    This is used to debug for label bias in binary classification
    So far this covers only the eval dataset! TODO: Implement for training dataset
    """
    label_0_counter = 0
    label_1_counter = 0
    for pred in predictions:
        if pred == 0:
            label_0_counter += 1
        elif pred == 1:
            label_1_counter += 1
        else:
            print(f"Invalid Prediction: {pred}")

    return label_0_counter, label_1_counter


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
    return {'accuracy': accuracy.compute(predictions=masked_preds.flatten(), references=masked_labels.flatten())["accuracy"]}
