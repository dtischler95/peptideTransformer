import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics as sk_metrics
from sklearn.metrics import (
    r2_score, mean_absolute_error, explained_variance_score, mean_squared_error,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    classification_report, ConfusionMatrixDisplay, matthews_corrcoef,
)

# Ordered most-specific first to avoid prefix collisions (e.g. staphylococcus_aureus before staphylococcus_epi)
_NAME_MAP = [
    ("staphylococcus_aureus", "Staphylococcus aureus"),
    ("staphylococcus_epi", "Staphylococcus epidermidis"),
    ("acineto", "Acinetobacter baumannii"),
    ("bacillus", "Bacillus subtilis"),
    ("candida", "Candida albicans"),
    ("enterobacter", "Enterobacter sp."),
    ("enterococc", "Enterococcus faecalis"),
    ("escher", "Escherichia coli"),
    ("klebsie", "Klebsiella pneumoniae"),
    ("micro", "Micrococcus luteus"),
    ("pseudo", "Pseudomonas aeruginosa"),
    ("salmonella", "Salmonella enterica"),
    ("threshold_style", "Threshold Dataset"),
    ("gram_dataset", "Gram Dataset"),
    ("gram", "Gram Dataset"),
    ("whitelab", "WhiteLab Dataset"),
]


def get_pretty_name(name: str) -> str:
    """Maps a file path or data tag to a human-readable label for plot titles.
    Falls back to title-casing the filename stem for unknown organisms."""
    stem = os.path.splitext(os.path.basename(name))[0].lower()
    for prefix, pretty in _NAME_MAP:
        if stem.startswith(prefix):
            return pretty
    return " ".join(p.capitalize() for p in stem.split("_")[:2])


def print_regression_metrics(y_true, y_pred, logger: logging.Logger, tag: str = ''):
    prefix = f"{tag} " if tag else ""
    logger.info(
        f"{prefix}Regression metrics:\n"
        f"    -> R2:  {r2_score(y_true=y_true, y_pred=y_pred):.5f}\n"
        f"    -> MAE: {mean_absolute_error(y_true=y_true, y_pred=y_pred):.5f}\n"
        f"    -> MSE: {mean_squared_error(y_true=y_true, y_pred=y_pred):.5f}\n"
        f"    -> VAR: {explained_variance_score(y_true=y_true, y_pred=y_pred):.5f}\n"
    )
    return r2_score(y_true=y_true, y_pred=y_pred), mean_squared_error(y_true=y_true, y_pred=y_pred)


def make_regression_plot(y_true, y_pred, path, file_name, tag):
    label = get_pretty_name(file_name)

    y_true = np.asarray(y_true).reshape(-1).astype(float)
    y_pred = np.asarray(y_pred).reshape(-1).astype(float)

    resid = y_true - y_pred
    sigma = resid.std(ddof=1)

    fig_reg, ax_reg = plt.subplots(figsize=(8, 6))
    ax_reg.scatter(y_pred, y_true, alpha=0.6, edgecolor='none')
    lo, hi = np.nanpercentile(np.concatenate([y_true, y_pred]), [0.5, 99.5])
    ax_reg.plot([lo, hi], [lo, hi], ls='--', c='green', label='Ideal')
    x_vals = np.linspace(lo, hi, 100)
    ax_reg.plot(x_vals, x_vals + sigma, ls='--', c='red', label='+σ')
    ax_reg.plot(x_vals, x_vals - sigma, ls='--', c='red', label='-σ')
    ax_reg.set_xlabel(r'Predicted Value $\log_{10}(\mathrm{MIC})\,[\mu\mathrm{M}]$')
    ax_reg.set_ylabel(r'Actual Value $\log_{10}(\mathrm{MIC})\,[\mu\mathrm{M}]$')
    ax_reg.legend(loc='best')
    fig_reg.suptitle(f"Regression Plot (σ={sigma:.2f}) ({tag})\nData: {label}")
    plt.tight_layout()
    base = Path(path).with_suffix('')
    plt.savefig(f"{base}_regression.pdf")
    plt.close(fig_reg)

    fig_res, ax_res = plt.subplots(figsize=(8, 6))
    ax_res.scatter(y_pred, resid, alpha=0.6, edgecolor='none')
    ax_res.axhline(0, color='k', ls=':')
    ax_res.axhline(+sigma, color='r', ls='--', label='+σ')
    ax_res.axhline(-sigma, color='r', ls='--', label='-σ')
    ax_res.set_xlabel(r'Predicted Value $\log_{10}(\mathrm{MIC})\,[\mu\mathrm{M}]$')
    ax_res.set_ylabel('Residual (Actual - Predicted)')
    ax_res.legend(loc='best')
    fig_res.suptitle(f"Residual Plot (σ={sigma:.2f}) ({tag})\nData: {label}")
    plt.tight_layout()
    plt.savefig(f"{base}_residuals.pdf")
    plt.close(fig_res)


def overall_stats(y_true, predictions, save_path, tag, file_name, model_name=None):
    label = get_pretty_name(file_name)
    model_suffix = f" Model: {model_name}" if model_name else ""

    plt.figure(figsize=(10, 6))
    sns.histplot(y_true, kde=True)
    plt.title(f'MIC Value Distribution ({tag})\nData: {label}{model_suffix}')
    plt.xlabel(r'$\log_{10}(\mathrm{MIC})\,[\mu\mathrm{M}]$')
    plt.ylabel('Count')
    plt.savefig(save_path + f'/{tag}_target_distribution.pdf')
    plt.close()
    plt.clf()

    residuals = y_true - predictions

    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title(f'Residual Distribution ({tag})\nData: {label}{model_suffix}')
    plt.xlabel(r'Residuals $\log_{10}(\mathrm{MIC})\,[\mu\mathrm{M}]$')
    plt.ylabel('Count')
    plt.savefig(save_path + f'/{tag}residuals_distribution.pdf')
    plt.close()
    plt.clf()

    df = pd.DataFrame({
        "MIC (log$_{10}$ µM)": y_true,
        "Residuals (log$_{10}$ µM)": residuals,
    })
    df["MIC Quartile"] = pd.qcut(df["MIC (log$_{10}$ µM)"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

    plt.figure(figsize=(8, 4))
    sns.boxplot(x="MIC Quartile", y="Residuals (log$_{10}$ µM)", data=df, color="skyblue")
    plt.xlabel(r"Quartiles of Actual MIC Values $\log_{10}(\mathrm{MIC})\,[\mu\mathrm{M}]$")
    plt.ylabel(r"Residuals $\log_{10}(\mathrm{MIC})\,[\mu\mathrm{M}]$")
    plt.title(f"Residual Quartile Plot ({tag})\nData: {label}{model_suffix}")
    plt.tight_layout()
    plt.savefig(save_path + f'/{tag}residuals_quantils.pdf')
    plt.close()
    plt.clf()


def get_model_stats(plot_dir, predictions, target_data, logger, file_name, tag):
    r2, mse = print_regression_metrics(y_true=target_data, y_pred=predictions, logger=logger, tag=tag)
    make_regression_plot(
        y_true=target_data, y_pred=predictions,
        path=f"{plot_dir}/{file_name}{tag}_regression.pdf",
        file_name=file_name, tag=tag,
    )
    return r2, mse


def evaluate_hemo(y_true, y_score, y_pred, plot_path, tag, file_name, model_name='', logger=None):
    """Shared hemolysis evaluation: ROC/PR curves, confusion matrix, classification report.

    :param plot_path: Directory to save plots into.
    :param model_name: Optional model name included in plot titles and filenames.
    """
    Path(plot_path).mkdir(parents=True, exist_ok=True)

    label = get_pretty_name(file_name)
    title_suffix = f"\nModel: {model_name} Data: {label}" if model_name else f"\nData: {label}"
    name_infix = f"_{model_name}" if model_name else ""

    def _log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    mcc = matthews_corrcoef(y_true, y_pred)

    _log(f"MCC: {mcc:.4f}")
    _log(f"AUROC: {auc:.4f}")
    _log(f"Average Precision (PR-AUC): {ap:.4f}")
    _log(classification_report(y_true, y_pred, digits=3))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUROC={auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title(f'ROC Curve ({tag}){title_suffix}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'{plot_path}/{tag}{name_infix}_roc.png')
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label=f'AP={ap:.3f}')
    plt.hlines(np.mean(y_true), 0, 1, linestyles='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({tag}){title_suffix}')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(f'{plot_path}/{tag}{name_infix}_pr.png')
    plt.close()

    confusion_matrix = sk_metrics.confusion_matrix(y_true, y_pred)
    with np.errstate(all='ignore'):
        confusion_matrix_normalized = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
    for i, matrix in enumerate([confusion_matrix, confusion_matrix_normalized], start=1):
        cm_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0, 1])
        cm_display.plot()
        plt.title(f"Confusion Matrix ({tag}){title_suffix}")
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.savefig(f'{plot_path}/{tag}_{i}_confusion_matrix{name_infix}.png')
        plt.close()
        plt.clf()
