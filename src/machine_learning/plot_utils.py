import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FixedLocator
from sklearn.inspection import permutation_importance



def make_regression(y_true, y_pred, file_name, path, tag):

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


    y_true = np.asarray(y_true).reshape(-1).astype(float)
    y_pred = np.asarray(y_pred).reshape(-1).astype(float)

    resid = y_true - y_pred
    sigma = resid.std(ddof=1)

    # --- Regression ---
    fig_reg, ax_reg = plt.subplots(figsize=(8, 6))

    ax_reg.scatter(y_pred, y_true, alpha=0.6, edgecolor='none')

    lo, hi = np.nanpercentile(np.concatenate([y_true, y_pred]), [0.5, 99.5])
    ax_reg.plot([lo, hi], [lo, hi], ls='--', c='green')

    x_vals = np.linspace(lo, hi, 100)
    ax_reg.plot(x_vals, x_vals + sigma, ls='--', c='red', label='+σ')
    ax_reg.plot(x_vals, x_vals - sigma, ls='--', c='red', label='-σ')

    ax_reg.set_xlabel(r'$\log_{10}(\mathrm{MIC})\,[\mu\mathrm{M}]$ (vorhergesagt)')
    ax_reg.set_ylabel(r'$\log_{10}(\mathrm{MIC})\,[\mu\mathrm{M}]$ (tatsächlich)')
    ax_reg.legend(loc='best')

    fig_reg.suptitle(f"Regressionsplot (σ={sigma:.2f}) ({tag})\nDaten: {file_name}")
    plt.tight_layout()
    plt.savefig(path.replace('.pdf', '_regression.pdf'))
    plt.close(fig_reg)

    # --- Residuen ---
    fig_res, ax_res = plt.subplots(figsize=(8, 6))

    ax_res.scatter(y_pred, resid, alpha=0.6, edgecolor='none')
    ax_res.axhline(0, color='k', ls=':')
    ax_res.axhline(+sigma, color='r', ls='--', label='+σ')
    ax_res.axhline(-sigma, color='r', ls='--', label='-σ')

    ax_res.set_xlabel(r'$\log_{10}(\mathrm{MIC})\,[\mu\mathrm{M}]$ (vorhergesagt)')
    ax_res.set_ylabel(r'Residuum $\log_{10}(\mathrm{MIC})\,[\mu\mathrm{M}]$')
    ax_res.legend(loc='best')

    fig_res.suptitle(f"Residuenplot (σ={sigma:.2f}) ({tag})\nDaten: {file_name}")
    plt.tight_layout()
    plt.savefig(path.replace('.pdf', '_residuals.pdf'))
    plt.close(fig_res)


