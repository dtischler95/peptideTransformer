import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FixedLocator
from sklearn.inspection import permutation_importance


def plot_feautre_importance_mdi(estimator,
                                feature_names: list[str],
                                plot_path: str,
                                features_to_display: int = 50):
    # Only tested with randomforest
    importances = estimator.feature_importances_
    print(f"Feature-Importances {importances}")

    std = np.std([estimator.feature_importances_ for tree in estimator.estimators_], axis=0)

    importance_dict = {}
    std_sorted = []
    feature_names_sorted = []
    for index, value, std_value in zip(feature_names, importances, std):
        # if value > 0.0005:
        importance_dict.update({index: value})
        std_sorted.append(std_value)
        feature_names_sorted.append(index)
    std_array = np.array(std_sorted)[:features_to_display]
    est_importances = pd.Series(importance_dict, index=feature_names_sorted).sort_values(ascending=False)[
                      :features_to_display]

    fig, ax = plt.subplots()
    # when using a lot of features, use fontsize 3 for xticklabels
    # when using small amount, use 8
    est_importances.plot.bar(yerr=std_array, ax=ax, align='edge', width=0.5)
    ax.set_title("Feaure importance using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=3)
    plt.tight_layout()
    # plt.show()
    plt.savefig(plot_path)
    plt.clf()


def plot_permutation_importance_from_est(estimator,
                                         x_data,
                                         y_data,
                                         plot_path: str,
                                         summary_path: str,
                                         file_name: str,
                                         feature_names: list[str],
                                         n_repeats: int = 10,
                                         random_state: bool = 42):



    feature_names = np.asarray(feature_names)
    importances = permutation_importance(
        estimator=estimator,
        X=x_data,
        y=y_data,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )


    with open(file=summary_path, mode="w", encoding="utf8") as outfile:
        outfile.write("# Permuation importance > 0 (mean importances!)\n")
        for i in importances.importances_mean.argsort()[::-1]:
            if importances.importances_mean[i] - 2 * importances.importances_std[i] > 0:
                outfile.write(f"* Feature: {feature_names[i]:<8}\n"
                              f"    * Mean importance:{importances.importances_mean[i]:.3f}\n"
                              f"    * Std: {importances.importances_std[i]:.3f}\n")

    imp_idx = importances.importances_mean.argsort()[:50]
    clf_importances_idx = np.argsort(estimator.feature_importances_)[:50]
    clf_idx = (np.arange(0, len(estimator.feature_importances_)) + 0.5)[:50]

    # when using a lot of features, use fontsize 4 for yticklabels
    # when using small amount, use 8
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.barh(clf_idx,
             estimator.feature_importances_[clf_importances_idx],
             height=0.7
             )
    ax1.set_yticks(clf_idx)
    ax1.set_yticklabels(feature_names[clf_importances_idx], fontsize=8)
    ax1.set_xticks(ax1.get_xticks())
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=8)
    ax1.xaxis.set_major_locator(FixedLocator(ax1.get_xticks()))
    # ax1.set_ylim(0, len(estimator.feature_importances_))
    ax2.boxplot(
        importances.importances[imp_idx].T[:100],
        vert=False,
        labels=feature_names[clf_importances_idx]
    )
    ax2.set_yticks(clf_idx)
    ax2.set_yticklabels(feature_names[clf_importances_idx], fontsize=8)
    fig.suptitle(f"Permutationsbasierte Merkmalswichtigkeit {n_repeats} Faltungen\nModel: rf Daten: {file_name}", fontsize=14)
    ax1.set_title("Balkendiagramm")
    ax1.set_ylabel("Merkmalsname")  # statt „Feaurename“
    ax1.set_xlabel("Wichtigkeit")

    ax2.set_title("Boxplot")
    ax2.set_xlabel("Wichtigkeit")

    fig.tight_layout()
    # plt.show()
    plt.savefig(plot_path)
    plt.clf()



def make_regression(y_true, y_pred, file_name, path, model_name):

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


    fig, (ax_reg, ax_res) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Regression (y_true vs y_pred)
    ax_reg.scatter(y_pred, y_true, alpha=0.6, edgecolor='none')
    lo, hi = np.nanpercentile(np.concatenate([y_true, y_pred]), [0.5, 99.5])
    ax_reg.plot([lo, hi], [lo, hi], ls='--', c='green')
    x_vals = np.linspace(lo, hi, endpoint=True)
    ax_reg.plot(x_vals, x_vals + sigma, ls='--', c='red', label='+σ')
    ax_reg.plot(x_vals, x_vals - sigma, ls='--', c='red', label='-σ')
    ax_reg.set_xlabel('Vorhergesagter Wert MIC (log$_{10}$ µM)')
    ax_reg.set_ylabel('Tatsächlicher Wert MIC (log$_{10}$ µM)')
    ax_reg.set_title(f'Tatsächlich vs. vorhergesagt')
    ax_reg.legend(loc='best')

    # --- Residuen (eigene Berechnung, kein residplot)
    ax_res.scatter(y_pred, resid, alpha=0.6, edgecolor='none')
    ax_res.axhline(0, color='k', ls=':')
    ax_res.axhline(+sigma, color='r', ls='--', label='+σ')
    ax_res.axhline(-sigma, color='r', ls='--', label='-σ')
    ax_res.set_xlabel('Vorhergesagter Wert MIC (log$_{10}$ µM)')
    ax_res.set_ylabel('Residuum MIC (log$_{10}$ µM)')
    ax_res.set_title(f'Residuen vs. Vorhersage')
    ax_res.legend(loc='best')

    fig.suptitle(f"Regressions- und Residuenplot (σ={sigma:.2f})\nModel: {model_name} Daten: {file_name}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plt.clf()
