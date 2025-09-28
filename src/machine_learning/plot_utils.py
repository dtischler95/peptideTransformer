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
                                         feature_names: list[str],
                                         n_repeats: int = 10,
                                         random_state: bool = 42):
    feature_names = np.asarray(feature_names)
    importances = permutation_importance(
        estimator=estimator,
        X=x_data,
        y=y_data,
        n_repeats=n_repeats,
        random_state=random_state
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
    ax1.set_yticklabels(feature_names[clf_importances_idx], fontsize=4)
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
    ax2.set_yticklabels(feature_names[clf_importances_idx], fontsize=4)
    fig.suptitle(f"Permutationsbasierte Merkmalswichtigkeit {n_repeats} Faltungen", fontsize=14)
    ax1.set_title("Balkendiagramm")
    ax1.set_ylabel("Merkmalsname")  # statt „Feaurename“
    ax1.set_xlabel("Wichtigkeit")

    ax2.set_title("Boxplot")
    ax2.set_xlabel("Wichtigkeit")

    fig.tight_layout()
    # plt.show()
    plt.savefig(plot_path)
    plt.clf()


def plot_with_seaborn(y_true, y_pred, path, tag):
    fig, axs = plt.subplots(ncols=2, figsize=(16, 8))

    # Regression part
    slope, intercept, r_value, p_value, std_err = linregress(y_pred, y_true)
    # reg_equation = "y = {:.2f}x".format(slope)

    # calculate the residuals
    residuals = y_true - y_pred
    std_residuals = np.std(residuals)

    print(f"Steigung: {slope}, Standartabweichung der Residuen: {std_residuals} log(µM) for {tag}")

    # generating residual plot
    sns.residplot(x=y_pred, y=residuals, ax=axs[1])
    axs[1].set_title(
        "Residuen gegen vorhergesagte Werte\nStandardabweichung der Residuen: {:.2f} log(µM)".format(std_residuals))
    axs[1].set_xlabel("Vorhergesagter Wert MIC/log(µM)")
    axs[1].set_ylabel("Residuum MHK/log(µM)")

    # Plot two red horizontal lines representing positive and negative standard deviations
    axs[1].axhline(std_residuals, color='red', linestyle='--')
    axs[1].axhline(-std_residuals, color='red', linestyle='--')

    # Plot manually added regression line with confidence interval
    y_pred_sorted = np.sort(y_pred)

    # generate scatter plot
    sns.regplot(x=y_pred, y=y_true, ax=axs[0], fit_reg=False)

    # plot the fitted line through the origin and also a line with slope 1 for comparison
    # axs[0].plot(y_pred_sorted, slope * y_pred_sorted, color='red')
    # standarf f(x) function for getting slope=1
    axs[0].plot([-1, 4], [-1, 4], linestyle='--', color='green', label='45-degree Line')

    # Calculate bounds for lines parallel to the regression line
    lower_bound = 1 * y_pred_sorted - std_residuals
    upper_bound = 1 * y_pred_sorted + std_residuals

    # reg_equation2 = f"y = {slope:.2f}x + {intercept:.2f}"
    # axs[0].text(0.05, 0.95, reg_equation2, transform=axs[0].transAxes, fontsize=12,
    #            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

    # Plot two lines parallel to the regression line representing positive and negative standard deviations
    axs[0].plot(y_pred_sorted, lower_bound, color='red', linestyle='--')
    axs[0].plot(y_pred_sorted, upper_bound, color='red', linestyle='--')

    axs[0].set_title("Tatsächliche gegen vorhergesagte Werte MIC/log(µM)")
    axs[0].set_xlabel("Vorhergesagter Wert MIC/log(µM)")
    axs[0].set_ylabel("Tatsächlicher Wert MIC/log(µM)")

    fig.suptitle("Regressions -und Residuenplot")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plt.clf()
