import eval_utils
from sklearn.inspection import permutation_importance  # noqa: F401  kept for callers


def make_regression(y_true, y_pred, file_name, path, tag, model_name=''):
    eval_utils.make_regression_plot(y_true=y_true, y_pred=y_pred, path=path, file_name=file_name, tag=tag)
