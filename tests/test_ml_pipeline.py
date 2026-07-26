"""End-to-end smoke of the classical-ML entry points on synthetic fixtures.

Exercises the full path per model: TF-IDF + SVD featurisation, the optional
scaler branch, 5-fold grid search, refit, evaluation and the pickled-model save.
Tiny grids keep it fast; the point is that the plumbing runs, not the scores.
"""

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from src.machine_learning.train_models import (
    run_regression, run_classification, default_classifiers, default_regressors,
)


def _assert_pkls(output_root, model_tags):
    saved = {p.stem for p in output_root.rglob("*.pkl")}
    for tag in model_tags:
        assert tag in saved, f"missing saved model for {tag} (found {saved})"


def test_default_model_factories_pair_models_and_grids():
    for factory in (default_classifiers, default_regressors):
        models, grids = factory()
        assert len(models) == len(grids) > 0
        for (tag, estimator), grid in zip(models, grids):
            assert isinstance(tag, str) and tag
            assert hasattr(estimator, "fit")
            assert isinstance(grid, dict)


def test_run_regression_smoke(mic_data_dir, tmp_path):
    output_root = tmp_path / "ml_plots"
    models = [
        ("dummy", DummyRegressor(strategy="mean")),
        ("ridge", Ridge()),
        ("svr", SVR()),
        ("xtra", ExtraTreesRegressor(n_jobs=1)),
        ("xgb", XGBRegressor(n_jobs=1, tree_method="hist")),
    ]
    grids = [
        {},
        {"alpha": [1.0]},
        {"svr__C": [1.0], "svr__kernel": ["rbf"], "svr__epsilon": [0.1]},
        {"n_estimators": [10]},
        {"n_estimators": [10]},
    ]
    run_regression(data_dir=mic_data_dir, output_root=output_root,
                   models=models, grids=grids, calculate_features=False, n_jobs=1)
    _assert_pkls(output_root, [t for t, _ in models])


def test_run_classification_smoke(hemo_data_dir, tmp_path):
    output_root = tmp_path / "ml_plots"
    models = [
        ("dummy", DummyClassifier(strategy="prior")),
        ("logreg", LogisticRegression(max_iter=1000)),
        ("svc", SVC(probability=True)),
        ("xtra", ExtraTreesClassifier(n_jobs=1)),
        ("xgb", XGBClassifier(n_jobs=1, tree_method="hist")),
    ]
    grids = [
        {},
        {"C": [1.0]},
        {"svc__C": [1.0], "svc__kernel": ["rbf"], "svc__gamma": ["scale"]},
        {"n_estimators": [10]},
        {"n_estimators": [10]},
    ]
    run_classification(data_dir=hemo_data_dir, output_root=output_root,
                       models=models, grids=grids, calculate_features=False, n_jobs=1)
    _assert_pkls(output_root, [t for t, _ in models])
