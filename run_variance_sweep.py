"""Variance-estimation sweep for the classical ML baseline.

For every seed it generates a fresh split (random via data_splitter, cluster via
MMseqs2 cluster_data_splitter) into results/_splits/, then trains the classical
ML pipeline on it. Each per-organism score therefore gets a spread across seeds,
for both splits symmetrically. Output dirs encode task, split, seed and feature
mode, so nothing collides and eval_utils.collect_run_metrics merges the whole
tree afterwards.

MMseqs2 must be on PATH for the cluster split (cluster_data_splitter raises a
clear install hint otherwise). The clustering itself is seed-independent, so the
per-seed regeneration re-clusters identically and only varies the cluster-to-
split assignment. That is wasted MMseqs compute but kept simple, the clustering
is fast on these short-peptide datasets.

Run from the repo root, e.g.:
    python run_variance_sweep.py --dry-run
    python run_variance_sweep.py --seeds 42 1 2 3 4
    python run_variance_sweep.py --seeds 42 1 2 3 4 --also-features
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from src.data_preprocessing import datasets
from src.data_preprocessing.data_splitter import data_splitter
from src.data_preprocessing.cluster_data_splitter import cluster_data_splitter
from src.machine_learning import config
from src.machine_learning.train_models import run_regression, run_classification

_ROOT = Path(__file__).resolve().parent


def _reg_models():
    # Baseline set. Models stay single-threaded; the grid search does the
    # parallelism (n_jobs), which avoids nested oversubscription and caps how
    # many memory-heavy SVR kernel fits run at once.
    models = [
        ("dummy", DummyRegressor(strategy="mean")),
        ("ridge", Ridge()),
        ("xtra", ExtraTreesRegressor(n_jobs=1)),
        ("xgb", XGBRegressor(n_jobs=1, tree_method="hist")),
        ("svr", SVR()),
    ]
    grids = [config.dummy_param_grid, config.ridge_param_grid, config.xtra_param_grid,
             config.xgb_param_grid, config.svr_param_grid]
    return models, grids


def _cls_models():
    models = [
        ("dummy", DummyClassifier(strategy="prior")),
        ("logreg", LogisticRegression(max_iter=1000)),
        ("xtra", ExtraTreesClassifier(n_jobs=1)),
        ("xgb", XGBClassifier(n_jobs=1)),
        ("svc", SVC()),
    ]
    grids = [config.dummy_param_grid, config.logreg_param_grid,
             config.xtra_cls_param_grid, config.xgb_cls_param_grid,
             config.svc_cls_param_grid]
    return models, grids


def _setup(task):
    """Return (run_fn, models_fn, splitter_task). ``task`` is the output label."""
    if task == "regression":
        return run_regression, _reg_models, "regression"
    return run_classification, _cls_models, "cls"


def _run_combo(task, split, seed, feat, results_root, dry_run, n_jobs=-1):
    run_fn, models_fn, splitter_task = _setup(task)
    suffix = "_feat" if feat else ""
    out_root = Path(results_root) / f"{task}_{split}_seed{seed}{suffix}"
    data_dir = Path(results_root) / "_splits" / f"{task}_{split}_seed{seed}"

    # Generate the split once. Reuse an existing one on resume so cluster runs do
    # not re-invoke mmseqs (the clustering is deterministic for a given seed anyway).
    have_split = data_dir.exists() and any(data_dir.glob("*_train.csv"))
    if have_split:
        print(f"[gen] {task}/{split}/seed{seed} -> reuse {data_dir}")
    else:
        print(f"[gen] {task}/{split}/seed{seed} -> {data_dir}")
        if not dry_run:
            if split == "random":
                data_splitter(task=splitter_task, out_dir=data_dir, random_state=seed)
            else:
                cluster_data_splitter(task=splitter_task, out_dir=data_dir, random_state=seed)

    n_models = len(models_fn()[0])
    # Organism set is split- and seed-independent, enumerate from the task default dir.
    organisms = [b.stem for b in datasets.split_basepaths_in(datasets.default_dir(splitter_task))]
    for org in organisms:
        slug = datasets.organism_slug(org)
        org_out = out_root / slug
        done = len(list(org_out.glob("*/metrics.json"))) if org_out.exists() else 0
        if done >= n_models:
            print(f"  [done] {task}/{split}/seed{seed}{suffix} {slug} ({done}/{n_models}) skip")
            continue
        print(f"  [run]  {task}/{split}/seed{seed}{suffix} {slug} -> {out_root}")
        if dry_run:
            continue
        models, grids = models_fn()  # fresh estimators per organism
        try:
            run_fn(data_dir=data_dir, output_root=out_root, models=models, grids=grids,
                   calculate_features=feat, seed=seed, organism=org, n_jobs=n_jobs)
        except Exception:
            print(f"    FAILED {slug}:\n{traceback.format_exc()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--tasks", nargs="+", default=["regression", "cls"],
                    choices=["regression", "cls"])
    ap.add_argument("--splits", nargs="+", default=["random", "cluster"],
                    choices=["random", "cluster"])
    ap.add_argument("--also-features", action="store_true",
                    help="additionally run the seq+feat variant (doubles runtime)")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="parallel jobs for the grid search, caps concurrent fits and thus peak memory (default -1 = all cores, use 3 to leave a core free and avoid OOM on big SVR fits)")
    ap.add_argument("--results-root", default=str(_ROOT / "results"))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned runs without splitting or training")
    args = ap.parse_args()

    feat_modes = [False, True] if args.also_features else [False]
    for task in args.tasks:
        for split in args.splits:
            for seed in args.seeds:
                for feat in feat_modes:
                    _run_combo(task, split, seed, feat, args.results_root, args.dry_run, n_jobs=args.n_jobs)

    print("\nDone. Merge with eval_utils.collect_run_metrics(results-root), "
          "then run paired_test / bootstrap.")


if __name__ == "__main__":
    main()
