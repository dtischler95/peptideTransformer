"""Paired Wilcoxon test across organisms from the merged metrics table.

Consumes the long-format DataFrame produced by ``ml_utils.collect_run_metrics``
(one row per organism/model/split/seed). For two model families it pairs the chosen
metric per organism (averaging over seeds first) and runs a Wilcoxon signed-rank
test, turning "architecture barely matters" from eyeballing tables into a stated
result. ``bert_vs_best_classical`` pairs BERT against the best classical model per
organism instead.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

try:
    from src.evaluation import eval_utils
except ImportError:  # allow running the file directly
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation import eval_utils  # type: ignore

# Metric direction, so "best classical" picks the right extreme.
_HIGHER_IS_BETTER = {"r2": True, "auroc": True, "mcc": True, "f1_macro": True,
                     "acc": True, "ap": True, "mse": False, "mae": False}


def _agg(df, metric, split, features):
    """Mean metric per (organism, model), so each organism contributes one value
    even when several seeds are present."""
    d = df[(df["split"] == split) & (df["features"] == bool(features))]
    return d.groupby(["data_name", "model"])[metric].mean().reset_index()


def paired_wilcoxon(df, model_a, model_b, metric, split="random", features=False,
                    alternative="two-sided"):
    agg = _agg(df, metric, split, features)
    pa = agg[agg["model"] == model_a].set_index("data_name")[metric]
    pb = agg[agg["model"] == model_b].set_index("data_name")[metric]
    common = sorted(pa.index.intersection(pb.index))
    a, b = pa.loc[common], pb.loc[common]
    diff = a - b
    if len(common) and (diff != 0).any():
        stat, p = wilcoxon(a.values, b.values, alternative=alternative)
    else:
        stat, p = float("nan"), float("nan")
    return {"model_a": model_a, "model_b": model_b, "metric": metric, "split": split,
            "features": bool(features), "n": len(common),
            "median_diff": float(diff.median()) if len(common) else float("nan"),
            "mean_diff": float(diff.mean()) if len(common) else float("nan"),
            "stat": float(stat), "p_value": float(p), "alternative": alternative}


def bert_vs_best_classical(df, metric, split="random", features=False, bert="bert",
                           exclude=("dummy", "ridge", "logreg"), alternative="two-sided"):
    agg = _agg(df, metric, split, features)
    higher = _HIGHER_IS_BETTER.get(metric, True)
    rows = []
    for org, g in agg.groupby("data_name"):
        if bert not in set(g["model"]):
            continue
        bert_val = g.loc[g["model"] == bert, metric].iloc[0]
        classical = g[~g["model"].isin({bert, *exclude})]
        if classical.empty:
            continue
        best = classical[metric].max() if higher else classical[metric].min()
        rows.append({"data_name": org, "bert": bert_val, "best_classical": best})
    res = pd.DataFrame(rows)
    if res.empty:
        return {"metric": metric, "split": split, "features": bool(features), "n": 0,
                "per_organism": res}
    diff = res["bert"] - res["best_classical"]
    if (diff != 0).any():
        stat, p = wilcoxon(res["bert"].values, res["best_classical"].values,
                           alternative=alternative)
    else:
        stat, p = float("nan"), float("nan")
    return {"metric": metric, "split": split, "features": bool(features),
            "n": int(len(res)), "median_diff_bert_minus_best": float(diff.median()),
            "mean_diff_bert_minus_best": float(diff.mean()),
            "stat": float(stat), "p_value": float(p), "alternative": alternative,
            "per_organism": res}


def run(metrics_root, model_a="bert", model_b=None, metric="r2", split="random",
        features=False, bert_vs_best=False):
    df = eval_utils.collect_run_metrics(metrics_root)
    if df.empty:
        print(f"No metrics.json found under {metrics_root}")
        return None

    if bert_vs_best:
        res = bert_vs_best_classical(df, metric=metric, split=split, features=features)
        print(f"Wilcoxon BERT vs best classical | metric={metric} split={split} feat={bool(features)}")
        if res.get("n"):
            print(f"  n_organisms={res['n']}  median(bert-best)={res['median_diff_bert_minus_best']:+.4f}"
                  f"  p={res['p_value']:.4f}")
        else:
            print("  not enough data")
        return res

    if model_b is None:
        raise ValueError("provide model_b, or set bert_vs_best=True")
    res = paired_wilcoxon(df, model_a, model_b, metric=metric, split=split, features=features)
    print(f"Wilcoxon {model_a} vs {model_b} | metric={metric} split={split} feat={bool(features)}")
    print(f"  n_organisms={res['n']}  median_diff={res['median_diff']:+.4f}  p={res['p_value']:.4f}")
    return res
