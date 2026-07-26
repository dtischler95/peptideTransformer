"""Rebuild RESULTS.md tables from the per-model metrics.json files.

One-shot scaffold. Re-run after new runs land (e.g. BERT/ESM) and extend the
model column lists below. Reads every results/**/metrics.json, aggregates
mean +/- SD over seeds, writes the full RESULTS.md.
"""
import json, glob, collections, statistics as st, itertools
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare, rankdata

OUT = r"E:\Test\peptideTransformer\RESULTS.md"

MIC_NAMES = {
    "acinetobacter_baumannii": "Acinetobacter baumannii",
    "bacillus_subtilis": "Bacillus subtilis",
    "candida_albicans": "Candida albicans",
    "enterobacter_for": "Enterobacter sp.",
    "enterococcus_faecalis": "Enterococcus faecalis",
    "escherichia_coli": "Escherichia coli",
    "klebsiella_pneumoniae": "Klebsiella pneumoniae",
    "micrococcus_luteus": "Micrococcus luteus",
    "pseudomonas_aeruginosa": "Pseudomonas aeruginosa",
    "salmonella_enterica": "Salmonella enterica",
    "staphylococcus_aureus": "Staphylococcus aureus",
    "staphylococcus_epidermidis": "Staphylococcus epidermidis",
}
HEMO_NAMES = {"whitelab_hemo": "WhiteLab", "threshold_style": "Threshold"}

MIC_MODELS = ["dummy", "ridge", "svr", "xgb", "xtra", "bert"]
HEMO_MODELS = ["dummy", "logreg", "svc", "xgb", "xtra"]
WILCOXON_MODELS = ["ridge", "svr", "xgb", "xtra"]
# The candidate models entering the significance analysis, BERT included now that its
# MIC sweep covers all 12 organisms. dummy stays out (it is the floor, not a contender).
MIC_STAT_MODELS = ["ridge", "svr", "xgb", "xtra", "bert"]
# Studentized range q at alpha=0.05 for the Nemenyi post-hoc (Demsar 2006, Table 5).
_NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
                7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

rows = [json.load(open(f)) for f in glob.glob("**/metrics.json", recursive=True)]

# BERT MIC coverage, derived so the status prose below cannot go stale.
MIC_ALL = list(MIC_NAMES)
_bert_done = {r["data_name"] for r in rows if r["model"] == "bert" and r["task"] == "mic"}
_bert_n = len(_bert_done)
_bert_pending = [MIC_NAMES[k] for k in MIC_ALL if k not in _bert_done]
_pending_disp = ", ".join(f"*{n}*" for n in _bert_pending) if _bert_pending else "none"


def collect(task, split, metric):
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    ntest = collections.defaultdict(int)
    for r in rows:
        if r["task"] == task and r["split"] == split and metric in r:
            d[r["data_name"]][r["model"]].append(r[metric])
            ntest[r["data_name"]] = max(ntest[r["data_name"]], r.get("n", 0))
    return d, ntest


def fmt(vals):
    if not vals:
        return "—"
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.2f} ±{s:.2f}"


def table_single(task, split, metric, models, names, corner):
    d, ntest = collect(task, split, metric)
    order = sorted(d.keys(), key=lambda k: -ntest[k])
    out = ["| " + corner + " | n test | " + " | ".join(models) + " |",
           "|" + "---|" * (len(models) + 2)]
    for k in order:
        cells = [fmt(d[k].get(mo, [])) for mo in models]
        disp = names.get(k, k)
        disp = f"*{disp}*" if task == "mic" else disp
        out.append(f"| {disp} | {ntest[k]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_split(task, metric, models, names, corner):
    drand, ntr = collect(task, "random", metric)
    order = sorted(drand.keys(), key=lambda k: -ntr[k])
    out = ["| " + corner + " | Split | n test | " + " | ".join(models) + " |",
           "|" + "---|" * (len(models) + 3)]
    for k in order:
        for split in ["random", "cluster"]:
            d, nt = collect(task, split, metric)
            cells = [fmt(d[k].get(mo, [])) for mo in models]
            out.append(f"| {names.get(k, k)} | {split} | {nt[k]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


# Wilcoxon helpers. Pair on per-organism mean over seeds, the organisms are the units.
def organism_means(task, split, metric, model):
    d, _ = collect(task, split, metric)
    return {k: st.mean(v[model]) for k, v in d.items() if v.get(model)}


def _pair(a, b):
    keys = sorted(set(a) & set(b))
    return [a[k] for k in keys], [b[k] for k in keys]


def _rank_biserial(diffs):
    """Matched-pairs rank-biserial effect size r = (R+ - R-) / (R+ + R-).
    Sign follows the difference direction, zeros dropped. |r| near 1 = uniform effect."""
    d = np.array([x for x in diffs if x != 0], dtype=float)
    if d.size == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    rp, rm = ranks[d > 0].sum(), ranks[d < 0].sum()
    tot = rp + rm
    return float((rp - rm) / tot) if tot else 0.0


def _paired_stats(a, b):
    diffs = [x - y for x, y in zip(a, b)]
    w, p = wilcoxon(a, b)
    return {"n": len(a), "mean_a": st.mean(a), "mean_b": st.mean(b),
            "mean_d": st.mean(diffs), "w": w, "p": p, "r": _rank_biserial(diffs),
            "gt": sum(d > 0 for d in diffs), "lt": sum(d < 0 for d in diffs)}


def _holm(labeled_p):
    """Holm-Bonferroni over [(label, p), ...]. Returns {label: adjusted p}."""
    m = len(labeled_p)
    adj, running = {}, 0.0
    for i, (lab, p) in enumerate(sorted(labeled_p, key=lambda kv: kv[1])):
        running = max(running, min(1.0, (m - i) * p))
        adj[lab] = running
    return adj


def friedman_nemenyi(task, metric, split, models):
    """Friedman omnibus over per-organism ranks (1 = best, higher metric better) plus the
    Nemenyi critical difference. Returns (rank_table_md, chi2, p, cd, N, significant_pairs)."""
    means = {mo: organism_means(task, split, metric, mo) for mo in models}
    orgs = sorted(set.intersection(*(set(means[mo]) for mo in models)))
    mat = np.array([[means[mo][o] for mo in models] for o in orgs], dtype=float)
    ranks = np.vstack([rankdata(-row, method="average") for row in mat])
    mean_ranks = ranks.mean(axis=0)
    chi2, p = friedmanchisquare(*[mat[:, j] for j in range(len(models))])
    k, N = len(models), len(orgs)
    cd = _NEMENYI_Q05[k] * (k * (k + 1) / (6 * N)) ** 0.5
    order = np.argsort(mean_ranks)  # lower rank = better
    tbl = ["| Rank | Model | mean rank |", "|---|---|---|"]
    for pos, j in enumerate(order, start=1):
        tbl.append(f"| {pos} | {models[j]} | {mean_ranks[j]:.2f} |")
    sig = []
    for i, jj in itertools.combinations(range(k), 2):
        if abs(mean_ranks[i] - mean_ranks[jj]) > cd:
            better, worse = (i, jj) if mean_ranks[i] < mean_ranks[jj] else (jj, i)
            sig.append(f"{models[better]} > {models[worse]}")
    return "\n".join(tbl), chi2, p, cd, N, sig


def friedman_block(split):
    tbl, chi2, p, cd, N, sig = friedman_nemenyi("mic", "r2", split, MIC_STAT_MODELS)
    sig_txt = ", ".join(sig) if sig else "no pair exceeds the critical difference"
    k = len(MIC_STAT_MODELS)
    return (f"Friedman χ²({k - 1}) = {chi2:.2f}, p = {p:.2e}, N = {N} organisms.\n\n"
            f"{tbl}\n\n"
            f"Nemenyi critical difference (α = 0.05): CD = {cd:.2f} mean-rank units. "
            f"Pairs beyond CD: {sig_txt}.")


def wilcox_split_table(task, metric, models):
    out = ["| Model | n | mean random | mean cluster | mean Δ | r | W | p |",
           "|---|---|---|---|---|---|---|---|"]
    for mo in models:
        r, c = _pair(organism_means(task, "random", metric, mo),
                     organism_means(task, "cluster", metric, mo))
        s = _paired_stats(r, c)
        out.append(f"| {mo} | {s['n']} | {s['mean_a']:.3f} | {s['mean_b']:.3f} | "
                   f"{s['mean_d']:+.3f} | {s['r']:+.2f} | {s['w']:.0f} | {s['p']:.4f} |")
    return "\n".join(out)


def wilcox_model_table(task, metric, split, models):
    rowstats = []
    for a, b in itertools.combinations(models, 2):
        va, vb = _pair(organism_means(task, split, metric, a),
                       organism_means(task, split, metric, b))
        rowstats.append((f"{a} vs {b}", _paired_stats(va, vb)))
    adj = _holm([(lab, s["p"]) for lab, s in rowstats])
    out = ["| Comparison | n | mean Δ | r | Δ>0 | Δ<0 | W | p | p (Holm) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for lab, s in rowstats:
        out.append(f"| {lab} | {s['n']} | {s['mean_d']:+.3f} | {s['r']:+.2f} | "
                   f"{s['gt']} | {s['lt']} | {s['w']:.0f} | {s['p']:.4f} | {adj[lab]:.4f} |")
    return "\n".join(out)


def hemo_split_seed_test(metric="mcc"):
    # Hemo has only 2 datasets, so no MIC-style cross-dataset pairing. Pair random vs
    # cluster over the 5 seeds within each dataset. One-sided: the drop direction is
    # pre-specified from the MIC result and the leakage mechanism.
    order = ["whitelab_hemo", "threshold_style"]
    models = [m for m in HEMO_MODELS if m != "dummy"]
    out = ["| Dataset | Model | mean random | mean cluster | mean Δ | seeds down | p (1-sided) |",
           "|---|---|---|---|---|---|---|"]
    for ds in order:
        for mo in models:
            per = {"random": {}, "cluster": {}}
            for r in rows:
                if r["task"] == "hemo" and r["data_name"] == ds and r["model"] == mo and metric in r:
                    per[r["split"]][r["seed"]] = r[metric]
            ks = sorted(set(per["random"]) & set(per["cluster"]))
            a = [per["random"][k] for k in ks]
            b = [per["cluster"][k] for k in ks]
            diffs = [x - y for x, y in zip(a, b)]
            try:
                _, p = wilcoxon(a, b, alternative="greater")
                pstr = f"{p:.4f}"
            except ValueError:
                pstr = "—"
            out.append(f"| {HEMO_NAMES[ds]} | {mo} | {st.mean(a):.3f} | {st.mean(b):.3f} | "
                       f"{st.mean(diffs):+.3f} | {sum(d > 0 for d in diffs)}/{len(ks)} | {pstr} |")
    return "\n".join(out)


SIZES = """### Dataset sizes

Sequence-level split, roughly 64 / 16 / 20 train / val / test. Totals are identical for the random and
cluster splits, since both use the same peptides in a different partition, and the cluster split differs
by at most two sequences per fold.

MIC regression:

| Organism | n train | n val | n test | n total |
|---|---|---|---|---|
| *Acinetobacter baumannii* | 1094 | 274 | 342 | 1710 |
| *Bacillus subtilis* | 1910 | 478 | 598 | 2986 |
| *Candida albicans* | 2227 | 557 | 696 | 3480 |
| *Enterobacter* sp. | 325 | 82 | 102 | 509 |
| *Enterococcus faecalis* | 1120 | 280 | 351 | 1751 |
| *Escherichia coli* | 6096 | 1524 | 1906 | 9526 |
| *Klebsiella pneumoniae* | 1478 | 370 | 462 | 2310 |
| *Micrococcus luteus* | 712 | 179 | 223 | 1114 |
| *Pseudomonas aeruginosa* | 4152 | 1038 | 1298 | 6488 |
| *Salmonella enterica* | 1052 | 264 | 330 | 1646 |
| *Staphylococcus aureus* | 5831 | 1458 | 1823 | 9112 |
| *Staphylococcus epidermidis* | 1475 | 369 | 462 | 2306 |

Hemolysis classification (positives = hemolytic fraction over the whole dataset):

| Dataset | n train | n val | n test | n total | Positives |
|---|---|---|---|---|---|
| WhiteLab | 3676 | 920 | 1150 | 5746 | about 16 % |
| Threshold | 2693 | 674 | 842 | 4209 | about 34 % |
"""

PROTOCOL = """## Evaluation protocol

Both model families select on validation data and report on the same untouched test split, so the
comparison is like-for-like in data budget.

- BERT trains on the train split, uses the validation split for early stopping and best-checkpoint
  selection, and is scored once on the test split.
- Classical ML chooses hyperparameters by 5-fold cross-validation over the combined train+val pool
  (stratified folds for classification, plain 5-fold for regression), refits the best configuration on
  all of train+val, and is scored once on the test split. The whole featuriser (k-mer TF-IDF, SVD,
  descriptor imputation, scaling) sits inside an sklearn Pipeline, so it is refit within each fold and
  nothing leaks from a fold's held-out data.

The test split is never seen during model selection in either pipeline. Each configuration is run for
five seeds (1, 2, 3, 4, 42) and the tables report mean ± SD across them. Per-model numbers are read from
each run's `metrics.json`. No cross-organism averages are reported: the organisms are clinically selected,
independent datasets, so each one stands on its own.

One asymmetry to keep in mind when reading the model comparison: the classical models are tuned by
per-organism grid search, while BERT runs one fixed optimizer configuration for every organism (learning
rate 1e-5, weight decay 0.01, ReduceLROnPlateau, up to 50 epochs with early stopping on validation loss,
patience 5). BERT is therefore evaluated at a single sensible setting, not a per-dataset-tuned optimum. A
BERT deficit on small datasets should be read with this in mind, it can reflect the fixed configuration
rather than the architecture.
"""

# Interpretive prose for the model comparison, computed so it cannot go stale on re-run.
_STRONG = ["svr", "xgb", "xtra", "bert"]


def _split_delta_range():
    ds = []
    for mo in MIC_STAT_MODELS:
        r, c = _pair(organism_means("mic", "random", "r2", mo),
                     organism_means("mic", "cluster", "r2", mo))
        ds.append(st.mean([x - y for x, y in zip(r, c)]))
    return min(ds), max(ds)


def _ranks_and_cd(split):
    means = {mo: organism_means("mic", split, "r2", mo) for mo in MIC_STAT_MODELS}
    orgs = sorted(set.intersection(*(set(means[mo]) for mo in MIC_STAT_MODELS)))
    mat = np.array([[means[mo][o] for mo in MIC_STAT_MODELS] for o in orgs], dtype=float)
    mr = np.vstack([rankdata(-row, method="average") for row in mat]).mean(axis=0)
    k, N = len(MIC_STAT_MODELS), len(orgs)
    cd = _NEMENYI_Q05[k] * (k * (k + 1) / (6 * N)) ** 0.5
    return dict(zip(MIC_STAT_MODELS, mr)), cd


def _model_prose():
    lo, hi = _split_delta_range()
    parts = []
    for split in ("random", "cluster"):
        mr, cd = _ranks_and_cd(split)
        seps = [f"{a} > {b}" if mr[a] < mr[b] else f"{b} > {a}"
                for a, b in itertools.combinations(_STRONG, 2) if abs(mr[a] - mr[b]) > cd]
        parts.append(f"on the {split} split it separates {', '.join(seps)} among the strong models"
                     if seps else
                     f"on the {split} split no two of the strong models (svr, xgb, xtra, bert) are separable")
    return (f"The split effect is uniform and large: every model loses between {lo:.2f} and {hi:.2f} R² "
            f"under clustering, with rank-biserial r at or near +1. The model ranking is the smaller lever. "
            f"By Nemenyi, {parts[0]}, and {parts[1]}. Read against the split drop, the gaps among the strong "
            f"models are second order, the split is the dominant axis.")


MODEL_PROSE = _model_prose()

doc = f"""<!-- Tables auto-generated by results/_make_results_tables.py from metrics.json. -->
<!-- Re-run after new runs land. -->

# Results

Status 2026-07-26. The classical ML baseline (sequence only) is complete and the ProtBERT MIC sweep now
covers {_bert_n} of the 12 organisms (random and cluster, five seeds each). Hemolysis BERT and ESM are
still pending.

All numbers are test-set metrics, mean ± SD over five seeds (1, 2, 3, 4, 42). The `dummy` column is a
mean / majority-class baseline (floor); `ridge` / `logreg` is a linear reference. `svr`/`svc`, `xgb` and
`xtra` (ExtraTrees) are the headline classical models.

{SIZES}
{PROTOCOL}
## MIC regression (sequence only)

Per organism, mean ± SD over five seeds. `dummy` predicts the train mean.

The `bert` column is the fine-tuned ProtBERT sweep, mean ± SD over the same five seeds. It currently
covers {_bert_n} of the 12 organisms. A `—` in the `bert` column means that organism's BERT run is still pending.

### Test R² (higher is better)

**Random split**

{table_single("mic", "random", "r2", MIC_MODELS, MIC_NAMES, "Organism")}

**Cluster split** (MMseqs2, min-seq-id 0.5)

{table_single("mic", "cluster", "r2", MIC_MODELS, MIC_NAMES, "Organism")}

### Test MSE (lower is better)

**Random split**

{table_single("mic", "random", "mse", MIC_MODELS, MIC_NAMES, "Organism")}

**Cluster split**

{table_single("mic", "cluster", "mse", MIC_MODELS, MIC_NAMES, "Organism")}

## Statistical comparison (MIC)

Paired tests over the 12 MIC organisms. Each organism contributes one value per model, its mean test R²
over the five seeds, and the 12 organisms are the paired units. All five candidate models are compared,
BERT included now that its sweep covers all 12 organisms. Values come from the full-precision per-run
metrics, not the rounded table cells above, so a few near-ties resolve differently.

### Model ranking — Friedman + Nemenyi

The Demšar (2006) protocol for comparing several models over several datasets: a Friedman omnibus on the
per-organism ranks, then the Nemenyi post-hoc. Two models differ significantly only if their mean ranks
are more than one critical difference (CD) apart. This is the primary model comparison, it controls the
family-wise error that running all pairwise tests separately does not.

**Random split**

{friedman_block("random")}

**Cluster split**

{friedman_block("cluster")}

### Random vs cluster split (per model)

Same model, the two splits paired per organism. A positive Δ means the random split scores higher, that
is a drop under clustering. `r` is the matched-pairs rank-biserial effect size (+1 = every organism drops).

{wilcox_split_table("mic", "r2", MIC_STAT_MODELS)}

### Model vs model, same split

All ten pairwise comparisons of the five models, Holm-corrected within each split. Δ = R²(first) −
R²(second), so a negative Δ means the second model is stronger. `r` is the rank-biserial effect size.
These pairwise tests are secondary to the Friedman + Nemenyi ranking above and should be read for
direction and effect size, not for the p-values alone: with 12 pairs the two-sided Wilcoxon floors at
p ≈ 0.0005, so a small but perfectly consistent gap and a large one land at the same p.

**Random split**

{wilcox_model_table("mic", "r2", "random", MIC_STAT_MODELS)}

**Cluster split**

{wilcox_model_table("mic", "r2", "cluster", MIC_STAT_MODELS)}

{MODEL_PROSE}

## Hemolysis classification — classical ML baseline (sequence only)

Per dataset and split, mean ± SD over five seeds. `dummy` = majority-class baseline.

### MCC

{table_split("hemo", "mcc", HEMO_MODELS, HEMO_NAMES, "Dataset")}

### AUROC

{table_split("hemo", "auroc", HEMO_MODELS, HEMO_NAMES, "Dataset")}

### Average precision (AP)

{table_split("hemo", "ap", HEMO_MODELS, HEMO_NAMES, "Dataset")}

### Accuracy

{table_split("hemo", "acc", HEMO_MODELS, HEMO_NAMES, "Dataset")}

### Macro-F1

{table_split("hemo", "f1_macro", HEMO_MODELS, HEMO_NAMES, "Dataset")}

The two hemolysis datasets are not independent: about 91 % of the Threshold sequences also appear in
WhiteLab, so this is one set of peptides under two labeling rules.

### Statistical comparison — cluster split effect (Hemolysis)

The MIC test pairs over 12 organisms. Hemolysis has only two datasets, and they overlap by about 91 % in
sequence, so there is no equivalent cross-dataset paired test. Instead the cluster split effect is tested
within each dataset, pairing random against cluster over the five seeds (MCC). The direction is fixed in
advance from the MIC result and the leakage mechanism, so the test is one-sided (random > cluster).

With five seeds the one-sided Wilcoxon bottoms out at p = 0.031 (all five seeds agreeing), and a two-sided
test cannot reach below 0.0625, so this is underpowered by construction. The consistent direction across
seeds carries more weight than the exact p-value.

{hemo_split_seed_test("mcc")}

The cluster split lowers MCC on both datasets by about 0.08 to 0.21, consistent across seeds. Same pattern
as MIC, the near-duplicate peptides in the random split flatter the test score.

---

## ProtBERT (partial)

The `bert` column in the MIC tables above holds the finished runs ({_bert_n} of 12 organisms, random and
cluster, five seeds each). Pending organisms: {_pending_disp}. Hemolysis BERT is not started yet.

## ESM (pending)

Queued after ProtBERT.
"""

with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(doc)

print("wrote", OUT, len(doc), "chars")
