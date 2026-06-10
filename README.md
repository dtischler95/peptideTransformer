# PeptideTransformer

Fine-tuning [ProtBERT](https://huggingface.co/Rostlab/prot_bert_bfd) for two antimicrobial peptide tasks:
- **Hemolysis classification** - binary prediction of hemolytic activity against human blood cells
- **MIC regression** - prediction of the minimum inhibitory concentration (log₁₀ µM) per organism

Created as part of my Master's thesis. Classical ML baselines (ExtraTrees, XGBoost, RF, SVC/SVR) are included for comparison.

---

## Repository structure

```
data/                   # Default input location for the commands below, not mandatory - custom paths via --data_dir possible
  hemo_train/           # Hemolysis classification data (pre-split)
  regression_data/      # MIC regression data per organism (pre-split)
  gram/                 # Gram staining classification data
  data_from_database/   # Raw data from public databases

src/
  cli.py                # CLI logic (all subcommands)
  __main__.py           # Entry point for `python -m src`
  bert_model/
    PeptideBERTClasses/ # Model, trainer, dataset, callbacks
    peptideBERT_configs/# YAML configs - one per organism / run
    fine_tune_protBERT.py
    fine_tune_utils.py
    transformer_metrics.py
  data_preprocessing/   # Data preparation and train/val/test split (see note below)
  data_analysis/        # Embedding clustering (PCA, t-SNE, UMAP) + data inspection plots
  evaluation/           # Shared evaluation and plotting utilities
  machine_learning/     # Classical ML training (train_models.py) and grid search
```

---

## Note on `data_preprocessing/`

These scripts rely on internal raw data and are not directly runnable without it.
They are included in the repository to make the preparation workflow traceable.

---

## Setup

**Option A - conda (recommended):**
```bash
conda env create -f environment.yml
conda activate peptide-transformer
```
> Adjust `pytorch-cuda` in `environment.yml` to match your driver version (11.8 / 12.1 / 12.4).
> For CPU-only, simply remove that line.

**Option B - pip:**
```bash
# First install PyTorch (adjust the CUDA version):
pip install torch==2.4.0 --extra-index-url https://download.pytorch.org/whl/cu121
# Then the rest:
pip install -r requirements.txt
```

**Install as an editable package**
```bash
pip install -e .
```
> `-e` makes code changes (e.g. new configs) take effect immediately, without reinstalling.

This makes the `peptide-transformer` command available system-wide, usable instead of `python -m src`, regardless of the current working directory:
```bash
peptide-transformer ml_classify --data_dir data/hemo_train/
```

Without installing as a package, all commands below (`python -m src ...`) assume execution from the **repo root**.

---

## Usage

### Splitting data

Creates `_train.csv`, `_val.csv`, `_test.csv` next to each source file.
Stratified by label (classification) or quantile bins (regression).

Must be run once before training so that fixed splits exist for all models/comparisons.

> **Important:** `train_file` (BERT configs) and `--data_dir` (ML baselines) reference the
> **base name without** the `_train`/`_val`/`_test` suffix - the suffixes are appended internally by the code.

```bash
python -m src data_init --task cls         # Hemolysis classification
python -m src data_init --task regression  # MIC regression
python -m src data_init --task gram        # Gram classification
```

Custom data directory:
```bash
python -m src data_init --task cls --data_dir path/to/csvs/
```

---

### Fine-tuning PeptideBERT

Generate a new YAML config:
```bash
python -m src generate_bert_model_config \
    --config_name your_config.yaml \
    --file_path path/to/your/configs/
```

Creates a fully commented template. Required fields to fill in: `train_file`, `model_save_path`, `output_dir`, `logging_dir`.

Single config (custom path):
```bash
python -m src bert_model --config_path path/to/your_config.yaml
```

Short name - looked up in `src/bert_model/peptideBERT_configs/`:
```bash
python -m src bert_model --config_path your_config.yaml
```

Batch - runs all `.yaml` files in a directory:
```bash
python -m src bert_model --pipe_configs path/to/your/config_dir/
python -m src bert_model --pipe_configs default   # default directory for batch runs: src/bert_model/peptideBERT_configs/config_pipe_dir/
```

---

### Classical ML baselines

Grid search over ExtraTrees, XGBoost, RF, SVC/SVR. Uses k-mer TF-IDF + optional peptide descriptors.
Results (plots, metrics) are saved to `final_plots/` by default.

```bash
python -m src ml_classify --data_dir data/hemo_train/      --output_dir final_plots/  # Hemolysis classification
python -m src ml_regress  --data_dir data/regression_data/ --output_dir final_plots/  # MIC regression
```

> `--features` additionally enables biochemical feature engineering (otherwise k-mer TF-IDF only).

---

## Config reference

Important fields of a training YAML:

| Field | Description |
|---|---|
| `model_class` | `binary_dense` (classification) or `regression` |
| `train_file` | Repo-root-relative path to the base CSV |
| `model_path` | HuggingFace model ID or local path |
| `model_save_path` | Where to save the trained model |
| `num_train_epochs` | Maximum number of epochs |
| `early_stopping_patience` | Epochs without improvement before stopping |
| `early_stop_warm_up` | Epochs before early stopping becomes active |
| `early_stop_metric` | Monitored metric (e.g. `eval_loss`) |
| `use_cpu` | `true` to force training on CPU |
| `fast_debug_mode` | `true` to truncate training data to 100 rows for pipeline tests |

For all other options, see an existing config in `src/bert_model/peptideBERT_configs/`.

---

## Reproducibility

Data splits, ML baselines and clustering use a fixed `random_state=42`.
For BERT training, the default seed (`42`) of HuggingFace `TrainingArguments` is used.
