# PeptideTransformer

Fine-Tuning von [ProtBERT](https://huggingface.co/Rostlab/prot_bert_bfd) für zwei antimikrobielle Peptid-Aufgaben:
- **Hämolyse-Klassifikation** - binäre Vorhersage hämolytischer Aktivität gegen menschliche Blutzellen
- **MIC-Regression** - Vorhersage der minimalen Hemmkonzentration (log₁₀ µM) pro Organismus

Entstanden im Rahmen meiner Masterarbeit. Klassische ML-Baselines (ExtraTrees, XGBoost, RF, SVC/SVR) sind zum Vergleich enthalten.

---

## Repository-Struktur

```
data/                   # Standard-Eingabeort für die Befehle unten, kein Zwang - eigene Pfade per --data_dir möglich
  hemo_train/           # Hämolyse-Klassifikationsdaten (vorab gesplittet)
  regression_data/      # MIC-Regressionsdaten pro Organismus (vorab gesplittet)
  gram/                 # Gram-Färbung Klassifikationsdaten
  data_from_database/   # Rohdaten aus öffentlichen Datenbanken

src/
  cli.py                # CLI-Logik (alle Subcommands)
  __main__.py           # Entry Point für `python -m src`
  bert_model/
    PeptideBERTClasses/ # Model, Trainer, Dataset, Callbacks
    peptideBERT_configs/# YAML-Configs — eine pro Organismus / Run
    fine_tune_protBERT.py
    fine_tune_utils.py
    transformer_metrics.py
  data_preprocessing/   # Datenaufbereitung und Train/Val/Test-Split (Showcase, s. u.)
  data_analysis/        # Embedding-Clustering (PCA, t-SNE, UMAP) + Daten-Inspektionsplots
  evaluation/           # Gemeinsame Evaluations- und Plot-Utilities
  machine_learning/     # Klassisches ML-Training (train_models.py) und Grid Search
```

---

## Hinweis zu `data_preprocessing/`

Diese Skripte basieren auf internen Rohdaten und sind ohne diese nicht direkt ausführbar. 
Sie sind im Repository enthalten, um den Aufbereitungs-Workflow nachvollziehbar zu machen.

---

## Setup

**Option A - conda (empfohlen):**
```bash
conda env create -f environment.yml
conda activate peptide-transformer
```
> `pytorch-cuda` in `environment.yml` an die eigene Treiberversion anpassen (11.8 / 12.1 / 12.4).
> Für CPU-only die Zeile einfach entfernen.

**Option B - pip:**
```bash
# Zuerst PyTorch installieren (CUDA-Version anpassen):
pip install torch==2.4.0 --extra-index-url https://download.pytorch.org/whl/cu121
# Danach den Rest:
pip install -r requirements.txt
```

**Als editierbares Paket installieren** 
```bash
pip install -e .
```
> `-e` sorgt dafür, dass Änderungen am Code (z. B. neue Configs) sofort wirksam sind, ohne Neuinstallation.

Damit steht der Befehl `peptide-transformer` systemweit zur Verfügung und kann anstelle von `python -m src` verwendet werden, unabhängig vom aktuellen Arbeitsverzeichnis:
```bash
peptide-transformer ml_classify --data_dir data/hemo_train/
```

Ohne Installation als Paket gehen alle folgenden Befehle (`python -m src ...`) davon aus, dass im **Repo-Root** ausgeführt wird.

---

## Nutzung

### Daten splitten

Erzeugt `_train.csv`, `_val.csv`, `_test.csv` neben jeder Quelldatei.
Stratifiziert nach Label (Klassifikation) bzw. Quantil-Bins (Regression).

Muss einmalig vor dem Training ausgeführt werden, damit feste Splits für alle Modelle/Vergleiche existieren.

> **Wichtig:** `train_file` (BERT-Configs) und `--data_dir` (ML-Baselines) referenzieren den
> **Basisnamen ohne** `_train`/`_val`/`_test`-Suffix - die Suffixe werden vom Code intern angehängt.

```bash
python -m src data_init --task cls         # Hämolyse-Klassifikation
python -m src data_init --task regression  # MIC-Regression
python -m src data_init --task gram        # Gram-Klassifikation
```

Eigenes Datenverzeichnis:
```bash
python -m src data_init --task cls --data_dir path/to/csvs/
```

---

### PeptideBERT fine-tunen

Neue YAML-Config generieren:
```bash
python -m src generate_bert_model_config \
    --config_name your_config.yaml \
    --file_path path/to/your/configs/
```

Erstellt ein vollständig kommentiertes Template. Pflichtfelder zum Ausfüllen: `train_file`, `model_save_path`, `output_dir`, `logging_dir`.

Einzelne Config (eigener Pfad):
```bash
python -m src bert_model --config_path path/to/your_config.yaml
```

Kurzname - wird in `src/bert_model/peptideBERT_configs/` gesucht:
```bash
python -m src bert_model --config_path your_config.yaml
```

Batch - führt alle `.yaml`-Dateien in einem Verzeichnis aus:
```bash
python -m src bert_model --pipe_configs path/to/your/config_dir/
python -m src bert_model --pipe_configs default   # Standard-Verzeichnis für Batch-Runs: src/bert_model/peptideBERT_configs/config_pipe_dir/
```

---

### Klassische ML-Baselines

Grid Search über ExtraTrees, XGBoost, RF, SVC/SVR. Nutzt k-mer TF-IDF + optionale Peptid-Deskriptoren.
Ergebnisse (Plots, Metriken) landen standardmäßig in `final_plots/`.

```bash
python -m src ml_classify --data_dir data/hemo_train/      --output_dir final_plots/  # Hämolyse-Klassifikation
python -m src ml_regress  --data_dir data/regression_data/ --output_dir final_plots/  # MIC-Regression
```

> Mit `--features` lässt sich zusätzlich biochemisches Feature Engineering aktivieren (sonst nur k-mer TF-IDF).

---

## Config-Referenz

Wichtige Felder einer Trainings-YAML:

| Feld | Beschreibung |
|---|---|
| `model_class` | `binary_dense` (Klassifikation) oder `regression` |
| `train_file` | Repo-root-relativer Pfad zur Basis-CSV |
| `model_path` | HuggingFace-Modell-ID oder lokaler Pfad |
| `model_save_path` | Speicherort des trainierten Modells |
| `num_train_epochs` | Maximale Anzahl Epochen |
| `early_stopping_patience` | Epochen ohne Verbesserung bis zum Abbruch |
| `early_stop_warm_up` | Epochen, bevor Early Stopping aktiv wird |
| `early_stop_metric` | Überwachte Metrik (z. B. `eval_loss`) |
| `use_cpu` | `true`, um Training auf CPU zu erzwingen |
| `fast_debug_mode` | `true`, um Trainingsdaten für Pipeline-Tests auf 100 Zeilen zu kürzen |

Für alle weiteren Optionen siehe eine bestehende Config in `src/bert_model/peptideBERT_configs/`.

---

## Reproduzierbarkeit

Datensplits, ML-Baselines und Clustering nutzen einen festen `random_state=42`. 
Beim BERT-Training wird der Standard-Seed (`42`) der HuggingFace `TrainingArguments` verwendet.
