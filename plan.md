# Plan: ESM-2 Backbone + fairer Vergleich (ML-Baseline vs ProtBERT vs ESM)

Arbeitsstand `add-esm-backbone`. Ziel: veröffentlichungsfähiger, fairer Vergleich.

## Status
- [x] ESM-2 Backbone via CLI-Overrides (`--backbone esm --model_path ... --model_name ...`), committed `6287cc8`.
- [x] `join_residues`-Flag entfernt → Space-Trennung wird aus `backbone` abgeleitet (ProtBERT: Spaces, ESM: rohe Sequenz). *Noch nicht committed.*
- [ ] ESM-Smoke-Test (noch offen, war beim Crash dran).

## Nächste Schritte (lokal zuhause)
1. **ESM-Smoke-Test**: 1 kleiner Organismus (*Enterobacter*), prüfen dass Tokenisierung + Training durchläuft.
   `python -m src bert_model --config_path <enterobacter>.yaml --backbone esm --model_path facebook/esm2_t33_650M_UR50D --model_name esm650m`
   - Sanity: ESM-Token-Counts plausibel (keine Spaces, ~1 Token/Residue).
2. **LR-Sweep** (siehe Protokoll unten).
3. **Großer Lauf**: gewählte LR × 12 Organismen × 5 Seeds (1,2,3,4,42) × 2 Splits.
4. RESULTS.md regenerieren, paired-Wilcoxon.

## LR-Sweep-Protokoll
- **Grid (log-spaced, 1e-5 beidseitig eingeklammert):** `5e-6, 1e-5, 3e-5`.
  - Adaptives Bracketing: Gewinner am Rand → 1 Punkt nachlegen (5e-6 gewinnt → 1e-6; 5e-5 gewinnt → 1e-4).
  - Hinweis: `reduce_lr_on_plateau` senkt LR nur → zu niedrige Start-LR (1e-6) underfittet, selten Gewinner.
- **Granularität: pro Backbone (Model), NICHT pro Organismus.** Eine geteilte LR uniform über alle 12.
- **Auswahl-Basis:** 1 großer (*E. coli*/*S. aureus*) + 1 kleiner (*Enterobacter*) Organismus, **1 Seed**, **nur Random-Split**, **val-Metrik**. Dieselben Repräsentanten für BERT und ESM.
- **Entscheidungsregel:** beste LR ODER innerhalb Seed-SD des Besten → behalten.
  - BERT-Default ist bereits 1e-5 → wenn 1e-5 gewinnt/innerhalb SD, **bestehende 5-Seed-bert-Läufe bleiben gültig (kein Neurechnen)**.
  - LR ehrlich nach val wählen, NICHT um Compute zu retten.
- **Seeds ≠ HP-Auswahl:** HP einmal auf val wählen, danach 5 Seeds nur zur Varianz der fixen Config. Seed-42-Lauf der Gewinner-LR als einen der 5 recyceln.

## Fairness-Notizen fürs Paper (offenlegen)
- ESM-Größe vs ProtBERT (~420M): Parameterzahl + Pretraining-Korpus je Backbone in Tabelle; ggf. ESM-Größenleiter als Scaling-Argument.
- Asymmetrie „klassisch = Grid-Search, Transformer = fixe Config (Early-Stopping auf val)" explizit benennen + begründen (Compute).
- FT-Regime identisch für BERT & ESM (aktuell full fine-tune).
- Primärer Endpunkt vorab festlegen (z. B. Cluster-Split R² / MCC) → Multiple-Comparison entschärfen.
- Bereits stark: gleiche Splits, Testset unangetastet, Cluster-Split (Homologie), 5 Seeds mean±SD, gepaarter Wilcoxon, kein Cross-Organismus-Mittel.

## Offene Entscheidungen (vor großem Lauf)
- ESM-Größe: Leiter vs eine gematchte Größe vs 650M.
- FT-Regime bestätigen (full vs frozen+Head).
- Kernaussage/Framing (PLM-Mehrwert kritisch? vs BERT-vs-ESM?).
