from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def make_seq_strat_labels(df: pd.DataFrame, task: str, target_col: str, n_bins: int = 10):
    """
    Erzeugt eine Sequenz-ebene Stratifikationsspalte:
    - Klassifikation: nutzt das (einheitliche) Klassenlabel pro Sequenz
    - Regression: nimmt Median pro Sequenz und packt ihn in Quantil-Bins
    """
    seq_grp = df.groupby('sequence', as_index=False)

    if task == "classification":
        # Mehrheitslabel pro Sequenz bilden und Duplikate entfernen
        seq_y = (
            df.groupby('sequence', as_index=False)[target_col]
            .agg(lambda x: x.value_counts().idxmax())  # Mehrheitslabel
            .rename(columns={target_col: 'majority_label'})
        )
        seq_y = seq_y.rename(columns={'majority_label': 'strat'})

    if task == "regression":
        seq_y = seq_grp[target_col].median().rename(columns={target_col: 'y_median'})
        # Quantil-Bins (robust gegen Ties)
        # Falls wenige unique Werte, reduziert qcut automatisch die Anzahl an Bins
        try:
            seq_y['strat'] = pd.qcut(seq_y['y_median'], q=min(n_bins, seq_y['y_median'].nunique()),
                                     duplicates='drop')
        except ValueError:
            # Fallback: alle in eine Bin (keine sinnvolle Stratifikation möglich)
            seq_y['strat'] = 0


    return seq_y[['sequence', 'strat']]

def split_with_val(tmp_df: pd.DataFrame, *, task: str, target_col: str,
                   test_size=0.20, val_size=0.16, random_state=42):


    # 1) Sequenz-Tabelle + Stratifikationslabels
    seq_df = tmp_df[['sequence', target_col]].copy()

    # nur eine Zeile pro Sequenz + strat label
    strat_df = make_seq_strat_labels(tmp_df, task=task, target_col=target_col)


    unique_seqs = strat_df['sequence'].to_frame()
    strat_labels = strat_df['strat']

    # 2) Test abspalten (z. B. 20 %)
    seq_trainval, seq_test = train_test_split(
        unique_seqs,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=strat_labels
    )

    # Mapping: welche Strat-Labels bleiben im Train/Val?
    strat_trainval = strat_df[strat_df['sequence'].isin(seq_trainval['sequence'])]['strat']


    val_rel = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
    seq_train, seq_val = train_test_split(
        seq_trainval,
        test_size=val_rel,
        random_state=random_state,
        shuffle=True,
        stratify=strat_trainval
    )

    train_df = seq_df[strat_df['sequence'].isin(seq_train['sequence'])]
    val_df   = seq_df[strat_df['sequence'].isin(seq_val['sequence'])]
    test_df  = seq_df[strat_df['sequence'].isin(seq_test['sequence'])]


    return train_df, val_df, test_df

def main(file_dir: str, task: str, target_col: str,
         test_size=0.20, val_size=0.16, random_state=42):
    for train_file in Path(file_dir).glob("*regression.csv"):
        tmp_df = pd.read_csv(train_file, sep=';')
        if task == "classification":
            try:
                tmp_df = tmp_df.drop(columns=["hemo_concentration", "hemo_percent"])
            except KeyError:
                pass

        train_df, val_df, test_df = split_with_val(
            tmp_df, task=task, target_col=target_col,
            test_size=test_size, val_size=val_size, random_state=random_state
        )

        out_base = train_file.with_suffix("")  # entfernt .csv
        train_df.to_csv(out_base.with_name(out_base.name + "_train.csv"), sep=';', index=False)
        val_df.to_csv(out_base.with_name(out_base.name + "_val.csv"), sep=';', index=False)
        test_df.to_csv(out_base.with_name(out_base.name + "_test.csv"), sep=';', index=False)

if __name__ == '__main__':

    main(file_dir='../../data/hemo_train/', task='classification', target_col='label')
    # Für Klassifikation wäre: task='classification', target_col='label'
