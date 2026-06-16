from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.data_preprocessing import datasets
except ImportError:  # allow running the file directly
    import datasets  # type: ignore

def make_seq_strat_labels(df: pd.DataFrame, task: str, target_col: str, n_bins: int = 10):
    """
    Creates a sequence-level stratification column:
    - Classification: uses the (uniform) class label per sequence
    - Regression: takes the median per sequence and bins it into quantile bins
    """
    seq_grp = df.groupby('sequence', as_index=False)

    if task == "cls":
        # build majority label per sequence and remove duplicates
        seq_y = (
            df.groupby('sequence', as_index=False)[target_col]
            .agg(lambda x: x.value_counts().idxmax())  # majority label
            .rename(columns={target_col: 'strat'})
        )


    if task == "regression":
        seq_y = seq_grp[target_col].median().rename(columns={target_col: 'y_median'})
        # quantile bins (robust against ties)
        # if few unique values, qcut automatically reduces the number of bins
        try:
            seq_y['strat'] = pd.qcut(seq_y['y_median'], q=min(n_bins, seq_y['y_median'].nunique()),
                                     duplicates='drop')
        except ValueError:
            # fallback: put everything into one bin (no meaningful stratification possible)
            seq_y['strat'] = 0

    if task == 'gram':
        seq_y = df.rename(columns={target_col: "strat"})
        return seq_y[['sequence', 'mic_log10', 'strat']]

    return seq_y[['sequence', 'strat']]

def split_with_val(tmp_df: pd.DataFrame, *, task: str, target_col: str,
                   test_size=0.20, val_size=0.16, random_state=42):


    strat_df = make_seq_strat_labels(tmp_df, task=task, target_col=target_col)


    unique_seqs = strat_df['sequence'].to_frame()
    strat_labels = strat_df['strat']

    # split off test set (e.g. 20%)
    seq_trainval, seq_test = train_test_split(
        unique_seqs,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=strat_labels
    )

    # strat-labels for the train/val portion
    strat_trainval = strat_df[strat_df['sequence'].isin(seq_trainval['sequence'])]['strat']


    val_rel = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
    seq_train, seq_val = train_test_split(
        seq_trainval,
        test_size=val_rel,
        random_state=random_state,
        shuffle=True,
        stratify=strat_trainval
    )

    if task == "cls":
        cls_strat = strat_df
    elif task == "gram":
        cls_strat = tmp_df[['sequence', 'mic_log10' ,target_col]]
    else:
        cls_strat = tmp_df[['sequence', target_col]]

    train_df = cls_strat[strat_df['sequence'].isin(seq_train['sequence'])].rename(columns={"strat": target_col})
    val_df   = cls_strat[strat_df['sequence'].isin(seq_val['sequence'])].rename(columns={"strat": target_col})
    test_df  = cls_strat[strat_df['sequence'].isin(seq_test['sequence'])].rename(columns={"strat": target_col})



    return train_df, val_df, test_df

def data_splitter(task: str,
                  data_dir: Path | str | None = None,
                  test_size=0.20, val_size=0.16, random_state=42):

    target_col = datasets.target_col(task)

    # Base files only (the registry's globs never match generated _train/_val/_test).
    for train_file in datasets.base_files(task, data_dir):
        tmp_df = pd.read_csv(train_file, sep=';')
        if task == "cls":
            try:
                tmp_df = tmp_df.drop(columns=["hemo_concentration", "hemo_percent"])
            except KeyError:
                pass

        train_df, val_df, test_df = split_with_val(
            tmp_df, task=task, target_col=target_col,
            test_size=test_size, val_size=val_size, random_state=random_state
        )

        out_base = train_file.with_suffix("")  # strip .csv suffix
        train_df.to_csv(out_base.with_name(out_base.name + "_train.csv"), sep=';', index=False)
        val_df.to_csv(out_base.with_name(out_base.name + "_val.csv"), sep=';', index=False)
        test_df.to_csv(out_base.with_name(out_base.name + "_test.csv"), sep=';', index=False)

if __name__ == '__main__':
    data_splitter(task='cls')
