import pandas as pd

"""
This script is meant to gether the data for the tamper dataset. 
"""

def prepare_tamper_data():

    train_negativ = "../../../data/tamper_data/tr_neg.faa"
    train_positiv = "../../../data/tamper_data/tr_pos.faa"
    val_negativ = "../../../data/tamper_data/val_neg.faa"
    val_positiv = "../../../data/tamper_data/val_pos.faa"

    train_neg_dict = _reformat_fasta(train_negativ, label=0)
    train_pos_dict = _reformat_fasta(train_positiv, label=1)

    train_df = pd.concat([pd.DataFrame(train_neg_dict), pd.DataFrame(train_pos_dict)])

    val_neg_dict = _reformat_fasta(val_negativ, label=0)
    val_pos_dict = _reformat_fasta(val_positiv, label=1)

    val_df = pd.concat([pd.DataFrame(val_neg_dict), pd.DataFrame(val_pos_dict)])

    train_df.to_csv("../../../data/tamper_data/train_tamper.csv", index=False, sep=';')
    val_df.to_csv("../../../data/tamper_data/val_tamper.csv", index=False, sep=';')


def _reformat_fasta(train_negativ, label):
    with open(train_negativ, "r") as f:
        id_counter = 0
        sequence_counter = 0
        sequences = []
        seq_label = []
        for line in f.readlines():
            if line.startswith(">"):
                id_counter += 1
            else:
                sequence_counter += 1
                sequences.append(line.strip())
                seq_label.append(label)
                if sequence_counter > id_counter:
                    sequence_counter -= 1
                    sequences[-1] += line.strip()
        return {
            "sequence": sequences,
            "label": seq_label
        }

if __name__ == "__main__":
    prepare_tamper_data()