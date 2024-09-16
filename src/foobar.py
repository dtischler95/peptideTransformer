from torch.utils.data import DataLoader
import numpy as np
import torch
import pandas as pd
from PeptideDataset import PeptideBERTDataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer



def load_data(config, file_path: str):
    print(f'{"="*30}{"DATA":^20}{"="*30}')

    df = pd.read_csv(file_path, sep=';')

    df_train, df_val_handler = train_test_split(df, test_size=0.2)
    df_val, df_test = train_test_split(df_val_handler, test_size=0.5)

    train_handler = df_train['sequence'].values
    train_handler_label = df_train['label'].values

    val_handler = df_val['sequence'].values
    val_handler_labels = df_val['label'].values

    test_handler = df_test['sequence'].values
    test_handler_labels = df_test['label'].values

    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', clean_up_tokenization_spaces=True)


    # TODO change max length to cofnig!!
    train_data_tokenized = tokenizer([' '.join(seq) for seq in train_handler], padding='max_length', truncation=True, return_tensors='pt', max_length=512)
    val_data_tokenized = tokenizer([' '.join(seq) for seq in val_handler], padding='max_length', truncation=True, return_tensors='pt', max_length=512)
    test_data_tokenized = tokenizer([' '.join(seq) for seq in test_handler], padding='max_length', truncation=True, return_tensors='pt', max_length=512)


    attention_mask = np.asarray(train_data_tokenized['attention_mask'], dtype=np.float64)
    attention_mask_val = np.asarray(val_data_tokenized['attention_mask'], dtype=np.float64)
    attention_mask_test = np.asarray(test_data_tokenized['attention_mask'], dtype=np.float64)

    train_inputs = train_data_tokenized['input_ids']
    val_inputs = val_data_tokenized['input_ids']
    test_inputs = test_data_tokenized['input_ids']

    train_dataset = PeptideBERTDataset(input_ids=train_inputs, attention_masks=attention_mask, labels=train_handler_label)
    val_dataset = PeptideBERTDataset(input_ids=val_inputs, attention_masks=attention_mask_val, labels=val_handler_labels)
    test_dataset = PeptideBERTDataset(input_ids=test_inputs, attention_masks=attention_mask_test, labels=test_handler_labels)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    print('Batch size: ', config['batch_size'])

    print('Train dataset samples: ', len(train_dataset))
    print('Validation dataset samples: ', len(val_dataset))
    print('Test dataset samples: ', len(test_dataset))

    print('Train dataset batches: ', len(train_data_loader))
    print('Validation dataset batches: ', len(val_data_loader))
    print('Test dataset batches: ', len(test_data_loader))

    print()

    return train_data_loader, val_data_loader, test_data_loader


if __name__ == '__main__':
    load_data(" ", "../data/base_data/whitelab_hemo_data.csv")
