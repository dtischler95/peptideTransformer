from transformers import BertTokenizer, BertForSequenceClassification, BertForMaskedLM, DataCollatorForLanguageModeling, DefaultDataCollator
import torch
import pandas as pd
from PeptideDatasets import PeptideDatasetBinary
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def fine_tune_supervised(show_encoding: bool = False,
                         model_save_path: str = './peptideBERT_model',
                         binary_or_mlm: str = 'binary'):
    """
    Fine-tunes the model on the hemo dataset, should contain basic functionality for fine-tuning
    Also includes the next sentence prediction, but can be turned off. I'm not sure if the next sentence prediction
    is actually needed for prediction on peptide sequences, but it is included here for completeness.

    :param show_encoding: if the encoding of the vocabulary should be shown
    :param model_save_path: path to save the model
    :param binary_or_mlm: if the model should be fine-tuned for binary classification or masked language modeling
                          Can be set to 'binary' or 'mlm' or 'both'

    :return:
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Load the data
    # Extract to method if i want to pipe binary and mlm fine-tuning
    df = pd.read_csv("../data/base_data/splitted_hemo_labeled.csv", sep=';')


    # Split the data into training, validation and test sets
    # TODO Create Accuracy Validation for Testdata
    df_train, df_val_handler = train_test_split(df, test_size=0.2)
    df_val, df_test = train_test_split(df_val_handler, test_size=0.5)

    sequence_data_train = df_train['sequence'].values
    sequence_data_val = df_val['sequence'].values
    sequence_data_test = df_test['sequence'].values

    label_data_train = df_train['label'].values
    label_data_val = df_val['label'].values
    label_data_test = df_test['label'].values

    # ------------------------------------------------------------------------------------------------------------------

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', clean_up_tokenization_spaces=True)

    # Load the model, the model is a BertForSequenceClassification model based on the Rostlab/prot_bert_bfd model
    # Based on https://pubs.acs.org/doi/10.1021/acs.jpclett.3c02398 PeptideBERT
    # Only Difference is, that we initiate the model not from BertModel class but from BertForSequenceClassification
    # Since this implementation integrated a classifier for the sequence classification task
    if binary_or_mlm == 'binary':
        model = BertForSequenceClassification.from_pretrained('Rostlab/prot_bert_bfd', num_labels=2)
        data_collator = DefaultDataCollator()
    elif binary_or_mlm == 'mlm':
        model = BertForMaskedLM.from_pretrained('Rostlab/prot_bert_bfd', num_labels=2)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    elif binary_or_mlm == 'both':
        NotImplementedError("Not implemented yet")
    else:
        raise ValueError("binary_or_mlm must be either 'binary' or 'mlm' or 'both'")

    # Create a Dataset Class for the training and validation data for our usecase
    # TODO Create Dataset Class for self-supervised learning
    train_dataset = PeptideDatasetBinary(peptides=sequence_data_train, labels=label_data_train, tokenizer=tokenizer)
    val_dataset = PeptideDatasetBinary(peptides=sequence_data_val, labels=label_data_val, tokenizer=tokenizer)

    # print out the encoding of the vocabulary used by the tokenizer if wanted
    get_encoding(tokenizer=tokenizer) if show_encoding else None


    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',  # Output directory
        num_train_epochs=3,  # Number of training epochs
        per_device_train_batch_size=4,  # Batch size for training
        per_device_eval_batch_size=4,  # Batch size for evaluation
        warmup_steps=500,  # Number of warmup steps
        weight_decay=0.01,  # Strength of weight decay
        logging_dir='./logs',  # Directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,  # The model to be trained
        args=training_args,  # Training arguments from above TODO check for PeptideBERT
        data_collator=data_collator, # Data collator for masking sequences if mlm is used
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model(model_save_path)


def get_encoding(tokenizer: BertTokenizer):
    """
    Prints the encoding of the vocabulary for the tokenizer after input is given

    :param tokenizer:
    :return:
    """

    for i in range(tokenizer.vocab_size):
        character = tokenizer.decode(i)
        print(f"{i}: {character}")


if __name__ == '__main__':
    fine_tune_supervised(show_encoding=False,
                         model_save_path='./peptideBERT_model',
                         binary_or_mlm='binary')
