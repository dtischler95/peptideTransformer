from tqdm import tqdm
from transformers import BertTokenizer, BertForPreTraining, BertForSequenceClassification, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import torch
from torch.optim import AdamW
import random
import pandas as pd
from torch.utils.data import DataLoader
from PeptideDataset import PeptideDataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def fine_tune(show_encoding: bool = False,
              mask_percentage: float = 0.15,
              max_length: int = 512,
              model_save_path: str = './peptideBERT_model'):
    """
    Fine-tunes the model on the hemo dataset, should contain basic functionality for fine-tuning
    Also includes the next sentence prediction, but can be turned off. I'm not sure if the next sentence prediction
    is actually needed for prediction on peptide sequences, but it is included here for completeness.

    :param show_encoding: if the encoding of the vocabulary should be shown
    :param mask_percentage: percentage of masking the input data
    :param max_length: max length of the sequences
    :param model_save_path: path to save the model

    :return:
    """
    # Load the data
    df = pd.read_csv("../data/base_data/splitted_hemo_labeled.csv", sep=';')
    df = df[:10]

    df_train, df_val_handler = train_test_split(df, test_size=0.2)
    df_val, df_test = train_test_split(df_val_handler, test_size=0.5)


    # Split the data into training and validation data

    sequence_data = df['sequence'].values

    sequence_data_train = df_train['sequence'].values
    sequence_data_val = df_val['sequence'].values
    sequence_data_test = df_test['sequence'].values

    label_data_train = df_train['label'].values
    label_data_val = df_val['label'].values
    label_data_test = df_test['label'].values


    # sequence_data_size = len(sequence_data)

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', clean_up_tokenization_spaces=True)

    model = BertForSequenceClassification.from_pretrained('Rostlab/prot_bert_bfd', num_labels=2)


    # for param in model.parameters(): param.data = param.data.contiguous()
    # create the mask for the input data
    # mask_inputs(inputs_train, mask_percentage)

    # create the dataset class
    train_dataset = PeptideDataset(peptides=sequence_data_train, labels=label_data_train, tokenizer=tokenizer)
    val_dataset = PeptideDataset(peptides=sequence_data_val, labels=label_data_val, tokenizer=tokenizer)

    get_encoding(tokenizer=tokenizer) if show_encoding else None




    # create the optimizer
    optim = AdamW(model.parameters(), lr=5e-5)

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
        args=training_args,  # Training arguments
        train_dataset=train_dataset,  # Training dataset
        eval_dataset=val_dataset,         # Optionally, add evaluation dataset
    )



    trainer.train()
    trainer.save_model(model_save_path)




def mask_inputs(inputs, mask_percentage):
    """
    Masks the input data with a mask_percentage and replaces the masked data with a [MASK] token inplace

    :param inputs: inputs from the tokenizer of the Bert model
    :param mask_percentage: chance of masking an amino acid in the sequence
    """
    rand = torch.rand(inputs.input_ids.shape)
    # mask the input data and ignoring the special tokens. ATTENTION! YOU NEED TO IDENTIFY THE SPECIAL TOKENS MANUALLY (Use get_encoding())
    mask_arr = (rand < mask_percentage) * (inputs.input_ids != 0) * (inputs.input_ids != 1) * (
            inputs.input_ids != 2) * (
                       inputs.input_ids != 3) * (inputs.input_ids != 4)
    # assigning the generated mask to the input data
    for i in range(inputs.input_ids.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        inputs.input_ids[i, selection] = 4


def compute_loss(batch, model: BertModel) -> BaseModelOutputWithPoolingAndCrossAttentions or \
                                                             tuple[torch.FloatTensor]:
    """
    Computes the loss of the model for the given batch

    :param batch: batch of data
    :param model: model to compute the loss on
    :param nsp_prediction: if the model is using next sentence prediction
    """
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    return loss


def prepare_sequences_for_nsp(sequence_data: list[str]) -> (list[int], list[str], list[str]):
    """
    Prepares the sequences for the next sentence prediction task

    :param sequence_data: data to prepare
    :return: nsp_label, sequence_end, sequence_start
    """

    sequence_start = []
    sequence_end = []
    nsp_label = []

    # Creates a nonsense sequence with a 50% chance for the next sentence prediction
    for seq in sequence_data:
        half_index = len(seq) // 2
        sequence_start.append(' '.join(seq[:half_index]))
        if random.random() > 0.5:
            sequence_end.append(' '.join(seq[:half_index]))
            nsp_label.append(0)
        else:
            sequence_end.append(' '.join(seq[half_index:]))
            nsp_label.append(1)
    return nsp_label, sequence_end, sequence_start


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
    fine_tune(show_encoding=False)
    # decoded_sequences_to_file('whitelab_negative.csv.txt')
