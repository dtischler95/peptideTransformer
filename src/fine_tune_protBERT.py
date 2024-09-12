from tqdm import tqdm
from transformers import BertTokenizer, BertForPreTraining, BertForSequenceClassification, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import torch
from torch.optim import AdamW
import random
import pandas as pd
from torch.utils.data import DataLoader
from PeptideDataset import PeptideDataset

device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cpu')


def fine_tune(nsp_prediction: bool = False,
              show_encoding: bool = False,
              mask_percentage: float = 0.15,
              max_length: int = 512,
              model_save_path: str = './peptideBERT_model'):
    """
    Fine-tunes the model on the hemo dataset, should contain basic functionality for fine-tuning
    Also includes the next sentence prediction, but can be turned off. I'm not sure if the next sentence prediction
    is actually needed for prediction on peptide sequences, but it is included here for completeness.

    :param nsp_prediction: if the model should use next sentence prediction
    :param show_encoding: if the encoding of the vocabulary should be shown
    :param mask_percentage: percentage of masking the input data
    :param max_length: max length of the sequences
    :param model_save_path: path to save the model

    :return:
    """
    # Load the data
    df = pd.read_csv("../data/hemo/splitted_hemo_labeled.csv", sep=';')
    df = df[:10]
    sequence_data = df['sequence'].values

    # sequence_data_size = len(sequence_data)

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', clean_up_tokenization_spaces=True)

    # choose of using the next sentence prediction
    if nsp_prediction:
        model = BertForPreTraining.from_pretrained('Rostlab/prot_bert_bfd')
        for param in model.parameters(): param.data = param.data.contiguous()
        nsp_label, sequence_end, sequence_start = prepare_sequences_for_nsp(sequence_data=sequence_data)

        # Tokenize the data, sequence_start and sequence_end are getting concatenated with a [SEP] token
        # TODO max_length is set to 512, but this should be changed to the max length of the sequences
        inputs = tokenizer(sequence_start, sequence_end, padding='max_length', truncation=True, return_tensors='pt',
                           max_length=max_length)

        get_encoding(tokenizer=tokenizer) if show_encoding else None

        # save the nsp_label to the input data. Labels are 0 for false_sequence and 1 for true_sequence
        inputs['next_sentence_label'] = torch.LongTensor([nsp_label]).T
        inputs['labels'] = inputs.input_ids.detach().clone()


    else:
        # if not using the next sentence prediction, we have to use the sequence classification model
        # TODO DO i really need the sequence classifier here? Since i only want a latent representation of the sequences
        # TODO Fix label, optimizer for classification wants a label array of shape [batch_size, 1] but since i mask
        # TODO AA's in the sequence the labels are in shape of [batch_size, max_length]
        model = BertForSequenceClassification.from_pretrained('Rostlab/prot_bert_bfd', num_labels=2)
        for param in model.parameters(): param.data = param.data.contiguous()
        # whitespaces between aa's of sequences needed for tokenization
        input_data_formatted = [' '.join(seq) for seq in sequence_data]
        inputs = tokenizer(input_data_formatted, padding='max_length', truncation=True, return_tensors='pt',
                           max_length=max_length)
        get_encoding(tokenizer=tokenizer) if show_encoding else None
        inputs['labels'] = torch.LongTensor(df['label'].values)

    # get the labels from the data before masking them
    # inputs['labels'] = inputs.input_ids.detach().clone()

    # create the mask for the input data
    mask_inputs(inputs, mask_percentage)

    # create the dataset class
    train_dataset = PeptideDataset(inputs)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # move the model to the device
    model.to(device)
    model.train()

    # create the optimizer
    optim = AdamW(model.parameters(), lr=5e-5)

    # train the model
    for epoch in range(2):
        loop = tqdm(train_dataloader, leave=True)
        for batch in loop:
            optim.zero_grad()
            loss = compute_loss(batch, model, nsp_prediction)
            loss.backward()
            optim.step()
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

    # save the model
    model.save_pretrained(model_save_path)


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


def compute_loss(batch, model: BertModel, nsp_prediction) -> BaseModelOutputWithPoolingAndCrossAttentions or \
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
    if nsp_prediction:
        next_sentence_label = batch['next_sentence_label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, next_sentence_label=next_sentence_label,
                        labels=labels)
    else:
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
    fine_tune(nsp_prediction=True, show_encoding=False)
    # decoded_sequences_to_file('whitelab_negative.csv.txt')
