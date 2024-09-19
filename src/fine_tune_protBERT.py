from transformers import BertTokenizer, BertForSequenceClassification, BertForMaskedLM, DataCollatorForLanguageModeling, \
    DefaultDataCollator
from transformers.utils.logging import enable_default_handler, enable_explicit_format
import sys
import torch
import pandas as pd
from PeptideDataset import PeptideDataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
import logging
from transformer_utils import compute_metrics
from PeptideTrainer import PeptideTrainer


# this line should be included in the TrainingArguments
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger = logging.getLogger(__name__)


def fine_tune(binary_or_mlm: str,
              model_path: str,
              train_file: str,
              show_encoding: bool = False,
              model_save_path: str = './peptideBERT_model'
              ):
    """
    Fine-tunes the model on the hemo dataset, should contain basic functionality for fine-tuning
    Also includes the next sentence prediction, but can be turned off. I'm not sure if the next sentence prediction
    is actually needed for prediction on peptide sequences, but it is included here for completeness.

    Script adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py


    :param binary_or_mlm: if the model should be fine-tuned for binary classification or masked language modeling
                      Can be set to 'binary' or 'mlm' or 'both'
    :param model_path: either provide path to the HuggingFace Repository or a local path of the model
    :param show_encoding: if the encoding of the vocabulary should be shown
    :param model_save_path: path to save the model


    :return:
    """

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Define training arguments
    training_args = TrainingArguments(
        do_train=True,  # Perform training
        do_eval=True,  # Perform evaluation
        do_predict=True,  # Perform prediction
        output_dir='./results',  # Output directory
        num_train_epochs=3,  # Number of training epochs
        per_device_train_batch_size=64,  # Batch size for training
        per_device_eval_batch_size=64,  # Batch size for evaluation
        warmup_steps=500,  # Number of warmup steps
        weight_decay=0.01,  # Strength of weight decay
        logging_dir='./logs',  # Directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        log_level='info',  # Set logging level
        seed=42,  # Seed for reproducibility
        dataloader_drop_last=False, # Drop the last incomplete batch
        dataloader_num_workers=4,  # Number of workers for data loading
        optim= 'adamw_torch',  # Optimizer to use
        lr_scheduler_type='linear',  # Learning rate scheduler type
        learning_rate=5e-5,  # Learning rate
        use_cpu = True # Only for local testing purposes
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # ------------------------------------------------------------------------------------------------------------------
    # Load the data
    # Extract to method if I want to pipe binary and mlm fine-tuning
    df = pd.read_csv(train_file, sep=';')
    df = df[:500]

    # Split the data into training, validation and test sets
    # TODO Create Accuracy Validation for Testdata
    df_train, df_val_handler = train_test_split(df, test_size=0.2)
    df_val, df_test = train_test_split(df_val_handler, test_size=0.5)

    sequence_data_train = df_train['sequence'].values
    sequence_data_val = df_val['sequence'].values
    sequence_data_test = df_test['sequence'].values

    label_data_train = df_train['label'].values if binary_or_mlm == 'binary' else None
    label_data_val = df_val['label'].values if binary_or_mlm == 'binary' else None
    label_data_test = df_test['label'].values if binary_or_mlm == 'binary' else None

    # ------------------------------------------------------------------------------------------------------------------

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', clean_up_tokenization_spaces=True)

    # Load the model, the model is a BertForSequenceClassification model based on the Rostlab/prot_bert_bfd model
    # Based on https://pubs.acs.org/doi/10.1021/acs.jpclett.3c02398 PeptideBERT
    # Only Difference is, that we initiate the model not from BertModel class but from BertForSequenceClassification
    # Since this implementation integrated a classifier for the sequence classification task
    if binary_or_mlm == 'binary':
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
        data_collator = DefaultDataCollator()
    elif binary_or_mlm == 'mlm':
        model = BertForMaskedLM.from_pretrained(model_path)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    elif binary_or_mlm == 'both':
        raise NotImplementedError("Not implemented yet") # TODO Think of Logic and how to implement
    else:
        raise ValueError("binary_or_mlm must be either 'binary' or 'mlm' or 'both'")

    # Create a Dataset Class for the training and validation data for our use case
    # TODO Create Dataset Class for self-supervised learning
    train_dataset = PeptideDataset(peptides=sequence_data_train, tokenizer=tokenizer, labels=label_data_train)
    val_dataset = PeptideDataset(peptides=sequence_data_val , tokenizer=tokenizer, labels=label_data_val)
    test_dataset = PeptideDataset(peptides=sequence_data_test, tokenizer=tokenizer, labels=label_data_test)

    # print out the encoding of the vocabulary used by the tokenizer if wanted
    get_encoding(tokenizer=tokenizer) if show_encoding else None

    # Initialize the Trainer
    trainer = PeptideTrainer(
        model=model,  # The model to be trained
        args=training_args,  # Training arguments from above TODO check for PeptideBERT
        data_collator=data_collator,  # Data collator for masking sequences if mlm is used
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics if binary_or_mlm == 'binary' else None  # TODO Get additional metrics for MLM
    )

    if training_args.do_train:
        logger.info("*** Train ***")
        # TODO May implement ReduceLROnPlateau, but need to step manually since Trainer class does not support it natively
        trainer.train()

        # params seems not to be contiguous, so we need to make them contiguous
        trainer.save_model(model_save_path)
        logger.info("*** Model saved ***")

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        eval_result = trainer.evaluate()
        logger.info(eval_result)
        logger.info("*** Evaluation finished ***")

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions = trainer.predict(test_dataset)
        # TODO Find a cool representation for the predictions
        # logger.info(predictions.predictions)
        logger.info("*** Prediction finished ***")


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
    fine_tune(binary_or_mlm='binary',
              train_file="../data/base_data/splitted_hemo_labeled.csv",
              model_path='Rostlab/prot_bert_bfd',
              model_save_path='./peptideBERT_model',
              show_encoding=False,
              )
