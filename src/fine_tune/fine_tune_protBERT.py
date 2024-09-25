from transformers import BertForSequenceClassification, BertForMaskedLM, DataCollatorForLanguageModeling, \
    DefaultDataCollator
from transformers.utils.logging import enable_default_handler, enable_explicit_format
import sys
import torch # pytorch in requirements.txt
import logging

from src.fine_tune.transformer_metrics import compute_metrics, mlm_metrics
from src.fine_tune.PeptideTrainer import PeptideTrainer
from src.fine_tune.fine_tune_utils import prepare_datasets, load_training_arguments, check_directory

# this line should be included in the TrainingArguments
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger = logging.getLogger(__name__)


def fine_tune(binary_or_mlm: str,
              model_path: str,
              train_file: str,
              show_encoding: bool = False,
              model_save_path: str = './peptideBERT_model',
              drop_duplicates: bool = False,
              use_cpu: bool = False,
              ignore_leakage: bool = False,
              mlm_probability: float = 0.15):
    """
    Fine-tunes the model on the hemo dataset, should contain basic functionality for fine-tuning
    Also includes the next sentence prediction, but can be turned off. I'm not sure if the next sentence prediction
    is actually needed for prediction on peptide sequences, but it is included here for completeness.

    Script adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py

    DataLoader also checks for data leakage between the datasets. If data leakage is detected, the training will stop if
    ignore_leakage is set to False. If ignore_leakage is set to True, the training will continue, but a warning will be
    printed with the number of leaked data points. Use only if you want to see how data leakage affects the training,
    since it seems like that PeptideBERT is affected by data leakage. [I used my check_data_loader_for_leakage function
    on PeptideBERTs train algorithm, and it seems like the model is affected by data leakage]


    :param binary_or_mlm: if the model should be fine-tuned for binary classification or masked language modeling
                      Can be set to 'binary' or 'mlm' or 'both'
    :param model_path: either provide path to the HuggingFace Repository or a local path of the model
    :param train_file : path to the training data
    :param show_encoding: if the encoding of the vocabulary should be shown
    :param model_save_path: path to save the model
    :param drop_duplicates: if duplicated sequences should be dropped
    :param use_cpu: if the CPU should be used for training
    :param ignore_leakage: if data leakage should be ignored or cause an error to stop training
    :param mlm_probability: probability of masking tokens for MLM. Only shows effect if binary_or_mlm is set to 'mlm'

    :return:
    """

    check_directory(model_save_path)

    # --------------------- Setup logging and configs ---------------------
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    training_args = load_training_arguments(config_file='fine_tune_config.yaml',
                                            training_type=binary_or_mlm,
                                            use_cpu=use_cpu)

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

    # --------------------- Prepare tokenizer and datasets ---------------------

    # Prepare tokenizer and datasets
    tokenizer, test_dataset, train_dataset, val_dataset = prepare_datasets(binary_or_mlm=binary_or_mlm,
                                                                           drop_duplicates=drop_duplicates,
                                                                           show_encoding=show_encoding,
                                                                           train_file=train_file,
                                                                           ignore_leakage=ignore_leakage)

    # --------------------- Prepare model and trainer ---------------------

    # Load the model, the model is a BertForSequenceClassification model based on the Rostlab/prot_bert_bfd model
    # Based on https://pubs.acs.org/doi/10.1021/acs.jpclett.3c02398 PeptideBERT
    # Only Difference is, that we initiate the model not from BertModel class but from BertForSequenceClassification
    # Since this implementation integrated a classifier for the sequence classification task
    if binary_or_mlm == 'binary':
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
        data_collator = DefaultDataCollator()

    # Load the model, the model is a BertForMaskedLM model based on the Rostlab/prot_bert_bfd model
    # Our Idea is to fine tune the ProtBERT model on MLM to further introduce the model to the peptide sequences instead
    # of the protein sequences. We hope to increase the binary classification performance by fine-tuning the model on MLM
    # first.
    elif binary_or_mlm == 'mlm':
        model = BertForMaskedLM.from_pretrained(model_path)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)
    else:
        raise ValueError(f"binary_or_mlm must be either 'binary' or 'mlm'. You provided: '{binary_or_mlm}'")

    # Initialize the Trainer
    trainer = PeptideTrainer(
        model=model,  # The model to be trained
        args=training_args,  # Training arguments from above TODO check for PeptideBERT
        data_collator=data_collator,  # Data collator for masking sequences if mlm is used
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics if binary_or_mlm == 'binary' else mlm_metrics  # TODO Get additional metrics for MLM
    )

    if training_args.do_train:
        logger.info("*** Train ***")
        # TODO May implement ReduceLROnPlateau, but need to step manually since Trainer class does not support it natively
        trainer.train()

        # TODO MCC !!!

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
        print(predictions)
        # TODO Find a cool representation for the predictions
        # logger.info(predictions.predictions)
        logger.info("*** Prediction finished ***")



if __name__ == '__main__':
    fine_tune(binary_or_mlm='mlm',  # Set to 'binary' for binary classification, 'mlm' for masked language modeling
              train_file="../../data/train_data/starpep_sequences.csv",  # Path to the training data
              model_path='Rostlab/prot_bert_bfd',  # Path to the model. Local path or HuggingFace Repository
              model_save_path='../our_BERT',  # Path to save the model
              show_encoding=False,  # Set to True if you want to see the encoding of the vocabulary
              drop_duplicates=True,  # Set to True if you want to drop duplicate sequences
              use_cpu=False,  # Set to True if you want to use the CPU instead of GPU for training
              ignore_leakage=False,  # Set to True if you want to compare how data leakage affects the training
              mlm_probability=0.15  # Probability of masking tokens for MLM.
              )
