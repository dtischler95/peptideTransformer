import warnings
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

from transformers import BertForSequenceClassification, BertForMaskedLM, DataCollatorForLanguageModeling, \
    DefaultDataCollator
from transformers.utils.logging import enable_default_handler, enable_explicit_format

import torch  # pytorch in requirements.txt
import logging

from src.fine_tune.transformer_metrics import binary_metrics, mlm_metrics
from src.fine_tune.PeptideBERTClasses.PeptideTrainer import PeptideTrainer
from src.fine_tune.fine_tune_utils import prepare_datasets, load_training_arguments
from src.fine_tune.PeptideBERTClasses.PeptideCallbackTrainer import MetricLogCallback

# this line should be included in the TrainingArguments
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger = logging.getLogger(__name__)


def fine_tune(binary_or_mlm: str,
              show_encoding: bool = False):
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
                      Can be set to 'binary' or 'mlm'
    :param show_encoding: if the encoding of the vocabulary should be shown

    :return:
    """

    # --------------------- Setup logging and configs ---------------------
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    training_args = load_training_arguments(config_file='fine_tune_config.yaml',
                                            training_type=binary_or_mlm)

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
                                                                           drop_duplicates=training_args.drop_duplicates,
                                                                           show_encoding=show_encoding,
                                                                           train_file=training_args.train_file,
                                                                           ignore_leakage=training_args.ignore_leakage,
                                                                           max_length=training_args.max_length)

    # --------------------- Prepare model and trainer ---------------------

    # Load the model, the model is a BertForSequenceClassification model based on the Rostlab/prot_bert_bfd model
    # Based on https://pubs.acs.org/doi/10.1021/acs.jpclett.3c02398 PeptideBERT
    # Only Difference is, that we initiate the model not from BertModel class but from BertForSequenceClassification
    # Since this implementation integrated a classifier for the sequence classification task
    if binary_or_mlm == 'binary':
        model = BertForSequenceClassification.from_pretrained(training_args.model_path, num_labels=2)
        data_collator = DefaultDataCollator()

    # Load the model, the model is a BertForMaskedLM model based on the Rostlab/prot_bert_bfd model
    # Our Idea is to fine tune the ProtBERT model on MLM to further introduce the model to the peptide sequences instead
    # of the protein sequences. We hope to increase the binary classification performance by fine-tuning the model on MLM
    # first.
    elif binary_or_mlm == 'mlm':
        model = BertForMaskedLM.from_pretrained(training_args.model_path)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                                        mlm_probability=training_args.mlm_probability)
    else:
        raise ValueError(f"binary_or_mlm must be either 'binary' or 'mlm'. You provided: '{binary_or_mlm}'")

    # Initialize the Trainer class
    trainer = PeptideTrainer(
        model=model,  # The model to be trained
        args=training_args,  # Training arguments from above
        data_collator=data_collator,  # Data collator for masking sequences if mlm is used
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=binary_metrics if binary_or_mlm == 'binary' else mlm_metrics,
        # callback Classes from transformers are a powerful tool to customize behavior during Training! Check the docs for more
        # https://huggingface.co/docs/transformers/main_classes/callback#transformers.TrainerCallback
        callbacks=[MetricLogCallback(plot_dir=training_args.plot_path, task_name=binary_or_mlm)]
    )

    if training_args.do_train:
        logger.info("*** Train ***")

        trainer.train()

        # TODO MCC !!!

        # params seems not to be contiguous, so we need to make them contiguous
        trainer.save_model(training_args.model_save_path)
        logger.info(f"*** Model saved to {training_args.model_save_path} ***")

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
              show_encoding=False,  # Set to True if you want to see the encoding of the vocabulary
              )
