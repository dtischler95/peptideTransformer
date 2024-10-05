import sys
import warnings


warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

from transformers import BertForSequenceClassification, BertForMaskedLM, DataCollatorForLanguageModeling, \
    DefaultDataCollator, BertConfig
from transformers.utils.logging import enable_default_handler, enable_explicit_format

import logging

from src.bert_model.transformer_metrics import binary_metrics, mlm_metrics
from src.bert_model.PeptideBERTClasses.PeptideTrainer import PeptideTrainer
from src.bert_model.fine_tune_utils import prepare_datasets, load_training_arguments, test_binary_label_bias
from src.bert_model.PeptideBERTClasses.PeptideCallbackTrainer import LearningCurveCallback, EarlyStoppingCallback
from src.bert_model.PeptideBERTClasses.PeptideBertForBinaryClassification import PeptideBertForBinaryClassification
# this line should be included in the TrainingArguments
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger = logging.getLogger(__name__)


def fine_tune(model_class: str,
              config_path: str,
              show_encoding: bool = False
              ):
    """
    Fine-tunes the model on the hemo dataset, should contain basic functionality for fine-tuning

    Script adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py

    DataLoader also checks for data leakage between the datasets. If data leakage is detected, the training will stop if
    ignore_leakage is set to False. If ignore_leakage is set to True, the training will continue, but a warning will be
    printed with the number of leaked data points. Use only if you want to see how data leakage affects the training,
    since it seems like that PeptideBERT is affected by data leakage. [I used my check_data_loader_for_leakage function
    on PeptideBERTs train algorithm, and it seems like the model is affected by data leakage]


    :param model_class: if the model should be fine-tuned for binary classification or masked language modeling
                      Can be set to 'binary' or 'mlm'
    :param config_path: path to the config file
    :param show_encoding: if the encoding of the vocabulary should be shown
    """

    # --------------------- Setup logging and configs ---------------------
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # load training params into PeptideTrainingArguments class. Adjust if we need other params
    training_args = load_training_arguments(config_file=config_path,
                                            training_type=model_class,
                                            logger=logger)

    # logging logging logging
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}"
    )

    # --------------------- Prepare tokenizer and datasets ---------------------

    # Prepare tokenizer and datasets
    tokenizer, train_dataset, val_dataset, test_dataset = prepare_datasets(binary_or_mlm=model_class,
                                                                           show_encoding=show_encoding,
                                                                           train_file=training_args.train_file,
                                                                           ignore_leakage=training_args.ignore_leakage,
                                                                           max_length=training_args.max_length,
                                                                           logger=logger,
                                                                           cut_df_for_faster_debug=training_args.fast_debug_mode,
                                                                           validation_data_size=training_args.validation_data_size,
                                                                           test_data_size=training_args.test_data_size)

    # --------------------- Prepare model and trainer ---------------------
    # Load the model, the model is a BertForSequenceClassification model based on the Rostlab/prot_bert_bfd model
    # Based on https://pubs.acs.org/doi/10.1021/acs.jpclett.3c02398 PeptideBERT
    # Only Difference is, that we initiate the model not from BertModel class but from BertForSequenceClassification
    # Since this implementation integrated a classifier for the sequence classification task
    if model_class == 'binary':
        config = BertConfig.from_pretrained(training_args.model_path)

        # TODO move to training args
        config.hidden_size = 480
        config.num_attention_heads = 12
        config.num_hidden_layers = 12
        config.classifier_dropout = 0.15
        model = PeptideBertForBinaryClassification(config, debug_label_plot_path=training_args.plot_path)
        data_collator = DefaultDataCollator()

    # Load the model, the model is a BertForMaskedLM model based on the Rostlab/prot_bert_bfd model
    # Our Idea is to fine tune the ProtBERT model on MLM to further introduce the model to the peptide sequences instead
    # of the protein sequences. We hope to increase the binary classification performance by fine-tuning the model on MLM
    # first.
    elif model_class == 'mlm':
        model = BertForMaskedLM.from_pretrained(training_args.model_path)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                                        mlm_probability=training_args.mlm_probability)
    elif model_class == 'custom':
        raise NotImplementedError("Custom task not implemented yet")
    else:
        raise ValueError(f"binary_or_mlm must be either 'binary' or 'mlm'. You provided: '{model_class}'")

    # Initialize the Trainer class most of the stuff should be handled by the PeptideTrainer class when an appropriate
    # configured PeptideTrainingArguments class is provided
    trainer = PeptideTrainer(
        model=model,  # The model to be trained
        args=training_args,  # Training arguments from above
        data_collator=data_collator,  # Data collator for masking sequences if mlm is used
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=binary_metrics if model_class == 'binary' else mlm_metrics,
        # calculate metrics based on the task
        # callback Classes from transformers are a powerful tool to customize behavior during Training! Check the docs for more
        # https://huggingface.co/docs/transformers/main_classes/callback#transformers.TrainerCallback
        callbacks=[
            # Custom Callback Class for plotting learning curves. STILL IN WORK
            LearningCurveCallback(args=training_args, task_name=model_class),
            # Custom Callback Class for early stopping.
            EarlyStoppingCallback()]
    )

    # --------------------- Train, evaluate and predict ---------------------
    if training_args.do_train:
        trainer.train()

        # TODO MCC !!!

        trainer.save_model(training_args.model_save_path)
        tokenizer.save_pretrained(training_args.model_save_path)
        logger.info(f"*** Model saved to {training_args.model_save_path} ***")

    if training_args.do_eval:
        eval_result = trainer.evaluate()
        logger.info(eval_result)
        if model_class == 'binary':
            from src.data_analysis.hemo_clustering import cluster_model_embedding
            # Custom Function for cluster the model embeddings with the whole dataset
            # TODO may provide custom file arg for this. But rn we dont have the data sadly
            # TODO Pass logger to the function
            cluster_model_embedding(file_path=training_args.train_file,
                                    batch_size=training_args.per_device_eval_batch_size,
                                    plot_path=training_args.plot_path,
                                    tokenizer_and_model=(tokenizer, trainer.model),
                                    device=training_args.device)

    if training_args.do_predict:

        # Test dataset not used so far. May remove it completely?

        if model_class == 'binary':
            test_binary_label_bias(tokenizer, trainer, training_args)







if __name__ == '__main__':
    fine_tune(model_class='binary',  # Set to 'binary' for binary classification, 'mlm' for masked language modeling
              config_path='peptideBERT_configs/debug_mlmBERT_config.yaml',  # Path to the config file
              show_encoding=False,  # Set to True if you want to see the encoding of the vocabulary
              )
