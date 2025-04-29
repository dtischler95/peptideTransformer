import sys
import warnings

warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
from transformers import BertForMaskedLM, DefaultDataCollator, BertConfig, DataCollatorForLanguageModeling
from transformers.utils.logging import enable_default_handler, enable_explicit_format
import logging
from src.bert_model.transformer_metrics import binary_metrics, mlm_metrics
from src.bert_model.PeptideBERTClasses.PeptideTrainer import PeptideTrainer
from src.bert_model.fine_tune_utils import prepare_datasets, load_training_arguments, \
    prepare_fisher_exact, get_bce_label_weight
from src.bert_model.PeptideBERTClasses.PeptideCallbackTrainer import LearningCurveCallback, EarlyStoppingCallback, \
    CurriculumLearningCallback, PlotMetricsCallback, CollectBatchWiseTrainMetrics
from src.bert_model.PeptideBERTClasses.PeptideBertForBinaryClassification import PeptideBertForBinaryClassification
from src.bert_model.PeptideBERTClasses.PeptideBertForRegression import PeptideBertForRegression
from src.bert_model.PeptideBERTClasses.PeptideDataCollator import PeptideCurriculumDataCollator

# this line should be included in the TrainingArguments
logger = logging.getLogger(__name__)


def fine_tune(config_path: str):
    """
    Fine-tunes the model on the hemo dataset, should contain basic functionality for fine-tuning

    Script adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py

    DataLoader also checks for data leakage between the datasets. If data leakage is detected, the training will stop if
    ignore_leakage is set to False. If ignore_leakage is set to True, the training will continue, but a warning will be
    printed with the number of leaked data points. Use only if you want to see how data leakage affects the training,
    since it seems like that PeptideBERT is affected by data leakage. [I used my check_data_loader_for_leakage function
    on PeptideBERTs train algorithm, and it seems like the model is affected by data leakage]


    :param config_path: path to the config file
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

    tokenizer, train_dataset, val_dataset, test_dataset = prepare_datasets(binary_or_mlm=training_args.model_class,
                                                                           model_path=training_args.model_path,
                                                                           show_encoding=training_args.run_verbose,
                                                                           train_file=training_args.train_file,
                                                                           ignore_leakage=training_args.ignore_leakage,
                                                                           max_length=training_args.max_length,
                                                                           logger=logger,
                                                                           cut_df_for_faster_debug=training_args.fast_debug_mode,
                                                                           validation_data_size=training_args.validation_data_size,
                                                                           test_data_size=training_args.test_data_size)
    # Load the model, the model is a BertForSequenceClassification model based on the Rostlab/prot_bert_bfd model
    # Based on https://pubs.acs.org/doi/10.1021/acs.jpclett.3c02398 PeptideBERT
    # Only Difference is, that we initiate the model not from BertModel class but from BertForSequenceClassification
    # Since this implementation integrated a classifier for the sequence classification task

    callback_list = [
        # Custom Callback Class for plotting learning curves. STILL IN WORK
        LearningCurveCallback(plot_path=training_args.plot_path),
        # Custom Callback Class for early stopping.
        EarlyStoppingCallback(),
        # following callbacks are essential for calculating metrics during training!
        CollectBatchWiseTrainMetrics(),
        PlotMetricsCallback()
    ]

    if training_args.model_class == 'binary':

        config = BertConfig.from_pretrained(training_args.model_path)#('GrimSqueaker/proteinBERT')
        #config2 = BertConfig.from_pretrained('Rostlab/prot_bert_bfd')#(training_args.model_path) 'Rostlab/prot_bert_bfd' 'GrimSqueaker/proteinBERT'
        model = PeptideBertForBinaryClassification(config,
                                                   loss_function=training_args.loss_function,
                                                   bce_logit_weight=get_bce_label_weight(
                                                       labels=train_dataset.labels).to(training_args.device))
        data_collator = DefaultDataCollator()
        run_metric = binary_metrics


    # Load the model, the model is a BertForMaskedLM model based on the Rostlab/prot_bert_bfd model
    # Our Idea is to fine tune the ProtBERT model on MLM to further introduce the model to the peptide sequences instead
    # of the protein sequences. We hope to increase the binary classification performance by fine-tuning the model on MLM
    # first.
    elif training_args.model_class == 'mlm':
        model = BertForMaskedLM.from_pretrained(training_args.model_path)
        # data_collator = PeptideCurriculumDataCollator(tokenizer=tokenizer,
        #                                               initial_prob=training_args.mlm_probability,
        #                                               increase_step=training_args.mlm_curriculum_increase_step,
        #                                               max_prob=training_args.mlm_curriculum_max_prob)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                                        mlm_probability=training_args.mlm_probability)

        # Add Curriculum Learning Callback if enabled
        # This callback can be adjusted if another metric for increasing/decreasing mlm_probability is needed
        #callback_list.append(CurriculumLearningCallback()) if training_args.mlm_curriculum_learning else ...
        run_metric = mlm_metrics
    elif training_args.model_class == 'custom':
        # raise NotImplementedError("Custom task not implemented yet")
        config = BertConfig.from_pretrained(training_args.model_path)
        model = PeptideBertForRegression(config)
        data_collator = DefaultDataCollator()
        run_metric = None  # TODO implement regression metrics
    else:
        raise ValueError(f"binary_or_mlm must be either 'binary' or 'mlm'. You provided: '{training_args.model_class}'")

    # Initialize the Trainer class most of the stuff should be handled by the PeptideTrainer class when an appropriate
    # configured PeptideTrainingArguments class is provided
    trainer = PeptideTrainer(
        model=model,  # The model to be trained
        args=training_args,  # Training arguments from above
        data_collator=data_collator,  # Data collator for masking sequences if mlm is used
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=run_metric,
        # calculate metrics based on the task
        # callback Classes from transformers are a powerful tool to customize behavior during Training! Check the docs for more
        # https://huggingface.co/docs/transformers/main_classes/callback#transformers.TrainerCallback
        callbacks=callback_list
    )

    # --------------------- Train, evaluate and predict ---------------------
    if training_args.do_train:
        trainer.train()
        trainer.save_model(training_args.model_save_path)
        tokenizer.save_pretrained(training_args.model_save_path)
        logger.info(f"*** Model saved to {training_args.model_save_path} ***")

    if training_args.do_eval:

        if training_args.model_class == 'binary':
            from src.data_analysis.hemo_clustering import cluster_model_embedding
            # Custom Function for cluster the model embeddings with the whole dataset
            cluster_model_embedding(file_path=training_args.train_file,
                                    data_tag=config_path.split('/')[-1].split('.')[0],
                                    batch_size=training_args.per_device_eval_batch_size,
                                    plot_path=training_args.plot_path,
                                    tokenizer_and_model=(tokenizer, trainer.model),
                                    device=training_args.device,
                                    label_0_cluster_data=training_args.label_0_cluster_data,
                                    label_1_cluster_data=training_args.label_1_cluster_data
                                    )

            prepare_fisher_exact(test_dataset=test_dataset,
                                 trainer=trainer,
                                 plot_path=training_args.plot_path)

    # not really needed for my case I guess
    if training_args.do_predict:
        ...
        # Test dataset not used so far. May remove it completely?
        """
        This part is only for debugging purposes.        
        """
        # if training_args.model_class == 'binary':
        #     test_binary_label_bias(tokenizer, trainer, training_args)


if __name__ == '__main__':
    fine_tune(config_path='peptideBERT_configs/debug_mlmBERT_config.yaml')  # Path to the config file
