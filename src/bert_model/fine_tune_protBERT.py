import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

from transformers.utils.logging import enable_default_handler, enable_explicit_format
import logging

from src.bert_model.PeptideBERTClasses.PeptideTrainer import PeptideTrainer
from src.bert_model.fine_tune_utils import prepare_datasets, load_training_arguments, \
    prepare_hemo_eval, init_model, overall_stats, get_model_stats
from src.bert_model.PeptideBERTClasses.PeptideCallbackTrainer import LearningCurveCallback, EarlyStoppingCallback, \
    PlotMetricsCallback, CollectBatchWiseTrainMetrics

import matplotlib as mpl
mpl.rcParams.update({
    'font.size': 13,        # base font size
    'axes.titlesize': 18,       # axes title size
    'axes.labelsize': 15,       # axes label size (xlabel, ylabel)
    'xtick.labelsize': 14,      # X-axis tick labels
    'ytick.labelsize': 14,      # Y-axis tick labels
})

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


    :param config_path: Path to the config file
    """

    # --------------------- Setup logging and configs ---------------------
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load training params into PeptideTrainingArguments class. Adjust if we need other params
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

    callback_list = [
        # Custom Callback Class for plotting learning curves. STILL IN WORK
        LearningCurveCallback(plot_path=training_args.plot_path),
        # Custom Callback Class for early stopping.
        EarlyStoppingCallback(),
        # following callbacks are essential for calculating metrics during training!
        CollectBatchWiseTrainMetrics(),
        PlotMetricsCallback()
    ]

    tokenizer, train_dataset, val_dataset, test_dataset, n_features = prepare_datasets(
        model_class=training_args.model_class,
        model_path=training_args.model_path,
        train_file=training_args.train_file,
        ignore_leakage=training_args.ignore_leakage,
        max_length=training_args.max_length,
        logger=logger,
        cut_df_for_faster_debug=training_args.fast_debug_mode,
        add_features=training_args.add_features
    )

    # Load the model, the model is a BertForSequenceClassification model based on the Rostlab/prot_bert_bfd model
    # Based on https://pubs.acs.org/doi/10.1021/acs.jpclett.3c02398 PeptideBERT


    data_collator, model, run_metric = init_model(train_dataset, training_args, n_features)

    # Initialize the Trainer class most of the stuff should be handled by the PeptideTrainer class when an appropriate
    # configured PeptideTrainingArguments class is provided
    trainer = PeptideTrainer(
        model=model,  # The model to be trained
        args=training_args,  # Training arguments from above
        data_collator=data_collator,
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
        file_name = Path(training_args.train_file).stem
        if training_args.model_class.startswith('binary'):
            from src.data_analysis.hemo_clustering import cluster_model_embedding
            # Custom Function for cluster the model embeddings with the whole dataset
            scaler = cluster_model_embedding(file_path=train_dataset,
                                             data_tag=Path(config_path).stem,
                                             batch_size=training_args.per_device_eval_batch_size,
                                             plot_path=training_args.plot_path + "/Training_",
                                             scaler=None,
                                             tokenizer_and_model=(tokenizer, trainer.model),
                                             device=training_args.device,
                                             sequence_max_length=training_args.max_length,
                                             label_0_cluster_data=training_args.label_0_cluster_data,
                                             label_1_cluster_data=training_args.label_1_cluster_data,
                                             logger=logger
                                             )

            cluster_model_embedding(file_path=val_dataset,
                                    data_tag=Path(config_path).stem,
                                    batch_size=training_args.per_device_eval_batch_size,
                                    plot_path=training_args.plot_path + "/val_",
                                    scaler=scaler,
                                    tokenizer_and_model=(tokenizer, trainer.model),
                                    device=training_args.device,
                                    sequence_max_length=training_args.max_length,
                                    label_0_cluster_data=training_args.label_0_cluster_data,
                                    label_1_cluster_data=training_args.label_1_cluster_data,
                                    logger=logger
                                    )

            cluster_model_embedding(file_path=test_dataset,
                                    data_tag=Path(config_path).stem,
                                    batch_size=training_args.per_device_eval_batch_size,
                                    plot_path=training_args.plot_path + "/test_",
                                    scaler=scaler,
                                    tokenizer_and_model=(tokenizer, trainer.model),
                                    device=training_args.device,
                                    sequence_max_length=training_args.max_length,
                                    label_0_cluster_data=training_args.label_0_cluster_data,
                                    label_1_cluster_data=training_args.label_1_cluster_data,
                                    logger=logger
                                    )

            prepare_hemo_eval(test_dataset=train_dataset,
                              trainer=trainer,
                              plot_path=training_args.plot_path + "/train_",
                              tag='Training',
                              file_name=file_name)
            prepare_hemo_eval(test_dataset=val_dataset,
                              trainer=trainer,
                              plot_path=training_args.plot_path + "/val_",
                              tag='Validation',
                              file_name=file_name)
            prepare_hemo_eval(test_dataset=test_dataset,
                              trainer=trainer,
                              plot_path=training_args.plot_path + "/test_",
                              tag='Test',
                              file_name=file_name)

        elif training_args.model_class.startswith('regression'):

            y_train_preds = trainer.predict(test_dataset=train_dataset)
            y_train_preds = y_train_preds.predictions.flatten()
            y_train_true = train_dataset.labels

            y_val_preds = trainer.predict(test_dataset=val_dataset)
            y_val_preds = y_val_preds.predictions.flatten()
            y_val_true = val_dataset.labels

            y_test_preds = trainer.predict(test_dataset=test_dataset)
            y_test_preds = y_test_preds.predictions.flatten()
            y_test_true = test_dataset.labels

            overall_stats(predictions=y_train_preds, y_true=y_train_true, save_path=training_args.plot_path,
                          tag='Training', file_name=file_name)
            overall_stats(predictions=y_val_preds, y_true=y_val_true, save_path=training_args.plot_path,
                          tag='Validation', file_name=file_name)
            overall_stats(predictions=y_test_preds, y_true=y_test_true, save_path=training_args.plot_path, tag='Test', file_name=file_name)

            train_r2, train_mse = get_model_stats(plot_dir=training_args.plot_path,
                                                  predictions=y_train_preds,
                                                  target_data=y_train_true,
                                                  logger=logger,
                                                  file_name=file_name,
                                                  tag='Training')

            val_r2, val_mse = get_model_stats(plot_dir=training_args.plot_path,
                                              predictions=y_val_preds,
                                              target_data=y_val_true,
                                              logger=logger,
                                              file_name=file_name,
                                              tag='Validation')

            test_r2, test_mse = get_model_stats(plot_dir=training_args.plot_path,
                                                predictions=y_test_preds,
                                                target_data=y_test_true,
                                                logger=logger,
                                                file_name=file_name,
                                                tag='Test')

            with open(f"{training_args.plot_path}/{Path(training_args.train_file).stem}_train.txt",
                      "w") as f:
                f.write(f"Filename: {file_name}\n"
                        f"Train R2: {round(train_r2, 4)}\n"
                        f"Train MSE: {round(train_mse, 4)}\n"
                        f"Validation R2: {round(val_r2, 4)}\n"
                        f"Validation MSE: {round(val_mse, 4)}\n"
                        f"Test R2: {round(test_r2, 4)}\n"
                        f"Test MSE: {round(test_mse, 4)}\n")



if __name__ == '__main__':
    fine_tune(config_path='peptideBERT_configs/debug_clsBERT_config.yaml')  # Path to the config file
