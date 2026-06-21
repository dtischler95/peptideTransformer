import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

from transformers.utils.logging import enable_default_handler, enable_explicit_format
import logging

from src.bert_model.PeptideBERTClasses.PeptideTrainer import PeptideTrainer
from src.bert_model.fine_tune_utils import prepare_datasets, load_training_arguments, \
    prepare_hemo_eval, init_model
from src.evaluation.eval_utils import overall_stats, get_model_stats, write_run_artifacts
from src.data_preprocessing import datasets
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


def _bert_split(train_file) -> str:
    """Infer the split from the train_file path (cluster dirs carry 'cluster')."""
    return "cluster" if "cluster" in str(train_file) else "random"


def fine_tune(config_path: str):
    """Fine-tune ProtBERT on a single dataset defined by a YAML config."""

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

            splits = [
                ("train_", train_dataset, "Training"),
                ("val_", val_dataset, "Validation"),
                ("test_", test_dataset, "Test"),
            ]

            # Custom function for clustering the model embeddings of each split.
            # The scaler is fit on the training split and reused (not refit) for val/test.
            train_scaler = None
            for split_dir, dataset, _ in splits:
                returned_scaler = cluster_model_embedding(file_path=dataset,
                                                  data_tag=Path(config_path).stem,
                                                  batch_size=training_args.per_device_eval_batch_size,
                                                  plot_path=training_args.plot_path + f"/{split_dir}/",
                                                  scaler=train_scaler,
                                                  tokenizer_and_model=(tokenizer, trainer.model),
                                                  device=training_args.device,
                                                  sequence_max_length=training_args.max_length,
                                                  label_0_cluster_data=training_args.label_0_cluster_data,
                                                  label_1_cluster_data=training_args.label_1_cluster_data,
                                                  logger=logger
                                                  )
                if train_scaler is None:
                    train_scaler = returned_scaler

            for split_dir, dataset, tag in splits:
                y_true, y_score, y_pred = prepare_hemo_eval(test_dataset=dataset,
                                  trainer=trainer,
                                  plot_path=training_args.plot_path + f"/{split_dir}",
                                  tag=tag,
                                  file_name=file_name)

                if tag == "Test":
                    write_run_artifacts(
                        out_dir=training_args.plot_path + f"/{split_dir}",
                        data_name=datasets.organism_slug(file_name),
                        model_name="bert",
                        task="hemo",
                        split=_bert_split(training_args.train_file),
                        seed=getattr(training_args, "seed", 42),
                        features=bool(training_args.add_features),
                        y_true=y_true,
                        y_pred=y_pred,
                        y_score=y_score,
                        sequences=[p.replace(" ", "") for p in dataset.peptides],
                    )

        elif training_args.model_class.startswith('regression'):

            splits = [
                ("Training", train_dataset),
                ("Validation", val_dataset),
                ("Test", test_dataset),
            ]

            stats = {}
            for tag, dataset in splits:
                preds = trainer.predict(test_dataset=dataset).predictions.flatten()
                y_true = dataset.labels

                overall_stats(predictions=preds, y_true=y_true, save_path=training_args.plot_path,
                              tag=tag, file_name=file_name)

                stats[tag] = get_model_stats(plot_dir=training_args.plot_path,
                                              predictions=preds,
                                              target_data=y_true,
                                              logger=logger,
                                              file_name=file_name,
                                              tag=tag)

                if tag == "Test":
                    write_run_artifacts(
                        out_dir=training_args.plot_path,
                        data_name=datasets.organism_slug(file_name),
                        model_name="bert",
                        task="mic",
                        split=_bert_split(training_args.train_file),
                        seed=getattr(training_args, "seed", 42),
                        features=bool(training_args.add_features),
                        y_true=y_true,
                        y_pred=preds,
                        sequences=[p.replace(" ", "") for p in dataset.peptides],
                    )

            with open(f"{training_args.plot_path}/{Path(training_args.train_file).stem}_train.txt",
                      "w") as f:
                f.write(f"Filename: {file_name}\n")
                for tag, (r2, mse) in stats.items():
                    f.write(f"{tag} R2: {round(r2, 4)}\n"
                            f"{tag} MSE: {round(mse, 4)}\n")



if __name__ == '__main__':
    fine_tune(config_path='peptideBERT_configs/debug_clsBERT_config.yaml')  # Path to the config file
