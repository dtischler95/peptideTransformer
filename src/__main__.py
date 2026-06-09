import argparse
import os
import sys

from src.bert_model.fine_tune_protBERT import fine_tune
from src.data_preprocessing.data_splitter import data_splitter

# Adding the src folder to python path so the module is runnable
sys.path.append(os.path.dirname(__file__))



def main():
    args, parser = parse_inputs()

    # Check if bert_model flags are proper set
    if args.config_path and args.pipe_configs:
        parser.error("Please provide either --config_path or --pipe_configs, not both.")
    if not args.config_path and not args.pipe_configs:
        parser.error("Please provide either --config_path or --pipe_configs.")

    if args.command == 'bert_model':
        # If we have configs in our config dir we can just pass the config name.
        if args.pipe_configs:
            # If no path provides use a default path
            path = './src/bert_model/peptideBERT_configs/config_pipe_dir/' if args.pipe_configs == 'default' else args.pipe_configs

            for file in os.listdir(path):
                if file.endswith('.yaml'):
                    tmp_config = os.path.join(path, file)
                    fine_tune(config_path=tmp_config)

        if args.config_path:
            if '/' not in args.config_path:
                args.config_path = f"./src/bert_model/peptideBERT_configs/{args.config_path}"
            # Main function Wrapper for the Training Pipeline. Any additional settings are done via the config.yaml inside peptideBERT_configs directory
            fine_tune(
                config_path=args.config_path
            )


    elif args.command == 'data_init':
        data_splitter(task=args.task)

    elif args.command == 'data_analysis':
        ...
        # from src.data_analysis.hemo_clustering import cluster_model_embedding
        # from transformers import AutoModel, AutoConfig, AutoTokenizer
        #
        # tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        # config = AutoConfig.from_pretrained(args.model_path)
        # model = AutoModel.from_pretrained(args.model_path, config=config)
        #
        #
        #
        # # Call your data_analysis function here
        # if args.cluster:
        #     #filterwarnings("ignore", category=UserWarning)
        #     cluster_model_embedding(file_path=args.file_path,
        #                             data_tag=args.file_path.split('/')[-1].split('.')[0],
        #                             batch_size=8,
        #                             plot_path=args.out_plot_path,
        #                             tokenizer_and_model=(tokenizer, model),
        #                             device=)  # Assuming you have a tokenizer and model to pass

    elif args.command == 'generate_bert_model_config':
        # Call your config generation function here
        from src.bert_model.fine_tune_utils import generate_custom_yaml_file
        generate_custom_yaml_file(config_name='',
                                  file_path='')
    else:
        parser.print_help()


def parse_inputs():
    parser = argparse.ArgumentParser(description='Peptide Transformers CLI')
    subparsers = parser.add_subparsers(dest='command')
    # Subparser for bert_model
    fine_tune_parser = subparsers.add_parser('bert_model', help='Fine-tune the model')
    fine_tune_parser.add_argument('--config_path', type=str, required=False, help='Path to the config file')
    fine_tune_parser.add_argument('--pipe_configs', type=str, required=False, help='Path to the Directory containing config files. Will use every config inside this dir.')
    # Subparser for data_preprocess
    data_preprocess_parser = subparsers.add_parser('data_init', help='Preprocess the data')
    data_preprocess_parser.add_argument('--task', type=str, help='task of the train data ["classification", "regression", "gram"]')
    # Add arguments for data_preprocess here
    # Subparser for data_analysis
    data_analysis_parser = subparsers.add_parser('data_analysis', help='Analyze the data')
    # Add arguments for data_analysis here
    data_analysis_parser.add_argument('--cluster', action='store_true', required=True, help='Cluster with model')
    data_analysis_parser.add_argument('--file_path', type=str, required=True, help='Path to the file')
    data_analysis_parser.add_argument('--out_plot_path', type=str, required=True, help='Path to the Plots')
    data_analysis_parser.add_argument('--model_path', type=str, required=False, help='Path to the Model')
    # Subparser for reproduce
    reproduce_parser = subparsers.add_parser('reproduce', help='Reproduce our results')
    # Add arguments for reproduce here
    # Subparser for autogernerating a config file
    generate_config_parser = subparsers.add_parser('generate_bert_model_config',
                                                   help='Generate a default set bert config yaml file')
    generate_config_parser.add_argument('--config_name', type=str, required=True, help='Name for the config file')
    generate_config_parser.add_argument('--file_path', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    return args, parser


if __name__ == '__main__':
    main()
