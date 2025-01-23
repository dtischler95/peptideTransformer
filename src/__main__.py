import argparse
from warnings import filterwarnings
from src.bert_model.fine_tune_protBERT import fine_tune
from src.data_analysis.hemo_clustering import cluster_model_embedding
# Import your data_preprocess and data_analysis functions here

def reproduce_our_work_steps():
    # Call your reproduce function here
    pass

def main():
    parser = argparse.ArgumentParser(description='Peptide Transformers CLI')
    subparsers = parser.add_subparsers(dest='command')

    # Subparser for bert_model
    fine_tune_parser = subparsers.add_parser('bert_model', help='Fine-tune the model')
    fine_tune_parser.add_argument('--model_class', type=str, required=True, choices=['binary', 'mlm'], help='Task type')
    fine_tune_parser.add_argument('--config_path', type=str, required=True, help='Path to the config file')
    fine_tune_parser.add_argument('--show_encoding', action='store_true', help='Show encoding of the vocabulary')

    # Subparser for data_preprocess
    data_preprocess_parser = subparsers.add_parser('data_preprocess', help='Preprocess the data')
    # Add arguments for data_preprocess here

    # Subparser for data_analysis
    data_analysis_parser = subparsers.add_parser('data_analysis', help='Analyze the data')
    # Add arguments for data_analysis here
    data_analysis_parser.add_argument('--cluster', action='store_true', required=True, help='Cluster with model')
    data_analysis_parser.add_argument('--file_path', type=str, required=True, help='Path to the file')
    data_analysis_parser.add_argument('--out_plot_path', type=str, required=True, help='Path to the Plots')

    # Subparser for reproduce
    reproduce_parser = subparsers.add_parser('reproduce', help='Reproduce our results')
    # Add arguments for reproduce here

    # Subparser for autogernerating a config file
    generate_config_parser = subparsers.add_parser('generate_bert_model_config', help='Generate a default set bert config yaml file')
    generate_config_parser.add_argument('--config_name', type=str, required=True, help='Name for the config file')
    generate_config_parser.add_argument('--file_path', type=str, required=True, help='Path to the config file')

    args = parser.parse_args()

    if args.command == 'bert_model':

        # If we have configs in our config dir we can just pass the config name.
        if '/' not in args.config_path:
            args.config_path = f"./src/bert_model/peptideBERT_configs/{args.config_path}"

        fine_tune(
            model_class=args.model_class,
            config_path=args.config_path,
            show_encoding=args.show_encoding
        )
    elif args.command == 'data_preprocess':
        # Call your data_preprocess function here
        pass
    elif args.command == 'data_analysis':
        # Call your data_analysis function here
        if args.cluster:
            filterwarnings("ignore", category=UserWarning)
            cluster_model_embedding(file_path=args.file_path,
                                    batch_size=64,
                                    plot_path=args.out_plot_path)
    elif args.command == 'reproduce':
        reproduce_our_work_steps()
    elif args.command == 'generate_bert_model_config':
        # Call your config generation function here
        from src.bert_model.fine_tune_utils import generate_custom_yaml_file
        generate_custom_yaml_file(config_name='',
                                  file_path='')
    else:
        parser.print_help()

if __name__ == '__main__':
    main()