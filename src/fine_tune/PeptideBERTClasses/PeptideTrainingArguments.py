from typing import Literal

from transformers import TrainingArguments
import sys
from pathlib import Path


class PeptideTrainingArguments(TrainingArguments):
    """
    Class for adjusting TrainingArguments for PeptideBERT to our needs and collecting all necessary arguments in one place.
    I tried to stick to commonly used values for the parameters, but feel free to adjust them to your needs.
    """
    def __init__(self, *args,
                 lro_factor:float=0.1,
                 lro_patience:int=4,
                 lro_min_lr:float=0.000001,
                 lro_mode:Literal['min', 'max']='min',
                 train_file=None,
                 model_path:str='Rostlab/prot_bert_bfd',
                 model_save_path:str='../default_path_BERT',
                 plot_path:str= '../../plots',
                 drop_duplicates: bool=True,
                 ignore_leakage: bool=False,
                 mlm_probability:float=0.15,
                 max_length:int=36,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lro_factor = lro_factor
        self.lro_patience = lro_patience
        self.lro_min_lr = lro_min_lr
        self.lro_mode = lro_mode
        self.train_file = train_file
        if self.train_file is None:
            # TODO may raise an error here and make a different inference function
            sys.stderr(
                "[warning] If you use model for Inference you can ignore this warning. Otherwise, you should provide a train_file in the config file.")
        self.model_path = model_path
        self.model_save_path = model_save_path
        self.plot_path = plot_path
        # Create the model and plot save path if it does not exist
        Path(self.model_save_path).mkdir(parents=True, exist_ok=True)
        Path(self.plot_path).mkdir(parents=True, exist_ok=True)
        self.drop_duplicates = drop_duplicates
        self.ignore_leakage = ignore_leakage
        self.mlm_probability = mlm_probability
        self.max_length = max_length


