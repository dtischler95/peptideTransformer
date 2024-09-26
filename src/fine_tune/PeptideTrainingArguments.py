from transformers import TrainingArguments

class PeptideTrainingArguments(TrainingArguments):
    def __init__(self, *args, lro_factor=0.1, lro_patience=4, lro_min_lr=1e-6, lro_mode='min', **kwargs):
        super().__init__(*args, **kwargs)
        self.lro_factor = lro_factor
        self.lro_patience = lro_patience
        self.lro_min_lr = lro_min_lr
        self.lro_mode = lro_mode
