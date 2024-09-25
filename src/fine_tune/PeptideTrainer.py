from transformers import Trainer

class PeptideTrainer(Trainer):
    def _save(self, output_dir: str):
        # Ensure that all model parameters are contiguous before saving
        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.contiguous()
        super()._save(output_dir)
