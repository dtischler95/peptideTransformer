from torch.utils.data import Dataset
import torch

class PeptideDataset(Dataset):
    def __init__(self, peptides, labels, tokenizer, max_length=128):
        self.peptides = peptides
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, idx):
        peptide = self.peptides[idx]
        label = self.labels[idx]

        # Tokenize the peptide sequence
        encoding = self.tokenizer(peptide, padding='max_length', truncation=True, max_length=self.max_length,
                                  return_tensors='pt')

        # Flatten the tensors (remove the batch dimension)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }