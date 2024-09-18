from torch.utils.data import Dataset
import torch


class PeptideDataset(Dataset):
    def __init__(self, peptides, tokenizer, labels=None, max_length=36):
        """
        Args:
            peptides: List of peptide sequences.
            labels: List of labels (for binary classification), can be None for self-supervised tasks like MLM.
            tokenizer: Tokenizer to tokenize the peptide sequences.
            max_length: Maximum length for padding/truncation.
        """
        self.peptides = peptides
        self.labels = labels  # Labels are optional for self-supervised learning tasks
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, idx):
        peptide = self.peptides[idx]

        # Tokenize the peptide sequence
        encoding = self.tokenizer(peptide, padding='max_length', truncation=True, max_length=self.max_length,
                                  return_tensors='pt')

        # Flatten the tensors (remove the batch dimension)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # For binary classification, return labels; otherwise, ignore
        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        if self.labels is not None:  # Only include labels for binary classification
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item
