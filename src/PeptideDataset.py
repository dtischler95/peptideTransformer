from torch.utils.data import Dataset


class PeptideDataset(Dataset):
    """
    Dataset class for the peptide sequences
    """

    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}
