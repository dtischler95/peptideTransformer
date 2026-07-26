"""Shared fixtures for the test suite.

Everything here is synthetic and tiny: the goal is to prove the pipeline runs
(tokenisation, model wiring, splits, ML grid search), not to produce meaningful
numbers. No pretrained weights are downloaded for the hermetic tests.
"""

import os
# Headless plotting for the ML pipeline tests (they save figures via matplotlib).
# Must be set before matplotlib is first imported anywhere.
os.environ.setdefault("MPLBACKEND", "Agg")

import random

import pandas as pd
import pytest
import torch
from transformers import BertConfig, EsmConfig


# Shared amino-acid motifs. Building sequences from a small motif pool guarantees
# k-mers recur across sequences, so the char 3-4gram TF-IDF (min_df=2) in the ML
# pipeline never produces an empty vocabulary on the small fixtures.
_MOTIFS = ["ACDE", "FGHI", "KLMN", "PQRS", "TVWY", "AGSC", "LIVM", "RKDE"]


def _make_sequence(rng: random.Random) -> str:
    return "".join(rng.choice(_MOTIFS) for _ in range(rng.randint(3, 5)))


def _unique_sequences(rng: random.Random, n: int) -> list[str]:
    seqs: set[str] = set()
    while len(seqs) < n:
        seqs.add(_make_sequence(rng))
    return list(seqs)


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    random.seed(0)


@pytest.fixture
def tiny_bert_config() -> BertConfig:
    return BertConfig(vocab_size=30, hidden_size=32, num_hidden_layers=2,
                      num_attention_heads=2, intermediate_size=64,
                      max_position_embeddings=64, pad_token_id=0)


@pytest.fixture
def tiny_esm_config() -> EsmConfig:
    return EsmConfig(vocab_size=33, hidden_size=32, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=64,
                     max_position_embeddings=64, pad_token_id=1)


@pytest.fixture
def token_batch():
    """A small input batch valid for both tiny configs (ids stay < vocab_size=30)."""
    batch, seq_len = 4, 12
    input_ids = torch.randint(1, 25, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    return input_ids, attention_mask


class FakeTokenizer:
    """Minimal stand-in so PeptideDataset can be exercised offline.

    Maps residues to ids deterministically and pads to max_length. Only the shape
    contract (batched input_ids/attention_mask) matters for the dataset tests.
    """

    def __call__(self, text, padding=None, truncation=None, max_length=36, return_tensors=None):
        ids = [ord(c) % 24 + 1 for c in text if c != ' '][:max_length]
        attn = [1] * len(ids)
        ids += [0] * (max_length - len(ids))
        attn += [0] * (max_length - len(attn))
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([attn])}


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


def _write_split(directory, stem, rows, sep=';'):
    df = pd.DataFrame(rows)
    df.to_csv(directory / f"{stem}.csv", sep=sep, index=False)


@pytest.fixture
def mic_data_dir(tmp_path):
    """A regression fixture dir with mini _train/_val/_test splits (sequence; mic_log10)."""
    rng = random.Random(1)
    directory = tmp_path / "regression_data"
    directory.mkdir()
    n_train, n_val, n_test = 18, 6, 6
    seqs = _unique_sequences(rng, n_train + n_val + n_test)
    targets = [round(rng.uniform(-1.0, 3.0), 3) for _ in seqs]
    rows = [{"sequence": s, "mic_log10": t} for s, t in zip(seqs, targets)]
    _write_split(directory, "mini_for_regression_train", rows[:n_train])
    _write_split(directory, "mini_for_regression_val", rows[n_train:n_train + n_val])
    _write_split(directory, "mini_for_regression_test", rows[n_train + n_val:])
    return directory


@pytest.fixture
def hemo_data_dir(tmp_path):
    """A classification fixture dir with balanced mini splits (sequence; label)."""
    rng = random.Random(2)
    directory = tmp_path / "hemo_train"
    directory.mkdir()

    def rows_for(n_per_class):
        seqs = _unique_sequences(rng, 2 * n_per_class)
        labels = [0] * n_per_class + [1] * n_per_class
        rng.shuffle(labels)
        return [{"sequence": s, "label": y} for s, y in zip(seqs, labels)]

    _write_split(directory, "mini_hemo_train", rows_for(10))
    _write_split(directory, "mini_hemo_val", rows_for(4))
    _write_split(directory, "mini_hemo_test", rows_for(4))
    return directory
