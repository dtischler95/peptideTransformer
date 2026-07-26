"""Opt-in end-to-end ESM smoke test with the real (small) esm2_t6_8M checkpoint.

Downloads ~30 MB, so it is skipped unless RUN_INTEGRATION=1. This is the "does the
ESM pipeline actually run" check (tokeniser + init_model + forward/backward), the
thing that kept OOM-ing with the 650M model. The 8M encoder runs on CPU.
"""

import logging
import os
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator

from src.modeling.fine_tune_utils import prepare_datasets, init_model

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION"),
    reason="set RUN_INTEGRATION=1 to run (downloads esm2_t6_8M)",
)

_ESM_SMALL = "facebook/esm2_t6_8M_UR50D"


def test_esm_pipeline_end_to_end(mic_data_dir):
    train_file = str(mic_data_dir / "mini_for_regression.csv")
    tokenizer, train_ds, _, _, n_features = prepare_datasets(
        model_class="regression", model_path=_ESM_SMALL, train_file=train_file,
        logger=logging.getLogger("esm-int"), backbone="esm", max_length=36,
    )
    # ESM tokenises raw residues: roughly one token per residue plus specials.
    assert tokenizer(train_ds.peptides[0])["input_ids"]

    training_args = SimpleNamespace(model_class="regression", model_path=_ESM_SMALL,
                                    backbone="esm", device=torch.device("cpu"))
    data_collator, model, _ = init_model(train_ds, training_args, n_features)

    loader = DataLoader(train_ds, batch_size=4, collate_fn=data_collator)
    batch = next(iter(loader))
    loss, logits = model(**batch)

    assert logits.shape[0] == 4
    assert torch.isfinite(loss)
    loss.backward()
