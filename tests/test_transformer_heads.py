"""Forward/backward smoke tests for the four transformer heads.

Random-initialised tiny encoders (pretrained=False) so no multi-GB checkpoint is
downloaded. Covers the model wiring both backbones share: pooled output, feature
concat, head, loss, and that gradients actually flow.
"""

import pytest
import torch

from src.bert_model.PeptideBERTClasses.PeptideBertForRegression import PeptideBertForRegression
from src.bert_model.PeptideBERTClasses.PeptideBertForBinaryClassification import PeptideBertForBinaryClassification
from src.bert_model.PeptideBERTClasses.PeptideEsmForRegression import PeptideEsmForRegression
from src.bert_model.PeptideBERTClasses.PeptideEsmForBinaryClassification import PeptideEsmForBinaryClassification


def _backward_has_grad(loss, model):
    loss.backward()
    return any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("backbone", ["bert", "esm"])
def test_regression_head_forward_backward(backbone, request, token_batch):
    config = request.getfixturevalue(f"tiny_{backbone}_config")
    model_cls = PeptideBertForRegression if backbone == "bert" else PeptideEsmForRegression
    model = model_cls(config, model_path="unused", n_features=0, pretrained=False)

    input_ids, attention_mask = token_batch
    labels = torch.randn(input_ids.shape[0])
    loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert logits.shape == (input_ids.shape[0], 1)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert _backward_has_grad(loss, model)


@pytest.mark.parametrize("backbone", ["bert", "esm"])
def test_classification_head_forward_backward(backbone, request, token_batch):
    config = request.getfixturevalue(f"tiny_{backbone}_config")
    model_cls = PeptideBertForBinaryClassification if backbone == "bert" else PeptideEsmForBinaryClassification
    model = model_cls(config, model_path="unused", n_features=0,
                      bce_logit_weight=torch.tensor([1.0]),
                      loss_function="bce_logit_loss", pretrained=False)

    input_ids, attention_mask = token_batch
    labels = torch.randint(0, 2, (input_ids.shape[0],)).float()
    loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert logits.shape == (input_ids.shape[0], 1)
    # Classification path sigmoids the logits before returning them.
    assert torch.all((logits >= 0) & (logits <= 1))
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert _backward_has_grad(loss, model)


@pytest.mark.parametrize("backbone", ["bert", "esm"])
def test_regression_head_with_features(backbone, request, token_batch):
    """The feature-concat branch (seq+feat) must accept n_features extra inputs."""
    config = request.getfixturevalue(f"tiny_{backbone}_config")
    model_cls = PeptideBertForRegression if backbone == "bert" else PeptideEsmForRegression
    n_features = 3
    model = model_cls(config, model_path="unused", n_features=n_features, pretrained=False)

    input_ids, attention_mask = token_batch
    features = torch.randn(input_ids.shape[0], n_features)
    labels = torch.randn(input_ids.shape[0])
    loss, logits = model(input_ids=input_ids, attention_mask=attention_mask,
                         features=features, labels=labels)

    assert logits.shape == (input_ids.shape[0], 1)
    assert torch.isfinite(loss)
