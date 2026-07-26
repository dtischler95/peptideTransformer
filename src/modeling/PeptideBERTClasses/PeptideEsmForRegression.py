from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn
from transformers import EsmModel, EsmPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling


class PeptideEsmForRegression(EsmPreTrainedModel):
    """
    ESM-2 counterpart of PeptideBertForRegression: a linear regression head on top of the
    CLS pooler output. Head, normalisation, dropout and loss are kept identical to the BERT
    variant for comparability; only the encoder is swapped (EsmModel) and token_type_ids are
    dropped (ESM has no segment embeddings). The hidden size is taken from the loaded ESM
    config and must not be overridden (ESM-2 sizes differ per checkpoint: 320/480/640/1280/...).
    """

    def __init__(self, config, model_path: str, n_features: int, pretrained: bool = True):
        config.return_dict = False
        super().__init__(config)
        # pretrained=False builds the encoder with random weights (no checkpoint download).
        # Only the test suite passes False, to exercise the pipeline without the multi-GB weights.
        self.esm = (EsmModel.from_pretrained(model_path, config=config, add_pooling_layer=True)
                    if pretrained else EsmModel(config, add_pooling_layer=True))

        # ----------------- Add regression head -----------------
        self.norm = nn.LayerNorm(config.hidden_size + n_features, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(0.15)
        self.regression = nn.Linear(config.hidden_size + n_features, 1)
        # -------------------------------------------------------

        # Initialize weights and apply final processing
        self.post_init()
        self.norm_seq = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            features: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None,
            return_pooler_output: Optional[bool] = False
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        """
        Mirror of PeptideBertForRegression.forward. token_type_ids is accepted to keep the
        signature stable for the shared Trainer/eval code but is not forwarded to the ESM
        encoder, which has no segment embeddings.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # ------- ADJUST IF REGRESSION HEAD IS CHANGED -------
        pooled_output = outputs[1]  # Use pooled output (CLS pooler)

        if return_pooler_output:
            return self.norm_seq(pooled_output)  # Return pooled output for visualization

        if features is not None:
            # Concatenate pooled output and concentration
            pooled_output = torch.cat((pooled_output, features), dim=1)

        pooled_output = self.norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.regression(pooled_output)
        # --------------------------------------------------

        # If labels are provided, compute the loss
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))

        if not return_dict:
            return (loss, logits) if loss is not None else logits

        return BaseModelOutputWithPooling(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
