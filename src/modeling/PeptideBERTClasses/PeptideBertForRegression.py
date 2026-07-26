from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling


class PeptideBertForRegression(BertPreTrainedModel):
    """
    Regression model using the BertModel as a base with a linear regression head on top of the [CLS] token embedding.
    """

    def __init__(self, config, model_path: str, n_features: int, pretrained: bool = True):
        config.return_dict = False
        super().__init__(config)
        # pretrained=False builds the encoder with random weights (no checkpoint download).
        # Only the test suite passes False, to exercise the pipeline without the multi-GB weights.
        self.bert = BertModel.from_pretrained(model_path, config=config) if pretrained else BertModel(config)


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
        Forward to propagate the input through the model adjusted to the single logit output for binary classification.
        We also added a return_pooler_output parameter to return the pooled output for visualization purposes.
        Also added a print_debug_graph parameter to plot the label bias in the predictions for debugging purposes.

        Function overwrote the forward function of the inherited class BertForSequenceClassification.
        We kept the Signature of the original function, so IDEs won't complain about the function signature.
        Many of them we don't handle, but you could implement them if needed.

        :param input_ids: input ids for the model
        :param attention_mask: attention mask for the model
        :param features: features for the model
        :param token_type_ids: token type ids for the model
        :param position_ids: position ids for the model
        :param head_mask: head mask for the model
        :param inputs_embeds: input embeddings for the model
        :param encoder_hidden_states: encoder hidden states for the model
        :param encoder_attention_mask: encoder attention mask for the model
        :param past_key_values: past key values for the model
        :param use_cache: use cache for the model
        :param labels: labels for the model
        :param output_attentions: output attentions for the model
        :param output_hidden_states: output hidden states for the model
        :param return_dict: return dict for the model
        :param return_pooler_output: return pooler output for the model
        :return: loss, logits or logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # with torch.no_grad():
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # ------- ADJUST IF REGRESSION HEAD IS CHANGED -------
        pooled_output = outputs[1]  # Use pooled output

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

        if return_pooler_output:
            return pooled_output  # Return pooled output for visualization

        if not return_dict:
            return (loss, logits) if loss is not None else logits

        return BaseModelOutputWithPooling(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
