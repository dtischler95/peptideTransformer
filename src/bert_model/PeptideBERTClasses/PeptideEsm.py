from typing import Optional, Union, Tuple

from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.esm.modeling_esm import EsmPreTrainedModel
from transformers.models.esm.modeling_esm import EsmModel
import torch.nn as nn
import torch


class PeptideEsm(EsmPreTrainedModel):
    """
    This class is a reproduction of PeptideBERTs implementation for binary classification using ESM.
    It added a Sigmoid activation function to the output logits to convert the output to probabilities.
    We now receive a single logit for binary classification instead of n logits for each class like in the inherited class.
    """

    def __init__(self, config, use_conc:bool=True):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.esm = EsmModel(config, add_pooling_layer=False)
        self.classifier = EsmClassificationHead(config, use_conc)

        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        concentration: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
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
        sequence_output = outputs[0]
        if concentration is not None:
            # if concentration is provided, we concatenate it to the sequence output

            # If concentration is a 1D tensor, we need to unsqueeze it to match the sequence output shape
            batch_size, seq_length, hidden_size = sequence_output.size()

            concentration = concentration.unsqueeze(1).repeat(1, seq_length).unsqueeze(-1)

            sequence_output = torch.cat((sequence_output, concentration), dim=-1)

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class EsmClassificationHead(nn.Module):
    """
    Classification head for ESM model.
    This head is used to classify the output of the ESM model into binary classes.
    """

    def __init__(self, config, use_conc: bool = True):
        super().__init__()
        hidden_size = config.hidden_size + (1 if use_conc else 0)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x