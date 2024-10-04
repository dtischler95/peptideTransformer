from typing import Optional, Union, Tuple
import torch
from transformers import BertForSequenceClassification, BertModel
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput


class PeptideBertForBinaryClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1  # Set num_labels to 1 for binary classification output
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size,
                                    self.num_labels)  # Output only one logit for binary classification
        self.sigmoid = nn.Sigmoid()  # Add sigmoid for binary classification

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_pooler_output: Optional[bool] = False  # New parameter
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        pooled_output = outputs[1]  # Use pooled output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # Apply sigmoid activation for binary classification
        logits = self.sigmoid(logits)

        # TODO plot logits

        # If labels are provided, compute the loss
        loss = None
        if labels is not None:
            loss_fct = nn.BCELoss()  # Use Binary Cross Entropy Loss
            loss = loss_fct(logits.view(-1), labels.view(-1).float())

        if return_pooler_output:
            return pooled_output # Return pooled output for visualization

        return (loss, logits) if loss is not None else logits
