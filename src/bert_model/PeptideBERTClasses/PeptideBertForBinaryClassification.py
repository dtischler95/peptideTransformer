from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn

class PeptideBertForBinaryClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]
        probs = self.classifier(pooled_output).squeeze(-1)  # Apply the sequential classifier

        return probs