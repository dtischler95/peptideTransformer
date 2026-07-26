from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling


class PeptideBertForBinaryClassification(BertPreTrainedModel):
    """
    This class is a reproduction of PeptideBERTs implementation for binary classification.
    It added a Sigmoid activation function to the output logits to convert the output to probabilities.
    We now receive a single logit for binary classification instead of n logits for each class like in the inherited class.
    """

    def __init__(self, config, model_path: str, n_features, bce_logit_weight, loss_function: str = 'bce',
                 pretrained: bool = True):
        config.num_labels = 1  # Set num_labels to 1 for binary classification output
        config.classifier_dropout = 0.15
        config.return_dict = False
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.loss_function = loss_function
        self.bce_logit_weight = bce_logit_weight
        # pretrained=False builds the encoder with random weights (no checkpoint download).
        # Only the test suite passes False, to exercise the pipeline without the multi-GB weights.
        self.bert = BertModel.from_pretrained(model_path, config=config) if pretrained else BertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.norm = nn.LayerNorm(config.hidden_size + n_features, eps=config.layer_norm_eps)

        self.classifier = nn.Linear(config.hidden_size + n_features,
                                    config.num_labels)  # Output only one logit for binary classification
        self.sigmoid = nn.Sigmoid()  # Add sigmoid for binary classification

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
        :param features: concentration for the model
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

        if return_pooler_output:
            return self.norm_seq(pooled_output)  # Return pooled output for visualization

        if features is not None:
            # Concatenate pooled output and concentration
            pooled_output = torch.cat((pooled_output, features), dim=1)

        pooled_output = self.norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        # Apply the classifier to the concatenated output
        logits = self.classifier(pooled_output)

        # If labels are provided, compute the loss
        loss = None
        if labels is not None:
            if self.loss_function == 'bce':
                # Apply sigmoid activation for binary classification
                logits = self.sigmoid(logits)  # Apply sigmoid activation for binary classification

                loss_fct = nn.BCELoss()  # Use Binary Cross Entropy Loss
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.loss_function == 'bce_logit_loss':
                # Move pos_weight to the logits' device so this also works under
                # nn.DataParallel (multi-GPU): the weight is created once on cuda:0, but
                # replicas run on other GPUs, which otherwise raises a device mismatch.
                pos_weight = self.bce_logit_weight
                if pos_weight is not None:
                    pos_weight = pos_weight.to(logits.device)
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # BCE with logits
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
                # this is done after loss calculation because bce_with_logit loss arleady implemented sigmoid
                # For further calculations we still need to apply sigmoid to the logits here since the loss function wont return the sigmoided logits
                logits = self.sigmoid(logits)  # Apply sigmoid activation for binary classification
            else:
                raise ValueError(f"Loss function {self.loss_function} not supported. Use 'bce' or 'bce_logit_loss'")

        if not return_dict:
            return (loss, logits) if loss is not None else logits

        return BaseModelOutputWithPooling(
            last_hidden_state=outputs.last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
