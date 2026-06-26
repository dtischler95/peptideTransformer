from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn
from transformers import EsmModel, EsmPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling


class PeptideEsmForBinaryClassification(EsmPreTrainedModel):
    """
    ESM-2 counterpart of PeptideBertForBinaryClassification. The head, normalisation,
    dropout, loss handling and pooled-output contract are kept byte-for-byte identical to
    the BERT variant so the two backbones are comparable; only the encoder is swapped
    (EsmModel instead of BertModel) and token_type_ids are dropped (ESM has no segment
    embeddings). The CLS pooler output (outputs[1]) is used exactly as in the BERT path.
    """

    def __init__(self, config, model_path: str, n_features, bce_logit_weight, loss_function: str = 'bce'):
        config.num_labels = 1  # Set num_labels to 1 for binary classification output
        config.classifier_dropout = 0.15
        config.return_dict = False
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.loss_function = loss_function
        self.bce_logit_weight = bce_logit_weight
        self.esm = EsmModel.from_pretrained(model_path, config=config, add_pooling_layer=True)
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
        Mirror of PeptideBertForBinaryClassification.forward. token_type_ids is accepted to
        keep the signature stable for the shared Trainer/eval code but is not forwarded to
        the ESM encoder, which has no segment embeddings.
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

        pooled_output = outputs[1]  # Use pooled output (CLS pooler)

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
                # this is done after loss calculation because bce_with_logit loss already implemented sigmoid
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
