# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC


import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
)
from transformers.models.bert.configuration_bert import BertConfig


class ClassificationModel(nn.Module):
    def __init__(self, base_model_name: str, num_classes: int):
        """
        Custom implementation for Single Multiclass Classification Head, similar to AutoModelForSequenceClassification

        Args:
            base_model_name (str): The name of the pre-trained model to use.
            num_classes (int): The number of output classes for classification.base_model_name (str): The name of the pretrained model.
        """

        super(ClassificationModel, self).__init__()

        self.base_model_name = base_model_name
        # Store the number of classes as attributes
        self.num_classes = num_classes

        if self.base_model_name in ["zhihan1996/DNABERT-2-117M"]:
            # Load Config if exists - need for DNABERT 2
            # https://www.kaggle.com/code/gabrielcabas/dnabert-for-classification
            # https://huggingface.co/zhihan1996/DNABERT-2-117M/discussions/26
            self.config = BertConfig.from_pretrained(base_model_name)
            self.base_model = AutoModel.from_pretrained(
                self.base_model_name, trust_remote_code=True, config=self.config
            )
        elif "nucleotide-transformer-v2" in base_model_name:
            self.base_model = AutoModelForMaskedLM.from_pretrained(
                base_model_name, trust_remote_code=True
            )
            self.config = self.base_model.config

        else:
            self.base_model = AutoModel.from_pretrained(
                self.base_model_name, trust_remote_code=True
            )
            self.config = self.base_model.config

        if "hyenadna" in base_model_name:
            # Single classification head
            self.classifier = nn.Linear(self.base_model.config.d_model, num_classes)
        else:
            # Single classification head
            self.classifier = nn.Linear(self.base_model.config.hidden_size, num_classes)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for input sequences.
            labels (torch.Tensor, optional): Target labels for training.

        Returns:
            dict: Contains logits and, if labels are provided, the computed loss.
        """

        # Forward pass through the base model
        if "hyenadna" in self.base_model_name:
            # Hyena models do not have attention mask
            outputs = self.base_model(input_ids=input_ids)
            hidden_states = outputs.last_hidden_state
            mean_embeddings = hidden_states.mean(
                dim=1
            )  # Simple mean pooling if no mask is provided
        elif "DNABERT" in self.base_model_name:
            outputs = self.base_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            hidden_states = outputs[0]
            mean_embeddings = self.mean_pooling(hidden_states, attention_mask)
        else:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]  # Extract last hidden state
            mean_embeddings = self.mean_pooling(hidden_states, attention_mask)

        # Compute logits for each taxonomic rank
        logits = self.classifier(mean_embeddings)

        # If labels are provided, compute loss for each taxonomic rank
        if labels is not None:
            loss = self.compute_loss(logits, labels)
            return {"loss": loss, "logits": logits}

        # If no labels are provided, return logits only
        return {"logits": logits}

    def mean_pooling(self, hidden_states, attention_mask):
        # Compute mean pooling for sequence embeddings
        attention_mask_expanded = attention_mask.unsqueeze(-1)  # Add embedding axis
        sum_hidden_states = torch.sum(hidden_states * attention_mask_expanded, dim=1)
        sum_attention_mask = torch.sum(attention_mask_expanded, dim=1)
        mean_embeddings = sum_hidden_states / sum_attention_mask.clamp(
            min=1e-9
        )  # Prevent division by zero
        return mean_embeddings

    def extract_cls_token(self, outputs):
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[
                :, 0, :
            ]  # Standard Hugging Face BERT output
        else:
            return outputs[0][:, 0, :]  # Fallback for models returning a tuple

    def compute_loss(self, logits, labels):  # Compute loss if labels are provided
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return loss

    # Override the state_dict method to make all tensors contiguous before saving
    # TODO I am not sure why we need this at all???
    def state_dict(self, *args, **kwargs):
        original_state_dict = super().state_dict(*args, **kwargs)
        for key, tensor in original_state_dict.items():
            if not tensor.is_contiguous():
                original_state_dict[key] = tensor.contiguous()
        return original_state_dict
