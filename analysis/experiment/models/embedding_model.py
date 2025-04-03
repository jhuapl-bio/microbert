# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC


import torch
from torch import nn
from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM
from transformers.models.bert.configuration_bert import BertConfig


class GenomicEmbeddingModel(nn.Module):
    def __init__(self, base_model_name):
        """
        Initialize the model.

        Args:
            base_model_name (str): The name of the pretrained model.
        """
        super(GenomicEmbeddingModel, self).__init__()

        self.base_model_name = base_model_name

        if base_model_name in ["zhihan1996/DNABERT-2-117M"]:
            # Load Config if exists - need for DNABERT 2
            # https://www.kaggle.com/code/gabrielcabas/dnabert-for-classification
            # https://huggingface.co/zhihan1996/DNABERT-2-117M/discussions/26
            self.config = BertConfig.from_pretrained(base_model_name)
            self.base_model = AutoModel.from_pretrained(
                base_model_name, trust_remote_code=True, config=self.config
            )
        elif "nucleotide-transformer-v2" in base_model_name:
            self.base_model = AutoModelForMaskedLM.from_pretrained(
                base_model_name, trust_remote_code=True
            )
            self.config = self.base_model.config
        elif "GenomeOcean" in base_model_name:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            self.config = self.base_model.config
        else:
            self.base_model = AutoModel.from_pretrained(
                base_model_name, trust_remote_code=True
            )
            self.config = self.base_model.config

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

        return {"embedding": mean_embeddings}

    def mean_pooling(self, hidden_states, attention_mask):
        attention_mask_expanded = attention_mask.unsqueeze(
            -1
        )  # Expand mask for broadcasting
        sum_hidden_states = torch.sum(hidden_states * attention_mask_expanded, dim=1)
        sum_attention_mask = torch.sum(
            attention_mask_expanded, dim=1
        )  # Keep dim for broadcasting
        return sum_hidden_states / sum_attention_mask.clamp(
            min=1e-9
        )  # Avoid division by zero

    def extract_cls_token(self, outputs):
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[
                :, 0, :
            ]  # Standard Hugging Face BERT output
        else:
            return outputs[0][:, 0, :]  # Fallback for models returning a tuple

    # Override the state_dict method to make all tensors contiguous before saving
    # TODO I am not sure why we need this at all???
    def state_dict(self, *args, **kwargs):
        original_state_dict = super().state_dict(*args, **kwargs)
        for key, tensor in original_state_dict.items():
            if not tensor.is_contiguous():
                original_state_dict[key] = tensor.contiguous()
        return original_state_dict
