import torch
import torch.nn as nn
from transformers import BertForSequenceClassification


class SoftPromptedBertForSequenceClassification(nn.Module):
    """
    A clean soft-prompt wrapper:
      - base_model is frozen BertForSequenceClassification
      - train only soft prompt P: (s, hidden_size)
      - forward uses inputs_embeds so we can prepend prompt embeddings
    """
    def __init__(self, base_model: BertForSequenceClassification, num_virtual_tokens: int):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.num_virtual_tokens = num_virtual_tokens

        hidden = self.config.hidden_size
        self.soft_prompt = nn.Parameter(torch.empty(num_virtual_tokens, hidden))
        nn.init.normal_(self.soft_prompt, mean=0.0, std=0.02)

        # Freeze entire base model (including classifier) — matches PromptDPSGD idea
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Only prompt trainable
        self.soft_prompt.requires_grad = True

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # Get token embeddings
        emb_layer = self.base_model.bert.embeddings.word_embeddings
        inputs_embeds = emb_layer(input_ids)

        bsz = inputs_embeds.shape[0]
        device = inputs_embeds.device

        # Expand prompt to batch: (bsz, s, hidden)
        prompt = self.soft_prompt.unsqueeze(0).expand(bsz, -1, -1).to(device)

        # Prepend embeddings
        inputs_embeds = torch.cat([prompt, inputs_embeds], dim=1)

        # Expand masks accordingly
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=device, dtype=torch.long)
        prompt_mask = torch.ones((bsz, self.num_virtual_tokens), device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        if token_type_ids is not None:
            prompt_tti = torch.zeros((bsz, self.num_virtual_tokens), device=device, dtype=token_type_ids.dtype)
            token_type_ids = torch.cat([prompt_tti, token_type_ids], dim=1)

        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )