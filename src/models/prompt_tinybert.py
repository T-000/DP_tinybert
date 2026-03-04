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
        self.config = base_model.config
        self.num_virtual_tokens = num_virtual_tokens

        hidden = self.config.hidden_size
        #self.soft_prompt = nn.Parameter(torch.empty(num_virtual_tokens, hidden))
        #nn.init.normal_(self.soft_prompt, mean=0.0, std=0.02)
        self.prompt_embed = nn.Embedding(num_virtual_tokens, hidden)
        nn.init.normal_(self.prompt_embed.weight, mean=0.0, std=0.02)

        # Freeze base model params
        for p in base_model.parameters():
            p.requires_grad = False

        # 不要注册为子模块（避免 Opacus 看到 buffers）
        self.__dict__["_frozen_base_model"] = base_model

    def move_frozen_base(self, device):
        # 因为 frozen base 不在 module tree 里，model.to(device) 不会搬它
        self.__dict__["_frozen_base_model"].to(device)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        base_model = self.__dict__["_frozen_base_model"]

        # embeddings
        emb_layer = base_model.bert.embeddings.word_embeddings
        inputs_embeds = emb_layer(input_ids)

        bsz = inputs_embeds.shape[0]
        device = inputs_embeds.device

        #prompt = self.soft_prompt.unsqueeze(0).expand(bsz, -1, -1)
        prompt_ids = torch.arange(self.num_virtual_tokens, device=device).unsqueeze(0).expand(bsz, -1)
        prompt = self.prompt_embed(prompt_ids)  # (bsz, s, hidden)
        inputs_embeds = torch.cat([prompt, inputs_embeds], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=device, dtype=torch.long)
        prompt_mask = torch.ones((bsz, self.num_virtual_tokens), device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        if token_type_ids is not None:
            prompt_tti = torch.zeros((bsz, self.num_virtual_tokens), device=device, dtype=token_type_ids.dtype)
            token_type_ids = torch.cat([prompt_tti, token_type_ids], dim=1)

        outputs = base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_dict=False,
            output_hidden_states=False,
            output_attentions=False,
        )

        if labels is not None:
            loss = outputs[0]
            return loss            # training loss tensor
        else:
            logits = outputs[0]
            return logits          # evaluation logits tensor