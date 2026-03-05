import torch
import torch.nn as nn
from transformers import BertForSequenceClassification


class SoftPromptedBertForSequenceClassification(nn.Module):
    def __init__(self, base_model: BertForSequenceClassification, num_virtual_tokens: int):
        super().__init__()
        self.config = base_model.config
        self.num_virtual_tokens = num_virtual_tokens
        hidden = self.config.hidden_size

        # trainable prompt
        self.prompt_embed = nn.Embedding(num_virtual_tokens, hidden)
        nn.init.normal_(self.prompt_embed.weight, mean=0.0, std=0.02)

        self.dropout = base_model.dropout
        self.classifier = base_model.classifier

        # freeze EVERYTHING first
        for p in base_model.parameters():
            p.requires_grad = False

        # unfreeze only classifier
        for p in self.classifier.parameters():
            p.requires_grad = True

        # keep backbone hidden from Opacus
        self.__dict__["_frozen_base_model"] = base_model

    def move_frozen_base(self, device):
        self.__dict__["_frozen_base_model"].to(device)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        base = self.__dict__["_frozen_base_model"]
        device = input_ids.device

        # embeddings
        emb_layer = base.bert.embeddings.word_embeddings
        inputs_embeds = emb_layer(input_ids)

        bsz = inputs_embeds.size(0)
        prompt_ids = torch.arange(self.num_virtual_tokens, device=device).unsqueeze(0).expand(bsz, -1)
        prompt = self.prompt_embed(prompt_ids)
        inputs_embeds = torch.cat([prompt, inputs_embeds], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=device, dtype=torch.long)
        prefix_mask = torch.ones((bsz, self.num_virtual_tokens), device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # run ONLY bert backbone (frozen)
        bert_out = base.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=None,
            return_dict=False,
        )
        pooled = bert_out[1]  

        logits = self.classifier(self.dropout(pooled))

        if labels is None:
            return logits

        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))