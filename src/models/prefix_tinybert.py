# Prefix-Tuning wrapper for TinyBERT.
from functools import cache

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
from transformers.cache_utils import DynamicCache


class PrefixedBertForSequenceClassification(nn.Module):
    """
    Clean Prefix-Tuning wrapper:
      - base_model frozen (including classifier)
      - trainable prefix as per-layer past_key_values (K/V)
      - forward passes past_key_values into BERT

    Trainable params ~ num_layers * 2 * prefix_len * hidden_size
    (very clean; no extra MLP)
    """
    def __init__(self, base_model: BertForSequenceClassification, num_virtual_tokens: int):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.num_virtual_tokens = num_virtual_tokens

        # Freeze whole base model
        for p in self.base_model.parameters():
            p.requires_grad = False

        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_attention_heads
        hidden = self.config.hidden_size
        head_dim = hidden // num_heads
        assert hidden % num_heads == 0

        # prefix_kv: (L, 2, prefix_len, num_heads, head_dim)
        self.prefix_kv = nn.Parameter(
            torch.empty(num_layers, 2, num_virtual_tokens, num_heads, head_dim)
        )
        nn.init.normal_(self.prefix_kv, mean=0.0, std=0.02)

    def _build_past_key_values(self, bsz: int, device):
        cache = DynamicCache()
        # 逐层写入 cache
        for layer in range(self.prefix_kv.shape[0]):
            k = self.prefix_kv[layer, 0].unsqueeze(0).expand(bsz, -1, -1, -1).to(device)
            v = self.prefix_kv[layer, 1].unsqueeze(0).expand(bsz, -1, -1, -1).to(device)
            
            # (bsz, prefix_len, heads, head_dim) -> (bsz, heads, prefix_len, head_dim)
            k = k.permute(0, 2, 1, 3).contiguous()
            v = v.permute(0, 2, 1, 3).contiguous()

            # DynamicCache.update(key_states, value_states, layer_idx)
            cache.update(k, v, layer_idx=layer)
        return cache

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        device = input_ids.device
        bsz, seq_len = input_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones((bsz, seq_len), device=device, dtype=torch.long)

        # only extend attention_mask for prefix length
        prefix_mask = torch.ones((bsz, self.num_virtual_tokens), device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # token_type_ids must stay ORIGINAL length (or just pass None)
        # let BERT create zeros internally
        token_type_ids = None

        past_key_values = self._build_past_key_values(bsz, device)

        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=True,
        )