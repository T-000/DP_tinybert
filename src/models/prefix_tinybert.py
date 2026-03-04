import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
from transformers.cache_utils import DynamicCache


class PrefixedBertForSequenceClassification(nn.Module):
    """
    Opacus-friendly Prefix-Tuning wrapper for BERT:
      - frozen backbone stored outside module tree (avoid buffers check)
      - trainable prefix implemented as Embedding + Linear "prefix encoder"
      - injects per-layer K/V via past_key_values (DynamicCache)
      - returns Tensor only:
          * if labels is not None -> loss Tensor
          * else -> logits Tensor
    """

    def __init__(self, base_model: BertForSequenceClassification, num_virtual_tokens: int):
        super().__init__()
        self.config = base_model.config
        self.num_virtual_tokens = num_virtual_tokens

        # Freeze base model params
        for p in base_model.parameters():
            p.requires_grad = False

        # Store frozen base outside module tree (avoid Opacus buffer restriction)
        self.__dict__["_frozen_base_model"] = base_model

        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_attention_heads
        hidden = self.config.hidden_size
        assert hidden % num_heads == 0
        head_dim = hidden // num_heads

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden = hidden
        self.head_dim = head_dim

        # Trainable prefix "tokens" in hidden space: (prefix_len, hidden)
        self.prefix_embed = nn.Embedding(num_virtual_tokens, hidden)
        nn.init.normal_(self.prefix_embed.weight, mean=0.0, std=0.02)

        # Prefix encoder: for each prefix token embedding (hidden),
        # produce all layers' K and V in hidden space.
        # Output per token: (num_layers * 2 * hidden)
        self.prefix_proj = nn.Linear(hidden, num_layers * 2 * hidden, bias=True)
        nn.init.normal_(self.prefix_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.prefix_proj.bias)

    def move_frozen_base(self, device):
        self.__dict__["_frozen_base_model"].to(device)

    def _build_past_key_values(self, bsz: int, device):
        """
        Build DynamicCache with per-layer prefix K/V.

        We generate:
          prefix_hidden: (prefix_len, hidden)
          proj: (prefix_len, num_layers*2*hidden)
        Reshape to:
          (prefix_len, num_layers, 2, num_heads, head_dim)
        Then expand batch and permute to:
          (bsz, num_heads, prefix_len, head_dim) for each layer's K/V
        """
        base_model = self.__dict__["_frozen_base_model"]

        # (prefix_len,)
        prefix_ids = torch.arange(self.num_virtual_tokens, device=device)
        # (prefix_len, hidden)
        prefix_hidden = self.prefix_embed(prefix_ids)

        # (prefix_len, num_layers*2*hidden)
        proj = self.prefix_proj(prefix_hidden)

        # (prefix_len, num_layers, 2, hidden)
        proj = proj.view(self.num_virtual_tokens, self.num_layers, 2, self.hidden)

        # (prefix_len, num_layers, 2, num_heads, head_dim)
        proj = proj.view(self.num_virtual_tokens, self.num_layers, 2, self.num_heads, self.head_dim)

        cache = DynamicCache()

        # Fill cache layer by layer
        for layer in range(self.num_layers):
            # (prefix_len, 2, num_heads, head_dim)
            layer_kv = proj[:, layer]  # index layer

            k = layer_kv[:, 0]  # (prefix_len, num_heads, head_dim)
            v = layer_kv[:, 1]  # (prefix_len, num_heads, head_dim)

            # Expand batch: (bsz, prefix_len, num_heads, head_dim)
            k = k.unsqueeze(0).expand(bsz, -1, -1, -1).contiguous()
            v = v.unsqueeze(0).expand(bsz, -1, -1, -1).contiguous()

            # Opacus/HF expects (bsz, heads, prefix_len, head_dim)
            k = k.permute(0, 2, 1, 3).contiguous()
            v = v.permute(0, 2, 1, 3).contiguous()

            cache.update(k, v, layer_idx=layer)

        return cache

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        base_model = self.__dict__["_frozen_base_model"]

        device = input_ids.device
        bsz, seq_len = input_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones((bsz, seq_len), device=device, dtype=torch.long)

        # Extend attention mask to account for prefix length
        prefix_mask = torch.ones((bsz, self.num_virtual_tokens), device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Simpler: let BERT create token_type_ids internally
        token_type_ids = None

        past_key_values = self._build_past_key_values(bsz, device)

        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=False,
            return_dict=False,
            output_hidden_states=False,
            output_attentions=False,
        )

        # return Tensor only (match trainer_dp)
        if labels is not None:
            loss = outputs[0]   # (loss, logits, ...)
            return loss
        else:
            logits = outputs[0] # (logits, ...)
            return logits