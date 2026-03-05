from sklearn import base
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
from transformers.cache_utils import DynamicCache


class PrefixedBertForSequenceClassification(nn.Module):
    """
    Opacus-friendly prefix tuning + trainable classifier:
      - frozen BERT backbone stored OUTSIDE module tree
      - trainable: prefix_embed + prefix_proj + classifier
      - prefix is built batch-aware to avoid grad_sample batch=1
      - returns Tensor only: loss or logits
    """

    def __init__(self, base_model: BertForSequenceClassification, num_virtual_tokens: int):
        super().__init__()
        self.config = base_model.config
        self.num_virtual_tokens = int(num_virtual_tokens)

        # freeze whole base
        for p in base_model.parameters():
            p.requires_grad = False

        # store frozen backbone outside module tree
        self.__dict__["_frozen_base_model"] = base_model

        hidden = self.config.hidden_size
        num_layers = self.config.num_hidden_layers
        num_heads = self.config.num_attention_heads
        assert hidden % num_heads == 0
        head_dim = hidden // num_heads

        self.hidden = hidden
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # trainable prefix (batch-aware usage in forward)
        self.prefix_embed = nn.Embedding(self.num_virtual_tokens, hidden)
        nn.init.normal_(self.prefix_embed.weight, mean=0.0, std=0.02)

        self.prefix_proj = nn.Linear(hidden, num_layers * 2 * hidden)
        nn.init.normal_(self.prefix_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.prefix_proj.bias)

        # trainable classifier head
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden, self.config.num_labels)
        if hasattr(base_model, "classifier") and isinstance(base_model.classifier, nn.Linear):
            self.classifier.load_state_dict(base_model.classifier.state_dict())

        # loss fn
        if self.config.num_labels == 1:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def move_frozen_base(self, device):
        self.__dict__["_frozen_base_model"].to(device)

    def _build_past_key_values(self, bsz: int, device):
        prefix_len = self.num_virtual_tokens

        # (bsz, prefix_len)
        prefix_ids = torch.arange(prefix_len, device=device).unsqueeze(0).expand(bsz, -1)

        # (bsz, prefix_len, hidden)
        prefix_hidden = self.prefix_embed(prefix_ids)

        # (bsz, prefix_len, num_layers*2*hidden)
        proj = self.prefix_proj(prefix_hidden)

        # (bsz, prefix_len, num_layers, 2, num_heads, head_dim)
        proj = proj.view(bsz, prefix_len, self.num_layers, 2, self.num_heads, self.head_dim)

        cache = DynamicCache()
        for layer in range(self.num_layers):
            # (bsz, prefix_len, 2, heads, head_dim)
            layer_kv = proj[:, :, layer]

            k = layer_kv[:, :, 0]  # (bsz, prefix_len, heads, head_dim)
            v = layer_kv[:, :, 1]

            # -> (bsz, heads, prefix_len, head_dim)
            k = k.permute(0, 2, 1, 3).contiguous()
            v = v.permute(0, 2, 1, 3).contiguous()

            cache.update(k, v, layer_idx=layer)

        return cache

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        base = self.__dict__["_frozen_base_model"]
        bert = base.bert

        device = input_ids.device
        bsz, seq_len = input_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones((bsz, seq_len), device=device, dtype=torch.long)

        # extend mask for prefix length
        prefix_mask = torch.ones((bsz, self.num_virtual_tokens), device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        token_type_ids = None
        past_key_values = self._build_past_key_values(bsz, device)

        outputs = bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
            use_cache=False,
            return_dict=False,
        )

        pooled = outputs[1]
        logits = self.classifier(self.dropout(pooled))

        if labels is None:
            return logits

        if self.config.num_labels == 1:
            loss = self.loss_fn(logits.view(-1), labels.view(-1).float())
        else:
            loss = self.loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))

        return loss