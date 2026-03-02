class LinearProbe:
    """
    Freeze the entire backbone; train only the classification head.

    For BertForSequenceClassification:
      - head is usually `classifier`
      - sometimes `score` exists (for some heads)
    """
    name = "linear_probe"

    def build(self, base_model, **kwargs):
        # Freeze everything
        for p in base_model.parameters():
            p.requires_grad = False

        # Unfreeze classifier head
        head_prefixes = ("classifier", "score")
        for n, p in base_model.named_parameters():
            if n.startswith(head_prefixes):
                p.requires_grad = True

        # Optional: some people also unfreeze pooler (rarely needed for benchmarks)
        # if kwargs.get("unfreeze_pooler", False):
        #     for n, p in base_model.named_parameters():
        #         if n.startswith("bert.pooler"):
        #             p.requires_grad = True

        return base_model