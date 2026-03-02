from peft import PrefixTuningConfig, TaskType, get_peft_model


class PrefixTuning:
    """
    Prefix tuning via PEFT PrefixTuning.
    Works like adding trainable prefix vectors to attention (past key values).

    Notes:
      - prefix_projection=True is often used to reduce parameters.
      - num_virtual_tokens is the prefix length.
    """
    name = "prefix"

    def build(self, base_model, **kwargs):
        # Explicitly freeze base model
        for p in base_model.parameters():
            p.requires_grad = False

        num_virtual_tokens = int(kwargs.get("num_virtual_tokens", 20))
        prefix_projection = bool(kwargs.get("prefix_projection", True))

        cfg = PrefixTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=num_virtual_tokens,
            prefix_projection=prefix_projection,
        )

        model = get_peft_model(base_model, cfg)
        return model
