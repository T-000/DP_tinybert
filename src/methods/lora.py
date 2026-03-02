from peft import LoraConfig, TaskType, get_peft_model


class LoRA:
    """
    PEFT LoRA for sequence classification.

    CLI passes: r, alpha, dropout
    Optional: target_modules (list[str]) - for BERT defaults to ["query", "value"]
    """
    name = "lora"

    def build(self, base_model, **kwargs):
        # Explicitly freeze base model
        for p in base_model.parameters():
            p.requires_grad = False

        r = int(kwargs.get("r", 8))
        alpha = int(kwargs.get("alpha", 16))
        dropout = float(kwargs.get("dropout", 0.0))
        target_modules = kwargs.get("target_modules", ["query", "value"])

        cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )

        model = get_peft_model(base_model, cfg)
        return model