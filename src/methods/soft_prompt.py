from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model


class SoftPrompt:
    """
    Soft prompt tuning via PEFT PromptTuning:
      - prepends trainable virtual tokens to the input
      - freezes original model weights by default
    """
    name = "soft_prompt"

    def build(self, base_model, **kwargs):
        # Explicitly freeze base model
        for p in base_model.parameters():
            p.requires_grad = False

        num_virtual_tokens = int(kwargs.get("num_virtual_tokens", 20))

        cfg = PromptTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=num_virtual_tokens,
            prompt_tuning_init=PromptTuningInit.RANDOM,  # avoids needing tokenizer/text init
        )

        model = get_peft_model(base_model, cfg)
        print("trainable:", [n for n,p in model.named_parameters() if p.requires_grad][:30])
        return model