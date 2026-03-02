class FullFineTuning:
    name = "full_ft"

    def build(self, base_model, **kwargs):
        # Train all parameters
        for p in base_model.parameters():
            p.requires_grad = True

        return base_model