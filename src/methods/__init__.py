from .full_ft import FullFineTuning
from .linear_probe import LinearProbe
from .lora import LoRA
from .soft_prompt import SoftPrompt
from .prefix import PrefixTuning

METHODS = {
    FullFineTuning.name: FullFineTuning(),
    LinearProbe.name: LinearProbe(),
    LoRA.name: LoRA(),
    SoftPrompt.name: SoftPrompt(),
    PrefixTuning.name: PrefixTuning(),
}