# DP TinyBERT Experiments

This repo contains a small experimental setup for comparing different parameter-efficient adaptation methods on TinyBERT, both **with and without Differential Privacy (DP)**.

Methods implemented:

- Full fine-tuning
- Linear probing
- LoRA
- Soft prompt tuning
- Prefix tuning

Privacy is implemented using **Opacus (DP-SGD)**.

---

# Tasks

Currently supported GLUE tasks:

- SST-2
- QNLI
- QQP
- MNLI

Each run logs results into:
results/tables/results.csv

---

# Setup

Create environment and install dependencies:

pip install torch transformers datasets opacus

Python 3.11 was used during development.

---

# Running Experiments

All experiments are launched through:
python -m src.cli

Example (non-private baseline):
python -m src.cli
--privacy none
--datasets qnli qqp mnli
--methods full_ft linear_probe lora soft_prompt prefix
--epochs 8
--batch_size 32
--lr 5e-4
--prompt_tokens 10

DP version:

python -m src.cli
--privacy dp
--datasets qnli qqp mnli
--methods full_ft linear_probe lora soft_prompt prefix
--epochs 8
--batch_size 32
--lr 5e-4
--prompt_tokens 10
--dp_target_epsilon 8
--dp_max_grad_norm 0.1

This will automatically:

1. Train models  
2. Evaluate on validation split  
3. Save best checkpoint  
4. Append results to `results.csv`

---

# Parameters

Some commonly adjusted arguments:

--epochs
--batch_size
--lr
--prompt_tokens

DP-specific arguments:

--dp_target_epsilon
--dp_max_grad_norm
--dp_noise_multiplier
--dp_delta

Typical DP configuration used in experiments:

epsilon = 8
max_grad_norm = 0.1

# Methods

Each method modifies TinyBERT differently.

### Full Fine-Tuning
All parameters are updated.

### Linear Probe
Backbone is frozen, only the classifier layer is trained.

### LoRA
Low-rank adapters added to attention layers.

### Soft Prompt
Trainable prompt embeddings prepended to the input sequence.

### Prefix Tuning
Trainable key/value prefixes injected into attention layers.

---

# Differential Privacy

DP training uses **DP-SGD** implemented via Opacus.

Main steps:

1. Per-sample gradient computation  
2. Gradient clipping  
3. Noise addition  

Privacy budget is tracked with the RDP accountant.

Results log the following fields:

epsilon
delta
noise_multiplier
max_grad_norm


# Notes

Some observations from early experiments:

- DP training can be unstable with full fine-tuning.
- Later epochs sometimes degrade performance due to noise accumulation.
- Using Adam optimizer significantly improves accuracy than SGD optimizer.


---

# Code Structure
src/
    -cli.py
    -data/
    -models/
    -methods/
    -training/
    -utils/
