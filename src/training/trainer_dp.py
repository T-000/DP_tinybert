import os
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier

from src.utils.metrics import get_metric


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, dataloader, device, metric_fn):
    model.eval()
    all_preds, all_labels = [], []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)

        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(batch["labels"].detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return metric_fn(all_preds, all_labels)


def train_dp(
    model,
    tokenizer,
    train_ds,
    val_ds,
    task_name,
    out_dir,
    epochs=5,
    batch_size=32,
    lr=5e-5,
    weight_decay=0.0,
    warmup_ratio=0.06,
    max_grad_norm=0.1,     # c in Algorithm 1 
    delta=None,            # δ (paper uses δ=1/N) 
    noise_multiplier=None, # σ in Algorithm 1 
    target_epsilon=None,   # optional: if set, compute noise_multiplier automatically
    accountant="rdp",
    secure_rng=False,
):
    """
    DP-SGD trainer that matches Algorithm 1 (PromptDPSGD):
      - Poisson sampling
      - per-sample gradients w.r.t. trainable params
      - clip to max_grad_norm (c)
      - add Gaussian noise with std = noise_multiplier * max_grad_norm (σ*c)
      - update trainable params only
      - track epsilon via accountant
    """
    os.makedirs(out_dir, exist_ok=True)

    device = get_device()
    model.to(device)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # IMPORTANT: Poisson sampling (Algorithm 1 line 3) 
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,  # Opacus will handle sampling; keep deterministic here
        collate_fn=collator,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    metric_fn, metric_name = get_metric(task_name)

    # Only optimize trainable parameters (your methods set requires_grad)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found. Check requires_grad flags.")

    optimizer = torch.optim.SGD(trainable_params, lr=lr, weight_decay=weight_decay)

    total_steps = epochs * max(1, math.ceil(len(train_loader)))
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # delta default: 1/N as in paper 
    N = len(train_ds)
    if delta is None:
        delta = 1.0 / max(1, N)

    # If user specifies target_epsilon, compute noise multiplier
    # Otherwise use provided noise_multiplier (σ)
    if target_epsilon is not None:
        # sampling rate q ~ batch_size / N (Poisson sampling) 
        sample_rate = batch_size / max(1, N)
        noise_multiplier = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant=accountant,
        )

    if noise_multiplier is None:
        raise ValueError("Provide either noise_multiplier (sigma) or target_epsilon.")

    privacy_engine = PrivacyEngine(accountant=accountant, secure_mode=secure_rng)

    # make_private with Poisson sampling behavior
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=float(noise_multiplier),
        max_grad_norm=float(max_grad_norm),
    )

    best_score = -1e9
    best_path = os.path.join(out_dir, "best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach().cpu())

        avg_loss = running_loss / max(1, len(train_loader))
        val_score = evaluate(model, val_loader, device, metric_fn)

        # Compute epsilon so far
        eps = privacy_engine.get_epsilon(delta)

        print(
            f"[{task_name}][DP] "
            f"Epoch {epoch} | "
            f"train_loss={avg_loss:.4f} | "
            f"val_{metric_name}={val_score:.4f} | "
            f"ε={eps:.3f} (δ={delta:.2e}, σ={float(noise_multiplier):.4f}, c={float(max_grad_norm):.3f})"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_path)

    final_eps = privacy_engine.get_epsilon(delta)

    summary = {
        "task": task_name,
        "metric": metric_name,
        "best": float(best_score),
        "dp": {
            "epsilon": float(final_eps),
            "delta": float(delta),
            "noise_multiplier": float(noise_multiplier),
            "max_grad_norm": float(max_grad_norm),
            "accountant": accountant,
        },
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary