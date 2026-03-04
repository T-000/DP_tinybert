import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.validators import ModuleValidator

from src.utils.metrics import get_metric


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, dataloader, device, metric_fn):
    """
    Assumes: model(**batch_without_labels) returns logits Tensor of shape (bsz, num_labels).
    """
    model.eval()
    all_preds, all_labels = [], []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        labels = batch.get("labels")
        if labels is None:
            raise KeyError("Batch has no 'labels'. Check dataset formatting / collator.")

        batch.pop("labels", None)

        logits = model(**batch)  # logits tensor
        preds = torch.argmax(logits, dim=-1)

        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

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
    max_grad_norm=0.1,        # clipping norm c
    delta=None,               # default 1/N
    noise_multiplier=None,    # sigma
    target_epsilon=None,      # if set, derive sigma
    accountant="rdp",
):
    """
    DP training using Opacus PrivacyEngine:
      - per-sample clipping (DP-SGD)
      - Gaussian noise
      - Poisson sampling
      - RDP accountant, report epsilon

    Assumes wrapper behavior:
      - model(**batch_with_labels) returns loss Tensor
      - model(**batch_without_labels) returns logits Tensor
      - if wrapper stores frozen base out of module tree, it provides move_frozen_base(device)
    """

    os.makedirs(out_dir, exist_ok=True)
    device = get_device()

    model.to(device)
    if hasattr(model, "move_frozen_base"):
        model.move_frozen_base(device)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
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

    # Only DP-update trainable params (wrappers set requires_grad)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found. Check requires_grad flags.")

    optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)

    # DP defaults
    N = len(train_ds)
    if delta is None:
        delta = 1.0 / max(1, N)

    # Used to derive sigma from target epsilon.
    # Under Poisson sampling, q ≈ batch_size / N
    sample_rate = batch_size / max(1, N)

    if target_epsilon is not None:
        noise_multiplier = get_noise_multiplier(
            target_epsilon=float(target_epsilon),
            target_delta=float(delta),
            sample_rate=float(sample_rate),
            epochs=int(epochs),
            accountant=accountant,
        )

    if noise_multiplier is None:
        raise ValueError("Provide either noise_multiplier or target_epsilon.")

    sigma = float(noise_multiplier)
    c = float(max_grad_norm)

    # Validate only (don't fix) to avoid losing wrapper's private fields
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print("[Opacus] ModuleValidator warnings (not fixed):")
        for e in errors:
            print(" -", e)

    privacy_engine = PrivacyEngine(accountant="rdp")

    # Wrap model/optimizer/loader for DP-SGD
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma,
        max_grad_norm=c,
        poisson_sampling=True,          # align with DP accountant assumptions
        grad_sample_mode="functorch",   # most compatible with transformers
    )

    # Scheduler AFTER make_private (train_loader may change with poisson sampling)
    total_steps = epochs * max(1, len(train_loader))
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_score = -1e9
    best_path = os.path.join(out_dir, "best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            # Wrapper returns loss Tensor when labels are present
            loss = model(**batch)
            loss.backward()

            optimizer.step()
            scheduler.step()

            running_loss += float(loss.detach().cpu()) * batch["input_ids"].shape[0]

        avg_loss = running_loss / max(1, len(train_ds))
        val_score = evaluate(model, val_loader, device, metric_fn)
        eps = privacy_engine.get_epsilon(delta)

        print(
            f"[{task_name}][DP] Epoch {epoch} | "
            f"train_loss={avg_loss:.4f} | val_{metric_name}={val_score:.4f} | "
            f"ε={eps:.3f} (δ={delta:.2e}, σ={sigma:.4f}, c={c:.3f})"
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
            "noise_multiplier": float(sigma),
            "max_grad_norm": float(c),
            "microbatch_size": None,
            "accountant": "rdp",
        },
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary