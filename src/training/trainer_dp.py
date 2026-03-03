import os
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

from opacus.accountants import RDPAccountant
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


def _global_l2_norm(tensors) -> torch.Tensor:
    # sqrt(sum_i ||g_i||^2)
    return torch.sqrt(sum((t.norm(2) ** 2) for t in tensors) + 1e-12)


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
    max_grad_norm=0.1,        # c
    delta=None,               # default 1/N
    noise_multiplier=None,    # sigma
    target_epsilon=None,      # if set, derive sigma
    accountant="rdp",
    microbatch_size=8,        # IMPORTANT: makes PromptDPSGD fast & readable
):
    """
    Fast, readable PromptDPSGD-style DP-SGD that matches Algorithm 1 conceptually:
      - compute per-(micro)batch gradient w.r.t. *trainable params only*
      - clip by global L2 norm to c
      - add Gaussian noise with std = sigma * c
      - update parameters
      - track epsilon with RDP accountant

    Notes:
      - If microbatch_size=1, this is the closest to true per-example clipping.
      - Larger microbatch_size is a common engineering compromise to speed up training.
    """

    os.makedirs(out_dir, exist_ok=True)
    device = get_device()
    model.to(device)

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

    # Only DP-update trainable params (methods define requires_grad)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found. Check requires_grad flags.")

    optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)

    total_steps = epochs * max(1, len(train_loader))
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    N = len(train_ds)
    if delta is None:
        delta = 1.0 / max(1, N)

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

    acc = RDPAccountant()

    best_score = -1e9
    best_path = os.path.join(out_dir, "best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            bsz = batch["input_ids"].shape[0]

            # We'll accumulate clipped grads over microbatches, then add noise once (Alg.1 line 6 idea)
            summed_grads = [torch.zeros_like(p, device=device) for p in params]
            mb_total = 0

            # microbatch loop
            for start in range(0, bsz, microbatch_size):
                mb = {k: v[start : start + microbatch_size] for k, v in batch.items()}
                mb_size = mb["input_ids"].shape[0]
                mb_total += mb_size

                optimizer.zero_grad(set_to_none=True)

                out = model(**mb)
                loss = out.loss
                loss.backward()

                grads = []
                for p in params:
                    if p.grad is None:
                        grads.append(torch.zeros_like(p, device=device))
                    else:
                        grads.append(p.grad.detach().clone())

                # clip microbatch gradient by global norm to c
                gnorm = _global_l2_norm(grads)
                coef = min(1.0, c / float(gnorm))
                grads = [g * coef for g in grads]

                for j in range(len(summed_grads)):
                    summed_grads[j] += grads[j]

                running_loss += float(loss.detach().cpu()) * mb_size

            # add noise once per step (Gaussian with std = sigma * c), then average
            std = sigma * c
            for j, p in enumerate(params):
                noise = torch.randn_like(p, device=device) * std
                p.grad = (summed_grads[j] + noise) / max(1, mb_total)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # accountant step
            acc.step(noise_multiplier=sigma, sample_rate=sample_rate)

        avg_loss = running_loss / max(1, len(train_ds))
        val_score = evaluate(model, val_loader, device, metric_fn)
        eps = acc.get_epsilon(delta)

        print(
            f"[{task_name}][DP] Epoch {epoch} | "
            f"train_loss={avg_loss:.4f} | val_{metric_name}={val_score:.4f} | "
            f"ε={eps:.3f} (δ={delta:.2e}, σ={sigma:.4f}, c={c:.3f}, micro={microbatch_size})"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_path)

    final_eps = acc.get_epsilon(delta)

    summary = {
        "task": task_name,
        "metric": metric_name,
        "best": float(best_score),
        "dp": {
            "epsilon": float(final_eps),
            "delta": float(delta),
            "noise_multiplier": float(sigma),
            "max_grad_norm": float(c),
            "microbatch_size": int(microbatch_size),
            "accountant": "rdp",
        },
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary