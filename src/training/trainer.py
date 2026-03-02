import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

from utils.metrics import get_metric


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, dataloader, device, metric_fn):
    model.eval()

    all_preds = []
    all_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)

        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(batch["labels"].detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return metric_fn(all_preds, all_labels)


def train(
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
    max_grad_norm=1.0,
):
    os.makedirs(out_dir, exist_ok=True)

    device = get_device()
    model.to(device)

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    metric_fn, metric_name = get_metric(task_name)

    # Only optimize trainable parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    total_steps = epochs * len(train_loader)
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

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))

        val_score = evaluate(model, val_loader, device, metric_fn)

        print(
            f"[{task_name}] "
            f"Epoch {epoch} | "
            f"train_loss={avg_loss:.4f} | "
            f"val_{metric_name}={val_score:.4f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_path)

    summary = {
        "task": task_name,
        "metric": metric_name,
        "best": float(best_score),
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary