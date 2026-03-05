import os
import json
from xml.parsers.expat import model
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


def _clear_grad_samples(module: torch.nn.Module) -> None:
    """
    IMPORTANT: Do NOT delete p.grad_sample (Opacus expects the attribute to exist).
    Instead, reset it to None to avoid stale shapes across steps.
    """
    for p in module.parameters():
        if hasattr(p, "grad_sample"):
            p.grad_sample = None
        if hasattr(p, "_current_grad_sample"):
            p._current_grad_sample = None

def _debug_check_grad_sample_shapes(model: torch.nn.Module, expected_bs: int) -> None:
    """
    Print the first parameter whose grad_sample batch dim != expected_bs.
    Call this after loss.backward() and BEFORE optimizer.step().
    """
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        gs = getattr(p, "grad_sample", None)
        if gs is None:
            # some params may legitimately have None; skip
            continue
        if gs.shape[0] != expected_bs:
            print(
                f"[DP DEBUG] grad_sample mismatch: {name} "
                f"got={gs.shape[0]} expected={expected_bs} "
                f"param_shape={tuple(p.shape)} grad_sample_shape={tuple(gs.shape)}"
            )
            return
    # If we reach here, all matched (or were None)
    # print("[DP DEBUG] grad_sample shapes OK")
@torch.no_grad()
def evaluate(model, dataloader, device, metric_fn):
    model.eval()
    all_preds, all_labels = [], []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        labels = batch.get("labels")
        if labels is None:
            raise KeyError("Batch has no 'labels'. Check dataset formatting / collator.")

        batch.pop("labels", None)

        out = model(**batch)

        if torch.is_tensor(out):
            logits = out
        elif hasattr(out, "logits"):
            logits = out.logits
        else:
            logits = out[0]

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
    max_grad_norm=0.1,
    delta=None,
    noise_multiplier=None,
    target_epsilon=None,
    accountant="rdp",
    method_name: str | None = None,
):
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()

    model.to(device)
    if hasattr(model, "move_frozen_base"):
        model.move_frozen_base(device)

    # Opacus requires train mode at make_private time
    model.train()

    # Make HF models return tuples instead of SequenceClassifierOutput
    if hasattr(model, "config"):
        model.config.return_dict = False

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
    
    # ------------------------------------------------------------
    # Full FT patch (Opacus hooks + broadcasted position embeddings)
    # ------------------------------------------------------------
    # In full fine-tuning, position_embeddings.weight may produce grad_sample with batch dim = 1
    # (broadcasted), while other params have batch dim = actual Poisson batch (e.g., 24).
    # This breaks Opacus' per-sample norm stacking. We freeze position embeddings ONLY for full_ft.
    if method_name == "full_ft":
        if hasattr(model, "bert") and hasattr(model.bert, "embeddings"):
            pe = model.bert.embeddings.position_embeddings
            for p in pe.parameters():
                p.requires_grad = False
            print("[DP] full_ft patch: froze bert.embeddings.position_embeddings to avoid grad_sample batch=1 mismatch.")

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found. Check requires_grad flags.")
    
    print("\n[TRAINABLE PARAMETERS]")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(" -", n, p.shape)
    print()

    # fix: try adam ???
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

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

    # Validate only (don't fix) to avoid losing wrapper's private fields
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print("[Opacus] ModuleValidator warnings (not fixed):")
        for e in errors:
            print(" -", e)

    privacy_engine = PrivacyEngine(accountant="rdp")

    # wrapper vs non-wrapper grad sampling mode
    is_wrapper = (
        hasattr(model, "__dict__")
        and ("_frozen_base_model" in model.__dict__ or "_frozen_bert" in model.__dict__)
    )
    gsm = "functorch" if is_wrapper else "hooks"

    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma,
        max_grad_norm=c,
        poisson_sampling=True,
        grad_sample_mode=gsm,
    )

    # Scheduler AFTER make_private (poisson sampling may change loader length)
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
            _clear_grad_samples(model)  # 清 stale grad_sample

            out = model(**batch)

            # A: wrapper returns loss Tensor
            if torch.is_tensor(out):
                loss = out
            # B: HF SequenceClassifierOutput
            elif hasattr(out, "loss") and out.loss is not None:
                loss = out.loss
            # C: tuple
            else:
                loss = out[0]

            loss.backward()
            
            expected_bs = batch["input_ids"].shape[0]
            _debug_check_grad_sample_shapes(model, expected_bs)

            optimizer.step()

            # step 后再清一次，防止残留影响下一步
            _clear_grad_samples(model)
            optimizer.zero_grad(set_to_none=True)

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