import argparse
import csv
import os
import random
import numpy as np
import torch

from src.data.glue import get_task, load_glue, finalize_format
from src.models.tinybert import load_tinybert
from src.methods import METHODS
from src.training.trainer import train


# ------------------------
# Reproducibility
# ------------------------
def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------
# Results CSV helpers
# ------------------------
def ensure_results_csv(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset",
                "method",
                "metric",
                "value",
                "epochs",
                "lr",
                "batch_size",
                "max_length",
                "seed",
                "prompt_tokens",
                "lora_r",
                "lora_alpha",
                "lora_dropout",
                "trainable_params",
                "total_params",
            ])


def append_results_csv(
    path: str,
    dataset: str,
    method: str,
    metric: str,
    value: float,
    epochs: int,
    lr: float,
    batch_size: int,
    max_length: int,
    seed: int,
    prompt_tokens: int | None,
    lora_r: int | None,
    lora_alpha: int | None,
    lora_dropout: float | None,
    trainable_params: int,
    total_params: int,
):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            dataset,
            method,
            metric,
            f"{value:.4f}",
            epochs,
            lr,
            batch_size,
            max_length,
            seed,
            "" if prompt_tokens is None else prompt_tokens,
            "" if lora_r is None else lora_r,
            "" if lora_alpha is None else lora_alpha,
            "" if lora_dropout is None else lora_dropout,
            trainable_params,
            total_params,
        ])

# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", nargs="+", default=["sst2", "qnli", "mnli", "qqp"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["soft_prompt", "prefix", "full_ft", "lora", "linear_probe"],
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out_dir", default="results/runs")
    parser.add_argument("--results_csv", default="results/tables/results.csv")

    # method-specific knobs
    parser.add_argument("--prompt_tokens", type=int, default=10)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_results_csv(args.results_csv)

    for dataset_name in args.datasets:
        task = get_task(dataset_name)
        raw = load_glue(task)

        # Load tokenizer once per dataset (fixed tinybert, only num_labels differs)
        tokenizer, _ = load_tinybert(num_labels=task.num_labels)

        s1_key = task.sentence1_key
        s2_key = task.sentence2_key

        def preprocess(batch):
            if s2_key is None:
                return tokenizer(
                    batch[s1_key],
                    truncation=True,
                    max_length=args.max_length,
                )
            return tokenizer(
                batch[s1_key],
                batch[s2_key],
                truncation=True,
                max_length=args.max_length,
            )

        raw = raw.map(preprocess, batched=True)
        raw = finalize_format(raw, keep_token_type_ids=True)

        train_ds = raw["train"]
        val_ds = raw[task.val_split]

        for method_name in args.methods:
            if method_name not in METHODS:
                raise ValueError(
                    f"Unknown method: {method_name}. Available: {sorted(METHODS.keys())}"
                )

            # fresh model for fairness (new weights each run)
            _, base_model = load_tinybert(num_labels=task.num_labels)

            method = METHODS[method_name]

            # build() returns a MODEL now (not BuiltModel)
            if method_name in ["soft_prompt", "prefix"]:
                model = method.build(base_model, num_virtual_tokens=args.prompt_tokens)
            elif method_name == "lora":
                model = method.build(
                    base_model,
                    r=args.lora_r,
                    alpha=args.lora_alpha,
                    dropout=args.lora_dropout,
                )
            else:
                model = method.build(base_model)

            run_name = f"{dataset_name}__{method_name}"
            out_dir = os.path.join(args.out_dir, run_name)

            # compute param stats here
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())

            print(f"\n=== {run_name} ===")
            print(f"Trainable params: {trainable_params}")
            print(f"Total params:     {total_params}")

            summary = train(
                model=model,
                tokenizer=tokenizer,
                train_ds=train_ds,
                val_ds=val_ds,
                task_name=dataset_name,
                out_dir=out_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )
            print("WRITE CSV:", dataset_name, method_name, summary["best"])
            # method-specific fields for logging
            prompt_tokens = args.prompt_tokens if method_name in ["soft_prompt", "prefix"] else None
            lora_r = args.lora_r if method_name == "lora" else None
            lora_alpha = args.lora_alpha if method_name == "lora" else None
            lora_dropout = args.lora_dropout if method_name == "lora" else None

            append_results_csv(
                args.results_csv,
                dataset_name,
                method_name,
                summary["metric"],
                summary["best"],
                args.epochs,
                args.lr,
                args.batch_size,
                args.max_length,
                args.seed,
                prompt_tokens,
                lora_r,
                lora_alpha,
                lora_dropout,
                trainable_params,
                total_params,
            )

    print("\nAll experiments finished.")
    print(f"Results saved to: {args.results_csv}")


if __name__ == "__main__":
    main()