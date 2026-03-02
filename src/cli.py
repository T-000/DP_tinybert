import argparse
import csv
import os
import random
import numpy as np
import torch

from datasets.glue import get_task, load_glue, finalize_format
from models.tinybert import load_tinybert
from methods import METHODS
from training.trainer import train_eval


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
            writer.writerow(
                ["dataset", "metric", "soft_prompt", "prefix", "full_ft", "lora", "linear_probe"]
            )


def update_results_csv(path: str, dataset: str, metric: str, method: str, value: float):
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    row = None
    for r in rows:
        if r["dataset"] == dataset:
            row = r
            break

    if row is None:
        row = {
            "dataset": dataset,
            "metric": metric,
            "soft_prompt": "",
            "prefix": "",
            "full_ft": "",
            "lora": "",
            "linear_probe": "",
        }
        rows.append(row)

    row["metric"] = metric
    row[method] = f"{value:.4f}"

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "metric", "soft_prompt", "prefix", "full_ft", "lora", "linear_probe"],
        )
        writer.writeheader()
        writer.writerows(rows)


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

            if method_name in ["soft_prompt", "prefix"]:
                built = method.build(base_model, num_virtual_tokens=args.prompt_tokens)
            elif method_name == "lora":
                built = method.build(
                    base_model,
                    r=args.lora_r,
                    alpha=args.lora_alpha,
                    dropout=args.lora_dropout,
                )
            else:
                built = method.build(base_model)

            run_name = f"{dataset_name}__{method_name}"
            out_dir = os.path.join(args.out_dir, run_name)

            print(f"\n=== {run_name} ===")
            print(f"Trainable params: {built.trainable_params}")
            print(f"Total params:     {built.total_params}")

            summary = train_eval(
                model=built.model,
                tokenizer=tokenizer,
                train_ds=train_ds,
                val_ds=val_ds,
                task_name=dataset_name,
                out_dir=out_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )

            update_results_csv(
                args.results_csv,
                dataset_name,
                summary["metric"],
                method_name,
                summary["best"],
            )

    print("\nAll experiments finished.")
    print(f"Results saved to: {args.results_csv}")


if __name__ == "__main__":
    main()