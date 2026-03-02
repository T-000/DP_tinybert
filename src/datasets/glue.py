from dataclasses import dataclass
from typing import Optional, Callable, Dict, List

from datasets import load_dataset


@dataclass(frozen=True)
class GlueTask:
    name: str
    subset: str
    num_labels: int
    sentence1_key: str
    sentence2_key: Optional[str]
    val_split: str   # MNLI uses validation_matched


def get_task(name: str) -> GlueTask:
    n = name.lower()
    if n == "sst2":
        return GlueTask(
            name="sst2",
            subset="sst2",
            num_labels=2,
            sentence1_key="sentence",
            sentence2_key=None,
            val_split="validation",
        )
    if n == "qnli":
        return GlueTask(
            name="qnli",
            subset="qnli",
            num_labels=2,
            sentence1_key="question",
            sentence2_key="sentence",
            val_split="validation",
        )
    if n == "mnli":
        return GlueTask(
            name="mnli",
            subset="mnli",
            num_labels=3,
            sentence1_key="premise",
            sentence2_key="hypothesis",
            val_split="validation_matched",
        )
    if n == "qqp":
        return GlueTask(
            name="qqp",
            subset="qqp",
            num_labels=2,
            sentence1_key="question1",
            sentence2_key="question2",
            val_split="validation",
        )
    raise ValueError(f"Unknown GLUE task: {name}. Choose from: sst2, qnli, mnli, qqp")


def load_glue(task: GlueTask):
    """Returns a DatasetDict with splits like train/validation/..."""
    return load_dataset("glue", task.subset)


def finalize_format(ds, keep_token_type_ids: bool = True):
    """
    Standardize column name 'label' -> 'labels', and set torch format.
    """
    if "label" in ds["train"].column_names:
        ds = ds.rename_column("label", "labels")

    # for segment embedding
    cols: List[str] = ["input_ids", "attention_mask", "labels"]
    if keep_token_type_ids and "token_type_ids" in ds["train"].column_names:
        cols.append("token_type_ids")

    ds.set_format(type="torch", columns=cols)
    return ds