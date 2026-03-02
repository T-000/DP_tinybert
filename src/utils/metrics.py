import numpy as np
from sklearn.metrics import f1_score
import evaluate


def get_metric(task_name: str):
    """
    Returns:
      metric_fn(preds: np.ndarray, labels: np.ndarray) -> float
      metric_name: str
    """
    task = task_name.lower()

    # GLUE SST2/QNLI/MNLI default metric: accuracy
    if task in {"sst2", "qnli", "mnli"}:
        acc = evaluate.load("accuracy")

        def metric_fn(preds, labels):
            preds = np.asarray(preds)
            labels = np.asarray(labels)
            return float(acc.compute(predictions=preds, references=labels)["accuracy"])

        return metric_fn, "accuracy"

    # GLUE QQP often reported with F1
    if task == "qqp":
        def metric_fn(preds, labels):
            preds = np.asarray(preds)
            labels = np.asarray(labels)
            return float(f1_score(labels, preds))

        return metric_fn, "f1"

    raise ValueError(f"Unknown task for metrics: {task_name}")