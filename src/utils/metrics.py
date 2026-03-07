import numpy as np
import evaluate


def get_metric(task_name: str):
    """
    Returns:
      metric_fn(preds: np.ndarray, labels: np.ndarray) -> float
      metric_name: str
    """
    acc = evaluate.load("accuracy")

    def metric_fn(preds, labels):
        preds = np.asarray(preds)
        labels = np.asarray(labels)
        return float(acc.compute(predictions=preds, references=labels)["accuracy"])

    return metric_fn, "accuracy"