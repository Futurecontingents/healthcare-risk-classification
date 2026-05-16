"""Evaluation helpers for binary classification models."""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classifier(name: str, model, X_test, y_test) -> dict[str, float | str]:
    """Evaluate a fitted classifier on the test set."""
    y_pred = model.predict(X_test)
    row: dict[str, float | str] = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }
    return row


def comparison_table(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    """Create a rounded comparison table for reporting."""
    table = pd.DataFrame(rows)
    metric_columns = ["accuracy", "precision", "recall", "f1_score"]
    table[metric_columns] = table[metric_columns].astype(float).round(3)
    return table


def model_confusion_matrix(model, X_test, y_test) -> np.ndarray:
    """Return a binary confusion matrix for a fitted classifier."""
    return confusion_matrix(y_test, model.predict(X_test))
