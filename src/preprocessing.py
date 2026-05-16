"""Preprocessing helpers for the healthcare risk classification project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region",
    "charges",
]

NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_insurance_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the insurance dataset and check the expected columns."""
    df = pd.read_csv(csv_path)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")
    return df


def add_healthcare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Add the binary high_cost target and simple grouping variables."""
    engineered = df.copy()
    high_cost_threshold = float(engineered["charges"].quantile(0.75))
    engineered["high_cost"] = (engineered["charges"] > high_cost_threshold).astype(int)

    engineered["bmi_category"] = np.select(
        [
            engineered["bmi"] < 25,
            (engineered["bmi"] >= 25) & (engineered["bmi"] < 30),
        ],
        ["normal", "overweight"],
        default="obese",
    )

    engineered["age_group"] = np.select(
        [
            engineered["age"] < 30,
            (engineered["age"] >= 30) & (engineered["age"] < 50),
        ],
        ["young", "middle"],
        default="older",
    )

    return engineered, high_cost_threshold


def get_feature_target(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = "high_cost",
) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and target while avoiding charge leakage."""
    if feature_columns is None:
        feature_columns = MODEL_FEATURES
    return df[feature_columns].copy(), df[target_column].copy()


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Create a stratified train/test split for stable class proportions."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
