"""Model builders for interpretable healthcare risk classification."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_one_hot_encoder(drop: str | None = None) -> OneHotEncoder:
    """Create a dense one-hot encoder across scikit-learn versions."""
    try:
        return OneHotEncoder(
            drop=drop,
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            drop=drop,
            handle_unknown="ignore",
            sparse=False,
        )


def make_preprocessor(drop: str | None = None) -> ColumnTransformer:
    """Create a small tabular preprocessor for numeric and categorical data."""
    transformer_args = dict(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", make_one_hot_encoder(drop=drop), CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0,
    )
    try:
        return ColumnTransformer(
            **transformer_args,
            verbose_feature_names_out=False,
        )
    except TypeError:
        return ColumnTransformer(**transformer_args)


def build_logistic_regression() -> Pipeline:
    """Build a logistic regression pipeline with interpretable coefficients."""
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(drop="first")),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def build_naive_bayes() -> Pipeline:
    """Build a Gaussian Naive Bayes pipeline on one-hot encoded features."""
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(drop=None)),
            ("model", GaussianNB()),
        ]
    )


def get_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Return feature names after preprocessing."""
    preprocessor = pipeline.named_steps["preprocess"]
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        categorical = preprocessor.named_transformers_["categorical"]
        categorical_names = list(categorical.get_feature_names(CATEGORICAL_FEATURES))
        return NUMERIC_FEATURES + categorical_names


def logistic_coefficient_table(pipeline: Pipeline) -> list[dict[str, float | str]]:
    """Return logistic regression coefficients and odds multipliers."""
    feature_names = get_transformed_feature_names(pipeline)
    coefficients = pipeline.named_steps["model"].coef_[0]
    rows = []
    for feature_name, coefficient in zip(feature_names, coefficients):
        rows.append(
            {
                "feature": feature_name,
                "coefficient_log_odds": float(coefficient),
                "odds_multiplier": float(np.exp(coefficient)),
            }
        )
    rows.sort(key=lambda row: abs(row["coefficient_log_odds"]), reverse=True)
    return rows
