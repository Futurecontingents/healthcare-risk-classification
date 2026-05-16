"""Matplotlib visualizations for the healthcare risk project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "blue": "#246B8F",
    "teal": "#2A9D8F",
    "coral": "#E76F51",
    "yellow": "#E9C46A",
    "gray": "#4A5568",
    "light_gray": "#E5E7EB",
}


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_figure(fig, filename: str, output_dirs: list[str | Path]) -> list[Path]:
    """Save one figure to one or more output directories."""
    saved_paths = []
    for output_dir in output_dirs:
        directory = ensure_directory(output_dir)
        path = directory / filename
        fig.savefig(path, dpi=180, bbox_inches="tight")
        saved_paths.append(path)
    return saved_paths


def plot_charges_distribution(df, threshold: float, output_dirs: list[str | Path]):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(df["charges"], bins=35, color=COLORS["blue"], edgecolor="white")
    ax.axvline(
        threshold,
        color=COLORS["coral"],
        linewidth=2,
        label=f"75th percentile = ${threshold:,.0f}",
    )
    ax.set_title("Distribution of Insurance Charges")
    ax.set_xlabel("Charges")
    ax.set_ylabel("Number of records")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    paths = save_figure(fig, "charges_distribution.png", output_dirs)
    return fig, paths


def plot_log_charges_distribution(df, output_dirs: list[str | Path]):
    """Plot the distribution after a log transform of charges."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(np.log(df["charges"]), bins=35, color=COLORS["teal"], edgecolor="white")
    ax.set_title("Distribution of Log Charges")
    ax.set_xlabel("log(charges)")
    ax.set_ylabel("Number of records")
    ax.grid(axis="y", alpha=0.25)
    paths = save_figure(fig, "log_charges_distribution.png", output_dirs)
    return fig, paths


def plot_class_balance(df, output_dirs: list[str | Path]):
    counts = df["high_cost"].value_counts().sort_index()
    labels = ["Not high cost", "High cost"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(labels, counts.values, color=[COLORS["teal"], COLORS["coral"]])
    ax.set_title("High-Cost Class Balance")
    ax.set_ylabel("Number of records")
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
        )
    paths = save_figure(fig, "high_cost_class_balance.png", output_dirs)
    return fig, paths


def plot_rate_by_group(
    df,
    group_column: str,
    title: str,
    filename: str,
    output_dirs: list[str | Path],
    order: list[str] | None = None,
):
    rates = df.groupby(group_column)["high_cost"].mean()
    if order is not None:
        rates = rates.reindex(order)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(rates.index.astype(str), rates.values, color=COLORS["blue"])
    ax.set_title(title)
    ax.set_xlabel(group_column.replace("_", " ").title())
    ax.set_ylabel("High-cost rate")
    ax.set_ylim(0, max(0.55, float(rates.max()) + 0.08))
    ax.grid(axis="y", alpha=0.25)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.0%}",
            ha="center",
            va="bottom",
        )
    paths = save_figure(fig, filename, output_dirs)
    return fig, paths


def plot_charges_scatter(
    df,
    x_column: str,
    title: str,
    filename: str,
    output_dirs: list[str | Path],
):
    colors = np.where(df["high_cost"] == 1, COLORS["coral"], COLORS["teal"])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(df[x_column], df["charges"], c=colors, alpha=0.65, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel(x_column.upper() if x_column == "bmi" else x_column.title())
    ax.set_ylabel("Charges")
    ax.grid(alpha=0.25)
    paths = save_figure(fig, filename, output_dirs)
    return fig, paths


def plot_correlation_matrix(df, columns: list[str], output_dirs: list[str | Path]):
    """Plot a small correlation matrix using matplotlib only."""
    corr = df[columns].corr()

    fig, ax = plt.subplots(figsize=(7, 5.8))
    image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix for Numerical Variables")
    ax.set_xticks(range(len(columns)), labels=columns, rotation=35, ha="right")
    ax.set_yticks(range(len(columns)), labels=columns)

    for row in range(len(columns)):
        for col in range(len(columns)):
            value = corr.values[row, col]
            color = "white" if abs(value) > 0.55 else "black"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Pearson correlation")
    fig.tight_layout()
    paths = save_figure(fig, "correlation_matrix.png", output_dirs)
    return fig, paths, corr


def plot_confusion_matrix_figure(
    matrix,
    title: str,
    filename: str,
    output_dirs: list[str | Path],
):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > matrix.max() / 2 else "black"
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    paths = save_figure(fig, filename, output_dirs)
    return fig, paths
