"""
evaluator.py — Evaluation utilities for Phase 1 and Phase 2 experiments.

Metrics:
    - Accuracy, Precision, Recall, Weighted F1, Macro F1
    - UAR (Unweighted Average Recall) — standard cross-corpus metric
    - Per-class F1 breakdown
    - Confusion matrix → seaborn heatmap PNG

Usage:
    from src.evaluator import Evaluator
    ev = Evaluator()
    ev.confusion_matrix_plot(y_true, y_pred, labels, title, "models/phase1/cm.png")
    df = ev.per_emotion_f1(y_true, y_pred, labels)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    Evaluation helper — confusion matrices, F1 breakdowns, UAR.
    """

    # ── Confusion Matrix ──────────────────────────────────────────────────

    def confusion_matrix_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[str],
        title: str = "Confusion Matrix",
        save_path: Optional[str | Path] = None,
        figsize: tuple[int, int] = (10, 8),
        normalize: bool = True,
    ) -> plt.Figure:
        """
        Produce a seaborn heatmap confusion matrix.

        Parameters
        ----------
        y_true, y_pred : array-like — true and predicted labels (string)
        labels : list[str] — ordered class names for axis ticks
        title : str
        save_path : str | Path | None — if given, save PNG to this path
        figsize : tuple[int, int]
        normalize : bool — show proportions (True) or raw counts (False)

        Returns
        -------
        matplotlib.figure.Figure
        """
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if normalize:
            cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt = ".2f"
            vmax = 1.0
        else:
            cm_plot = cm
            fmt = "d"
            vmax = None

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm_plot,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            vmin=0.0,
            vmax=vmax,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            logger.info("Confusion matrix saved → %s", save_path)

        return fig

    # ── Per-emotion F1 ────────────────────────────────────────────────────

    def per_emotion_f1(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[str],
    ) -> pd.DataFrame:
        """
        Compute per-class precision, recall, and F1 score.

        Parameters
        ----------
        y_true, y_pred : array-like
        labels : list[str] — class names

        Returns
        -------
        pd.DataFrame with index=labels and columns [precision, recall, f1, support]
        """
        precision = precision_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        support = cm.sum(axis=1)

        df = pd.DataFrame(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            },
            index=labels,
        )
        df.index.name = "emotion"
        return df

    # ── Classification report → DataFrame ────────────────────────────────

    def classification_report_to_df(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[str],
    ) -> pd.DataFrame:
        """
        Convert sklearn classification_report to a tidy pd.DataFrame.

        Parameters
        ----------
        y_true, y_pred : array-like
        labels : list[str]

        Returns
        -------
        pd.DataFrame
        """
        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        df = pd.DataFrame(report).T
        df.index.name = "class"
        return df

    # ── Summary metrics ───────────────────────────────────────────────────

    def summary(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """
        Compute all summary metrics at once.

        Returns
        -------
        dict with keys: accuracy, weighted_f1, macro_f1, uar
        """
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "uar": self.compute_uar(y_true, y_pred),
        }

    # ── UAR ───────────────────────────────────────────────────────────────

    @staticmethod
    def compute_uar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Unweighted Average Recall (UAR).

        UAR = (1/C) * Σ recall_c

        This is the standard evaluation metric for cross-corpus speech emotion
        recognition (Schuller et al.; Madanian et al.).
        """
        return float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # ── Cross-corpus evaluation report ───────────────────────────────────

    def cross_corpus_report(
        self,
        results: list[dict],
        save_path: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """
        Build a formatted comparison table for cross-corpus experiments.

        Parameters
        ----------
        results : list[dict]
            Each dict must have keys: system, train_corpus, test_corpus,
            accuracy, macro_f1, uar, notes (optional)
        save_path : str | Path | None

        Returns
        -------
        pd.DataFrame
        """
        df = pd.DataFrame(results)
        numeric_cols = [c for c in ["accuracy", "macro_f1", "weighted_f1", "uar"] if c in df.columns]
        df[numeric_cols] = df[numeric_cols].round(4)

        if save_path is not None:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            df.to_csv(str(save_path), index=False)
            logger.info("Cross-corpus report saved → %s", save_path)

        return df

    # ── Comparison plot ───────────────────────────────────────────────────

    def comparison_bar_plot(
        self,
        results_df: pd.DataFrame,
        x_col: str = "classifier",
        y_col: str = "weighted_f1",
        hue_col: str = "overlap_mode",
        title: str = "Classifier Performance Comparison",
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """
        Bar plot comparing classifiers across overlap modes.

        Parameters
        ----------
        results_df : pd.DataFrame
        x_col, y_col, hue_col : str — column names for axes and grouping
        title, save_path : str

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=results_df,
            x=x_col,
            y=y_col,
            hue=hue_col,
            palette="muted",
            ax=ax,
        )
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Classifier", fontsize=12)
        ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(title=hue_col.replace("_", " ").title())
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            logger.info("Comparison plot saved → %s", save_path)

        return fig
