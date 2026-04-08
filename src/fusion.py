"""
fusion.py — Phase 2 hybrid fusion of acoustic and text sentiment branches.

Two methods:
1. Weighted Average (Yurtay et al. validated): text × 0.75 + acoustic × 0.25
2. Logistic Regression meta-learner trained on concatenated 6-dim probabilities

Design Decision (Thesis §DD-3):
    Yurtay et al.'s 75/25 split is validated on real call center data.
    Our ablation study over (0.6, 0.75, 0.90) provides empirical evidence
    for the optimal weight in the German-language Allianz context.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

# ── sklearn for LR meta-learner ────────────────────────────────────────────
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# SentimentFusion
# ─────────────────────────────────────────────────────────────────────────────

class SentimentFusion:
    """
    Fuses text and acoustic 3-class sentiment probability vectors.

    Both input dicts must have keys: 'positive', 'negative', 'neutral'.

    Parameters
    ----------
    text_weight : float — weight for text branch (default 0.75)
    """

    LABELS: list[str] = ["positive", "negative", "neutral"]

    def __init__(self, text_weight: float = 0.75):
        if not 0.0 <= text_weight <= 1.0:
            raise ValueError("text_weight must be in [0, 1]")
        self.text_weight = text_weight
        self.acoustic_weight = 1.0 - text_weight
        self._lr_model: Optional[LogisticRegression] = None
        self._lr_label_encoder = LabelEncoder() if _SKLEARN_AVAILABLE else None

    # ── Method 1: Weighted average ────────────────────────────────────────

    def weighted_fusion(
        self,
        text_proba: dict[str, float],
        acoustic_proba: dict[str, float],
        text_weight: Optional[float] = None,
    ) -> tuple[dict[str, float], str]:
        """
        Fuse text and acoustic probability vectors with a fixed weight.

        Parameters
        ----------
        text_proba : dict — {'positive': p, 'negative': q, 'neutral': r}
        acoustic_proba : dict — same structure
        text_weight : float | None — override default if given

        Returns
        -------
        fused : dict[str, float] — fused probabilities (sum ≈ 1.0)
        predicted_class : str — argmax label
        """
        w_text = text_weight if text_weight is not None else self.text_weight
        w_acoustic = 1.0 - w_text

        fused: dict[str, float] = {}
        for label in self.LABELS:
            t = text_proba.get(label, 0.0)
            a = acoustic_proba.get(label, 0.0)
            fused[label] = w_text * t + w_acoustic * a

        # Re-normalize
        total = sum(fused.values())
        if total > 0:
            fused = {k: v / total for k, v in fused.items()}

        predicted_class = max(fused, key=fused.get)
        return fused, predicted_class

    def fuse_batch(
        self,
        text_probas: list[dict[str, float]],
        acoustic_probas: list[dict[str, float]],
        text_weight: Optional[float] = None,
    ) -> tuple[list[dict[str, float]], list[str]]:
        """
        Fuse lists of probability dicts.

        Returns
        -------
        fused_probas : list[dict]
        predicted_classes : list[str]
        """
        results = [
            self.weighted_fusion(t, a, text_weight)
            for t, a in zip(text_probas, acoustic_probas)
        ]
        fused_probas = [r[0] for r in results]
        predicted_classes = [r[1] for r in results]
        return fused_probas, predicted_classes

    # ── Method 2: Logistic Regression meta-learner ────────────────────────

    def logistic_regression_fusion(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        C: float = 1.0,
        max_iter: int = 1000,
    ) -> np.ndarray:
        """
        Train a logistic regression meta-learner on concatenated probabilities.

        Input features: [text_positive, text_negative, text_neutral,
                          acoustic_positive, acoustic_negative, acoustic_neutral]
        → shape (n_samples, 6)

        Parameters
        ----------
        X_train : np.ndarray shape (n_train, 6)
        y_train : np.ndarray shape (n_train,) — string labels
        X_test : np.ndarray shape (n_test, 6)
        C : float — inverse regularisation strength
        max_iter : int

        Returns
        -------
        predictions : np.ndarray shape (n_test,) — string labels
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for LR meta-learner.")

        assert X_train.shape[1] == 6, f"Expected 6 features, got {X_train.shape[1]}"

        self._lr_label_encoder.fit(y_train)
        y_train_enc = self._lr_label_encoder.transform(y_train)

        self._lr_model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42,
        )
        self._lr_model.fit(X_train, y_train_enc)

        y_pred_enc = self._lr_model.predict(X_test)
        predictions = self._lr_label_encoder.inverse_transform(y_pred_enc)
        logger.info("LR meta-learner predictions: %d samples", len(predictions))
        return predictions

    @staticmethod
    def build_feature_matrix(
        text_probas: list[dict[str, float]],
        acoustic_probas: list[dict[str, float]],
        labels: list[str] = ("positive", "negative", "neutral"),
    ) -> np.ndarray:
        """
        Concatenate text and acoustic probability dicts into a (n, 6) matrix.

        Row = [text_positive, text_negative, text_neutral,
               acoustic_positive, acoustic_negative, acoustic_neutral]

        Parameters
        ----------
        text_probas, acoustic_probas : list[dict]
        labels : tuple[str]

        Returns
        -------
        np.ndarray shape (n_samples, 6)
        """
        rows = []
        for t, a in zip(text_probas, acoustic_probas):
            row = [t.get(l, 0.0) for l in labels] + [a.get(l, 0.0) for l in labels]
            rows.append(row)
        return np.array(rows, dtype=np.float32)

    # ── Acoustic probability from 7-class to 3-class ──────────────────────

    @staticmethod
    def collapse_acoustic_proba(
        proba_7class: dict[str, float],
    ) -> dict[str, float]:
        """
        Collapse a 7-class acoustic probability dict to 3-class (Yurtay mapping).

        7-class → 3-class:
            positive  ← happiness
            negative  ← anger, disgust, fear, boredom_calm, sadness
            neutral   ← neutral

        Parameters
        ----------
        proba_7class : dict — keys are 7-class emotion labels, values are probabilities

        Returns
        -------
        dict with keys: 'positive', 'negative', 'neutral'
        """
        from src.label_mapper import LabelMapper

        collapsed = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for emotion, prob in proba_7class.items():
            three_class = LabelMapper.THREE_CLASS_MAP.get(emotion)
            if three_class:
                collapsed[three_class] += prob

        total = sum(collapsed.values())
        if total > 0:
            collapsed = {k: v / total for k, v in collapsed.items()}

        return collapsed

    # ── Ablation study ────────────────────────────────────────────────────

    def ablation_study(
        self,
        text_probas: list[dict[str, float]],
        acoustic_probas: list[dict[str, float]],
        y_true: np.ndarray,
        text_weights: list[float] = (0.60, 0.75, 0.90),
    ) -> pd.DataFrame:
        """
        Evaluate fusion performance across multiple text weight values.

        Parameters
        ----------
        text_probas : list[dict]
        acoustic_probas : list[dict]
        y_true : np.ndarray — true labels (strings)
        text_weights : list[float]

        Returns
        -------
        pd.DataFrame — comparison table (weight × metrics)
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for ablation study.")

        rows = []
        for w in text_weights:
            _, y_pred = self.fuse_batch(text_probas, acoustic_probas, text_weight=w)
            acc = accuracy_score(y_true, y_pred)
            wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            rows.append(
                {
                    "text_weight": w,
                    "acoustic_weight": round(1 - w, 2),
                    "accuracy": round(acc, 4),
                    "weighted_f1": round(wf1, 4),
                    "macro_f1": round(mf1, 4),
                }
            )
            logger.info(
                "Ablation w=%.2f → Acc=%.4f, wF1=%.4f, mF1=%.4f", w, acc, wf1, mf1
            )

        return pd.DataFrame(rows)

    # ── Sentiment score for dashboard timeline ────────────────────────────

    @staticmethod
    def compute_sentiment_score(proba: dict[str, float]) -> float:
        """
        Map a 3-class probability dict to a continuous sentiment score.

        sentiment_score = P(positive) - P(negative)
        Range: [-1.0, +1.0]

        Colour coding in the Gradio dashboard:
            > +0.3  → green  (positive)
            ≥ -0.3  → gray   (neutral)
            < -0.3  → red    (negative)

        Parameters
        ----------
        proba : dict — {'positive': p, 'negative': q, 'neutral': r}

        Returns
        -------
        float in [-1, 1]
        """
        return float(proba.get("positive", 0.0) - proba.get("negative", 0.0))
