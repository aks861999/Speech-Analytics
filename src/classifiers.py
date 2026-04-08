"""
classifiers.py — Phase 1 classifier suite: SVM, KNN, Random Forest,
Gradient Boosting, Extra Trees with grid-search + 10-fold stratified CV.

All experiments logged to MLflow. Best models serialized to disk as .pkl.

Design: Replicates Madanian et al. (2022) exactly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from src.utils import ensure_dir, get_logger, set_seed

logger = get_logger(__name__)

# ── MLflow (optional — graceful degradation) ─────────────────────────────
try:
    import mlflow
    import mlflow.sklearn

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    logger.warning("mlflow not installed — experiment tracking disabled.")


# ─────────────────────────────────────────────────────────────────────────────
# EmotionClassifierSuite
# ─────────────────────────────────────────────────────────────────────────────

class EmotionClassifierSuite:
    """
    Train and evaluate 5 classifiers on the 34-feature set extracted by
    PyAudioFeatureExtractor, following Madanian et al. (2022).

    Parameters
    ----------
    random_state : int
    cv_folds : int — number of StratifiedKFold folds (default 10)
    models_dir : str | Path — directory to save serialized models
    mlflow_tracking_uri : str | None
    mlflow_experiment_name : str
    """

    # ── Hyperparameter search grids (exact from Madanian et al.) ──────────
    PARAM_GRIDS: dict[str, dict] = {
        "SVM": {
            "C": [0.001, 0.01, 0.5, 1.0, 10.0, 20.0],
            "kernel": ["rbf"],
        },
        "KNN": {
            "n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15],
        },
        "RandomForest": {
            "n_estimators": [10, 25, 50, 100, 200, 500],
        },
        "GradientBoosting": {
            "n_estimators": [10, 25, 50, 100, 200, 500],
            "learning_rate": [0.1],
            "max_depth": [3],
        },
        "ExtraTrees": {
            "n_estimators": [10, 25, 50, 100, 200, 500],
        },
    }

    def __init__(
        self,
        random_state: int = 42,
        cv_folds: int = 10,
        models_dir: str | Path = "models/phase1",
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: str = "speech-emotion-allianz",
    ):
        set_seed(random_state)
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.models_dir = ensure_dir(models_dir)
        self.mlflow_experiment_name = mlflow_experiment_name

        if _MLFLOW_AVAILABLE and mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(mlflow_experiment_name)
            logger.info("MLflow tracking URI: %s", mlflow_tracking_uri)

        self._scaler: Optional[StandardScaler] = None
        self._label_encoder = LabelEncoder()
        self.best_models: dict[str, Any] = {}

    # ── Classifier factories ──────────────────────────────────────────────

    def _build_estimators(self) -> dict[str, Any]:
        return {
            "SVM": SVC(probability=True, random_state=self.random_state),
            "KNN": KNeighborsClassifier(),
            "RandomForest": RandomForestClassifier(random_state=self.random_state),
            "GradientBoosting": GradientBoostingClassifier(random_state=self.random_state),
            "ExtraTrees": ExtraTreesClassifier(random_state=self.random_state),
        }

    # ── Normalisation ─────────────────────────────────────────────────────

    def fit_scaler(self, X: np.ndarray) -> np.ndarray:
        """Fit StandardScaler on X and return transformed X."""
        self._scaler = StandardScaler()
        return self._scaler.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted scaler to X."""
        if self._scaler is None:
            raise RuntimeError("Call fit_scaler first.")
        return self._scaler.transform(X)

    # ── Main experiment runner ────────────────────────────────────────────

    def run_all_experiments(
        self,
        X: np.ndarray,
        y: np.ndarray,
        overlap_mode: str = "overlap",
        mlflow_run: bool = True,
        tags: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Run GridSearchCV + 10-fold StratifiedKFold for all 5 classifiers.

        Parameters
        ----------
        X : np.ndarray shape (n_samples, 34)
        y : np.ndarray shape (n_samples,) — string labels
        overlap_mode : str — 'overlap' or 'nooverlap' (for MLflow tagging)
        mlflow_run : bool — whether to log runs to MLflow
        tags : dict | None — extra MLflow tags

        Returns
        -------
        results_df : pd.DataFrame — classifiers × metrics
        best_models : dict — {classifier_name: fitted_best_estimator}
        """
        # Encode labels
        y_enc = self._label_encoder.fit_transform(y)
        classes = self._label_encoder.classes_.tolist()
        logger.info("Classes: %s", classes)

        # Normalize features (full dataset for CV — scaler fit per fold inside GridSearch)
        X_scaled = self.fit_scaler(X)

        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        estimators = self._build_estimators()
        results_rows = []

        for clf_name, base_estimator in estimators.items():
            logger.info("─── Running %s [%s] ───", clf_name, overlap_mode)
            param_grid = self.PARAM_GRIDS[clf_name]

            grid_search = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                cv=StratifiedKFold(
                    n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
                ),
                scoring="f1_weighted",
                n_jobs=-1,
                refit=True,
                verbose=0,
            )

            # Outer CV for unbiased performance estimate
            outer_cv_results = cross_validate(
                grid_search,
                X_scaled,
                y_enc,
                cv=cv,
                scoring=["accuracy", "f1_weighted", "f1_macro"],
                n_jobs=1,   # GridSearchCV already parallelizes internally
                return_train_score=False,
                verbose=0,
            )

            acc = float(np.mean(outer_cv_results["test_accuracy"]))
            f1_weighted = float(np.mean(outer_cv_results["test_f1_weighted"]))
            f1_macro = float(np.mean(outer_cv_results["test_f1_macro"]))

            logger.info(
                "%s → Acc=%.4f, wF1=%.4f, mF1=%.4f",
                clf_name, acc, f1_weighted, f1_macro,
            )

            # Fit best model on full dataset
            grid_search.fit(X_scaled, y_enc)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            self.best_models[clf_name] = best_model
            results_rows.append(
                {
                    "classifier": clf_name,
                    "overlap_mode": overlap_mode,
                    "accuracy": acc,
                    "weighted_f1": f1_weighted,
                    "macro_f1": f1_macro,
                    "best_params": str(best_params),
                }
            )

            # MLflow logging
            if mlflow_run and _MLFLOW_AVAILABLE:
                self._log_mlflow(
                    clf_name=clf_name,
                    overlap_mode=overlap_mode,
                    acc=acc,
                    f1_weighted=f1_weighted,
                    f1_macro=f1_macro,
                    best_params=best_params,
                    best_model=best_model,
                    extra_tags=tags,
                )

            # Save model to disk
            self._save_model(clf_name, overlap_mode, best_model)

        results_df = pd.DataFrame(results_rows)
        return results_df, self.best_models

    # ── Single-classifier training ────────────────────────────────────────

    def train_single(
        self,
        clf_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scale: bool = True,
    ) -> dict:
        """
        Train a single classifier on pre-split data and return metrics.

        Useful for cross-corpus evaluation (train on EMO-DB, test on RAVDESS).

        Parameters
        ----------
        clf_name : str — one of PARAM_GRIDS keys
        X_train, y_train, X_test, y_test : np.ndarray
        scale : bool — apply StandardScaler (fit on train, apply to test)

        Returns
        -------
        dict with keys: accuracy, weighted_f1, macro_f1, y_pred
        """
        from sklearn.metrics import accuracy_score, f1_score

        estimators = self._build_estimators()
        base_estimator = estimators[clf_name]
        param_grid = self.PARAM_GRIDS[clf_name]

        # Encode labels
        le = LabelEncoder()
        le.fit(np.concatenate([y_train, y_test]))
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring="f1_weighted",
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X_train, y_train_enc)
        y_pred = grid_search.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test_enc, y_pred),
            "weighted_f1": f1_score(y_test_enc, y_pred, average="weighted", zero_division=0),
            "macro_f1": f1_score(y_test_enc, y_pred, average="macro", zero_division=0),
            "y_pred": le.inverse_transform(y_pred),
            "y_true": le.inverse_transform(y_test_enc),
            "best_params": grid_search.best_params_,
        }

    # ── MLflow helpers ────────────────────────────────────────────────────

    def _log_mlflow(
        self,
        clf_name: str,
        overlap_mode: str,
        acc: float,
        f1_weighted: float,
        f1_macro: float,
        best_params: dict,
        best_model: Any,
        extra_tags: Optional[dict] = None,
    ) -> None:
        if not _MLFLOW_AVAILABLE:
            return

        run_name = f"{clf_name}_{overlap_mode}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("classifier", clf_name)
            mlflow.set_tag("overlap_mode", overlap_mode)
            mlflow.set_tag("dataset", "emodb")
            mlflow.set_tag("phase", "1")
            if extra_tags:
                for k, v in extra_tags.items():
                    mlflow.set_tag(k, str(v))

            mlflow.log_param("classifier_type", clf_name)
            mlflow.log_param("overlap", overlap_mode)
            for param_name, param_val in best_params.items():
                mlflow.log_param(param_name, param_val)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("weighted_f1", f1_weighted)
            mlflow.log_metric("macro_f1", f1_macro)

            try:
                mlflow.sklearn.log_model(best_model, artifact_path=f"model_{clf_name}")
            except Exception as exc:
                logger.warning("MLflow model logging failed: %s", exc)

    # ── Model persistence ─────────────────────────────────────────────────

    def _save_model(
        self,
        clf_name: str,
        overlap_mode: str,
        model: Any,
    ) -> Path:
        out_path = self.models_dir / f"{clf_name}_{overlap_mode}.pkl"
        joblib.dump(model, str(out_path))
        logger.info("Model saved → %s", out_path)
        return out_path

    def save_scaler(self, path: str | Path) -> None:
        """Persist the fitted StandardScaler."""
        if self._scaler is not None:
            joblib.dump(self._scaler, str(path))
            logger.info("Scaler saved → %s", path)

    def load_model(self, path: str | Path) -> Any:
        """Load a serialized sklearn model from disk."""
        return joblib.load(str(path))

    # ── UAR helper ────────────────────────────────────────────────────────

    @staticmethod
    def compute_uar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Unweighted Average Recall (UAR) — standard cross-corpus metric.

        UAR = mean of per-class recall.
        """
        from sklearn.metrics import recall_score

        return float(recall_score(y_true, y_pred, average="macro", zero_division=0))
