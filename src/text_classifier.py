"""
text_classifier.py — Text-based sentiment classification for Phase 2.

Models:
    German:  oliverguhr/german-sentiment-bert  (3-class: positiv/negativ/neutral)
    English: cardiffnlp/twitter-roberta-base-sentiment-latest (3-class: pos/neg/neu)

⚠️  The German model returns German label strings ("positiv", "negativ", "neutral").
    These are mapped to English equivalents before use in fusion.

Usage:
    clf = GermanSentimentClassifier()
    proba = clf.predict_proba("Das war ein tolles Gespräch!")
    # → {"positive": 0.87, "neutral": 0.09, "negative": 0.04}
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

# ── transformers import ────────────────────────────────────────────────────
try:
    from transformers import pipeline as hf_pipeline

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed — text classification unavailable.")


# ─────────────────────────────────────────────────────────────────────────────
# Label normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

# German BERT output labels → canonical 3-class English
_GERMAN_LABEL_MAP: dict[str, str] = {
    "positiv": "positive",
    "negativ": "negative",
    "neutral": "neutral",
    # lowercase variants
    "positive": "positive",
    "negative": "negative",
}

# cardiffnlp/twitter-roberta-base-sentiment-latest labels
_CARDIFF_LABEL_MAP: dict[str, str] = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    # Some versions use LABEL_0/1/2
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


def _normalize_label(label: str, label_map: dict[str, str]) -> str:
    return label_map.get(label, label_map.get(label.lower(), "neutral"))


def _scores_to_dict(scores: list[dict], label_map: dict[str, str]) -> dict[str, float]:
    """
    Convert HuggingFace return_all_scores output to a normalized probability dict.

    Returns
    -------
    dict with keys: 'positive', 'negative', 'neutral' (summing to ~1.0)
    """
    result = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for item in scores:
        key = _normalize_label(item["label"], label_map)
        result[key] = float(item["score"])
    # Re-normalize to ensure sum == 1 (floating point)
    total = sum(result.values())
    if total > 0:
        result = {k: v / total for k, v in result.items()}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GermanSentimentClassifier (primary — EMO-DB German transcripts)
# ─────────────────────────────────────────────────────────────────────────────

class GermanSentimentClassifier:
    """
    3-class sentiment classifier for German text using oliverguhr/german-sentiment-bert.

    Parameters
    ----------
    model_name : str — HuggingFace model ID
    device : str — "auto" | "cuda" | "cpu"
    """

    DEFAULT_MODEL = "oliverguhr/german-sentiment-bert"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
    ):
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. pip install transformers==4.41.2")

        self.model_name = model_name

        # Resolve device
        if device == "auto":
            try:
                import torch
                self._device = 0 if torch.cuda.is_available() else -1
            except ImportError:
                self._device = -1
        elif device == "cuda":
            self._device = 0
        else:
            self._device = -1

        logger.info("Loading text classifier: %s [device=%s]", model_name, self._device)
        self._pipeline = hf_pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True,
            device=self._device,
        )
        logger.info("Text classifier loaded.")

    def predict_proba(self, text: str) -> dict[str, float]:
        """
        Predict sentiment probability distribution for a single text.

        Parameters
        ----------
        text : str — German (or English) text

        Returns
        -------
        dict with keys 'positive', 'negative', 'neutral' (float, sum ≈ 1.0)
        """
        if not text or not text.strip():
            logger.warning("Empty text passed to predict_proba — returning uniform distribution.")
            return {"positive": 1 / 3, "negative": 1 / 3, "neutral": 1 / 3}

        try:
            scores = self._pipeline(text[:512])[0]  # truncate to 512 tokens
            return _scores_to_dict(scores, _GERMAN_LABEL_MAP)
        except Exception as exc:
            logger.error("Text classification failed: %s", exc)
            return {"positive": 1 / 3, "negative": 1 / 3, "neutral": 1 / 3}

    def predict(self, text: str) -> str:
        """Return the argmax class label."""
        proba = self.predict_proba(text)
        return max(proba, key=proba.get)

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[dict[str, float]]:
        """
        Batch prediction for efficiency.

        Parameters
        ----------
        texts : list[str]
        batch_size : int

        Returns
        -------
        list[dict[str, float]]
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            # Replace None / empty with placeholder
            clean_batch = [t[:512] if (t and t.strip()) else "unbekannt" for t in batch]

            try:
                batch_scores = self._pipeline(clean_batch)
                for j, scores in enumerate(batch_scores):
                    if texts[i + j] and texts[i + j].strip():
                        results.append(_scores_to_dict(scores, _GERMAN_LABEL_MAP))
                    else:
                        results.append({"positive": 1 / 3, "negative": 1 / 3, "neutral": 1 / 3})
            except Exception as exc:
                logger.error("Batch prediction failed at index %d: %s", i, exc)
                results.extend(
                    [{"positive": 1 / 3, "negative": 1 / 3, "neutral": 1 / 3}] * len(batch)
                )

            if (i // batch_size + 1) % 10 == 0:
                logger.info(
                    "Text classification progress: %d/%d", min(i + batch_size, len(texts)), len(texts)
                )

        return results

    def predict_manifest(
        self,
        manifest_df: pd.DataFrame,
        transcript_col: str = "transcript",
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """
        Predict sentiment for all transcripts in a manifest DataFrame.

        Returns manifest_df with columns:
            text_positive, text_negative, text_neutral, text_predicted_class
        """
        texts = manifest_df[transcript_col].fillna("").tolist()
        probas = self.predict_batch(texts, batch_size=batch_size)

        result_df = manifest_df.copy()
        result_df["text_positive"] = [p["positive"] for p in probas]
        result_df["text_negative"] = [p["negative"] for p in probas]
        result_df["text_neutral"] = [p["neutral"] for p in probas]
        result_df["text_predicted_class"] = [max(p, key=p.get) for p in probas]

        return result_df


# ─────────────────────────────────────────────────────────────────────────────
# EnglishSentimentClassifier (for RAVDESS transcripts)
# ─────────────────────────────────────────────────────────────────────────────

class EnglishSentimentClassifier(GermanSentimentClassifier):
    """
    3-class sentiment classifier for English text using
    cardiffnlp/twitter-roberta-base-sentiment-latest.

    Inherits from GermanSentimentClassifier — only the model and label map differ.
    """

    DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
    ):
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required.")

        self.model_name = model_name

        if device == "auto":
            try:
                import torch
                self._device = 0 if torch.cuda.is_available() else -1
            except ImportError:
                self._device = -1
        elif device == "cuda":
            self._device = 0
        else:
            self._device = -1

        logger.info("Loading English classifier: %s [device=%s]", model_name, self._device)
        self._pipeline = hf_pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True,
            device=self._device,
        )
        logger.info("English classifier loaded.")

    def predict_proba(self, text: str) -> dict[str, float]:
        if not text or not text.strip():
            return {"positive": 1 / 3, "negative": 1 / 3, "neutral": 1 / 3}
        try:
            scores = self._pipeline(text[:512])[0]
            return _scores_to_dict(scores, _CARDIFF_LABEL_MAP)
        except Exception as exc:
            logger.error("English classification failed: %s", exc)
            return {"positive": 1 / 3, "negative": 1 / 3, "neutral": 1 / 3}

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[dict[str, float]]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            clean_batch = [t[:512] if (t and t.strip()) else "unknown" for t in batch]
            try:
                batch_scores = self._pipeline(clean_batch)
                for j, scores in enumerate(batch_scores):
                    if texts[i + j] and texts[i + j].strip():
                        results.append(_scores_to_dict(scores, _CARDIFF_LABEL_MAP))
                    else:
                        results.append({"positive": 1 / 3, "negative": 1 / 3, "neutral": 1 / 3})
            except Exception as exc:
                logger.error("English batch prediction error at %d: %s", i, exc)
                results.extend(
                    [{"positive": 1 / 3, "negative": 1 / 3, "neutral": 1 / 3}] * len(batch)
                )
        return results





class GermanEmotionClassifier:
    """
    7-class emotion classifier for German speech transcripts.
    Pipeline: German text → Helsinki translation → English emotion classifier.
    Uses j-hartmann/emotion-english-distilroberta-base (7-class).
    Maps to 3-class: positive/negative/neutral via LabelMapper.
    """
    EMOTION_3CLASS_MAP = {
        'joy':      'positive',
        'surprise': 'positive',
        'anger':    'negative',
        'disgust':  'negative',
        'fear':     'negative',
        'sadness':  'negative',
        'neutral':  'neutral',
    }

    def __init__(self, device: str = 'auto'):
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required.")

        if device == 'auto':
            try:
                import torch
                self._device = 0 if torch.cuda.is_available() else -1
            except ImportError:
                self._device = -1
        else:
            self._device = 0 if device == 'cuda' else -1

        logger.info("Loading Helsinki DE→EN translator...")
        from transformers import MarianMTModel, MarianTokenizer
        self._tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        self._translator = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        if self._device == 0:
            self._translator = self._translator.cuda()

        logger.info("Loading emotion classifier: j-hartmann/emotion-english-distilroberta-base")
        self._emotion_clf = hf_pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
            device=self._device,
        )
        logger.info("GermanEmotionClassifier ready.")

    def predict_proba(self, text: str) -> dict[str, float]:
        """Returns dict with keys: positive, negative, neutral."""
        if not text or not text.strip():
            return {'positive': 1/3, 'negative': 1/3, 'neutral': 1/3}
        try:

            inputs = self._tokenizer(
            [text[:512]], return_tensors="pt", padding=True, truncation=True
        )
            if self._device == 0:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            translated_tokens = self._translator.generate(**inputs)
            translated = self._tokenizer.decode(
                translated_tokens[0], skip_special_tokens=True
            )



            
            raw = self._emotion_clf(translated[:512])
            # Unwrap nesting: pipeline returns [[{...},...]] or [{...},...]
            if isinstance(raw[0], list):
                scores = raw[0]   # batch wrapper present
            else:
                scores = raw      # already flat list of dicts

            collapsed = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            
            for item in scores:
                three = self.EMOTION_3CLASS_MAP.get(item['label'].lower(), 'neutral')
                collapsed[three] += float(item['score'])
            total = sum(collapsed.values())
            if total > 0:
                collapsed = {k: v / total for k, v in collapsed.items()}
            return collapsed
        except Exception as exc:
            logger.error("GermanEmotionClassifier failed: %s", exc)
            return {'positive': 1/3, 'negative': 1/3, 'neutral': 1/3}

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict[str, float]]:
        return [self.predict_proba(t) for t in texts]






# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_classifier(language: str = "de", device: str = "auto") -> GermanSentimentClassifier:
    """
    Factory: return the appropriate classifier for the given language.

    Parameters
    ----------
    language : str — "de" for German, "en" for English
    device : str

    Returns
    -------
    GermanSentimentClassifier or EnglishSentimentClassifier
    """
    if language == "de":
        return GermanSentimentClassifier(device=device)
    else:
        return EnglishSentimentClassifier(device=device)
