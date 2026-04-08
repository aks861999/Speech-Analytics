"""
label_mapper.py — Emotion label definitions and 7-class → 3-class collapsing.

Design Decision (Thesis §DD-1):
    For call-center deployment 7-class fine-grained emotion is impractical.
    Following Yurtay et al. (2024), we collapse to 3 actionable classes:
        Positive  → maintain
        Neutral   → observe
        Negative  → escalate / de-escalation protocol

Cross-corpus note (Thesis §DD-4):
    RAVDESS "calm" (code 02) has no direct EMO-DB equivalent.
    We map it to `boredom_calm` — the same class as EMO-DB "Boredom" — and
    document this design decision explicitly.
    RAVDESS "surprised" (code 08) is excluded from cross-corpus evaluation
    because EMO-DB has no equivalent class.
"""

from __future__ import annotations

from typing import Optional


class LabelMapper:
    """Central source-of-truth for all label mappings in the project."""

    # ── EMO-DB: emotion letter → English label ────────────────────────────
    # Character at index 5 (0-indexed) of the filename stem.
    # Example: 03a01Fa.wav  →  stem = "03a01Fa"  →  stem[5] = 'F'
    EMODB_7CLASS: dict[str, str] = {
        "W": "anger",
        "L": "boredom_calm",
        "E": "disgust",
        "A": "fear",
        "F": "happiness",
        "T": "sadness",
        "N": "neutral",
    }

    # ── EMO-DB: emotion letter → German name (for documentation) ─────────
    EMODB_GERMAN_NAME: dict[str, str] = {
        "W": "Ärger/Wut",
        "L": "Langeweile",
        "E": "Ekel",
        "A": "Angst",
        "F": "Freude",
        "T": "Trauer",
        "N": "Neutral",
    }

    # ── RAVDESS: emotion code (string) → English label ────────────────────
    # Modality 03, channel 01 only.
    # Code 08 (surprised) is excluded from cross-corpus evaluation.
    RAVDESS_7CLASS: dict[str, str] = {
        "01": "neutral",
        "02": "boredom_calm",   # "calm" mapped to boredom_calm for cross-corpus alignment
        "03": "happiness",
        "04": "sadness",
        "05": "anger",
        "06": "fear",
        "07": "disgust",
        "08": "surprised",      # EXCLUDED from cross-corpus eval — no EMO-DB equivalent
    }

    # ── Shared 7-class label space (intersection) ─────────────────────────
    HARMONIZED_LABELS: list[str] = [
        "anger",
        "boredom_calm",
        "disgust",
        "fear",
        "happiness",
        "neutral",
        "sadness",
    ]

    # ── 3-class collapse map (Yurtay et al. 2024) ─────────────────────────
    THREE_CLASS_MAP: dict[str, str] = {
        "happiness":    "positive",
        "anger":        "negative",
        "disgust":      "negative",
        "fear":         "negative",
        "boredom_calm": "negative",
        "sadness":      "negative",
        "neutral":      "neutral",
    }

    # ── Numeric encoding for 3-class ─────────────────────────────────────
    THREE_CLASS_INT: dict[str, int] = {
        "positive": 2,
        "neutral":  1,
        "negative": 0,
    }

    # ─────────────────────────────────────────────────────────────────────
    # Methods
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def emodb_label_from_filename(cls, filename_stem: str) -> Optional[str]:
        """
        Extract the 7-class English emotion label from an EMO-DB filename stem.

        The emotion code is at character position 5 (0-indexed) of the stem.

        Parameters
        ----------
        filename_stem : str
            Filename without extension, e.g. "03a01Fa"

        Returns
        -------
        str | None
            English emotion label or None if code is unrecognised.
        """
        if len(filename_stem) < 6:
            return None
        code = filename_stem[5]  # position 5, not 6 — critical off-by-one
        return cls.EMODB_7CLASS.get(code)

    @classmethod
    def ravdess_label_from_filename(cls, filename_stem: str) -> Optional[str]:
        """
        Extract the 7-class English emotion label from a RAVDESS filename stem.

        RAVDESS format: {modality}-{channel}-{emotion}-{intensity}-{stmt}-{rep}-{actor}
        Emotion code is the 3rd field (index 2) when splitting by '-'.

        Parameters
        ----------
        filename_stem : str
            Filename without extension, e.g. "03-01-06-01-02-01-12"

        Returns
        -------
        str | None
            English emotion label or None (including if surprised/08).
        """
        parts = filename_stem.split("-")
        if len(parts) < 3:
            return None
        code = parts[2]
        label = cls.RAVDESS_7CLASS.get(code)
        # Return None for "surprised" — excluded from cross-corpus eval
        if label == "surprised":
            return None
        return label

    @classmethod
    def collapse_to_3class(cls, emotion_label: str) -> Optional[str]:
        """
        Collapse a 7-class harmonized emotion label to the 3-class scheme.

        Parameters
        ----------
        emotion_label : str
            One of the HARMONIZED_LABELS or 'surprised'.

        Returns
        -------
        str | None
            'positive', 'negative', or 'neutral', or None if unmapped.
        """
        return cls.THREE_CLASS_MAP.get(emotion_label)

    @classmethod
    def is_valid_ravdess_speech_file(cls, filename_stem: str) -> bool:
        """
        Return True only for audio-only speech files:
            modality == '03'  (audio-only)
            channel  == '01'  (speech, not song)

        Parameters
        ----------
        filename_stem : str
            Filename stem, e.g. "03-01-06-01-02-01-12"
        """
        parts = filename_stem.split("-")
        if len(parts) < 2:
            return False
        return parts[0] == "03" and parts[1] == "01"

    @classmethod
    def get_3class_labels(cls) -> list[str]:
        """Return ordered 3-class label list."""
        return ["negative", "neutral", "positive"]

    @classmethod
    def get_harmonized_labels(cls) -> list[str]:
        """Return the 7-class harmonized label list."""
        return cls.HARMONIZED_LABELS.copy()
