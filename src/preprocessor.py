"""
preprocessor.py — Audio preprocessing: resampling, silence removal (VAD),
and data augmentation using audiomentations (Python 3.12 compatible).

Note: nlpaug is NOT used — it is incompatible with Python 3.12.
      audiomentations provides equivalent functionality.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

# ── audiomentations import (graceful fallback if not installed) ────────────
try:
    from audiomentations import Compose, AddGaussianNoise, TimeStretch

    _AUGMENTATION_AVAILABLE = True
except ImportError:
    logger.warning(
        "audiomentations not installed — augmentation will be skipped. "
        "Install with: pip install audiomentations==0.36.0"
    )
    _AUGMENTATION_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# AudioPreprocessor
# ─────────────────────────────────────────────────────────────────────────────

class AudioPreprocessor:
    """
    Handles all audio preprocessing steps:
        1. Resampling to 16kHz mono (already in data_loader; exposed here too)
        2. Simple energy-based VAD to trim leading/trailing silence
        3. Data augmentation via audiomentations
    """

    def __init__(
        self,
        target_sr: int = 16000,
        aug_gaussian_noise_p: float = 0.5,
        aug_gaussian_min_amp: float = 0.001,
        aug_gaussian_max_amp: float = 0.015,
        aug_time_stretch_p: float = 0.3,
        aug_time_stretch_min: float = 0.9,
        aug_time_stretch_max: float = 1.1,
    ):
        self.target_sr = target_sr

        # Build augmentation pipeline
        if _AUGMENTATION_AVAILABLE:
            self._augment = Compose(
                [
                    AddGaussianNoise(
                        min_amplitude=aug_gaussian_min_amp,
                        max_amplitude=aug_gaussian_max_amp,
                        p=aug_gaussian_noise_p,
                    ),
                    TimeStretch(
                        min_rate=aug_time_stretch_min,
                        max_rate=aug_time_stretch_max,
                        p=aug_time_stretch_p,
                    ),
                ]
            )
        else:
            self._augment = None

    # ── Core helpers ──────────────────────────────────────────────────────

    def load(self, filepath: str | Path) -> np.ndarray:
        """Load audio file as mono 16kHz numpy array."""
        audio, _ = librosa.load(str(filepath), sr=self.target_sr, mono=True)
        return audio.astype(np.float32)

    def save(self, audio: np.ndarray, filepath: str | Path) -> None:
        """Save float32 numpy array as 16-bit PCM WAV."""
        ensure_dir(Path(filepath).parent)
        sf.write(str(filepath), audio, self.target_sr, subtype="PCM_16")

    # ── VAD: simple energy-based silence trimming ─────────────────────────

    def trim_silence(
        self,
        audio: np.ndarray,
        top_db: float = 20.0,
    ) -> np.ndarray:
        """
        Trim leading and trailing silence using librosa's energy-based VAD.

        Parameters
        ----------
        audio : np.ndarray — float32 mono signal
        top_db : float — threshold below peak (dB); higher = more aggressive

        Returns
        -------
        np.ndarray — trimmed signal (may be same length if no silence found)
        """
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed

    # ── Augmentation ──────────────────────────────────────────────────────

    def augment(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply the audiomentations augmentation pipeline to a single clip.

        Parameters
        ----------
        audio : np.ndarray — float32 mono at self.target_sr

        Returns
        -------
        np.ndarray — augmented audio (same shape)
        """
        if self._augment is None:
            logger.warning("Augmentation skipped — audiomentations not available.")
            return audio
        augmented = self._augment(samples=audio, sample_rate=self.target_sr)
        return augmented.astype(np.float32)

    def augment_dataset(
        self,
        manifest_df,
        output_dir: str | Path,
        source_col: str = "processed_filepath",
    ):
        """
        Augment every file in manifest_df and save to output_dir with '_aug' suffix.

        Parameters
        ----------
        manifest_df : pd.DataFrame with 'processed_filepath' column
        output_dir : str | Path — where to write augmented files
        source_col : str — column name containing source file paths

        Returns
        -------
        list[str] — paths of generated augmented files
        """
        output_dir = ensure_dir(output_dir)
        aug_paths: list[str] = []

        for _, row in manifest_df.iterrows():
            src = row.get(source_col)
            if src is None or not Path(src).exists():
                logger.warning("Source file missing, skipping augmentation: %s", src)
                continue

            stem = Path(src).stem
            dst = output_dir / f"{stem}_aug.wav"

            if dst.exists():
                aug_paths.append(str(dst))
                continue

            try:
                audio = self.load(src)
                aug_audio = self.augment(audio)
                self.save(aug_audio, dst)
                aug_paths.append(str(dst))
            except Exception as exc:
                logger.error("Augmentation failed for %s: %s", src, exc)

        logger.info(
            "Augmented %d files → %s", len(aug_paths), output_dir
        )
        return aug_paths

    # ── Convenience: full preprocessing pipeline for one file ────────────

    def preprocess_file(
        self,
        filepath: str | Path,
        output_path: Optional[str | Path] = None,
        trim: bool = True,
    ) -> np.ndarray:
        """
        Load → resample → optionally trim silence → optionally save.

        Parameters
        ----------
        filepath : str | Path
        output_path : str | Path | None — if given, saves processed audio
        trim : bool — apply VAD silence trimming

        Returns
        -------
        np.ndarray — processed float32 mono signal
        """
        audio = self.load(filepath)
        if trim:
            audio = self.trim_silence(audio)
        if output_path is not None:
            self.save(audio, output_path)
        return audio
