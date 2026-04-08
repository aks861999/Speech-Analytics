"""
feature_extractor.py — PyAudioAnalysis 34-feature extraction wrapper.

Features extracted (34 total, as in Madanian et al. 2022):
    13 MFCCs
     1 Zero Crossing Rate
     1 Short-term Energy
     1 Energy Entropy
     5 Spectral (Centroid, Spread, Entropy, Flux, Rolloff)
    12 Chroma
    ─────
    33 short-term features + 1 chroma = 34

Both overlap and non-overlap extraction modes are supported.

Design note: PyAudioAnalysis must be installed from GitHub for Python 3.12
    pip install git+https://github.com/tyiannak/pyAudioAnalysis.git
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

# ── pyAudioAnalysis import (graceful degradation) ─────────────────────────
try:
    from pyAudioAnalysis import audioFeatureExtraction as afe
    from pyAudioAnalysis import audioTrainTest as att
    from pyAudioAnalysis import MidTermFeatures as mtf

    _PAA_AVAILABLE = True
    logger.info("pyAudioAnalysis loaded successfully.")
except ImportError:
    _PAA_AVAILABLE = False
    logger.warning(
        "pyAudioAnalysis not installed. Install from GitHub:\n"
        "  pip install git+https://github.com/tyiannak/pyAudioAnalysis.git\n"
        "Feature extraction will fall back to librosa-based implementation."
    )

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Librosa-based 34-feature fallback
# ─────────────────────────────────────────────────────────────────────────────

def _librosa_extract_34(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extract the same 34 features using librosa when pyAudioAnalysis is
    unavailable. This ensures the codebase is functional even during
    environment setup.

    Returns a 1D array of shape (34,) — mean over all frames.
    """
    if not _LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for fallback feature extraction.")

    features = []

    # 1. ZCR (1)
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features.append(float(np.mean(zcr)))

    # 2. Short-term energy (1)
    energy = np.array([
        np.sum(audio[i:i + 512] ** 2)
        for i in range(0, len(audio), 512)
    ])
    features.append(float(np.mean(energy)))

    # 3. Energy entropy (1) — entropy of normalised energy sub-frame distribution
    if energy.sum() > 0:
        p = energy / energy.sum()
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log2(p)))
    else:
        entropy = 0.0
    features.append(entropy)

    # 4. Spectral Centroid (1)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features.append(float(np.mean(centroid)))

    # 5. Spectral Spread (1) — spectral bandwidth
    spread = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    features.append(float(np.mean(spread)))

    # 6. Spectral Entropy (1) — entropy of power spectrum
    stft = np.abs(librosa.stft(audio)) ** 2
    power = stft.sum(axis=0)
    if power.sum() > 0:
        p = power / power.sum()
        p = p[p > 0]
        sp_entropy = float(-np.sum(p * np.log2(p)))
    else:
        sp_entropy = 0.0
    features.append(sp_entropy)

    # 7. Spectral Flux (1)
    flux = np.mean(np.diff(stft, axis=1) ** 2)
    features.append(float(flux))

    # 8. Spectral Rolloff (1)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    features.append(float(np.mean(rolloff)))

    # 9. MFCCs 1–13 (13)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features.extend(float(np.mean(m)) for m in mfccs)

    # 10. Chroma 1–12 (12)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12)
    features.extend(float(np.mean(c)) for c in chroma)

    assert len(features) == 34, f"Expected 34 features, got {len(features)}"
    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class PyAudioFeatureExtractor:
    """
    34-feature extractor wrapping pyAudioAnalysis (primary) with a librosa
    fallback for environments where pyAudioAnalysis is not yet installed.

    Parameters
    ----------
    target_sr : int   — expected sample rate of input files (16000)
    use_fallback : bool — force librosa fallback even if pyAudioAnalysis available
    """

    FEATURE_NAMES: list[str] = (
        ["ZCR", "Energy", "EnergyEntropy"]
        + ["SpectralCentroid", "SpectralSpread", "SpectralEntropy",
           "SpectralFlux", "SpectralRolloff"]
        + [f"MFCC_{i+1}" for i in range(13)]
        + [f"Chroma_{i+1}" for i in range(12)]
    )

    def __init__(
        self,
        target_sr: int = 16000,
        use_fallback: bool = False,
    ):
        self.target_sr = target_sr
        self._use_fallback = use_fallback or not _PAA_AVAILABLE

        if self._use_fallback:
            logger.info(
                "FeatureExtractor: using librosa fallback (34 features via librosa)."
            )
        else:
            logger.info(
                "FeatureExtractor: using pyAudioAnalysis."
            )

    # ── pyAudioAnalysis extraction ────────────────────────────────────────

    def _extract_paa(
        self,
        audio: np.ndarray,
        st_win: float,
        st_step: float,
    ) -> np.ndarray:
        """
        Extract mid-term features via pyAudioAnalysis and return mean over time.

        pyAudioAnalysis stFeatureExtraction returns a matrix (n_features × n_frames).
        We reduce to a single vector by taking the mean over frames.
        """
        # pyAudioAnalysis expects integer array at the target sample rate
        signal = audio.astype(float)
        n_win = int(st_win * self.target_sr)
        n_step = int(st_step * self.target_sr)

        # stFeatureExtraction returns (34 × n_frames) numpy array
        try:
            feat_matrix, _ = afe.stFeatureExtraction(signal, self.target_sr, n_win, n_step)
        except Exception:
            # Newer API
            feat_matrix = afe.stFeatureExtraction(signal, self.target_sr, n_win, n_step)

        # Take mean across time frames → shape (n_features,)
        if feat_matrix.ndim == 2:
            feature_vector = np.mean(feat_matrix, axis=1)
        else:
            feature_vector = feat_matrix

        return feature_vector[:34].astype(np.float32)

    # ── librosa fallback extraction ───────────────────────────────────────

    def _extract_librosa(self, audio: np.ndarray) -> np.ndarray:
        return _librosa_extract_34(audio, sr=self.target_sr)

    # ── Public extraction methods ─────────────────────────────────────────

    def extract_file(
        self,
        filepath: str | Path,
        overlap: bool = True,
    ) -> np.ndarray:
        """
        Extract 34 features from a single audio file.

        Parameters
        ----------
        filepath : str | Path
        overlap : bool
            True  → st_win=0.05s, st_step=0.025s (50% overlap)
            False → st_win=0.05s, st_step=0.05s  (no overlap)

        Returns
        -------
        np.ndarray shape (34,)
        """
        import librosa as _librosa
        audio, _ = _librosa.load(str(filepath), sr=self.target_sr, mono=True)

        if self._use_fallback:
            return self._extract_librosa(audio)

        st_win = 0.05
        st_step = 0.025 if overlap else 0.05
        return self._extract_paa(audio, st_win, st_step)

    def extract_from_array(
        self,
        audio: np.ndarray,
        overlap: bool = True,
    ) -> np.ndarray:
        """
        Extract 34 features from a pre-loaded numpy array.

        Parameters
        ----------
        audio : np.ndarray — float32 mono at target_sr
        overlap : bool

        Returns
        -------
        np.ndarray shape (34,)
        """
        if self._use_fallback:
            return self._extract_librosa(audio)

        st_win = 0.05
        st_step = 0.025 if overlap else 0.05
        return self._extract_paa(audio, st_win, st_step)

    def extract_manifest(
        self,
        manifest_df: pd.DataFrame,
        filepath_col: str = "processed_filepath",
        label_col: str = "emotion_label_en",
        overlap: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Extract features for every file in manifest_df.

        Parameters
        ----------
        manifest_df : pd.DataFrame
        filepath_col : str
        label_col : str
        overlap : bool

        Returns
        -------
        X : np.ndarray shape (n_samples, 34)
        y : np.ndarray shape (n_samples,) — string labels
        filenames : list[str]
        """
        X_rows, y_rows, names = [], [], []

        for _, row in manifest_df.iterrows():
            fpath = row.get(filepath_col)
            label = row.get(label_col)

            if fpath is None or not Path(str(fpath)).exists():
                logger.warning("Skipping missing file: %s", fpath)
                continue
            if label is None:
                logger.warning("Skipping file with no label: %s", fpath)
                continue

            try:
                feats = self.extract_file(fpath, overlap=overlap)
                X_rows.append(feats)
                y_rows.append(str(label))
                names.append(Path(str(fpath)).name)
            except Exception as exc:
                logger.error("Feature extraction failed for %s: %s", fpath, exc)

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows)
        logger.info(
            "Extracted features: X=%s, overlap=%s", X.shape, overlap
        )
        return X, y, names

    def extract_and_save_arff(
        self,
        manifest_df: pd.DataFrame,
        output_arff_path: str | Path,
        filepath_col: str = "processed_filepath",
        label_col: str = "emotion_label_en",
        overlap: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract features and save to ARFF format for compatibility with
        pyAudioAnalysis / Weka pipelines.

        Parameters
        ----------
        manifest_df : pd.DataFrame
        output_arff_path : str | Path
        filepath_col : str
        label_col : str
        overlap : bool

        Returns
        -------
        X : np.ndarray shape (n_samples, 34)
        y : np.ndarray shape (n_samples,) — string labels
        """
        X, y, _ = self.extract_manifest(
            manifest_df,
            filepath_col=filepath_col,
            label_col=label_col,
            overlap=overlap,
        )

        output_arff_path = Path(output_arff_path)
        ensure_dir(output_arff_path.parent)

        with open(str(output_arff_path), "w", encoding="utf-8") as f:
            f.write("@RELATION emotion_features\n\n")
            for feat_name in self.FEATURE_NAMES:
                f.write(f"@ATTRIBUTE {feat_name} NUMERIC\n")
            unique_labels = sorted(set(y))
            f.write(f"@ATTRIBUTE class {{{','.join(unique_labels)}}}\n\n")
            f.write("@DATA\n")
            for feats, label in zip(X, y):
                values = ",".join(f"{v:.6f}" for v in feats)
                f.write(f"{values},{label}\n")

        logger.info("ARFF saved → %s (%d samples)", output_arff_path, len(X))
        return X, y

    @staticmethod
    def load_arff(arff_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """
        Load features and labels from an ARFF file.

        Returns
        -------
        X : np.ndarray shape (n_samples, n_features)
        y : np.ndarray shape (n_samples,) — string labels
        """
        import re

        arff_path = Path(arff_path)
        X_rows, y_rows = [], []
        in_data = False

        with open(str(arff_path), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.lower() == "@data":
                    in_data = True
                    continue
                if in_data and line and not line.startswith("%"):
                    parts = line.rsplit(",", 1)
                    if len(parts) == 2:
                        feats = [float(v) for v in parts[0].split(",")]
                        label = parts[1].strip()
                        X_rows.append(feats)
                        y_rows.append(label)

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows)
        logger.info("ARFF loaded: %s — X=%s", arff_path.name, X.shape)
        return X, y

    # ── Augmentation-aware extraction ─────────────────────────────────────

    def augment_dataset(
        self,
        raw_dir: str | Path,
        output_dir: str | Path,
        target_sr: int = 16000,
    ) -> list[str]:
        """
        Apply audiomentations augmentation to all .wav files in raw_dir and
        save to output_dir with '_aug' suffix.

        This is a convenience wrapper that internally uses AudioPreprocessor.

        Parameters
        ----------
        raw_dir : str | Path
        output_dir : str | Path
        target_sr : int

        Returns
        -------
        list[str] — paths of generated augmented files
        """
        from src.preprocessor import AudioPreprocessor

        preprocessor = AudioPreprocessor(target_sr=target_sr)
        output_dir = ensure_dir(output_dir)
        aug_paths = []

        for wav_path in sorted(Path(raw_dir).glob("*.wav")):
            dst_path = output_dir / f"{wav_path.stem}_aug.wav"
            if dst_path.exists():
                aug_paths.append(str(dst_path))
                continue
            try:
                audio = preprocessor.load(wav_path)
                aug_audio = preprocessor.augment(audio)
                preprocessor.save(aug_audio, dst_path)
                aug_paths.append(str(dst_path))
            except Exception as exc:
                logger.error("Augmentation failed for %s: %s", wav_path.name, exc)

        logger.info("Augmented %d files → %s", len(aug_paths), output_dir)
        return aug_paths
