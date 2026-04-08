"""
data_loader.py — Dataset loading and manifest creation for EMO-DB and RAVDESS.

Manifest columns (EMO-DB):
    filepath, speaker_id, text_code, emotion_code, emotion_label_de,
    emotion_label_en, split

Manifest columns (RAVDESS):
    filepath, modality, channel, emotion_code, intensity, statement,
    repetition, actor_id, emotion_label, split
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from src.label_mapper import LabelMapper
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# EMO-DB Loader
# ─────────────────────────────────────────────────────────────────────────────

class EmoDB_Loader:
    """
    Load EMO-DB dataset, build manifest CSV, and resample audio files.

    EMO-DB filename convention (each position is 0-indexed on the stem):
        Positions 0-1 : speaker ID  (e.g. "03")
        Positions 2-4 : text code   (e.g. "a01")
        Position  5   : emotion code letter (e.g. 'F' = Freude/Happiness)
        Position  6   : version letter      (e.g. 'a')

    Example: "03a01Fa.wav" → speaker=03, text=a01, emotion=F, version=a
    """

    EMOTION_DE: dict[str, str] = {
        "W": "Ärger",
        "L": "Langeweile",
        "E": "Ekel",
        "A": "Angst",
        "F": "Freude",
        "T": "Trauer",
        "N": "Neutral",
    }

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._lm = LabelMapper()

    def load_manifest(self, raw_dir: str | Path) -> pd.DataFrame:
        """
        Scan raw_dir for .wav files and build a manifest DataFrame.

        Parameters
        ----------
        raw_dir : str | Path
            Directory containing the raw EMO-DB .wav files.

        Returns
        -------
        pd.DataFrame with columns:
            filepath, speaker_id, text_code, emotion_code,
            emotion_label_de, emotion_label_en
        """
        raw_dir = Path(raw_dir)
        if not raw_dir.exists():
            raise FileNotFoundError(f"EMO-DB raw directory not found: {raw_dir}")

        records = []
        wav_files = sorted(raw_dir.glob("*.wav"))

        if not wav_files:
            logger.warning("No .wav files found in %s", raw_dir)
            return pd.DataFrame()

        skipped = 0
        for wav_path in wav_files:
            stem = wav_path.stem  # filename without extension
            if len(stem) < 7:
                logger.debug("Skipping malformed filename: %s", wav_path.name)
                skipped += 1
                continue

            emotion_code = stem[5]  # CRITICAL: index 5, not 6
            emotion_en = LabelMapper.EMODB_7CLASS.get(emotion_code)
            if emotion_en is None:
                logger.debug("Unknown emotion code '%s' in %s", emotion_code, wav_path.name)
                skipped += 1
                continue

            records.append(
                {
                    "filepath": str(wav_path.resolve()),
                    "filename": wav_path.name,
                    "speaker_id": stem[0:2],
                    "text_code": stem[2:5],
                    "emotion_code": emotion_code,
                    "emotion_label_de": self.EMOTION_DE.get(emotion_code, ""),
                    "emotion_label_en": emotion_en,
                }
            )

        df = pd.DataFrame(records)
        logger.info(
            "EMO-DB manifest: %d files loaded, %d skipped. Classes: %s",
            len(df),
            skipped,
            df["emotion_label_en"].value_counts().to_dict() if len(df) else {},
        )
        return df

    def resample_all(
        self,
        raw_dir: str | Path,
        output_dir: str | Path,
        target_sr: int = 16000,
        manifest_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Resample all EMO-DB .wav files to target_sr mono and save to output_dir.

        Parameters
        ----------
        raw_dir : str | Path
        output_dir : str | Path
        target_sr : int
        manifest_df : pd.DataFrame | None
            If provided, adds 'processed_filepath' column and returns it.

        Returns
        -------
        pd.DataFrame — manifest with 'processed_filepath' column appended.
        """
        raw_dir = Path(raw_dir)
        output_dir = ensure_dir(output_dir)

        if manifest_df is None:
            manifest_df = self.load_manifest(raw_dir)

        processed_paths = []
        for _, row in manifest_df.iterrows():
            src_path = Path(row["filepath"])
            dst_path = output_dir / src_path.name

            if dst_path.exists():
                processed_paths.append(str(dst_path))
                continue

            try:
                audio, _ = librosa.load(str(src_path), sr=target_sr, mono=True)
                sf.write(str(dst_path), audio, target_sr, subtype="PCM_16")
                processed_paths.append(str(dst_path))
            except Exception as exc:
                logger.error("Failed to resample %s: %s", src_path.name, exc)
                processed_paths.append(None)

        manifest_df = manifest_df.copy()
        manifest_df["processed_filepath"] = processed_paths
        logger.info(
            "Resampled %d/%d files to %dHz in %s",
            sum(p is not None for p in processed_paths),
            len(processed_paths),
            target_sr,
            output_dir,
        )
        return manifest_df

    def save_manifest(self, df: pd.DataFrame, output_path: str | Path) -> None:
        """Persist manifest DataFrame as CSV."""
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        df.to_csv(str(output_path), index=False)
        logger.info("Manifest saved → %s (%d rows)", output_path, len(df))

    def load_saved_manifest(self, csv_path: str | Path) -> pd.DataFrame:
        """Load a previously saved manifest CSV."""
        return pd.read_csv(str(csv_path))


# ─────────────────────────────────────────────────────────────────────────────
# RAVDESS Loader
# ─────────────────────────────────────────────────────────────────────────────

class RAVDESS_Loader:
    """
    Load RAVDESS dataset, build manifest CSV, and resample audio files.

    RAVDESS naming convention (7 fields separated by '-'):
        [modality]-[channel]-[emotion]-[intensity]-[statement]-[repetition]-[actor]
        Example: 03-01-06-01-02-01-12.wav
                 ↑  ↑  ↑  ↑  ↑  ↑  ↑
                 mod ch  emo int  stm rep actor

    We use ONLY:
        modality == '03'  (audio-only; excludes video)
        channel  == '01'  (speech; excludes song)
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def load_manifest(self, raw_dir: str | Path) -> pd.DataFrame:
        """
        Scan raw_dir recursively for RAVDESS .wav speech files and build manifest.

        RAVDESS is structured as Actor_01/, Actor_02/, ... Actor_24/ subdirectories.

        Parameters
        ----------
        raw_dir : str | Path

        Returns
        -------
        pd.DataFrame with columns:
            filepath, filename, modality, channel, emotion_code, intensity,
            statement, repetition, actor_id, emotion_label
        """
        raw_dir = Path(raw_dir)
        if not raw_dir.exists():
            raise FileNotFoundError(f"RAVDESS raw directory not found: {raw_dir}")

        records = []
        skipped = 0

        for wav_path in sorted(raw_dir.rglob("*.wav")):
            stem = wav_path.stem
            parts = stem.split("-")

            if len(parts) != 7:
                skipped += 1
                continue

            modality, channel, emotion_code, intensity, statement, repetition, actor_id = parts

            # Filter: audio-only speech files only
            if modality != "03" or channel != "01":
                skipped += 1
                continue

            emotion_label = LabelMapper.RAVDESS_7CLASS.get(emotion_code)
            # Exclude "surprised" from cross-corpus eval (no EMO-DB equivalent)
            if emotion_label == "surprised":
                skipped += 1
                continue

            records.append(
                {
                    "filepath": str(wav_path.resolve()),
                    "filename": wav_path.name,
                    "modality": modality,
                    "channel": channel,
                    "emotion_code": emotion_code,
                    "intensity": intensity,
                    "statement": statement,
                    "repetition": repetition,
                    "actor_id": actor_id,
                    "emotion_label": emotion_label,
                }
            )

        df = pd.DataFrame(records)
        logger.info(
            "RAVDESS manifest: %d speech files, %d excluded. Classes: %s",
            len(df),
            skipped,
            df["emotion_label"].value_counts().to_dict() if len(df) else {},
        )
        return df

    def resample_all(
        self,
        raw_dir: str | Path,
        output_dir: str | Path,
        target_sr: int = 16000,
        manifest_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Resample RAVDESS speech files to target_sr mono.

        Parameters
        ----------
        raw_dir : str | Path
        output_dir : str | Path
        target_sr : int
        manifest_df : pd.DataFrame | None

        Returns
        -------
        pd.DataFrame — manifest with 'processed_filepath' column.
        """
        raw_dir = Path(raw_dir)
        output_dir = ensure_dir(output_dir)

        if manifest_df is None:
            manifest_df = self.load_manifest(raw_dir)

        processed_paths = []
        for _, row in manifest_df.iterrows():
            src_path = Path(row["filepath"])
            dst_path = output_dir / src_path.name

            if dst_path.exists():
                processed_paths.append(str(dst_path))
                continue

            try:
                audio, _ = librosa.load(str(src_path), sr=target_sr, mono=True)
                sf.write(str(dst_path), audio, target_sr, subtype="PCM_16")
                processed_paths.append(str(dst_path))
            except Exception as exc:
                logger.error("Failed to resample %s: %s", src_path.name, exc)
                processed_paths.append(None)

        manifest_df = manifest_df.copy()
        manifest_df["processed_filepath"] = processed_paths
        logger.info(
            "Resampled %d/%d RAVDESS files to %dHz",
            sum(p is not None for p in processed_paths),
            len(processed_paths),
            target_sr,
        )
        return manifest_df

    def save_manifest(self, df: pd.DataFrame, output_path: str | Path) -> None:
        """Persist manifest DataFrame as CSV."""
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        df.to_csv(str(output_path), index=False)
        logger.info("RAVDESS manifest saved → %s (%d rows)", output_path, len(df))

    def load_saved_manifest(self, csv_path: str | Path) -> pd.DataFrame:
        """Load a previously saved manifest CSV."""
        return pd.read_csv(str(csv_path))
