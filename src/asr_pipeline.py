"""
asr_pipeline.py — ASR using faster-whisper (Python 3.12 compatible).

⚠️  Do NOT use openai-whisper — it has a pkg_resources.__version__ KeyError on
    Python 3.13 and is no longer maintained. faster-whisper is a drop-in
    replacement that is faster on both CPU and GPU.

Usage:
    asr = FasterWhisperASR(model_size="medium", device="auto")
    transcript = asr.transcribe_file("audio.wav", language="de")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

# ── faster-whisper import ──────────────────────────────────────────────────
try:
    from faster_whisper import WhisperModel

    _FW_AVAILABLE = True
except ImportError:
    _FW_AVAILABLE = False
    logger.warning(
        "faster-whisper not installed. Install with:\n"
        "  pip install faster-whisper==1.0.3\n"
        "ASR functionality will be unavailable."
    )


# ─────────────────────────────────────────────────────────────────────────────
# FasterWhisperASR
# ─────────────────────────────────────────────────────────────────────────────

class FasterWhisperASR:
    """
    Thin wrapper around faster-whisper for batch transcription of EMO-DB
    (German) and RAVDESS (English) audio files.

    Parameters
    ----------
    model_size : str
        One of: "tiny", "base", "small", "medium", "large-v2", "large-v3"
        Recommended: "medium" for accuracy/speed balance
    device : str
        "auto" — detects CUDA and falls back to CPU
        "cuda" — force GPU
        "cpu"  — force CPU
    compute_type : str
        "float16" for GPU, "int8" for CPU (auto-selected when device="auto")
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "auto",
        compute_type: str = "float16",
    ):
        if not _FW_AVAILABLE:
            raise ImportError(
                "faster-whisper is not installed.\n"
                "Install with: pip install faster-whisper==1.0.3"
            )

        self.model_size = model_size

        # Auto-detect device
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                    self.compute_type = compute_type
                else:
                    self.device = "cpu"
                    self.compute_type = "int8"
            except ImportError:
                self.device = "cpu"
                self.compute_type = "int8"
        else:
            self.device = device
            self.compute_type = compute_type if device != "cpu" else "int8"

        logger.info(
            "Loading faster-whisper model: %s [device=%s, compute_type=%s]",
            model_size, self.device, self.compute_type,
        )
        self._model = WhisperModel(
            model_size_or_path=model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("faster-whisper model loaded.")

    # ── Single file transcription ─────────────────────────────────────────

    def transcribe_file(
        self,
        audio_path: str | Path,
        language: str = "de",
        beam_size: int = 5,
        word_timestamps: bool = False,
    ) -> Optional[str]:
        """
        Transcribe a single audio file.

        Parameters
        ----------
        audio_path : str | Path — path to .wav file
        language : str — "de" for German (EMO-DB), "en" for English (RAVDESS)
        beam_size : int
        word_timestamps : bool

        Returns
        -------
        str | None — transcript text, or None if empty / transcription failed
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error("Audio file not found: %s", audio_path)
            return None

        try:
            segments, info = self._model.transcribe(
                str(audio_path),
                language=language,
                beam_size=beam_size,
                word_timestamps=word_timestamps,
                task="transcribe",
            )
            transcript = " ".join(seg.text.strip() for seg in segments)
            transcript = transcript.strip()

            # Handle empty transcripts
            words = transcript.split()
            if len(words) < 3:
                logger.warning(
                    "Short/empty transcript (%d words) for %s: '%s'",
                    len(words), audio_path.name, transcript,
                )
                return transcript if transcript else None

            return transcript

        except Exception as exc:
            logger.error("Transcription failed for %s: %s", audio_path.name, exc)
            return None

    # ── Batch transcription ───────────────────────────────────────────────

    def transcribe_batch(
        self,
        manifest_df: pd.DataFrame,
        filepath_col: str = "processed_filepath",
        language: str = "de",
        language_col: Optional[str] = None,
        output_csv_path: Optional[str | Path] = None,
        beam_size: int = 5,
    ) -> pd.DataFrame:
        """
        Transcribe all files in manifest_df and return DataFrame with transcripts.

        Parameters
        ----------
        manifest_df : pd.DataFrame
        filepath_col : str — column containing audio file paths
        language : str — default language ("de" or "en")
        language_col : str | None — if given, per-row language override
        output_csv_path : str | Path | None — save result CSV here
        beam_size : int

        Returns
        -------
        pd.DataFrame — manifest_df with 'transcript' and 'duration_sec' columns appended
        """
        import librosa

        transcripts = []
        durations = []

        total = len(manifest_df)
        for i, (_, row) in enumerate(manifest_df.iterrows()):
            fpath = row.get(filepath_col)
            lang = row.get(language_col, language) if language_col else language

            if fpath is None or not Path(str(fpath)).exists():
                logger.warning("[%d/%d] Missing file: %s", i + 1, total, fpath)
                transcripts.append(None)
                durations.append(None)
                continue

            # Duration
            try:
                duration = librosa.get_duration(path=str(fpath))
            except Exception:
                duration = None

            # Transcribe
            transcript = self.transcribe_file(fpath, language=str(lang), beam_size=beam_size)
            transcripts.append(transcript)
            durations.append(duration)

            if (i + 1) % 50 == 0 or (i + 1) == total:
                logger.info(
                    "[%d/%d] Transcription progress — latest: '%s'",
                    i + 1, total, transcript[:60] if transcript else "None",
                )

        result_df = manifest_df.copy()
        result_df["transcript"] = transcripts
        result_df["duration_sec"] = durations

        n_ok = sum(t is not None for t in transcripts)
        logger.info(
            "Batch transcription complete: %d/%d successful", n_ok, total
        )

        if output_csv_path is not None:
            output_csv_path = Path(output_csv_path)
            ensure_dir(output_csv_path.parent)
            result_df.to_csv(str(output_csv_path), index=False)
            logger.info("Transcripts saved → %s", output_csv_path)

        return result_df

    # ── Streaming chunk transcription (for Gradio dashboard) ─────────────

    def transcribe_chunk(
        self,
        audio: "np.ndarray",
        sample_rate: int = 16000,
        language: str = "de",
    ) -> Optional[str]:
        """
        Transcribe a raw numpy audio chunk (e.g. from Gradio microphone stream).

        Parameters
        ----------
        audio : np.ndarray — float32 mono array
        sample_rate : int
        language : str

        Returns
        -------
        str | None
        """
        import io
        import tempfile

        import numpy as np
        import soundfile as sf

        if audio is None or (hasattr(audio, '__len__') and len(audio) == 0):
            return None

        # Write to a temp file (faster-whisper expects a file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            if isinstance(audio, np.ndarray):
                audio_float = audio.astype(np.float32)
                if audio_float.max() > 1.0:
                    audio_float = audio_float / 32768.0  # int16 → float
            else:
                audio_float = np.array(audio, dtype=np.float32)

            sf.write(tmp_path, audio_float, sample_rate, subtype="PCM_16")
            transcript = self.transcribe_file(tmp_path, language=language)
            return transcript
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
