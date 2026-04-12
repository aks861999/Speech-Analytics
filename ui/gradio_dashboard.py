"""
gradio_dashboard.py — Phase 3: Gradio 5.x Real-Time Agent Dashboard.

Design Decision (Thesis §DD-2 — State-Change Alerting):
    A threshold-only approach generates too many false positives.
    The state-machine approach fires an alert only when sentiment
    *transitions* to negative — signalling a change in call dynamics.

Architecture per audio chunk:
    Audio chunk → faster-whisper ASR → German BERT text classification
                                     → PyAudioAnalysis 34-feat extraction
               → 75/25 Fusion → State machine → UI update

Gradio 5.x streaming note:
    Use gr.Audio(streaming=True, stream_every=N) in gr.Blocks().
    The old queue/stream() API from Gradio 4.x no longer exists.
"""

from __future__ import annotations
import datetime
import logging
import os
import sys
import time
from pathlib import Path





# ── Add project root to sys.path ──────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Optional
import numpy as np
import pandas as pd
import logging as _logging
from src.fusion import SentimentFusion
from src.utils import get_logger, load_config
import gradio as gr
import plotly.graph_objects as go
import librosa



import shutil as _shutil
import os as _os
_orig_shutil_move = _shutil.move

def _win_safe_move(src, dst, copy_function=_shutil.copy2):
    try:
        _orig_shutil_move(src, dst, copy_function=copy_function)
    except (PermissionError, OSError):
        try:
            _os.replace(src, dst)
        except (PermissionError, OSError):
            pass  # cache move failed; audio already in numpy buffer, safe to skip

_shutil.move = _win_safe_move


logger = get_logger(__name__)

# ── Pipeline log file handler ──────────────────────────────────────────────

_log_formatter = _logging.Formatter(
    "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_file_handler = _logging.FileHandler(
    _PROJECT_ROOT / "pipeline.log",   # saves to Masters-Thesis-Voice-AI/pipeline.log
    mode="a",                          # append — preserves logs across restarts
    encoding="utf-8",
)
_file_handler.setFormatter(_log_formatter)
_file_handler.setLevel(_logging.DEBUG)  # capture DEBUG and above to file

# Attach to root logger so ALL src.* module logs are captured
_logging.getLogger().addHandler(_file_handler)
_logging.getLogger().setLevel(_logging.DEBUG)
logger.info("Pipeline log started → %s", _PROJECT_ROOT / "pipeline.log")

# ── Graceful imports ───────────────────────────────────────────────────────

try:
    import gradio as gr

    _GRADIO_AVAILABLE = True
except ImportError:
    _GRADIO_AVAILABLE = False
    logger.error("gradio not installed. Run: pip install gradio==5.4.0")

try:
    from src.asr_pipeline import FasterWhisperASR
    _ASR_AVAILABLE = True
except Exception:
    _ASR_AVAILABLE = False

try:
    from src.text_classifier import GermanSentimentClassifier   # oliverguhr — German-native, no translation
    from src.text_classifier import GermanEmotionClassifier     # j-hartmann — kept for reference/offline use
    _TEXT_CLF_AVAILABLE = True
except Exception:
    _TEXT_CLF_AVAILABLE = False

try:
    from src.feature_extractor import PyAudioFeatureExtractor
    _FEAT_EXT_AVAILABLE = True
except Exception:
    _FEAT_EXT_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded ML components (only instantiated when the app starts)
# ─────────────────────────────────────────────────────────────────────────────

_asr: Optional["FasterWhisperASR"] = None
_text_clf: Optional["GermanSentimentClassifier"] = None
_feat_extractor: Optional["PyAudioFeatureExtractor"] = None
_fusion = SentimentFusion(text_weight=0.75)
_svm = None           # loaded from models/phase1/
_svm_scaler = None    # loaded from models/phase1/
_svm_classes = None
_svm_le = None



def _build_sentiment_plot(history: list):
    """Build a plotly figure — thread-safe and natively supported by gr.Plot in Gradio 5.x."""
    

    fig = go.Figure()

    if history:
       
        df = pd.DataFrame(history)
        times  = df["time"].tolist()
        scores = df["sentiment_score"].tolist()

        # Positive fill area
        fig.add_trace(go.Scatter(
            x=times, y=[max(0.0, s) for s in scores],
            fill="tozeroy", fillcolor="rgba(34,197,94,0.12)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))

        # Negative fill area
        fig.add_trace(go.Scatter(
            x=times, y=[min(0.0, s) for s in scores],
            fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))

        # Main sentiment line
        fig.add_trace(go.Scatter(
            x=times, y=scores,
            mode="lines+markers",
            line=dict(color="#4f98a3", width=2),
            marker=dict(size=6, color="#4f98a3"),
            name="Sentiment Score",
            hovertemplate="Time: %{x}s<br>Score: %{y:.3f}<extra></extra>",
        ))

    # Zero baseline
    fig.add_hline(y=0, line_dash="dash", line_color="#555", line_width=1)

    fig.update_layout(
        paper_bgcolor="rgba(28,27,25,1)",
        plot_bgcolor="rgba(28,27,25,1)",
        font=dict(color="#cdccca", size=12),
        xaxis=dict(
            title="Time (seconds)",
            gridcolor="#393836",
            zerolinecolor="#393836",
            color="#cdccca",
        ),
        yaxis=dict(
            title="Sentiment Score",
            range=[-1.05, 1.05],
            gridcolor="#393836",
            zerolinecolor="#393836",
            color="#cdccca",
        ),
        margin=dict(l=55, r=15, t=15, b=50),
        height=280,
        showlegend=False,
    )
    return fig


def _init_models(config: dict) -> None:
    """Lazy-initialise ML models once at app startup."""
    global _asr, _text_clf, _feat_extractor, _svm, _svm_scaler, _svm_classes, _svm_le


    asr_cfg = config.get("asr", {})
    txt_cfg = config.get("text_classifier", {})

    if _ASR_AVAILABLE and _asr is None:
        try:
            _asr = FasterWhisperASR(
                model_size=asr_cfg.get("model_size", "base"),  # use "base" for speed in demo
                device=asr_cfg.get("device", "auto"),
            )
            logger.info("ASR model loaded. Attr: %s",
                        "model" if hasattr(_asr, "model") else
                        "_model" if hasattr(_asr, "_model") else "UNKNOWN — transcribe_chunk fallback will be used")
        except Exception as e:
            logger.warning("ASR model load failed: %s", e)

    if _TEXT_CLF_AVAILABLE and _text_clf is None:
        try:
            # GermanSentimentClassifier: oliverguhr/german-sentiment-bert
            # German-native → no Helsinki DE→EN translation step
            # 3-class output (positive/negative/neutral) — exactly what fusion needs
            # ~2-3x faster per chunk than the translate+classify pipeline
            _text_clf = GermanSentimentClassifier(device='auto')
            logger.info("Text classifier loaded: GermanSentimentClassifier (oliverguhr, German-native, 3-class)")
        except Exception as e:
            logger.warning("Text classifier load failed: %s", e)

    if _FEAT_EXT_AVAILABLE and _feat_extractor is None:
        _feat_extractor = PyAudioFeatureExtractor(target_sr=16000)

    import joblib
    global _svm, _svm_scaler, _svm_classes, _svm_le
    _MODEL_DIR   = _PROJECT_ROOT / 'models' / 'phase1'
    _svm_path    = _MODEL_DIR / 'SVM_overlap.pkl'
    _scaler_path = _MODEL_DIR / 'scaler_overlap.pkl'
    _le_path     = _MODEL_DIR / 'label_encoder_overlap.pkl'

    if _svm is None and _svm_path.exists():
        _svm = joblib.load(_svm_path)
        logger.info("Loaded SVM from %s", _svm_path)

    if _svm_scaler is None and _scaler_path.exists():
        _svm_scaler = joblib.load(_scaler_path)
        logger.info("Loaded scaler from %s", _scaler_path)
    
    
    # inside init_models:
    if _svm_le is None and _le_path.exists():
        _svm_le = joblib.load(_le_path)
        _svm_classes = _svm_le.classes_
        logger.info("Loaded label encoder, classes: %s", _svm_classes)

    # Only warn if the SVM itself is still missing after attempting load
    if _svm is None:
        logger.warning("SVM model not found — acoustic branch uses heuristic fallback")





# ── Dashboard CSS (module-level so main() can pass it to launch()) ────────────
_DASHBOARD_CSS = """
.gradio-container { font-family: 'Inter', sans-serif; }
.sentiment-positive { color: #22c55e; font-weight: bold; }
.sentiment-negative { color: #ef4444; font-weight: bold; }
.sentiment-neutral  { color: #94a3b8; font-weight: bold; }
.header-bar {
    background: linear-gradient(90deg, #1e3a8a, #1d4ed8);
    color: white; padding: 20px 30px;
    border-radius: 10px; margin-bottom: 20px;
}
"""

DEMO_SEQUENCE = [
    {
        "text_proba": {"positive": 0.05, "negative": 0.10, "neutral": 0.85},
        "label": "neutral",
        "transcript": "Ja, guten Tag. Wie kann ich Ihnen helfen?",
        "description": "Agent greets customer",
    },
    {
        "text_proba": {"positive": 0.10, "negative": 0.15, "neutral": 0.75},
        "label": "neutral",
        "transcript": "Ich verstehe Ihr Anliegen. Lassen Sie mich das prüfen.",
        "description": "Agent acknowledges issue",
    },
    {
        "text_proba": {"positive": 0.05, "negative": 0.80, "neutral": 0.15},
        "label": "negative",
        "transcript": "Das ist inakzeptabel! Ich warte schon seit drei Wochen!",
        "description": "⚠️ Customer becomes frustrated — ALERT fires",
    },
    {
        "text_proba": {"positive": 0.10, "negative": 0.60, "neutral": 0.30},
        "label": "negative",
        "transcript": "Niemand hat mir geantwortet. Das ist absolut unzumutbar.",
        "description": "Customer remains negative",
    },
    {
        "text_proba": {"positive": 0.25, "negative": 0.20, "neutral": 0.55},
        "label": "neutral",
        "transcript": "Okay, ich verstehe. Ich warte auf Ihre Rückmeldung.",
        "description": "Recovery begins — alert clears",
    },
    {
        "text_proba": {"positive": 0.65, "negative": 0.10, "neutral": 0.25},
        "label": "positive",
        "transcript": "Vielen Dank für Ihre schnelle Hilfe. Das ist sehr nett.",
        "description": "✅ Customer sentiment recovered to positive",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Audio processing: state machine
# ─────────────────────────────────────────────────────────────────────────────

def _deduplicate_transcript(previous: str, new: str) -> str:
    """
    Strip the portion of `new` that was already in `previous`.
    Whisper re-transcribes the full ring buffer every chunk, so the
    beginning of each new transcript overlaps with the previous one.
    Punctuation is stripped before comparison so "fühle." matches "fühle".
    """
    if not previous or not new:
        return new

    import string
    def _clean(w: str) -> str:
        return w.strip(string.punctuation).lower()

    prev_words = previous.split()
    new_words  = new.split()
    prev_clean = [_clean(w) for w in prev_words]
    new_clean  = [_clean(w) for w in new_words]

    max_overlap = min(len(prev_clean), len(new_clean))
    for overlap in range(max_overlap, 0, -1):
        if prev_clean[-overlap:] == new_clean[:overlap]:
            return " ".join(new_words[overlap:]).strip()

    return new


def process_audio_chunk(
    audio_chunk,
    stream_state: dict,
    language: str = "de",
) -> tuple:
    """
    Process one audio chunk through the full pipeline.

    Parameters
    ----------
    audio_chunk : tuple (sample_rate, np.ndarray) from Gradio 5.x Audio
    stream_state : dict with keys:
        previous_sentiment : str
        history : list[dict]
        transcript_buffer : str
        alert_active : bool
        alert_timestamp : float | None
    language : str

    Returns
    -------
    tuple of (
        label_update,        # dict for gr.Label
        transcript_update,   # str
        plot_data,           # pd.DataFrame for gr.LinePlot
        alert_html,          # str
        log_text,            # str
        fusion_scores,       # dict
        new_state,           # dict
    )
    """


    if audio_chunk is None:
        return _empty_output(stream_state)

    # Unpack audio
    if isinstance(audio_chunk, tuple):
        sample_rate, audio_array = audio_chunk
    else:
        return _empty_output(stream_state)

    if audio_array is None or len(audio_array) == 0:
        return _empty_output(stream_state)

    # Normalize audio
    if audio_array.dtype != np.float32:
        if np.abs(audio_array).max() > 1.0:
            audio_array = audio_array.astype(np.float32) / 32768.0
        else:
            audio_array = audio_array.astype(np.float32)

    # ── Ring buffer: accumulate audio, always pass last 4s to Whisper ─────
    # Resample FIRST so the buffer is always uniformly 16kHz
    if sample_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    audio_buffer = stream_state.get("audio_buffer", [])
    audio_buffer.append(audio_array)

    _MAX_BUFFER_SAMPLES = 4 * 16000  # 4 seconds at 16kHz
    combined = np.concatenate(audio_buffer)
    if len(combined) > _MAX_BUFFER_SAMPLES:
        combined = combined[-_MAX_BUFFER_SAMPLES:]
    stream_state = dict(stream_state)
    stream_state["audio_buffer"] = [combined]   # store as single flat array

    asr_audio = combined   # full buffer → Whisper gets full context

    # ── ASR ────────────────────────────────────────────────────────────────
    # Call faster-whisper directly with numpy array — avoids writing temp wav
    # files to disk, which causes WinError 32 (file locked) on Windows when
    # Gradio's cache manager tries to move the same temp file our ASR holds open.
    transcript = None
    if _asr is not None:
        try:
            # faster-whisper WhisperModel.transcribe() accepts float32 numpy arrays natively.
            # getattr handles both .model and ._model depending on how FasterWhisperASR wraps it.
            _model = getattr(_asr, "model", None) or getattr(_asr, "_model", None)
            if _model is not None:
                segments, _ = _model.transcribe(
                    asr_audio,
                    language=language,
                    task="transcribe",
                    beam_size=3,       # 3 is fast enough on CPU; 5 adds latency with little gain
                    vad_filter=True,   # skip silent segments — big speedup on CPU
                    vad_parameters=dict(min_silence_duration_ms=300),
                )
                transcript = " ".join(seg.text.strip() for seg in segments).strip()
            else:
                # Fallback if model attribute name differs — still avoids temp file via chunk
                transcript = _asr.transcribe_chunk(asr_audio, sample_rate=16000, language=language)
        except Exception as e:
            logger.warning("ASR chunk failed: %s", e)

    if not transcript:
        transcript = "[Audio received — transcription pending]"

    # ── Text classification ────────────────────────────────────────────────
    if _text_clf is not None and transcript and "[Audio" not in transcript:
        text_proba = _text_clf.predict_proba(transcript)
    else:
        # Fallback: uniform
        text_proba = {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

    # ── Acoustic feature extraction → 3-class probability ─────────────────
    # Use only the latest chunk (not the full 4s buffer) — acoustic features
    # like energy/ZCR should reflect current moment, not accumulated history.
    acoustic_proba_3class = _compute_acoustic_proba(audio_array, sample_rate)

    # ── Fusion ────────────────────────────────────────────────────────────
    fused_proba, predicted_class = _fusion.weighted_fusion(text_proba, acoustic_proba_3class)
    sentiment_score = SentimentFusion.compute_sentiment_score(fused_proba)

    # ── State machine ─────────────────────────────────────────────────────
    previous_sentiment = stream_state.get("previous_sentiment", "neutral")
    alert_active = stream_state.get("alert_active", False)
    alert_log = stream_state.get("alert_log", "")
    history = stream_state.get("history", [])
    transcript_buffer = stream_state.get("transcript_buffer", "")

    new_alert = False
    recovery = False
    now = datetime.datetime.now().strftime("%H:%M:%S")

    if predicted_class == "negative" and previous_sentiment != "negative":
        new_alert = True
        alert_active = True
        ts_msg = f"[{now}] ⚠️  Sentiment turned NEGATIVE — consider de-escalation"
        alert_log = ts_msg + "\n" + alert_log

    if predicted_class != "negative" and previous_sentiment == "negative":
        recovery = True
        alert_active = False
        ts_msg = f"[{now}] ✅  Sentiment recovered to {predicted_class.upper()}"
        alert_log = ts_msg + "\n" + alert_log

    # Update transcript buffer — deduplicate so ring-buffer re-transcriptions don't repeat
    if transcript and "[Audio" not in transcript:
        novel = _deduplicate_transcript(transcript_buffer, transcript)
        if novel:
            transcript_buffer = (transcript_buffer + " " + novel).strip()
        transcript_buffer = transcript_buffer[-800:]

    # Update history
    history.append(
        {
            "time": len(history) * 3,  # 3 seconds per chunk
            "sentiment_score": sentiment_score,
            "predicted_class": predicted_class,
        }
    )
    # Keep last 60 entries (3 min of 3-second chunks)
    history = history[-60:]

    new_state = {
        "previous_sentiment": predicted_class,
        "alert_active": alert_active,
        "alert_log": alert_log,
        "history": history,
        "transcript_buffer": transcript_buffer,
        "audio_buffer": stream_state.get("audio_buffer", []),  # carry buffer forward!
        "audio_sr": 16000,
    }

    # ── Build outputs ──────────────────────────────────────────────────────

    # gr.Label expects raw 0-1 probabilities — it renders them as % automatically
    confidence_pct = {
        k.upper(): round(v, 4) for k, v in fused_proba.items()
    }

    # Alert HTML
    alert_html = _build_alert_html(alert_active, predicted_class, recovery)

    # Plot data
    plot_fig = _build_sentiment_plot(history)

    # Fusion scores for transparency panel
    fusion_scores = {
        "text": {k: round(v, 3) for k, v in text_proba.items()},
        "acoustic": {k: round(v, 3) for k, v in acoustic_proba_3class.items()},
        "fused": {k: round(v, 3) for k, v in fused_proba.items()},
        "predicted": predicted_class,
        "sentiment_score": round(sentiment_score, 3),
    }

    return (
        confidence_pct,
        transcript_buffer,
        plot_fig,
        alert_html,
        alert_log,
        fusion_scores,
        new_state,
    )


def _compute_acoustic_proba(
    audio_array: np.ndarray,
    sample_rate: int,
) -> dict[str, float]:
    """
    Extract acoustic features and convert to a 3-class probability dict.

    When the full feature extractor + classifier is available, use it.
    Otherwise fall back to a simple energy/ZCR heuristic for the demo.
    """
    try:
        if _feat_extractor is not None:
            # Resample if needed
            if sample_rate != 16000:
                
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000

            feats = _feat_extractor.extract_from_array(audio_array, overlap=True)


            # Use trained Phase 1 SVM if available
            if _svm is not None and _svm_le is not None:

                feats_2d = feats.reshape(1, -1)

                import sklearn.pipeline
                if _svm_scaler is not None and not isinstance(_svm, sklearn.pipeline.Pipeline):
                    feats_2d = _svm_scaler.transform(feats_2d)
                proba_raw = _svm.predict_proba(feats_2d)[0]
                _svm_clf_step = _svm.named_steps['clf'] if hasattr(_svm, 'named_steps') else _svm
                proba_7class = {_svm_le.classes_[int(i)]: float(p) for i, p in zip(_svm_clf_step.classes_, proba_raw)}

                
    
                return SentimentFusion.collapse_acoustic_proba(proba_7class)

            # Fallback heuristic if SVM not loaded yet
            zcr = feats[0]
            energy = feats[1]
            neg_score = float(np.clip(energy * 10 - zcr * 5, 0, 1))
            pos_score = float(np.clip(zcr * 3 - energy * 5, 0, 1))
            neu_score = max(0.0, 1.0 - neg_score - pos_score)
            total = neg_score + pos_score + neu_score
            if total > 0:
                return {'positive': pos_score / total, 'negative': neg_score / total, 'neutral': neu_score / total}



    
    except Exception as e:
        logger.debug("Acoustic feature extraction error: %s", e)

    # Fallback: neutral
    return {"positive": 0.15, "negative": 0.20, "neutral": 0.65}


def _build_alert_html(
    alert_active: bool,
    predicted_class: str,
    recovery: bool = False,
) -> str:
    """Generate the alert banner HTML."""
    if alert_active and predicted_class == "negative":
        return """
        <div style="
            background: linear-gradient(135deg, #ff4444, #cc0000);
            color: white;
            padding: 16px 24px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            text-align: center;
            box-shadow: 0 4px 12px rgba(255,0,0,0.3);
            animation: pulse 1.5s infinite;
        ">
            ⚠️ KUNDENSTIMMUNG NEGATIV — Bitte De-Eskalationsprotokoll anwenden
            <br>
            <small style="font-weight: normal; font-size: 13px;">
                CUSTOMER SENTIMENT TURNED NEGATIVE — Consider escalating or applying de-escalation protocol
            </small>
        </div>
        <style>
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.8; }
                100% { opacity: 1; }
            }
        </style>
        """
    elif recovery:
        return """
        <div style="
            background: linear-gradient(135deg, #22c55e, #16a34a);
            color: white;
            padding: 16px 24px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,200,0,0.3);
        ">
            ✅ Stimmung erholt — Gespräch normalisiert / Sentiment recovered
        </div>
        """
    else:
        return """
        <div style="
            background: #1e293b;
            color: #64748b;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 14px;
            text-align: center;
        ">
            🟢 System active — monitoring customer sentiment
        </div>
        """


def _empty_output(stream_state: dict) -> tuple:
    """Return no-op outputs when no audio is available."""

    history = stream_state.get("history", [])
    plot_fig = _build_sentiment_plot(history)
    return (
        {"NEUTRAL": 1.0},
        stream_state.get("transcript_buffer", ""),
        plot_fig,
        _build_alert_html(False, "neutral"),
        stream_state.get("alert_log", ""),
        {},
        stream_state,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode processing
# ─────────────────────────────────────────────────────────────────────────────

def process_demo_step(
    demo_index: int,
    stream_state: dict,
) -> tuple:
    """
    Advance the demo by one step, returning the same output signature as
    process_audio_chunk.
    """

    step = DEMO_SEQUENCE[demo_index % len(DEMO_SEQUENCE)]
    text_proba = step["text_proba"]
    acoustic_proba = {"positive": 0.15, "negative": 0.20, "neutral": 0.65}

    fused_proba, predicted_class = _fusion.weighted_fusion(text_proba, acoustic_proba)
    sentiment_score = SentimentFusion.compute_sentiment_score(fused_proba)

    previous_sentiment = stream_state.get("previous_sentiment", "neutral")
    alert_active = stream_state.get("alert_active", False)
    alert_log = stream_state.get("alert_log", "")
    history = stream_state.get("history", [])

    now = datetime.datetime.now().strftime("%H:%M:%S")
    recovery = False

    if predicted_class == "negative" and previous_sentiment != "negative":
        alert_active = True
        alert_log = f"[{now}] ⚠️  Sentiment turned NEGATIVE — Demo step {demo_index + 1}\n" + alert_log

    if predicted_class != "negative" and previous_sentiment == "negative":
        recovery = True
        alert_active = False
        alert_log = f"[{now}] ✅  Recovery — sentiment is now {predicted_class.upper()}\n" + alert_log

    history.append(
        {"time": len(history) * 3, "sentiment_score": sentiment_score, "predicted_class": predicted_class}
    )
    history = history[-60:]

    new_state = {
        "previous_sentiment": predicted_class,
        "alert_active": alert_active,
        "alert_log": alert_log,
        "history": history,
        "transcript_buffer": step["transcript"],
    }

    # gr.Label expects raw 0-1 probabilities — it renders them as % automatically
    confidence_pct = {k.upper(): round(v, 4) for k, v in fused_proba.items()}
    alert_html = _build_alert_html(alert_active, predicted_class, recovery)
    plot_fig = _build_sentiment_plot(history)
    fusion_scores = {
        "text": text_proba,
        "acoustic": acoustic_proba,
        "fused": {k: round(v, 3) for k, v in fused_proba.items()},
        "predicted": predicted_class,
        "description": step["description"],
    }

    return (
        confidence_pct,
        step["transcript"],
        plot_fig,
        alert_html,
        alert_log,
        fusion_scores,
        new_state,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard(config: Optional[dict] = None) -> "gr.Blocks":
    """
    Build and return the Gradio Blocks dashboard.

    Parameters
    ----------
    config : dict | None — loaded from configs/config.yaml

    Returns
    -------
    gr.Blocks
    """
    if not _GRADIO_AVAILABLE:
        raise ImportError("gradio not installed.")

    if config is None:
        try:
            config = load_config()
        except Exception:
            config = {}

    gradio_cfg = config.get("gradio", {})
    stream_every = gradio_cfg.get("stream_every", 3.0)

    # Pre-load models
    _init_models(config)



    with gr.Blocks(
        title="Speech Analytics Dashboard — Allianz Thesis Prototype",
        theme=gr.themes.Base(),
        css=_DASHBOARD_CSS,
    ) as demo:

        # ── Header ─────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-bar">
            <h2 style="margin:0;">🎙️ Speech Analytics — Real-Time Emotion Monitor</h2>
            <p style="margin:4px 0 0; opacity:0.8; font-size:14px;">
                Allianz Call Center Prototype | Master Thesis — Akash Biswas | KIT / BHT Berlin 2026
            </p>
        </div>
        """)

        # ── State ──────────────────────────────────────────────────────────
        stream_state = gr.State(
            {
                "previous_sentiment": "neutral",
                "alert_active": False,
                "alert_log": "",
                "history": [],
                "transcript_buffer": "",
                "audio_buffer": [],      # ring buffer: accumulates raw audio chunks
                "audio_sr": 16000,       # sample rate of buffered audio
            }
        )
        demo_index = gr.State(0)

        # ── Alert Banner (full width) ───────────────────────────────────────
        alert_banner = gr.HTML(value=_build_alert_html(False, "neutral"))

        # ── Main 3-column layout ───────────────────────────────────────────
        with gr.Row():

            # LEFT — Call controls
            with gr.Column(scale=1):
                gr.Markdown("### 📞 Call Controls")

                mode_selector = gr.Radio(
                    choices=["Live Microphone", "Demo Mode"],
                    value="Demo Mode",
                    label="Input Mode",
                    interactive=True,
                )

                audio_input = gr.Audio(
                    sources=["microphone"],
                    streaming=True,
                    label="🎤 Microphone Input",
                    visible=False,
                    type="numpy",
                )

                # Demo controls
                with gr.Column(visible=True) as demo_controls:
                    demo_next_btn = gr.Button("▶️  Next Demo Step", variant="primary", size="lg")
                    demo_reset_btn = gr.Button("🔄  Reset Demo", variant="secondary")
                    demo_status = gr.Textbox(
                        value="Demo Mode: Click ▶️ to advance through emotion sequence",
                        label="Demo Status",
                        interactive=False,
                        lines=2,
                    )

                language_selector = gr.Dropdown(
                    choices=["de (German / EMO-DB)", "en (English / RAVDESS)"],
                    value="de (German / EMO-DB)",
                    label="🌐 Language",
                )

                speaker_selector = gr.Dropdown(
                    choices=["Customer", "Agent"],
                    value="Customer",
                    label="👤 Analysed Speaker",
                )

                gr.Markdown(
                    """
                    **Phase 1**: EMO-DB baseline  
                    **Phase 2**: Hybrid fusion (ASR + BERT)  
                    **Phase 3**: Real-time agent dashboard  
                    """
                )

            # CENTER — Sentiment display
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Live Sentiment Analysis")

                sentiment_label = gr.Label(
                    label="Current Customer Sentiment",
                    num_top_classes=3,
                )

                transcript_display = gr.Textbox(
                    label="📝 Live Transcript",
                    lines=4,
                    interactive=False,
                    placeholder="Transcript will appear here as speech is analysed...",
                )

                alert_log_display = gr.Textbox(
                    label="🚨 Alert Log",
                    lines=6,
                    interactive=False,
                    placeholder="Alerts will appear here when sentiment turns negative...",
                )

            # RIGHT — History & scores
            with gr.Column(scale=2):
                gr.Markdown("### 📈 Sentiment History")

                sentiment_plot = gr.Plot(
                    label="Sentiment Trend (positive_prob − negative_prob)",
                )

                fusion_scores_display = gr.JSON(
                    label="🔬 Fusion Scores (Text / Acoustic / Fused)",
                    value={},
                )

        # ── Footer ─────────────────────────────────────────────────────────
        gr.Markdown(
            """
            ---
            **System**: 75/25 Weighted Fusion (Yurtay et al. 2024) | 
            **ASR**: faster-whisper (medium) | 
            **Text**: oliverguhr/german-sentiment-bert |
            **Text**: j-hartmann/emotion-english-distilroberta-base (DE→EN) |
            **Acoustic**: PyAudioAnalysis 34 features |
            **Alert logic**: State-change detection (Thesis Design Decision §DD-2)
            """
        )

        # ── Event handlers ─────────────────────────────────────────────────

        def toggle_mode(mode: str):
            is_live = mode == "Live Microphone"
            return (
                gr.update(visible=is_live),   # audio_input
                gr.update(visible=not is_live),  # demo_controls
            )

        mode_selector.change(
            fn=toggle_mode,
            inputs=[mode_selector],
            outputs=[audio_input, demo_controls],
        )

        # Live microphone streaming
        def handle_audio(audio, state, lang_sel):
            lang = "de" if "de" in lang_sel else "en"
            return process_audio_chunk(audio, state, language=lang)

        audio_input.stream(
            fn=handle_audio,
            inputs=[audio_input, stream_state, language_selector],
            outputs=[
                sentiment_label,
                transcript_display,
                sentiment_plot,
                alert_banner,
                alert_log_display,
                fusion_scores_display,
                stream_state,
            ],
            stream_every=1.5,   # trigger every 1.5s — lower latency than 3s
        )

        # Demo mode: next step button
        def handle_demo_next(idx, state):
            outputs = process_demo_step(idx, state)
            confidence_pct, transcript, plot_fig, alert_html, log, scores, new_state = outputs
            step_info = DEMO_SEQUENCE[idx % len(DEMO_SEQUENCE)]
            status = f"Step {(idx % len(DEMO_SEQUENCE)) + 1}/{len(DEMO_SEQUENCE)}: {step_info['description']}"
            new_idx = (idx + 1) % len(DEMO_SEQUENCE)
            return (
                confidence_pct,
                transcript,
                plot_fig,
                alert_html,
                log,
                scores,
                new_state,
                new_idx,
                status,
            )

        demo_next_btn.click(
            fn=handle_demo_next,
            inputs=[demo_index, stream_state],
            outputs=[
                sentiment_label,
                transcript_display,
                sentiment_plot,
                alert_banner,
                alert_log_display,
                fusion_scores_display,
                stream_state,
                demo_index,
                demo_status,
            ],
        )

        # Reset
        def handle_reset():
            init_state = {
                "previous_sentiment": "neutral",
                "alert_active": False,
                "alert_log": "",
                "history": [],
                "transcript_buffer": "",
                "audio_buffer": [],
                "audio_sr": 16000,
            }
            empty_plot = _build_sentiment_plot([]) 
            return (
                {"NEUTRAL": 1.0},
                "",
                empty_plot,
                _build_alert_html(False, "neutral"),
                "",
                {},
                init_state,
                0,
                "Demo reset. Click ▶️ to start.",
            )

        demo_reset_btn.click(
            fn=handle_reset,
            inputs=[],
            outputs=[
                sentiment_label,
                transcript_display,
                sentiment_plot,
                alert_banner,
                alert_log_display,
                fusion_scores_display,
                stream_state,
                demo_index,
                demo_status,
            ],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Suppress the HF Hub unauthenticated request warning — not needed for local inference
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # avoids tokenizer fork warning on Windows

    try:
        config = load_config("configs/config.yaml")
    except FileNotFoundError:
        logger.warning("configs/config.yaml not found — using defaults.")
        config = {}

    gradio_cfg = config.get("gradio", {})
    port = gradio_cfg.get("server_port", 7860)
    host = gradio_cfg.get("server_name", "0.0.0.0")

    dashboard = build_dashboard(config)
    logger.info("Launching Gradio dashboard on http://%s:%d", host, port)






    dashboard.launch(
        server_name=host,
        server_port=port,
        share=False,
        show_api=False,
    )


if __name__ == "__main__":
    main()