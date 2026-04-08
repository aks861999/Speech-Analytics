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
from typing import Optional

import numpy as np

# ── Add project root to sys.path ──────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.fusion import SentimentFusion
from src.utils import get_logger, load_config

logger = get_logger(__name__)

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
    from src.text_classifier import GermanSentimentClassifier
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


def _init_models(config: dict) -> None:
    """Lazy-initialise ML models once at app startup."""
    global _asr, _text_clf, _feat_extractor

    asr_cfg = config.get("asr", {})
    txt_cfg = config.get("text_classifier", {})

    if _ASR_AVAILABLE and _asr is None:
        try:
            _asr = FasterWhisperASR(
                model_size=asr_cfg.get("model_size", "base"),  # use "base" for speed in demo
                device=asr_cfg.get("device", "auto"),
            )
            logger.info("ASR model loaded.")
        except Exception as e:
            logger.warning("ASR model load failed: %s", e)

    if _TEXT_CLF_AVAILABLE and _text_clf is None:
        try:
            _text_clf = GermanSentimentClassifier(device="auto")
            logger.info("Text classifier loaded.")
        except Exception as e:
            logger.warning("Text classifier load failed: %s", e)

    if _FEAT_EXT_AVAILABLE and _feat_extractor is None:
        _feat_extractor = PyAudioFeatureExtractor(target_sr=16000)


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode samples (simulated emotion sequence for thesis presentation)
# ─────────────────────────────────────────────────────────────────────────────

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
    import pandas as pd

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

    # ── ASR ────────────────────────────────────────────────────────────────
    transcript = None
    if _asr is not None:
        try:
            transcript = _asr.transcribe_chunk(audio_array, sample_rate=sample_rate, language=language)
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

    # Update transcript buffer
    if transcript and "[Audio" not in transcript:
        transcript_buffer = transcript_buffer + " " + transcript
        transcript_buffer = transcript_buffer.strip()[-800:]  # keep last 800 chars

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
    }

    # ── Build outputs ──────────────────────────────────────────────────────

    # gr.Label input: dict of {label: confidence}
    confidence_pct = {
        k.upper(): round(v * 100, 1) for k, v in fused_proba.items()
    }

    # Alert HTML
    alert_html = _build_alert_html(alert_active, predicted_class, recovery)

    # Plot data
    plot_df = pd.DataFrame(history)

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
        plot_df,
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
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000

            feats = _feat_extractor.extract_from_array(audio_array, overlap=True)

            # Simple softmax over ZCR/Energy features as acoustic proxy
            zcr = feats[0]   # ZCR
            energy = feats[1]  # Energy

            # Heuristic: high energy + low ZCR → anger (negative)
            #            low energy            → neutral
            #            moderate              → positive
            neg_score = float(np.clip(energy * 10 - zcr * 5, 0, 1))
            pos_score = float(np.clip(zcr * 3 - energy * 5, 0, 1))
            neu_score = max(0.0, 1.0 - neg_score - pos_score)

            total = neg_score + pos_score + neu_score
            if total > 0:
                return {
                    "positive": pos_score / total,
                    "negative": neg_score / total,
                    "neutral": neu_score / total,
                }
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
    import pandas as pd
    history = stream_state.get("history", [])
    plot_df = pd.DataFrame(history) if history else pd.DataFrame(
        {"time": [], "sentiment_score": []}
    )
    return (
        {"NEUTRAL": 100.0},
        stream_state.get("transcript_buffer", ""),
        plot_df,
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
    import pandas as pd

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

    confidence_pct = {k.upper(): round(v * 100, 1) for k, v in fused_proba.items()}
    alert_html = _build_alert_html(alert_active, predicted_class, recovery)
    plot_df = pd.DataFrame(history)
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
        plot_df,
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

    css = """
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

    with gr.Blocks(
        title="Speech Analytics Dashboard — Allianz Thesis Prototype",
        css=css,
        theme=gr.themes.Base(),
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
                    stream_every=stream_every,
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

                sentiment_plot = gr.LinePlot(
                    label="Sentiment Trend (positive_prob − negative_prob)",
                    x="time",
                    y="sentiment_score",
                    x_title="Time (seconds)",
                    y_title="Sentiment Score",
                    y_lim=[-1.0, 1.0],
                    height=280,
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
        )

        # Demo mode: next step button
        def handle_demo_next(idx, state):
            outputs = process_demo_step(idx, state)
            confidence_pct, transcript, plot_df, alert_html, log, scores, new_state = outputs
            step_info = DEMO_SEQUENCE[idx % len(DEMO_SEQUENCE)]
            status = f"Step {(idx % len(DEMO_SEQUENCE)) + 1}/{len(DEMO_SEQUENCE)}: {step_info['description']}"
            new_idx = (idx + 1) % len(DEMO_SEQUENCE)
            return (
                confidence_pct,
                transcript,
                plot_df,
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
            }
            import pandas as pd
            empty_plot = pd.DataFrame({"time": [], "sentiment_score": []})
            return (
                {"NEUTRAL": 100.0},
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
