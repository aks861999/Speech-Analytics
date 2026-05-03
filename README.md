# Speech Analytics System for Emotions in Employee–Customer Interactions

**Author**: Akash Biswas | 2026  
**Research Question**: *How should a speech analytics system for the analysis of emotions in employee–customer interactions be designed?*

---

## Overview

This system implements a 3-phase speech analytics pipeline targeting the German insurance call-center context. It extends and fills the gaps of two prior papers:

| Prior Work | Gap | This System |
|---|---|---|
| Madanian et al. (2022) | English only, no text branch | German (EMO-DB) + Whisper ASR + BERT |
| Yurtay et al. (2024) | Turkish data, no cross-corpus eval | Cross-corpus EMO-DB → RAVDESS + German deployment |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 — Acoustic Baseline (Replicating Madanian et al.)      │
│  EMO-DB (German, 535 files) → 34 features → 5 ML classifiers    │
│  10-fold CV, MLflow tracking, confusion matrices                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  PHASE 2 — Hybrid Fusion (Extending Yurtay et al.)              │
│  ASR (faster-whisper) → German BERT sentiment                   │
│  Acoustic 34-feat → 3-class collapse                            │
│  75/25 Weighted Fusion + LR meta-learner                        │
│  Cross-corpus: EMO-DB ↔ RAVDESS                                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  PHASE 3 — Real-Time Agent Dashboard (Gradio 5.x)               │
│  Microphone → 3s chunks → ASR + BERT + Fusion → state machine   │
│  Alert fires when sentiment TRANSITIONS to negative             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment (Python 3.12 ONLY — not 3.13)
conda env create -f environment.yml
conda activate speech-analytics

# Install pyAudioAnalysis from GitHub (not PyPI)
pip install git+https://github.com/tyiannak/pyAudioAnalysis.git

# Verify
python -c "import librosa, transformers, gradio; print('OK')"
```

### 2. Dataset Download

**EMO-DB** (535 German .wav files):
```bash
# Download from http://emodb.bilderbar.info
# Place .wav files in: data/emodb/raw/
```

**RAVDESS** (1440 English .wav files):
```bash
# Download from https://zenodo.org/record/1188976
# Extract Actor_01/ … Actor_24/ into: data/ravdess/raw/
```

### 3. Run Phase 1 (Baseline)

```python
from src.data_loader import EmoDB_Loader
from src.feature_extractor import PyAudioFeatureExtractor
from src.classifiers import EmotionClassifierSuite
from src.evaluator import Evaluator

# Build manifest
loader = EmoDB_Loader()
df = loader.load_manifest("data/emodb/raw")
df = loader.resample_all("data/emodb/raw", "data/emodb/processed/emodb_16khz", manifest_df=df)
loader.save_manifest(df, "data/emodb/processed/emodb_manifest.csv")

# Extract features (both overlap modes)
extractor = PyAudioFeatureExtractor()
X, y, _ = extractor.extract_manifest(df, overlap=True)
extractor.extract_and_save_arff(df, "data/emodb/features/emodb_34features.arff", overlap=True)

# Train classifiers
suite = EmotionClassifierSuite(mlflow_tracking_uri="sqlite:///models/mlruns/mlflow.db")
results_df, best_models = suite.run_all_experiments(X, y, overlap_mode="overlap")
print(results_df[["classifier", "accuracy", "weighted_f1"]])
```

### 4. Run Phase 2 (Fusion)

```python
from src.asr_pipeline import FasterWhisperASR
from src.text_classifier import GermanSentimentClassifier
from src.fusion import SentimentFusion

# Transcribe EMO-DB files
asr = FasterWhisperASR(model_size="medium")
df_with_transcripts = asr.transcribe_batch(df, language="de")

# Text classification
clf = GermanSentimentClassifier()
df_with_preds = clf.predict_manifest(df_with_transcripts)

# Fusion
fusion = SentimentFusion(text_weight=0.75)
fused, predicted = fusion.weighted_fusion(
    text_proba={"positive": 0.7, "negative": 0.1, "neutral": 0.2},
    acoustic_proba={"positive": 0.3, "negative": 0.4, "neutral": 0.3},
)
```

### 5. Launch Dashboard (Phase 3)

```bash
cd speech-analytics
python ui/gradio_dashboard.py
# Open http://localhost:7860
```

---

## Project Structure

```
speech-analytics
├── configs/
│   └── config.yaml                ← All hyperparameters and paths
├── data/
│   ├── emodb/{raw, processed, features}
│   └── ravdess/{raw, processed, features}
├── src/
│   ├── utils.py                   ← Logging, config, seed
│   ├── label_mapper.py            ← 7-class ↔ 3-class mapping
│   ├── data_loader.py             ← EMO-DB & RAVDESS manifest creation
│   ├── preprocessor.py            ← Resampling, VAD, augmentation
│   ├── feature_extractor.py       ← 34-feature extraction (pyAudioAnalysis)
│   ├── classifiers.py             ← SVM, KNN, RF, GB, ExtraTrees + GridSearch
│   ├── evaluator.py               ← Metrics, confusion matrices, UAR
│   ├── asr_pipeline.py            ← faster-whisper ASR (de/en)
│   ├── text_classifier.py         ← German BERT + English RoBERTa
│   └── fusion.py                  ← Weighted fusion + LR meta-learner
├── ui/
│   └── gradio_dashboard.py        ← Phase 3: Gradio 5.x real-time dashboard
├── notebooks/
│   ├── 01_eda_emodb.ipynb
│   ├── 02_baseline_experiments.ipynb
│   ├── 03_cross_corpus_eval.ipynb
│   └── 04_fusion_evaluation.ipynb
├── models/
│   ├── phase1/                    ← Serialized sklearn models (.pkl)
│   ├── phase2/                    ← Fusion model checkpoints
│   └── mlruns/                    ← MLflow tracking
├── requirements.txt
└── environment.yml
```

---

## Key Design Decisions (Thesis Contributions)

### DD-1 — 7-class vs. 3-class
For call-center deployment, 7-class emotion is impractical for agents. Following Yurtay et al.'s 3-class scheme:  
**Positive** → maintain | **Neutral** → observe | **Negative** → escalate

### DD-2 — State-Change Alerting
Alert fires when sentiment **transitions** to negative (not when P(negative) > threshold). This reduces false positives and signals a change in call dynamics.

### DD-3 — Fusion Branch Weighting
75/25 text/acoustic split follows Yurtay's validated result. Ablation study over 60/40, 75/25, 90/10 provides German-context empirical evidence.

### DD-4 — Why EMO-DB
The only publicly available German emotional speech corpus at scale. EMO-DB's acted nature is acknowledged as a limitation; spontaneous call recordings are recommended for future work.

---

## Technology Stack

| Component | Library | Version |
|---|---|---|
| Audio features | pyAudioAnalysis | GitHub HEAD |
| Audio preprocessing | librosa | 0.10.2 |
| Data augmentation | audiomentations | 0.36.0 |
| ASR | faster-whisper | 1.0.3 |
| German BERT | transformers | 4.41.2 |
| ML classifiers | scikit-learn | 1.5.2 |
| Experiment tracking | mlflow | 2.13.2 |
| Dashboard | gradio | 5.4.0 |
| Deep learning | torch | 2.3.1 |

> ⚠️ Python 3.12 only — `openai-whisper` has a breaking `pkg_resources` incompatibility on Python 3.13.

---

## MLflow Tracking

Start the MLflow server:

```bash
mlflow server \
  --backend-store-uri sqlite:///models/mlruns/mlflow.db \
  --default-artifact-root ./models/mlruns/artifacts \
  --host 0.0.0.0 --port 5000
```

Open http://localhost:5000 to view experiment comparison tables for the thesis.

---

## Evaluation Metrics

| Metric | Use Case |
|---|---|
| Accuracy | Within-corpus balanced datasets |
| Weighted F1 | Primary metric (EMO-DB is imbalanced) |
| Macro F1 | Compare across class counts |
| UAR (Unweighted Average Recall) | Cross-corpus standard |
| Confusion Matrix | Identify confused emotion pairs |

---

## Citation / References

- Madanian et al. (2022). *Speech emotion recognition using PyAudioAnalysis*  
- Yurtay et al. (2024). *Hybrid acoustic-text sentiment analysis for call centers*  
- EMO-DB: http://emodb.bilderbar.info  
- RAVDESS: https://zenodo.org/record/1188976  
- German BERT: https://huggingface.co/oliverguhr/german-sentiment-bert
