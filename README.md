# Fine-tuned Spark-TTS: German Emotional Speech Synthesis

This repository contains a fine-tuned implementation of the **Spark-TTS (0.5B)** model, specialized for German speech synthesis with advanced support for **emotional cues** and **non-verbal audio tokens**.

The model was fine-tuned using LoRA (Low-Rank Adaptation) on a curated German dataset containing high-quality audio with diverse emotional expressions and non-verbal cues.

## 🚀 Key Highlights

*   **57.14% Loss Improvement:** Reduced test loss from **10.0074** (Base) to **4.2891** (Fine-tuned).
*   **Emotional Support:** Handles stylistic tags like `[happy]`, `[angry]`, and `[thoughtful]`.
*   **Non-Verbal Tokens:** Accurately synthesizes non-speech sounds like `[sighs]`, `[laughter]`, `[yawn]`, and `[growl]`.
*   **Optimized for RTX 5070:** Configured for 12GB VRAM with efficient 4-bit loading and gradient accumulation.

## 🛠️ Installation & Setup

### 1. Environment Setup
```powershell
# Create and activate virtual environment
python -m venv spark_tts_venv
.\spark_tts_venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.1+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies and setup Spark-TTS
python setup_environment.py
```

### 2. Model Download
```powershell
python 1_download_model.py
```

## 🏋️ Training Pipeline

The project follows a systematic 7-step pipeline:

1.  **Data Preparation:** `python 2_data_preparation.py` (Tokenizes German text/audio).
2.  **Model Setup:** `python 3_model_setup.py` (Applies LoRA adapters).
3.  **Hyperparameter Tuning:** `python 4_hyperparameter_tuning.py` (Grid search for LR, Rank, Alpha).
4.  **Full Training:** `python 5_train_full.py` (Uses optimal parameters: LR 0.0005, Rank 64, Alpha 64).
5.  **Plots & Verification:** `python generate_test_plot.py` (Visualizes loss convergence).
6.  **Comparative Evaluation:** `python 7_comparative_evaluation.py` (Benchmarks against base model).

## 🔊 Inference Examples

Use `inference.py` to generate speech. The script handles the specific token extraction required for the BiCodec tokenizer.

```python
# Example Prompt
prompt = "[sighs] Endlich Wochenende, ich brauch echt mal Pause! [thoughtful]"
```

**Run Inference:**
```powershell
python inference.py
```

## 📊 Performance Comparison

| Metric | Base Model (0.5B) | Fine-tuned (German) | Improvement |
| :--- | :--- | :--- | :--- |
| **Test Loss** | 10.0074 | **4.2891** | **57.14%** |
| **German Prosody** | Basic | Advanced | High |
| **Non-Verbal Support**| Minimal | Complete | N/A |

## 📁 Directory Structure

*   `best_model/`: Saved LoRA adapters and config.
*   `outputs/`: Training logs and loss visualizations.
*   `processed_dataset/`: Tokenized Arrow dataset.
*   `Spark-TTS/`: Core model architecture and modules.
*   `output_audio/`: Generated WAV samples from inference.

## 📜 Credits
Developed as part of a German TTS fine-tuning project using the Spark-TTS architecture by SparkAudio.
