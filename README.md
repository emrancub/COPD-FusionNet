# COPD-FusionNet: A Multimodal Deep Learning Framework for Robust Respiratory Disease Detection

COPD-FusionNet is a multimodal deep learning framework designed to detect Chronic Obstructive Pulmonary Disease (COPD) by fusing respiratory audio signals (coughs, breathing sounds) with patient metadata (symptoms, demographics).  
👉 **Note:** At this stage, the repository contains **pseudocode / high-level templates** for all main programs. After the final decision from the target journal, we will release the **full implementation** and deploy a public **Streamlit demo (free community URL)**.

This repository contains the official implementation of the paper:

> **“COPD-FusionNet: Robust Multimodal Learning for Respiratory Disease Diagnosis”**  
> Submitted to **IEEE Journal of Biomedical and Health Informatics (J-BHI)**, 2026.

---

## 📌 Table of Contents

- [Overview](#-overview)  
- [System Architecture](#️-system-architecture)  
- [Key Features](#-key-features)  
- [Installation](#-installation)  
- [Usage](#-usage)  
  - [Preprocessing](#1-preprocessing)  
  - [Training](#2-training)  
  - [External Validation](#3-external-validation)  
  - [Ablation Study](#4-ablation-study)  
  - [Visualization](#5-visualization)  
- [Interactive Dashboard](#-interactive-dashboard)  
- [Tutorials](#-tutorials)  
- [Results](#-results)  
- [Citation](#-citation)  
- [Contact](#-contact)  

---

## 🚀 Overview

Detecting COPD early is critical but challenging due to the variability and incompleteness of real-world clinical data. **Unimodal** models (audio-only or tabular-only) often fail when one source of data is noisy, missing, or ambiguous.

**COPD-FusionNet** addresses this limitation by:

- Extracting **spatial features** from audio spectrograms using **EfficientNet-B2**.  
- Capturing **temporal dynamics** using a custom **CRNN** (CNN + Bi-GRU + Attention).  
- Modeling **tabular interactions** using a **Residual MLP** and a **Transformer Encoder**.  
- **Fusing modalities** via a learnable fusion head to produce robust COPD predictions.

---

## 🏗️ System Architecture

The proposed **FusionNet** architecture consists of **four parallel branches** that process audio and tabular data independently before fusing them for final classification:

- **Audio Branch 1 (EffNet-B2)**  
  Learns hierarchical spatial patterns from Mel-spectrograms (e.g., wheezes, crackles).

- **Audio Branch 2 (CRNN with Attention)**  
  Captures temporal evolution of cough/breath events using convolution + Bi-GRU + attention pooling.

- **Tabular Branch 1 (Residual MLP)**  
  Learns additive and low-order interactions from symptom and risk-factor features.

- **Tabular Branch 2 (Transformer Encoder)**  
  Models higher-order and context-dependent interactions among clinical features.

These branches are combined via a **learnable fusion head** (gating + dense layers) to output a COPD risk score.

> _[Placeholder: Insert multimodal architecture figure here]_  

### Pipeline Workflow

1. **Data Collection**  
   - Audio: Respiratory sound database and related sources.  
   - Tabular: Symptom and risk-factor metadata (e.g., smoking history, chronic cough).

2. **Preprocessing**  
   - Audio: Resampling, segmentation, Mel-spectrogram conversion, SpecAugment (time/frequency masking).  
   - Tabular: Normalization, one-hot encoding, and feature engineering.

3. **Training**  
   - 5-fold **stratified cross-validation** for robust evaluation.  
   - Joint training of all four branches with a multimodal loss.

4. **Evaluation**  
   - Standard metrics (AUC, accuracy, sensitivity, specificity, F1, MCC).  
   - Stress-testing with **noise injection** and **missing data** simulation.  
   - External validation on independent datasets.

---

## ✨ Key Features

- **Multimodal Fusion**  
  Combines CNNs, RNNs, Transformers, and MLPs for holistic COPD screening from audio + metadata.

- **Robustness**  
  Evaluated under varying noise levels (e.g., SNR 0–20 dB) and missing data conditions; FusionNet degrades more gracefully than unimodal baselines.

- **Explainability (XAI)**  
  - **SHAP Analysis:** Explains which symptoms and risk factors drive predictions.  
  - **Grad-CAM Saliency:** Highlights key regions in spectrograms (e.g., wheezes, abnormal patterns).

- **User Dashboard**  
  Fully interactive **Streamlit** web app for real-time demonstrations and clinical prototyping (public URL will be shared upon code release).

---

## 💻 Installation

> ⚠️ Currently, scripts in this repository are **pseudocode / templates**. Commands below reflect the intended workflow for the full release.

```bash
# Clone the repository
git clone https://github.com/YourUsername/COPD-FusionNet.git
cd COPD-FusionNet

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
````

**Main dependencies (planned):**
`torch`, `torchaudio`, `timm`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`, `streamlit`.

---

## 🛠️ Usage

> Note: The following commands describe the **intended pipeline**. In the current version, you will find pseudocode and high-level script structure.

### 1. Preprocessing

Prepare the raw audio and tabular data for training. This script will:

* Generate Mel-spectrograms.
* Build 5-fold stratified splits for cross-validation.

```bash
python preprocess.py
```

---

### 2. Training

Train the **FusionNet** model alongside baseline models:

* Audio-only (EffNet-B2, CRNN).
* Tabular-only (ResMLP, Transformer).

```bash
python train.py
```

**Outputs (in full release):**

* Trained weights in `checkpoints/`
* Metrics and logs in `results/`

---

### 3. External Validation

Evaluate trained models on an unseen external dataset and perform robustness tests:

* Noise injection experiments.
* Missing-data simulations.

```bash
python external_validation.py
```

---

### 4. Ablation Study

Run a component-wise ablation to quantify the contribution of:

* CRNN branch
* Transformer branch
* Data augmentation (SpecAugment)
* Fusion mechanism

```bash
python run_ablation.py
```

---

### 5. Visualization

Generate publication-quality plots and interpretability visualizations:

```bash
# ROC, PR, calibration, confusion matrices, t-SNE
python plot_results.py

# SHAP analysis for tabular features
python shap_tabular.py

# Grad-CAM saliency maps on spectrograms
python audio_saliency.py
```

---

## 🖥️ Interactive Dashboard

We provide a **Streamlit** dashboard for real-time COPD-FusionNet demonstration:

```bash
streamlit run dashboard.py
```

The dashboard will allow you to:

* **Upload** respiratory audio files (`.wav`).
* **Input** patient-level metadata (age, sex, symptoms, risk factors).
* **View**:

  * Mel-spectrograms of the input audio.
  * Predicted COPD risk and confidence.
  * Saliency overlays on the spectrogram.
  * Symptom/risk-factor contribution plots.

> After journal decisions, we plan to deploy the dashboard on **Streamlit Community Cloud** and share a public URL here.

---

## 📖 Tutorials

### COPD-FusionNet Dashboard

This repository contains the code for the **COPD-FusionNet Dashboard**, an interactive tool for Chronic Obstructive Pulmonary Disease (COPD) detection using multimodal AI. The dashboard demonstrates how to use respiratory audio data and patient metadata for predicting COPD risk, as well as visualizing the results through intuitive plots and analyses.

### 🖥️ Interactive Dashboard

We provide a Streamlit dashboard for real-time COPD-FusionNet demonstration:

```bash
streamlit run dashboard.py
````

The dashboard will allow you to:

* **Upload respiratory audio files (.wav)** for analysis.
* **Input patient-level metadata** (age, sex, symptoms, risk factors).
* **View**:

  * Mel-spectrograms of the input audio.
  * Predicted COPD risk and confidence.
  * Saliency overlays on the spectrogram.
  * Symptom/risk-factor contribution plots.

### After journal decisions, we plan to deploy the dashboard on Streamlit Community Cloud and share a public URL here.


* **Step-by-step tutorial notebooks (planned)** explaining how to:

  * Prepare data
  * Run training and evaluation
  * Use the Streamlit dashboard in practice

We will also add **video walkthroughs** that demonstrate:

* How to launch and interact with the dashboard.
* How to interpret the visual explanations (SHAP, Grad-CAM).
* End-to-end usage from raw audio to COPD risk prediction.

### You can refer to the full **COPD-FusionNet Dashboard Tutorial PDF** [here](COPD-FusionNet_Dashboard_Tutorial.pdf) for further instructions.

### Watch the **YouTube tutorial** below to get a hands-on introduction:

[Watch the tutorial](https://youtu.be/K4Qa2jtJcdc?autoplay=1)

---
## 📊 Results

Extensive experiments in the associated manuscript show that **FusionNet** outperforms unimodal baselines, especially under noisy audio conditions.

Example summary (internal 5-fold cross-validation):

| Model                 | Modality   | ***    | ***   | ***   | ***   | ***   |
|-----------------------| ---------- |------------------| ----------- | ----------- |----------- |----------- |
| Audio (EffNet-B2)     | Audio      | ***              | ***          | ***       | ***   | ***   |
| Audio (CRNN)          | Audio      | ***       | ***        |***       | ***   | ***   |
| Tabular (ResMLP)      | Tabular    | ***      | ***         |***       | ***   | ***   |
| Tabular (Transformer) | Tabular    | ***      | ***         |***       | ***   | ***   |
| **COPD-FusionNet (Ours)**  | Multimodal | ***    | ***       |***       | ***   | ***   |

> *Note:* These values summarize internal experiments as described in the paper. See the manuscript and `results/` (in the full code release) for full metrics: accuracy, F1, sensitivity, specificity, MCC, external validation, and ablation details.

---

## 📜 Citation

If you use this framework or ideas from this work, please cite:

> The final BibTeX entry and author list will be updated after acceptance.

---

## 📧 Contact

For questions, feedback, or collaboration:

* **Md Emran Hasan** – [mdemranhasan@njust.edu.cn](mailto:mdemranhasan@njust.edu.cn)

You are welcome to open an issue or pull request if you have suggestions or would like to contribute improvements. Full source code and a live Streamlit demo will be released after the journal’s decision.
