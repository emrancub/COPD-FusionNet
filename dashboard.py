import os
import uuid # For unique temp files

# --- CRITICAL FIX: OMP Runtime Conflict ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
import torchaudio
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import config
from model import (
    AudioModel_EffNet,
    AudioModel_CRNN,
    TabularModel_MLP,
    TabularModel_Transformer,
    FusionNet_Quad
)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="COPD-FusionNet Diagnostic Tool",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLE & CSS ---
st.markdown("""
<style>
    /* Center the main title */
    .main-header {
        font-size: 3rem; 
        font-weight: 800; 
        color: #d62728; 
        text-align: center; 
        margin-bottom: 10px;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .sub-header-center {
        font-size: 1.2rem; 
        font-weight: 400; 
        color: #555; 
        text-align: center; 
        margin-bottom: 40px;
    }
    .section-header {
        font-size: 1.5rem; 
        font-weight: bold; 
        color: #1f77b4; 
        margin-top: 20px;
        border-bottom: 2px solid #eee;
        padding-bottom: 10px;
    }
    .stButton>button {
        font-size: 1.2rem;
        font-weight: bold;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_feature_list():
    path = config.TABULAR_FEATURES_JSON
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        return json.load(f)

@st.cache_resource
def load_model(model_name, num_feats):
    device = torch.device('cpu')

    if model_name == "COPD-FusionNet":
        model = FusionNet_Quad(num_tab_features=num_feats)
    elif model_name == "Audio_EffNet":
        model = AudioModel_EffNet(pretrained=False)
    elif model_name == "Audio_CRNN":
        model = AudioModel_CRNN()
    elif model_name == "Tab_MLP":
        model = TabularModel_MLP(input_dim=num_feats)
    elif model_name == "Tab_Trans":
        model = TabularModel_Transformer(input_dim=num_feats)
    else:
        return None

    weight_path = os.path.join(config.CHECKPOINTS_DIR, f"{model_name}_fold0.pth")

    if os.path.exists(weight_path):
        try:
            try:
                state = torch.load(weight_path, map_location=device, weights_only=True)
            except:
                state = torch.load(weight_path, map_location=device, weights_only=False)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading {model_name}: {e}")
            return None
    else:
        st.warning(f"Weights for {model_name} not found. Train model first.")
        return None

# --- PROCESSING ---
def process_audio(uploaded_file):
    try:
        # Unique temp file to prevent collisions
        temp_filename = f"temp_{uuid.uuid4().hex}.wav"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())

        waveform, sr = torchaudio.load(temp_filename)

        try: os.remove(temp_filename)
        except: pass

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
            waveform = resampler(waveform)

        if waveform.shape[1] < config.NUM_SAMPLES:
            padding = config.NUM_SAMPLES - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > config.NUM_SAMPLES:
            waveform = waveform[:, :config.NUM_SAMPLES]

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS
        )
        db_transform = torchaudio.transforms.AmplitudeToDB()

        mel_spec = mel_transform(waveform)
        mel_spec = db_transform(mel_spec)
        return mel_spec.unsqueeze(0)

    except Exception as e:
        st.error(f"Audio processing failed: {e}")
        return None

def process_tabular(user_inputs, feature_names):
    vector = np.zeros(len(feature_names), dtype=np.float32)
    for i, feat in enumerate(feature_names):
        if feat == 'age':
            vector[i] = user_inputs.get('Age', 0)
        elif feat == 'is_male':
            vector[i] = 1 if user_inputs.get('Sex') == 'Male' else 0
        elif feat == 'is_female':
            vector[i] = 1 if user_inputs.get('Sex') == 'Female' else 0
        else:
            clean_name = feat.replace('symptoms_', '').replace('_', ' ').title()
            if clean_name in user_inputs.get('Symptoms', []):
                vector[i] = 1.0
    return torch.tensor(vector, dtype=torch.float32).unsqueeze(0)

# --- MAIN ---
def main():
    # --- Header Section ---
    st.markdown('<p class="main-header">🫁 COPD-Fusion-Net: Diagnostic Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header-center">Advanced Multimodal AI for Respiratory Disease Detection</p>', unsafe_allow_html=True)

    feature_names = load_feature_list()
    if not feature_names:
        st.error("Feature list missing. Please run preprocess.py.")
        st.stop()

    with st.sidebar:
        # st.image("https://avatars.githubusercontent.com/u/32496196?v=4", width=150) # Placeholder Lung Icon
        st.header("System Configuration")
        model_choice = st.selectbox(
            "Select AI Model",
            ["COPD-FusionNet", "Audio_EffNet", "Audio_CRNN", "Tab_MLP", "Tab_Trans"]
        )
        st.info(f"Active Model: **{model_choice}**")
        st.markdown("---")
        st.markdown("### Model Capabilities")
        if "Fusion" in model_choice:
            st.success("✅ Audio Analysis")
            st.success("✅ Patient Metadata")
            st.write("Combines all available data for maximum accuracy.")
        elif "Audio" in model_choice:
            st.success("✅ Audio Analysis")
            st.write("Specialized in spectrogram pattern recognition.")
        else:
            st.success("✅ Patient Metadata")
            st.write("Analyzes risk factors and symptoms.")

    # Create two main columns for input
    col1, col2 = st.columns([1, 1], gap="large")

    audio_tensor = None
    tab_tensor = None

    # 1. Audio Section
    with col1:
        st.markdown('<p class="section-header">1. Respiratory Audio Input</p>', unsafe_allow_html=True)
        st.write("Upload a `.wav` recording of patient breathing/coughing.")
        uploaded_file = st.file_uploader("Drag and drop file here", type=["wav"])

        if uploaded_file:
            st.audio(uploaded_file, format='audio/wav')
            audio_tensor = process_audio(uploaded_file)
            if audio_tensor is not None:
                st.success("Audio Processed Successfully")
                with st.expander("View Mel Spectrogram"):
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.imshow(audio_tensor.squeeze().numpy(), aspect='auto', origin='lower', cmap='inferno')
                    ax.set_title("Mel Spectrogram Feature Map")
                    ax.set_ylabel("Frequency")
                    ax.set_xlabel("Time")
                    st.pyplot(fig)

    # 2. Patient Data Section
    with col2:
        st.markdown('<p class="section-header">2. Patient Metadata Input</p>', unsafe_allow_html=True)
        st.write("Enter demographic info and observed symptoms.")

        c_age, c_sex = st.columns(2)
        with c_age:
            age = st.number_input("Patient Age", 0, 120, 65)
        with c_sex:
            sex = st.radio("Sex", ["Male", "Female"], horizontal=True)

        symptom_options = []
        for f in feature_names:
            if f.startswith('symptoms_'):
                symptom_options.append(f.replace('symptoms_', '').replace('_', ' ').title())

        selected_symptoms = st.multiselect("Observed Symptoms", options=sorted(symptom_options))

        # Reactive Data Processing
        user_data = {"Age": age, "Sex": sex, "Symptoms": selected_symptoms}

        if "Fusion" in model_choice or "Tab" in model_choice:
            tab_tensor = process_tabular(user_data, feature_names)
            st.info(f"Vectorized {len(selected_symptoms)} symptoms + demographics.")

    # 3. Analysis Section
    st.markdown("---")
    st.markdown('<p class="section-header" style="text-align:center;">3. Diagnostic Analysis</p>', unsafe_allow_html=True)

    # Center the button using columns
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        run_analysis = st.button("RUN DIAGNOSTIC MODEL", type="primary", use_container_width=True)

    if run_analysis:
        # Validation
        needs_audio = "Audio" in model_choice or "Fusion" in model_choice
        needs_tab = "Tab" in model_choice or "Fusion" in model_choice

        if needs_audio and audio_tensor is None:
            st.error("❌ Error: Please upload an audio file for this model.")
            st.stop()
        if needs_tab and tab_tensor is None:
            st.error("❌ Error: Patient data vector is empty.")
            st.stop()

        model = load_model(model_choice, len(feature_names))

        if model:
            with st.spinner(f"Running inference with {model_choice}..."):
                try:
                    # Inference
                    if model_choice == "FusionNet":
                        logits = model(audio_tensor, tab_tensor)
                    elif "Audio" in model_choice:
                        logits = model(audio_tensor)
                    else:
                        logits = model(tab_tensor)

                    prob = torch.sigmoid(logits).item()

                    # --- RESULTS DISPLAY ---
                    st.markdown("### 🔍 Analysis Results")

                    res_c1, res_c2 = st.columns([1, 1])

                    with res_c1:
                        if prob > 0.5:
                            st.error(f"### ⚠️ COPD DETECTED")
                            st.markdown(f"**Confidence Score:** `{prob:.2%}`")
                            st.write("The model has identified patterns highly consistent with Chronic Obstructive Pulmonary Disease.")
                        else:
                            st.success(f"### ✅ HEALTHY/OTHERS")
                            st.markdown(f"**Confidence Score:** `{(1-prob):.2%}`")
                            st.write("No significant signs of COPD were detected in the provided data.")

                    with res_c2:
                        # Gauge Chart using Matplotlib
                        fig, ax = plt.subplots(figsize=(6, 2))

                        # Color bar
                        gradient = np.linspace(0, 1, 256).reshape(1, -1)
                        ax.imshow(gradient, aspect='auto', cmap='RdYlGn_r', extent=[0, 1, 0, 1])

                        # Pointer
                        ax.axvline(prob, color='black', linewidth=5)
                        ax.plot(prob, 1.1, marker='v', color='black', markersize=15, clip_on=False)

                        # Labels
                        ax.text(0.05, 0.5, "Healthy/Others", color='white', fontweight='bold', va='center')
                        ax.text(0.95, 0.5, "COPD", color='white', fontweight='bold', ha='right', va='center')

                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.axis('off')
                        st.pyplot(fig)

                        # st.caption(f"Model Output Probability: {prob:.4f}")

                except Exception as e:
                    st.error(f"Analysis Error: {e}")

if __name__ == "__main__":
    main()