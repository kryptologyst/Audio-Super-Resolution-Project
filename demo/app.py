"""
Streamlit Demo for Audio Super-Resolution

This demo provides an interactive interface for testing audio super-resolution models.
Users can upload audio files, process them with different models, and compare results.
"""

import streamlit as st
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import create_model
from metrics import AudioMetrics
from utils import get_device, load_config


# Page configuration
st.set_page_config(
    page_title="Audio Super-Resolution Demo",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Privacy disclaimer
st.warning("""
**PRIVACY AND ETHICS DISCLAIMER**

This is a research and educational demonstration. This software is NOT intended for:
- Biometric identification or verification systems
- Voice cloning or impersonation without explicit consent
- Any form of surveillance or monitoring
- Commercial deployment without proper ethical review

By using this demo, you agree to use it responsibly and in accordance with applicable laws and ethical guidelines.
""")

# Title and description
st.title("🎵 Audio Super-Resolution Demo")
st.markdown("""
This demo allows you to test different audio super-resolution models on your own audio files.
Upload an audio file and select a model to enhance its quality by recovering high-frequency details.
""")

# Sidebar for model selection and settings
st.sidebar.header("Model Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    ["SRCNN", "UNet", "WaveNet"],
    help="Choose the neural network architecture for super-resolution"
)

# Load model configuration
config_path = Path(__file__).parent.parent / 'configs' / f'{model_type.lower()}.yaml'
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
else:
    st.error(f"Configuration file not found: {config_path}")
    st.stop()

# Model parameters
st.sidebar.subheader("Model Parameters")
upsampling_factor = st.sidebar.slider(
    "Upsampling Factor",
    min_value=2,
    max_value=8,
    value=config['model'].get('upsampling_factor', 2),
    help="Factor by which to increase the sample rate"
)

# Audio processing parameters
st.sidebar.subheader("Audio Processing")
target_sample_rate = st.sidebar.selectbox(
    "Target Sample Rate",
    [16000, 22050, 32000, 44100, 48000],
    index=2,  # Default to 32000
    help="Target sample rate for super-resolution"
)

# Load model
@st.cache_resource
def load_model(model_type, config):
    """Load and cache the model."""
    device = get_device()
    model = create_model(model_type.lower(), config['model'])
    model.to(device)
    model.eval()
    return model, device

try:
    model, device = load_model(model_type, config)
    st.sidebar.success(f"✅ {model_type} model loaded successfully")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 Upload Audio")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Upload an audio file to enhance with super-resolution"
    )
    
    if uploaded_file is not None:
        # Load uploaded audio
        try:
            audio_data, sr = librosa.load(uploaded_file, sr=None)
            
            st.success(f"✅ Audio loaded successfully")
            st.info(f"**Original:** {len(audio_data)} samples, {sr} Hz, {len(audio_data)/sr:.2f}s")
            
            # Display original audio
            st.subheader("🎧 Original Audio")
            st.audio(uploaded_file, format='audio/wav')
            
            # Create waveform plot
            fig = go.Figure()
            time_axis = np.linspace(0, len(audio_data)/sr, len(audio_data))
            fig.add_trace(go.Scatter(x=time_axis, y=audio_data, name='Original'))
            fig.update_layout(
                title="Original Audio Waveform",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to load audio file: {e}")
            st.stop()

with col2:
    st.header("🔧 Processing")
    
    if uploaded_file is not None:
        # Process button
        if st.button("🚀 Enhance Audio", type="primary"):
            with st.spinner("Processing audio..."):
                try:
                    # Convert to target sample rate
                    if sr != target_sample_rate:
                        audio_resampled = librosa.resample(
                            audio_data, orig_sr=sr, target_sr=target_sample_rate
                        )
                    else:
                        audio_resampled = audio_data
                    
                    # Create low-resolution version (simulate)
                    low_sr = target_sample_rate // upsampling_factor
                    audio_low = librosa.resample(
                        audio_resampled, orig_sr=target_sample_rate, target_sr=low_sr
                    )
                    
                    # Convert to tensor
                    audio_tensor = torch.from_numpy(audio_low).float().unsqueeze(0).unsqueeze(0)
                    audio_tensor = audio_tensor.to(device)
                    
                    # Run model
                    with torch.no_grad():
                        enhanced_tensor = model(audio_tensor)
                        enhanced_audio = enhanced_tensor.squeeze().cpu().numpy()
                    
                    # Ensure same length as target
                    min_length = min(len(enhanced_audio), len(audio_resampled))
                    enhanced_audio = enhanced_audio[:min_length]
                    target_audio = audio_resampled[:min_length]
                    
                    st.success("✅ Audio enhancement completed!")
                    
                    # Display enhanced audio
                    st.subheader("🎵 Enhanced Audio")
                    
                    # Convert to bytes for audio player
                    enhanced_bytes = io.BytesIO()
                    sf.write(enhanced_bytes, enhanced_audio, target_sample_rate, format='WAV')
                    enhanced_bytes.seek(0)
                    
                    st.audio(enhanced_bytes, format='audio/wav')
                    
                    # Create comparison plot
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Low-Resolution Input', 'Enhanced Output'),
                        vertical_spacing=0.1
                    )
                    
                    time_low = np.linspace(0, len(audio_low)/low_sr, len(audio_low))
                    time_enhanced = np.linspace(0, len(enhanced_audio)/target_sample_rate, len(enhanced_audio))
                    
                    fig.add_trace(
                        go.Scatter(x=time_low, y=audio_low, name='Low-Res Input'),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=time_enhanced, y=enhanced_audio, name='Enhanced Output'),
                        row=2, col=1
                    )
                    
                    fig.update_layout(
                        title="Audio Enhancement Comparison",
                        height=600,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Compute metrics
                    st.subheader("📊 Quality Metrics")
                    
                    metrics_calc = AudioMetrics(sample_rate=target_sample_rate)
                    metrics = metrics_calc.compute_all_metrics(enhanced_audio, target_audio)
                    
                    # Display metrics in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("PESQ", f"{metrics['pesq']:.3f}")
                        st.metric("STOI", f"{metrics['stoi']:.3f}")
                    
                    with col2:
                        st.metric("SI-SDR (dB)", f"{metrics['si_sdr']:.2f}")
                        st.metric("SNR (dB)", f"{metrics['snr']:.2f}")
                    
                    with col3:
                        st.metric("Spectral MSE", f"{metrics['spectral_mse']:.6f}")
                        st.metric("Spectral L1", f"{metrics['spectral_l1']:.6f}")
                    
                    # Download enhanced audio
                    st.subheader("💾 Download Enhanced Audio")
                    st.download_button(
                        label="Download Enhanced Audio",
                        data=enhanced_bytes.getvalue(),
                        file_name=f"enhanced_{uploaded_file.name}",
                        mime="audio/wav"
                    )
                    
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    st.exception(e)

# Additional information
st.markdown("---")
st.markdown("""
### About Audio Super-Resolution

Audio super-resolution aims to enhance low-quality audio by recovering high-frequency details
that are lost during downsampling or compression. This demo uses deep learning models to:

- **SRCNN**: Lightweight CNN for fast processing
- **UNet**: U-shaped architecture with skip connections for detail preservation  
- **WaveNet**: Dilated convolutions for capturing long-range dependencies

### How It Works

1. **Upload** an audio file (WAV, MP3, FLAC, M4A)
2. **Select** a model and processing parameters
3. **Process** the audio to create a low-resolution version
4. **Enhance** using the selected neural network model
5. **Compare** results and download the enhanced audio

### Technical Details

- Models are trained to map low-resolution audio features to high-resolution features
- Evaluation metrics include PESQ, STOI, SI-SDR, and spectral measures
- Processing is done in real-time using PyTorch models
- All processing happens locally - no data is sent to external servers
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Audio Super-Resolution Demo | Research & Educational Use Only</p>
    <p>Built with PyTorch, Streamlit, and Librosa</p>
</div>
""", unsafe_allow_html=True)
