import os
import glob
import io
import streamlit as st
import numpy as np
import torch
from pydub import AudioSegment
import soundfile as sf
from kokoro_onnx import Kokoro
from kokoro_onnx.config import SAMPLE_RATE
from misaki import en, zh, ja, espeak
from misaki.espeak import EspeakG2P
from time import time
import requests

st.set_page_config(
    page_title="Kokoro Voice Composer",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    .block-container { padding-top: 1rem; padding-bottom: 0rem; max-width: 1200px; }
    h1 { font-size: 1.8rem !important; margin-bottom: 0.5rem !important; }
    h2 { font-size: 1.4rem !important; margin-top: 0.8rem !important; margin-bottom: 0.5rem !important; }
    h3 { font-size: 1.2rem !important; margin-top: 0.8rem !important; margin-bottom: 0.3rem !important; }
    div[data-testid="column"] > div > div > div > div > div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
    div[data-testid="stVerticalBlock"] > div { padding-bottom: 0; }
    div[data-testid="stHorizontalBlock"] { gap: 1rem; }
    .stMultiSelect { margin-bottom: 0.2rem; }
    div.stButton > button { margin-top: 0.2rem; }
    hr { margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# Dynamically set BASE_DIR (works for Hugging Face Spaces or local)
#BASE_DIR = "/data" if os.path.exists("/data") else os.getcwd() # for Hugging Face Spaces
BASE_DIR = os.getcwd()
GENERATED_DIR = os.path.join(BASE_DIR, "generated")
MODEL_V1 = os.path.join(BASE_DIR, "kokoro-v1.0.onnx")
MODEL_V1_ZH = os.path.join(BASE_DIR, "kokoro-v1.1-zh.onnx")
VOICES_V1 = os.path.join(BASE_DIR, "voices-v1.0.bin")
VOICES_V1_ZH = os.path.join(BASE_DIR, "voices-v1.1-zh.bin")

# Model file URLs
MODEL_URLS = {
    MODEL_V1: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    MODEL_V1_ZH: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.1-zh.onnx",
    VOICES_V1: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
    VOICES_V1_ZH: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.1-zh.bin"
}

# Download file if it doesn’t exist
def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        st.info(f"Downloading {os.path.basename(dest_path)}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success(f"Downloaded {os.path.basename(dest_path)} to {dest_path}")

# Create directories and ensure model files are present
def initialize_directories():
    os.makedirs(GENERATED_DIR, exist_ok=True)
    for path, url in MODEL_URLS.items():
        download_file(url, path)

# Language options
LANGUAGE_MAP = {
    'American English': {'code': 'en-us', 'g2p': lambda: en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False)), 'model': MODEL_V1, 'voices': VOICES_V1},
    'British English': {'code': 'en-gb', 'g2p': lambda: en.G2P(trf=False, british=True, fallback=espeak.EspeakFallback(british=True)), 'model': MODEL_V1, 'voices': VOICES_V1},
    'Spanish': {'code': 'es', 'g2p': lambda: EspeakG2P(language='es'), 'model': MODEL_V1, 'voices': VOICES_V1},
    'French': {'code': 'fr-fr', 'g2p': lambda: EspeakG2P(language='fr-fr'), 'model': MODEL_V1, 'voices': VOICES_V1},
    'Italian': {'code': 'it', 'g2p': lambda: EspeakG2P(language='it'), 'model': MODEL_V1, 'voices': VOICES_V1},
    'Brazilian Portuguese': {'code': 'pt-br', 'g2p': lambda: EspeakG2P(language='pt-br'), 'model': MODEL_V1, 'voices': VOICES_V1},
    'Hindi': {'code': 'hi', 'g2p': lambda: EspeakG2P(language='hi'), 'model': MODEL_V1, 'voices': VOICES_V1},
    'Chinese': {'code': 'zh', 'g2p': lambda: zh.ZHG2P(), 'model': MODEL_V1_ZH, 'voices': VOICES_V1_ZH},
    'Japanese': {'code': 'ja', 'g2p': lambda: ja.JAG2P(), 'model': MODEL_V1_ZH, 'voices': VOICES_V1_ZH}
}

# KokoroStyleTransfer class
class KokoroStyleTransfer:
    def __init__(self, model_path, voices_path):
        self.kokoro = Kokoro(model_path, voices_path)
        self.voice_cache = {}

    def load_original_voice(self, voice_name: str) -> np.ndarray:
        return self.kokoro.get_voice_style(voice_name)

    def load_generated_voice(self, voice_name: str) -> np.ndarray:
        if voice_name in self.voice_cache:
            return self.voice_cache[voice_name]
        bin_path = os.path.join(GENERATED_DIR, f"{voice_name}.bin")
        if os.path.exists(bin_path):
            with np.load(bin_path) as data:
                voice_array = data["voice"]
            st.write(f"Loaded generated voice: {bin_path}, shape: {voice_array.shape}")
            self.voice_cache[voice_name] = voice_array
            return voice_array
        raise ValueError(f"Generated voice {voice_name} not found in {GENERATED_DIR}")

    def generate_speech(self, text: str, voice: str | np.ndarray, g2p, speed: float = 0.85, lang: str = "en-us"):
        try:
            phonemes, _ = g2p(text)
            samples, sample_rate = self.kokoro.create(phonemes, voice=voice, speed=speed, lang=lang, is_phonemes=True)
            return {"success": True, "samples": samples, "sample_rate": sample_rate}
        except Exception as e:
            return {"success": False, "error": str(e)}

# Transformation functions
def blend_voices(voice_list):
    if len(voice_list) == 1 and len(voice_list[0]) == 1:
        return voice_list[0][0]
    voices = [item[0] for item in voice_list]
    weights = [item[1] if len(item) > 1 else 1.0 for item in voice_list]
    total_weight = sum(weights)
    if total_weight == 0:
        weights = [1.0 / len(voices) for _ in voices]
    else:
        weights = [w / total_weight for w in weights]
    blended = np.zeros_like(voices[0])
    for voice, weight in zip(voices, weights):
        blended += voice * weight
    st.write(f"Blended {len(voices)} voices with weights {weights}, shape: {blended.shape}")
    return blended

def save_voice_bin(voice_tensor, filename):
    if not filename.endswith('.bin'):
        filename += '.bin'
    save_path = os.path.join(GENERATED_DIR, filename)
    np.savez(save_path, voice=voice_tensor)
    os.rename(save_path + '.npz', save_path)  # Rename to .bin
    st.write(f"Saved voice to {save_path}, shape: {voice_tensor.shape}")
    return save_path

def save_voice_pt(voice_tensor, filename):
    if not filename.endswith('.pt'):
        filename += '.pt'
    save_path = os.path.join(BASE_DIR, filename)
    voice_tensor_torch = torch.from_numpy(voice_tensor).unsqueeze(1)
    torch.save(voice_tensor_torch, save_path)
    st.write(f"Saved voice to {save_path}, shape: {voice_tensor_torch.shape}")
    return save_path

def transfer_style(voice, style, perc=0.3):
    perc = max(0.0, min(1.0, 1 - perc))
    split_point = int(voice.shape[-1] * perc)
    result = voice.copy()
    result[..., split_point:] = style[..., split_point:]
    return result

def add_noise(embedding, strength=0.26, seed=None):
    if seed is not None:
        np.random.seed(seed)
    feature_noise = np.random.randn(embedding.shape[-1]) * strength
    noisy_embedding = embedding.copy()
    for i in range(embedding.shape[0]):
        noisy_embedding[i, :] += feature_noise
    return noisy_embedding

def pca_perturbation(embedding, strength=0.1, n_components=5):
    if embedding.shape[1] == 1:
        embedding_squeezed = embedding[:, 0, :]
    else:
        embedding_squeezed = embedding
    flat_embedding = embedding_squeezed.T
    n_samples, n_features = flat_embedding.shape
    n_components = min(n_components, n_samples)
    mean = np.mean(flat_embedding, axis=1)
    centered = flat_embedding - mean[:, np.newaxis]
    cov = np.cov(centered, rowvar=True)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    top_components = eigenvectors[:, idx[:n_components]]
    perturbation = np.random.randn(n_components) * strength
    perturbation_vector = np.dot(top_components, perturbation)
    perturbed_embedding = embedding.copy()
    perturbed_embedding += perturbation_vector[np.newaxis, np.newaxis, :]
    return perturbed_embedding

def spectral_envelope_modification(base_voice, formant_shift=0.1, brightness=0.8):
    result = base_voice.copy()
    feature_dim = base_voice.shape[-1]
    shift_amount = int(feature_dim * formant_shift)
    brightness_slope = np.linspace(0, brightness, feature_dim)
    for i in range(base_voice.shape[0]):
        features = result[i, :]
        if shift_amount > 0:
            shifted = np.concatenate([features[-shift_amount:], features[:-shift_amount]])
        else:
            shift = abs(shift_amount)
            shifted = np.concatenate([features[shift:], features[:shift]])
        features = features * 0.3 + shifted * 0.7
        features *= (1.0 + brightness_slope)
        result[i, :] = features
    return result

# Load voices
def load_voices(voices_path):
    voices = {'original': [], 'generated': []}
    with np.load(voices_path) as data:
        voices['original'] = sorted(list(data.keys()))
    bin_files = glob.glob(os.path.join(GENERATED_DIR, '*.bin'))
    voices['generated'] = [os.path.splitext(os.path.basename(f))[0] for f in bin_files]
    return voices

# Cache style transfer
@st.cache_resource
def get_style_transfer(model_path, voices_path):
    return KokoroStyleTransfer(model_path, voices_path)

# Main UI
def main():
    initialize_directories()
    
    st.title("Kokoro Voice Composer")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        lang_speed_cols = st.columns(2)
        with lang_speed_cols[0]:
            st.write("<small>Language</small>", unsafe_allow_html=True)
            language = st.selectbox("Language", options=list(LANGUAGE_MAP.keys()), index=0, label_visibility="collapsed", key="language_select")
        with lang_speed_cols[1]:
            st.write("<small>Speed</small>", unsafe_allow_html=True)
            speed = st.slider("Speed", min_value=0.5, max_value=1.2, value=0.85, step=0.05, label_visibility="collapsed", key="speed_slider")
            
        lang_config = LANGUAGE_MAP[language]
        voices = load_voices(lang_config['voices'])
        style_transfer = get_style_transfer(lang_config['model'], lang_config['voices'])
        g2p = lang_config['g2p']()
        
        st.header("Voice Blending")
        
        original_selected = st.multiselect("Original Voices", voices['original'], key="original_select")
        original_weights = {}
        for voice in original_selected:
            original_weights[voice] = st.slider(f"Weight for {voice}", min_value=0.0, max_value=1.0, value=1.0, key=f"ow_{voice}")
        
        generated_selected = st.multiselect("Generated Voices", voices['generated'], key="generated_select")
        generated_weights = {}
        for voice in generated_selected:
            generated_weights[voice] = st.slider(f"Weight for {voice}", min_value=0.0, max_value=1.0, value=1.0, key=f"gw_{voice}")
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        text = st.text_area("Text to Generate", "The lawyer was a man of a rugged countenance that was never lighted by a smile; cold, scanty and embarrassed in discourse; backward in sentiment; lean, long, dusty, dreary and yet somehow lovable.", height=100, key="text_input")
        
        play_button = st.button("Generate", use_container_width=True, key="play_button")
        
        # Audio player placeholder for persistent audio
        audio_placeholder = st.empty()

    with col2:
        st.subheader("Seed Transform")
        noise_enabled = st.checkbox("Enable Seed Transform", value=False, key="noise_checkbox")
        if noise_enabled:
            noise_strength = st.slider("Strength", 0.0, 0.5, 0.26, key="noise_strength")
            noise_seed = st.number_input("Seed", min_value=0, value=42, step=1, key="noise_seed")
        else:
            noise_strength = 0.26
            noise_seed = 42
        
        st.subheader("PCA Perturbation")
        pca_enabled = st.checkbox("Enable PCA Perturbation", value=False, key="pca_checkbox")
        if pca_enabled:
            pca_strength = st.slider("Strength", -3.5, 3.5, -0.5, step=0.1, key="pca_strength")
            n_components = st.slider("Number of Components", 1, 22, 2, step=1, key="n_components")
        else:
            pca_strength = 0.1
            n_components = 5
        
        st.subheader("Spectral Transform")
        spectral_enabled = st.checkbox("Enable Spectral Transform", value=False, key="spectral_checkbox")
        if spectral_enabled:
            formant_shift = st.slider("Formant Shift", -0.5, 0.5, 0.1, step=0.05, key="formant_shift")
            brightness = st.slider("Brightness", 0.0, 2.0, 0.8, step=0.1, key="brightness")
        else:
            formant_shift = 0.1
            brightness = 0.8
        
        st.subheader("Inflection Transfer")
        style_enabled = st.checkbox("Enable Inflection Transfer", value=False, key="style_checkbox")
        if style_enabled:
            all_voices = voices['original'] + voices['generated']
            inflection_voice = st.selectbox("Inflection Voice", all_voices, key="inflection_select")
            style_perc = st.slider("Style Percentage", 0.0, 1.0, 0.3, key="style_perc")
        else:
            inflection_voice = None
            style_perc = 0.3
        
        st.markdown('<hr>', unsafe_allow_html=True)
        
        name = st.text_input("Enter name for the voice:", key="name_input")
        
        download_col1, download_col2 = st.columns(2)
        with download_col1:
            save_button = st.button("Save", use_container_width=True, key="save_button")
            download_pt_button = st.button("Download .pt", use_container_width=True, key="download_pt_button")
        with download_col2:
            download_mp3_button = st.button("Download MP3", use_container_width=True, key="download_mp3_button")
            download_bin_button = st.button("Download .bin", use_container_width=True, key="download_bin_button")
        
        download_placeholder = st.empty()

    # Initialize session state for audio
    if 'audio_bytes' not in st.session_state:
        st.session_state.audio_bytes = None

    # Render persistent audio player with autoplay
    if st.session_state.audio_bytes and not play_button:
        with audio_placeholder.container():
            st.audio(st.session_state.audio_bytes, format="audio/mp3", start_time=0)

    # Processing Logic
    selected_voices = []
    
    for voice, weight in original_weights.items():
        try:
            tensor = style_transfer.load_original_voice(voice)
            selected_voices.append((tensor, weight))
        except ValueError as e:
            st.error(f"Error loading original voice {voice}: {str(e)}")
    
    for voice, weight in generated_weights.items():
        try:
            tensor = style_transfer.load_generated_voice(voice)
            selected_voices.append((tensor, weight))
        except ValueError as e:
            st.error(f"Error loading generated voice {voice}: {str(e)}")

    if selected_voices:
        current_voice = blend_voices(selected_voices)
        
        if noise_enabled:
            current_voice = add_noise(current_voice, noise_strength, noise_seed)
        
        if pca_enabled:
            current_voice = pca_perturbation(current_voice, pca_strength, n_components)
        
        if spectral_enabled:
            current_voice = spectral_envelope_modification(current_voice, formant_shift, brightness)
        
        if style_enabled and inflection_voice:
            try:
                if inflection_voice in voices['original']:
                    style_tensor = style_transfer.load_original_voice(inflection_voice)
                else:
                    style_tensor = style_transfer.load_generated_voice(inflection_voice)
                current_voice = transfer_style(current_voice, style_tensor, style_perc)
            except ValueError as e:
                st.error(f"Error loading inflection voice {inflection_voice}: {str(e)}")

        if play_button:
            with st.spinner("Generating audio..."):
                if not (noise_enabled or pca_enabled or spectral_enabled or style_enabled) and len(original_selected) == 1 and not generated_selected:
                    voice_to_use = original_selected[0]
                else:
                    voice_to_use = current_voice
                result = style_transfer.generate_speech(text, voice_to_use, g2p, speed, lang=lang_config['code'])
                
                if result["success"]:
                    # Process audio in memory (no temp file)
                    wav_buffer = io.BytesIO()
                    sf.write(wav_buffer, result["samples"], result["sample_rate"], format="wav")
                    wav_buffer.seek(0)
                    
                    # Convert to MP3 in memory
                    audio = AudioSegment.from_file(wav_buffer, format="wav")
                    mp3_buffer = io.BytesIO()
                    audio.export(mp3_buffer, format="mp3", bitrate="128k", parameters=["-q:a", "0"])
                    mp3_bytes = mp3_buffer.getvalue()
                    wav_buffer.close()
                    mp3_buffer.close()
                    
                    # Store in session state and play with autoplay
                    st.session_state.audio_bytes = mp3_bytes
                    
                    with audio_placeholder.container():
                        st.audio(mp3_bytes, format="audio/mp3", start_time=0)
                    st.success("Audio generated")

        if save_button:
            if name:
                try:
                    save_path = save_voice_bin(current_voice, name)
                    if os.path.exists(save_path):
                        st.success(f"Voice '{name}' successfully saved to {save_path}!", icon="✅")
                    else:
                        st.error(f"Failed to save '{name}': File was not created.", icon="❌")
                except Exception as e:
                    st.error(f"Failed to save '{name}': {str(e)}", icon="❌")
                st.rerun()
            else:
                st.warning("Please enter a name for the voice before saving.", icon="⚠️")
                
        if download_bin_button:
            if name:
                with st.spinner("Preparing BIN file..."):
                    bin_buffer = io.BytesIO()
                    np.savez(bin_buffer, voice=current_voice)
                    bin_file_bytes = bin_buffer.getvalue()
                    bin_buffer.close()
                    with download_placeholder.container():
                        st.download_button(
                            label=f"Download {name}.bin",
                            data=bin_file_bytes,
                            file_name=f"{name}.bin",
                            mime="application/octet-stream",
                            key="actual_bin_download"
                        )
            else:
                st.warning("Please enter a name for the voice before downloading.")
                
        if download_pt_button:
            if name:
                with st.spinner("Preparing PT file..."):
                    voice_tensor_torch = torch.from_numpy(current_voice).unsqueeze(1)
                    pt_buffer = io.BytesIO()
                    torch.save(voice_tensor_torch, pt_buffer)
                    pt_buffer.seek(0)
                    pt_file_bytes = pt_buffer.getvalue()
                    pt_buffer.close()
                    with download_placeholder.container():
                        st.download_button(
                            label=f"Download {name}.pt",
                            data=pt_file_bytes,
                            file_name=f"{name}.pt",
                            mime="application/octet-stream",
                            key="actual_pt_download"
                        )
            else:
                st.warning("Please enter a name for the voice before downloading.")
                
        if download_mp3_button:
            if st.session_state.audio_bytes:
                if name:
                    mp3_filename = f"{name}.mp3"
                else:
                    mp3_filename = "generated_audio.mp3"
                with download_placeholder.container():
                    st.download_button(
                        label=f"Download {mp3_filename}",
                        data=st.session_state.audio_bytes,
                        file_name=mp3_filename,
                        mime="audio/mp3",
                        key="download_mp3"
                    )
            else:
                st.warning("Please generate audio first before downloading MP3.")

    else:
        if play_button or save_button or download_bin_button or download_pt_button or download_mp3_button:
            st.warning("Please select at least one voice.")

if __name__ == "__main__":
    main()
