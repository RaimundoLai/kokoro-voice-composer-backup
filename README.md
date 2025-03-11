# Kokoro Voice Composer

Kokoro Voice Composer is a Python Streamlit UI for creating and customizing text-to-speech (TTS) voices for the Kokoro TTS model. Blend existing voices, apply transformations, and export them for use with `kokoro` and `kokoro-onnx`. It will run on CPU or GPU.

[View the example](https://huggingface.co/spaces/alasdairforsythe/kokoro-voice-composer) on Hugging Face Spaces.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/alasdairforsythe/kokoro-voice-composer.git
   cd kokoro-voice-composer
   ```

2. Install the required Python packages using pip:
   ```bash
   pip install kokoro-onnx onnxruntime misaki[en,zh,ja] streamlit pydub soundfile torch numpy ordered_set pypinyin cn2an jieba fugashi jaconv mojimoji
   ```
   For GPU add `kokoro-onnx[gpu]`

3. Ensure `eSpeak` is installed (e.g., on Ubuntu: `sudo apt install espeak-ng`).
   For GPU also install `portaudio19-dev`

4. Run the application:
   ```bash
   streamlit run voice-composer.py --server.port 5544 --server.fileWatcherType none
   ```
   The app will automatically download available voice models upon first startup. Refresh the page to get rid of the notifications.

---

## Features
- **Voice Blending**: Combine multiple voice embeddings with adjustable weights.
- **Seed Transformation**: Add controlled noise to create unique voice variations.
- **Inflection Transfer**: Transfer vocal inflections from one voice to another.
- **PCA Perturbation and Spectral Transform**: Other transformations.
- **Export Options**: Save voices as `.pt` files or `.bin` files compatible with `kokoro-onnx`.
- **Real-Time Playback**: Preview and download generated audio with customizable text, language, and speed.

---

## Recommended Workflow

To efficiently create high-quality custom voices, follow this recommended workflow:

1. **Voice Discovery**:
   - Select an existing voice.
   - Experiment with different seeds using **Seed Transform** until a desired voice quality or inflection is achieved.
   - Save desirable voices; they will appear in your 'Generated' list.

2. **Voice Refinement**:
   - If the newly created voice has distortions, blend it lightly (weights around 0.1 - 0.2) with similar existing voices to smooth out imperfections.

3. **Inflection Optimization**:
   - Replace the voice’s inflection with a more preferred option using **Inflection Transfer**. Keep the default strength of 0.3 for optimal results.

4. **Reusing Inflections**:
   - Store any seed transformations with good inflections for future use in other voices.

---

## Using Exported Voices with kokoro-onnx

Exported ONNX voices can be integrated easily:

```python
kokoro = Kokoro("kokoro-v1.0.onnx", "your_voice.bin")
samples, sample_rate = kokoro.create("Yes! It works!", voice="voice", speed=1.0, lang="en-us")
```

## License

This project is licensed under the [MIT License](LICENSE).
