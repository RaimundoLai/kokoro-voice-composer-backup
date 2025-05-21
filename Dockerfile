FROM python:3.12

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    espeak-ng \
    portaudio19-dev \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY $PWD/voice-composer.py /app 

RUN cd /app

# Install Python dependencies
# For CPU-only version:
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Uncomment the line below and comment the above RUN command if you want GPU support
# RUN pip install --no-cache-dir kokoro-onnx[gpu] onnxruntime misaki[en,zh,ja] streamlit pydub \
#     soundfile torch numpy ordered_set pypinyin cn2an jieba fugashi jaconv mojimoji

# Download unidic dictionary for Japanese language processing
RUN python -m unidic download

# Expose the Streamlit port
EXPOSE 5544

# Set the entrypoint
CMD ["streamlit", "run", "voice-composer.py", "--server.port", "5544", "--server.fileWatcherType", "none"]