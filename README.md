# Multichannel ASR

**[English](README.md) | [简体中文](README_zh-CN.md)**

---

## Overview

Multichannel ASR is a training-free solution for transcribing multi-channel audio recordings (e.g., call center conversations, remote meetings) using standard mono-channel ASR models like OpenAI Whisper. It achieves accurate speaker diarization through a novel VAD-based channel merging and silence insertion strategy.

## Key Features

- **Training-Free**: No model training or fine-tuning required
- **Model Agnostic**: Works with any mono-channel ASR model (Whisper, FunASR, SenseVoice, etc.)
- **Efficient**: Multi-processing support for batch processing

## How It Works

The pipeline follows six steps to transform multi-channel audio into diarized transcripts:

<div align="center">
  <img src="./assets/workflow_diagram.jpg" alt="Workflow" width="600">
</div>

1. **Multi-Channel Input**: Load stereo or multi-channel audio
2. **VAD & Segmentation**: Detect voice activity in each channel separately
3. **Silence Insertion & Merging**: Merge channels with strategic silence gaps:
   - 1000ms silence between different speakers
   - 50ms silence within same speaker
4. **ASR Transcription**: Transcribe merged audio using Whisper
5. **Timestamp-based Diarization**: Match transcription timestamps to original speakers
6. **Structured Output**: Generate Excel files with speaker labels and timestamps

<div align="center">
  <img src="./assets/technical_illustration.jpg" alt="Technical Details" width="700">
</div>

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.2+ (for GPU acceleration)
- FFmpeg: `sudo apt-get install ffmpeg` (Ubuntu) or `brew install ffmpeg` (macOS)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/Deng-GuiFeng/multichannel-asr.git
cd multichannel-asr

# Install PyTorch with CUDA support (CUDA 12.6 is compatible with CUDA 12.2)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py \
    --model_id openai/whisper-large-v3-turbo \
    --device cuda:0 \
    --src_dir ./data/input \
    --tgt_dir ./data/output
```

**Parameters:**
- `--model_id`: Hugging Face model ID (e.g., `openai/whisper-large-v3-turbo`)
- `--device`: Device to use (`cuda:0` for GPU, `cpu` for CPU)
- `--src_dir`: Input directory containing audio files
- `--tgt_dir`: Output directory for transcripts

## Patent Notice

This project implements the method described in Chinese Patent **CN120895028A** (Published). The code is released under the MIT License, but commercial use of the patented method may require separate authorization.

## License

[MIT License](LICENSE.txt)

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [WebRTC VAD](https://github.com/wiseman/py-webrtcvad)
