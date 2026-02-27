# 🎨 Realistic Vision v5.1 — AI Image Generator

A powerful web-based image generation application built with **FastAPI** and **Stable Diffusion**. Optimized for high-quality, photorealistic images with facial identity preservation using **IP-Adapter Face**.

## ✨ Features
- **Photorealistic Output**: Uses Realistic Vision v5.1 for stunning clarity and detail.
- **Text-to-Image**: Generate images from text prompts only.
- **Image-to-Image + IP-Adapter Face**: Transform existing photos while keeping the original facial identity.
- **Auto Quality Boost**: Built-in prompt engineering for professional-grade results.
- **Multi-GPU Support**: Auto-detects and uses the best available GPU:
  - 🟢 **NVIDIA** (CUDA) — Full support, fastest performance
  - 🔴 **AMD** (ROCm on Linux / DirectML on Windows)
  - 🔵 **Intel** (XPU / DirectML on Windows)
  - 🍎 **Apple Silicon** (MPS on macOS)
  - 🖥️ **CPU** — Fallback, works on any computer
- **Hi-Res Fix**: Optional upscale via img2img refinement pass for higher resolution output.
- **Multiple Schedulers**: Choose between Euler, DPM++, or DDIM samplers.
- **History Tracking**: Automatically keeps track of your last 20 generated images.
- **Glassmorphism UI**: Modern, responsive frontend with a premium glass-style design.

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.10+** and **Git** installed
- **GPU** recommended for faster generation (NVIDIA, AMD, Intel, or Apple Silicon)
  - NVIDIA users: Install **CUDA Toolkit 12.1** ([Download](https://developer.nvidia.com/cuda-12-1-0-download-archive))
  - AMD users on Windows: Install `pip install torch-directml`
  - Works on **CPU** too (just slower)

### 2. Installation
Clone the repository and install the dependencies:
```powershell
git clone https://github.com/ryant00/ProjectImageGenerator.git
cd ProjectImageGenerator
python -m venv venv
.\venv\Scripts\activate   # For Windows
pip install -r requirements.txt
```

### 3. Run the Application
Start the FastAPI server:
```powershell
python app.py
```
Open your browser and visit: `http://localhost:7860`

## 📥 Automatic Model Download
**Note:** When you run the application for the first time, it will automatically download approximately **3-5 GB** of AI models from Hugging Face:
- **Realistic Vision v5.1** — Base Model (`SG161222/Realistic_Vision_V5.1_noVAE`)
- **SD-VAE-ft-mse** — Better Color Fidelity (`stabilityai/sd-vae-ft-mse`)
- **IP-Adapter Plus Face** — Facial Identity Preservation (`h94/IP-Adapter`)
- **CLIP Vision Encoder** — Image understanding for IP-Adapter

These files will be stored in the `models_cache/` folder.

## ⚙️ GPU & Memory Optimization
This application is optimized for NVIDIA GPUs with CUDA 12.1:
- **Float16 precision** for faster inference and lower VRAM usage
- **Attention slicing** to reduce memory peaks
- **VAE tiling & slicing** for memory-efficient decoding
- **xformers** support for memory-efficient attention (if installed)
- **Low-VRAM mode** (auto-enabled for GPUs < 6GB VRAM): applies aggressive memory saving
- **CUDA memory allocator** tuned with `max_split_size_mb:128` for better memory management

## 🛠️ Tech Stack
| Component | Technology |
|---|---|
| **Backend** | FastAPI, Uvicorn |
| **AI Engine** | Diffusers, PyTorch (CUDA 12.1), Transformers |
| **Face Preservation** | IP-Adapter Plus Face (SD 1.5) |
| **Image Processing** | Pillow, NumPy |
| **Frontend** | Vanilla HTML5, CSS3 (Glassmorphism), JavaScript |

## 📋 Requirements
```
torch==2.1.2+cu121
torchvision==0.16.2+cu121
diffusers==0.25.1
transformers==4.36.2
accelerate==0.25.0
safetensors==0.4.1
huggingface_hub==0.20.3
fastapi==0.109.0
uvicorn[standard]==0.25.0
python-multipart==0.0.6
Pillow==10.2.0
numpy==1.26.3
```

## 📄 License
This project is for educational purposes as part of the "Projek Jurusan".
