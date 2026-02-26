# 🎨 Realistic Vision v5.1 — Image-to-Image Generator

A powerful web-based image generation application built with **FastAPI** and **Stable Diffusion**. Optimized for high-quality, photorealistic images with facial identity preservation using **IP-Adapter Face**.

## ✨ Features
- **Photorealistic Output**: Uses Realistic Vision v5.1 for stunning clarity and detail.
- **Image-to-Image + IP-Adapter**: Transform existing photos while keeping the original facial identity.
- **Auto Quality Boost**: Built-in prompt engineering for professional-grade results.
- **VRAM Optimized**: Runs efficiently even on lower-end GPUs (e.g. RTX 3050 4GB).
- **History Tracking**: Automatically keeps track of your last 20 generated images.

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have **Python 3.10+** and **Git** installed. An NVIDIA GPU with at least 4GB of VRAM is highly recommended.

### 2. Installation
Clone the repository and install the dependencies:
```powershell
git clone https://github.com/ryant00/Project-ImageGenerator.git
cd Project-ImageGenerator
python -m venv venv
.\venv\Scripts\activate   # For Windows
pip install -r requirements.txt
```

### 3. Troubleshooting GPU
Jika GPU tidak terdeteksi (muncul pesan "CPU Mode" di aplikasi), jalankan script perbaikan otomatis:
```powershell
python fix_gpu.py
```

### 4. Run the Application
Start the FastAPI server:
```powershell
python app.py
```
Open your browser and visit: `http://localhost:7860`


### 📥 Automatic Model Download
**Note:** When you run the application for the first time, it will automatically download approximately **3-5 GB** of AI models from Hugging Face:
- Realistic Vision v5.1 (Base Model)
- SD-VAE-ft-mse (Color Fidelity)
- IP-Adapter Plus Face (Identity Preservation)

These files will be stored in the `models_cache/` folder.

## 🛠️ Tech Stack
- **Backend:** FastAPI, Uvicorn
- **AI Engine:** Diffusers, PyTorch, Transformers
- **Frontend:** Vanilla HTML5, CSS3 (Glassmorphism), JavaScript

## 📄 License
This project is for educational purposes as part of the "Projek Jurusan".
