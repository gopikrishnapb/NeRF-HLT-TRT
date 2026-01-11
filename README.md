# NeRF from Scratch with LayerFusion-TensorRT Acceleration

This repository contains an implementation of **Neural Radiance Fields (NeRF)** trained entirely from scratch using PyTorch, followed by **GPU-optimized inference using NVIDIA TensorRT**.  
The project includes training, PyTorch-based inference, and TensorRT-accelerated inference with performance benchmarking

---

## Dataset

We use the **Lego synthetic NeRF dataset**, which can be downloaded from:

🔗 https://inst.eecs.berkeley.edu/~cs180/fa23/hw/proj5/assets/lego_200x200.npz

Place the dataset in the following directory:
```text
data/lego_200x200.npz

## Here's the result:
<div style="display: flex; justify-content: space-between;"> <div style="flex: 0 0 calc(50% - 5px); padding: 5px;"> <img src="assets/rgb.gif" alt="Image 2" style="max-width: 100%; height: auto;"> <p>RGB</p> </div> <div style="flex: 0 0 calc(50% - 5px); padding: 5px;"> <img src="assets/depth.gif" alt="Image 2" style="max-width: 100%; height: auto;"> <p>Depth</p> </div> </div>

## To install and run the code:
conda create -n nerf -y python=3.10
conda activate nerf
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchdata==0.6.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
CUDA 11.8 is recommended.
## To install CUDA
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
            torchaudio==2.0.2+cu118 torchdata==0.6.1 \
            --index-url https://download.pytorch.org/whl/cu118
## Install TensorRT
🔗 TensorRT Installation Guide:
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
After installation, verify:
python -c "import tensorrt as trt; print(trt.__version__)"

## Project Structure
Scripts/
├── nerf.py                  # NeRF training
├── inference.py             # PyTorch inference
├── tensorrt_inference.py    # TensorRT-accelerated inference
├── trt_model.py             # TensorRT engine utilities

assets/                      # GIFs and visualizations
data/                        # Dataset (not included)
checkpoints/                 # Saved models (not included)

# Run
python Scripts/nerf.py
python Scripts/inference.py
python Scripts/tensorrt_inference.py

