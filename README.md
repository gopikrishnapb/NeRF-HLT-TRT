 NeRF from Scratch with LayerFusion-TensorRT Acceleration

This repository contains an implementation of **Neural Radiance Fields (NeRF)** trained entirely from scratch using PyTorch, followed by **GPU-optimized inference using NVIDIA TensorRT with Layer Fusion optimizations**.  
The project includes NeRF training, PyTorch-based inference, and TensorRT-accelerated inference with performance benchmarking.

---

## Results

| RGB Rendering | Depth Rendering |
| :---: | :---: |
| ![RGB Rendering](assets/rgb.gif) | ![Depth Rendering](assets/depth.gif) |
| **RGB** | **Depth** |

---

## Dataset

This project uses the **Lego synthetic NeRF dataset**.

1. **Download:**  
   https://inst.eecs.berkeley.edu/~cs180/fa23/hw/proj5/assets/lego_200x200.npz
2. **Placement:**  
   Place the file at:
   ```text
   data/lego_200x200.npz

# Environment Setup
## 1. Create Conda Environment
```text
conda create -n nerf -y python=3.10
conda activate nerf
```
## 2. Install CUDA
CUDA 11.8 is recommended

## 3. Install PyTorch (CUDA 11.8)
```
 pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
 torchaudio==2.0.2+cu118 torchdata==0.6.1 \
 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118) 
```

## 4. Install Remaining Dependencies
```
pip install -r requirements.txt
```

#  TensorRT Installation
🔗 TensorRT Installation Guide:
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

Verify your installation:
```python -c "import tensorrt as trt; print(trt.__version__)"
```

# Run
```
python Scripts/nerf.py  
python Scripts/inference.py
python Scripts/tensorrt_inference.py
```
## Disclaimer

This code is released for **research and academic purposes only**.
The authors do not grant any license for commercial use.
If you intend to use this code for commercial purposes, please contact the authors.
