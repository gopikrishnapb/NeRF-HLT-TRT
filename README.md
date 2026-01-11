# NeRF from Scratch with TensorRT Acceleration

This repository contains an implementation of **Neural Radiance Fields (NeRF)** trained entirely from scratch using PyTorch, followed by **GPU-optimized inference using NVIDIA TensorRT**.  
The project includes training, PyTorch-based inference, and TensorRT-accelerated inference with performance benchmarking.

This implementation was developed as part of the NeRF project described in the following specification:  
🔗 https://inst.eecs.berkeley.edu/~cs180/fa23/hw/proj5/

---

## Dataset

We use the **Lego synthetic NeRF dataset**, which can be downloaded from:

🔗 https://inst.eecs.berkeley.edu/~cs180/fa23/hw/proj5/assets/lego_200x200.npz

Place the dataset in the following directory:
```text
data/lego_200x200.npz
