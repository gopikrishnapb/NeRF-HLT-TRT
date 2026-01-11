# NeRF from Scratch with LayerFusion-TensorRT Acceleration

This repository contains an implementation of **Neural Radiance Fields (NeRF)** trained entirely from scratch using PyTorch, followed by **GPU-optimized inference using NVIDIA TensorRT with Layer Fusion optimizations**.  
The project includes NeRF training, standard PyTorch-based inference, and TensorRT-accelerated inference with performance benchmarking.

---

## Dataset

We use the **Lego synthetic NeRF dataset**, which can be downloaded from:

🔗 https://inst.eecs.berkeley.edu/~cs180/fa23/hw/proj5/assets/lego_200x200.npz

Place the dataset in the following directory:

```text
data/lego_200x200.npz
