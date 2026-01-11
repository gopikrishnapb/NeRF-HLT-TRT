# NeRF from Scratch with LayerFusion-TensorRT Acceleration

This repository contains an implementation of **Neural Radiance Fields (NeRF)** trained entirely from scratch using PyTorch, followed by **GPU-optimized inference using NVIDIA TensorRT with Layer Fusion optimizations**.

The project includes the full NeRF training pipeline, standard PyTorch-based inference, and TensorRT-accelerated inference with performance benchmarking to demonstrate the speedups achieved through hardware-specific optimization.

---

## Results

| RGB Rendering | Depth Rendering |
| :---: | :---: |
| ![RGB Rendering](assets/rgb.gif) | ![Depth Rendering](assets/depth.gif) |
| **RGB** | **Depth** |

---

## Dataset

This project uses the **Lego synthetic NeRF dataset**.

1. **Download:** [lego_200x200.npz](https://inst.eecs.berkeley.edu/~cs180/fa23/hw/proj5/assets/lego_200x200.npz)
2. **Placement:** Place the downloaded file in the following directory:
   ```text
   data/lego_200x200.npz
