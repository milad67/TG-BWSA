# TG-BWSA: Trainable Gated Block-Wise Shared Adapter for Multimodal Brain Tumor Segmentation

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00)

Official implementation of **TG-BWSA**, introduced in:

> **Improved Multimodal Fusion for Brain Tumor Segmentation via Trainable Gated Block-Wise Shared Adapters**  
> **Milad Rostamian**, **Mohammadreza Mohammadi**, **Ender Konukoglu**, **Mohsen Soryani**

<!-- Uncomment/add when available -->
<!-- [[Paper (arXiv)]](https://arxiv.org/abs/XXXX.XXXXX) • [[Project Page]](https://milad67.github.io/TG-BWSA) -->

---

## 🔍 Overview

**TG-BWSA** is a lightweight, architecture-agnostic adapter for multimodal MRI fusion. It performs **block-wise** feature adaptation with **trainable dual gates** (α for the fused path, β for the identity path), plus **depthwise-separable convolutions**.

- Works with pretrained **and** randomly initialized encoders  
- Robust to **missing modalities** and modality noise  
- Validated on **BraTS 2020** and **BraTS 2023**  
- Parameter overhead **<~4% in our configs**
- Available in **PyTorch** and **TensorFlow/Keras**

### ✨ Highlights

| Capability | TG-BWSA |
|---|---|
| Per-stage, per-modality gates (α, β) | ✓ |
| Depthwise-separable conv | ✓ |
| Plug-and-play (UNet/Transformer) | ✓ |
| Robust to missing modalities | ✓ |
| PyTorch + TensorFlow | ✓ |

---

## 📦 Installation

### PyTorch Version

