# TG-BWSA: Trainable Gated Block-Wise Shared Adapter for Multimodal Brain Tumor Segmentation

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange)
![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)

Official PyTorch implementation of **TG-BWSA**, introduced in:

> **Improved Multimodal Fusion for Brain Tumor Segmentation via Trainable Gated Block-Wise Shared Adapters**  
> **Milad Rostamian**, **Mohammadreza Mohammadi**, **Ender Konukoglu**, **Mohsen Soryani**  
> *IEEE Transactions on Medical Imaging (TMI), under review*

<!-- Uncomment when available -->
<!-- [[Paper (arXiv)]](https://arxiv.org/abs/XXXX.XXXXX) • [[Project Page]](https://milad67.github.io/TG-BWSA) -->

---

## 🔍 Overview

**TG-BWSA** introduces a novel **parameter-efficient adapter** for multimodal MRI brain tumor segmentation that combines:

- **Stage-wise dual-scalar blending** via trainable gates (α for fused path, β for identity path)
- **Shared adapter per stage** to minimize parameter overhead
- **Squeeze-and-Excitation (SE) attention** for channel-wise recalibration
- **Depthwise-separable convolutions** for efficient spatial feature extraction
- **Compatibility** with both pretrained and randomly initialized encoders

### Key Advantages

✅ **Lightweight**: ~4% parameter overhead  
✅ **Flexible**: Works with U-Net, TransUNet, Swin-UNet, and other architectures  
✅ **Robust**: Handles missing modalities and inter-modality noise  
✅ **Adaptive**: Depth-aware fusion without dense attention maps  
✅ **Validated**: BraTS 2020 and BraTS 2023 benchmarks

---

## 🏗️ Architecture

