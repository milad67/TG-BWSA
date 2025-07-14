# TG-BWSA

# TG-BWSA: Trainable Gated Block-Wise Shared Adapter for Multimodal Brain Tumor Segmentation

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Official implementation of the TG-BWSA module from our paper:

> **Improved Multimodal Fusion for Brain Tumor Segmentation via Trainable Gated Block-Wise Shared Adapters**  
> Milad Rostamian, Mohsen Soryani, Mohammad Reza Mohammadi, Ender Konukoglu  
> *IEEE Transactions on Medical Imaging (TMI), 2025 (under review)*

[[Paper (PDF)]](https://arxiv.org/abs/xxxx.xxxxx) • [[Project Page]](https://milad67.github.io) • [[License]](#license)

---

## 🔍 Overview

**TG-BWSA** is a lightweight, architecture-agnostic module designed for multimodal MRI fusion. It performs dynamic block-wise feature adaptation using learnable gating, SE attention, and depthwise separable convolutions. TG-BWSA improves segmentation—especially of challenging tumor subregions—while preserving encoder compatibility and remaining under 4% parameter overhead.

> ✅ Architecture-agnostic  
> ✅ Pretrained or randomly initialized encoders  
> ✅ Robust to modality noise and missing data  
> ✅ Validated on BraTS 2020 & BraTS 2023  

---

## 🧠 Highlights

| Feature              | TG-BWSA                   |
|---------------------|---------------------------|
| Dynamic Gating      | ✓ (α, β per block & modality) |
| SE Attention        | ✓ |
| Depthwise Conv      | ✓ |
| Plug-and-Play       | ✓ |
| Overhead            | < 4% |
| Missing Modalities  | Robust |
| Code                | PyTorch 1.13+ |

---

## 📦 Installation

```bash
git clone https://github.com/milad67/TG-BWSA.git
cd TG-BWSA
pip install -r requirements.txt
