#!/usr/bin/env python
#coding=utf-8

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models import Model  # your RFNet model class
from scipy.ndimage import label, generate_binary_structure


import logging
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import label, generate_binary_structure

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
CHECKPOINT_PATH = "/scratch_net/maus/mrostamian/project/RFNet-main_org_base/checkpoints/model_300.pth"  # Best checkpoint path
BRATS_VAL_DIR   = "/scratch_net/maus/mrostamian/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
OUTPUT_DIR      = "/scratch_net/maus/mrostamian/project/RFNet-main_org_base/online_predictions_base_new_pp_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




import torch
import time
from fvcore.nn import FlopCountAnalysis

def benchmark_model(model, input_tensor):
    model.eval()
    input_tensor = input_tensor.cuda()
    model = model.cuda()

    # Measure Latency
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        _ = model(input_tensor, mask=torch.tensor([[True, True, True, True]], device='cuda'))[0]
    torch.cuda.synchronize()
    latency = time.time() - start_time  # seconds per volume

    # Measure Peak GPU Memory
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(input_tensor, mask=torch.tensor([[True, True, True, True]], device='cuda'))[0]
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

    # Measure GFLOPs
    flops = FlopCountAnalysis(model, (input_tensor, torch.tensor([[True, True, True, True]], device='cuda')))
    gflops = flops.total() / 1e9

    print(f"Latency (s/vol): {latency:.4f}")
    print(f"Peak GPU Memory (GB): {peak_memory:.2f}")
    print(f"GFLOPs: {gflops:.2f}")
    return latency, peak_memory, gflops

# Example use:
# from models import Model  # your actual RFNet+TG-BWSA model class
# model = Model(num_cls=4)
# dummy_input = torch.randn(1, 4, 128, 128, 128)
# benchmark_model(model, dummy_input)













# --------------------------------------------------------------------
# 1. Normalization
# --------------------------------------------------------------------
def normalize(vol):
    """
    Normalizes each modality channel independently.
    Expects vol with shape [4, H, W, D].
    """
    mask = vol.sum(0) > 0
    for k in range(4):
        x = vol[k]
        y = x[mask]
        if y.size > 0:
            vol[k] = (x - y.mean()) / (y.std() + 1e-8)
    return vol

# --------------------------------------------------------------------
# 2. Save Final Segmentation
# --------------------------------------------------------------------
def save_prediction(volume, sample_paths, save_dir):
    """
    Saves the predicted segmentation volume (shape: [H, W, D], labels in {0,1,2,3})
    as a NIfTI (.nii.gz) file.
    
    Uses the 't1c' file to extract the affine transformation.
    """
    os.makedirs(save_dir, exist_ok=True)
    try:
        orig_nii = nib.load(sample_paths['t1c'])
        affine = orig_nii.affine
    except Exception as e:
        print("Error loading affine from t1c; using identity.", e)
        affine = np.eye(4)

    base_name = os.path.basename(sample_paths['t1c'])
    parts = base_name.split('-')
    if len(parts) >= 4:
        case_id   = parts[2]
        timepoint = parts[3][:3]
    else:
        case_id   = "unknown"
        timepoint = "000"

    out_name = f"{case_id}-{timepoint}.nii.gz"
    out_path = os.path.join(save_dir, out_name)

    brats_vol = np.zeros_like(volume, dtype=np.uint8)
    brats_vol[volume == 1] = 1
    brats_vol[volume == 2] = 2
    brats_vol[volume == 3] = 3

    nib.save(nib.Nifti1Image(brats_vol, affine), out_path)
    print(f"Saved prediction to {out_path}")

# --------------------------------------------------------------------
# 3. Remove Small False Positives
# --------------------------------------------------------------------
# def remove_small_false_positives_3d(prediction, probability_volume):
#     """
#     Remove small connected components that don't meet class-specific thresholds.
#     """
#     cleaned = np.zeros_like(prediction)
#     structure = generate_binary_structure(3, 3)
    
#     class_params = {
#         1: {'min_voxels': 450, 'min_prob': 0.32},  # NCR
#         2: {'min_voxels': 700, 'min_prob': 0.30},  # EDEMA
#         3: {'min_voxels': 250, 'min_prob': 0.35}   # ET
#     }
    
#     for class_idx in [2, 1, 3]:
#         class_mask = (prediction == class_idx)
#         labeled_mask, num_features = label(class_mask, structure=structure)
#         for i in range(1, num_features + 1):
#             lesion_mask = (labeled_mask == i)
#             vcount = lesion_mask.sum()
#             prob_mask = probability_volume[..., class_idx]
#             mean_prob = prob_mask[lesion_mask].mean() if vcount > 0 else 0
#             params = class_params[class_idx]
#             if vcount >= params['min_voxels'] and mean_prob >= params['min_prob']:
#                 cleaned[lesion_mask & (cleaned == 0)] = class_idx
#     return cleaned


def remove_small_false_positives_3d(
    prediction: np.ndarray,
    probability_volume: np.ndarray,
    *,
    # ───────── thresholds ────────────────────────────────────────────
    large_voxel_thr: int = 1500,
    min_prob_large: float = .65,                # used for large & regular
    min_prob_regular: float = .8,                # used for large & regular
    small_voxel_thr: Dict[int, int] = None,    # min size for small/high-conf
    high_conf_thr: Dict[int, float] = None,    # prob thr for small/high-conf
    class_voxel_thresholds: Optional[Dict[int, int]] = None,
    connectivity: int = 3,
    verbose: bool = True,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """
    Remove small or low-confidence CCs based on class-specific rules:
      1) large & confident → always keep
      2) small but very confident (cls=1,3) → keep
      3) regular class thresholds → keep
    """

    # default per-class ‘regular’ minimum size
    if class_voxel_thresholds is None:
        class_voxel_thresholds = {1: 400, 2: 800, 3: 150}

    # for cls 1 & 3: minimum size *and* very high prob to accept small CCs
    if small_voxel_thr is None:
        small_voxel_thr = {1: 200, 3: 70}
    if high_conf_thr is None:
        high_conf_thr   = {1: 0.85, 3: 0.90}

    # setup logger
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(h)
        logger.setLevel(logging.INFO if verbose else logging.WARNING)

    cleaned   = np.zeros_like(prediction, dtype=prediction.dtype)
    structure = generate_binary_structure(3, connectivity)

    for cls in (3, 1, 2):  # ET first, then NCR, then ED
        mask   = prediction == cls
        labeled, n_cc = label(mask, structure=structure)
        logger.info(f"Class {cls}: {n_cc} components")

        prob_map = probability_volume[..., cls]
        reg_min  = class_voxel_thresholds[cls]

        for i in range(1, n_cc + 1):
            lesion   = (labeled == i)
            vcount   = int(lesion.sum())
            if vcount == 0:
                continue
            mean_prob = float(prob_map[lesion].mean())

            keep, reason = False, "discarded"

            # 1) large & confident
            if vcount > large_voxel_thr and mean_prob >= min_prob_large:
                keep, reason = True, "large & confident"

            # 2) small but very confident (only for cls=1 or cls=3)
            elif cls in small_voxel_thr and cls in high_conf_thr:
                if vcount >= small_voxel_thr[cls] and mean_prob >= high_conf_thr[cls]:
                    keep, reason = True, "small & very confident"

            # 3) regular class threshold, just for class 2
            elif vcount >= reg_min and mean_prob >= min_prob_regular:
                keep, reason = True, "class threshold"


            # # 2) small but very confident (only for cls=1 or cls=3)
            # elif (
            #     cls in small_voxel_thr
            #     and cls in high_conf_thr
            #     and vcount >= small_voxel_thr[cls]
            #     and mean_prob >= high_conf_thr[cls]
            # ):
            #     keep, reason = True, "small & very confident"

            # # 3) regular class threshold
            # elif vcount >= reg_min and mean_prob >= min_prob_regular:
            #     keep, reason = True, "class threshold"






            if keep:
                cleaned[lesion] = cls
                logger.info(f"  • CC{i:02d}: {vcount} vox, p={mean_prob:.3f} → kept ({reason})")
            else:
                logger.info(f"  • CC{i:02d}: {vcount} vox, p={mean_prob:.3f} → discarded")

    return cleaned


# --------------------------------------------------------------------
# 4. Inference Pipeline with Sliding Window
# --------------------------------------------------------------------
def sliding_window_inference(model, vol, brain_mask, window_size=80, overlap=0.5):
    """
    Performs sliding window inference.
    
    Inputs:
      - vol: numpy array with shape [4, H, W, D] (normalized modalities)
      - brain_mask: numpy array with shape [H, W, D]
      - window_size: size of sliding patch (cubic)
      - overlap: fractional overlap between patches
      
    Returns:
      - pred_softmax: softmax probabilities, shape (H, W, D, 4)
      - pred_labels: final label predictions (H, W, D)
    """
    _, H, W, D = vol.shape
    stride = int(window_size * (1 - overlap))
    device = next(model.parameters()).device

    pred = np.zeros((4, H, W, D), dtype=np.float32)
    count = np.zeros((1, H, W, D), dtype=np.float32)

    h_steps = list(range(0, H - window_size + 1, stride))
    if h_steps[-1] != H - window_size:
        h_steps.append(H - window_size)
    w_steps = list(range(0, W - window_size + 1, stride))
    if w_steps[-1] != W - window_size:
        w_steps.append(W - window_size)
    d_steps = list(range(0, D - window_size + 1, stride))
    if d_steps[-1] != D - window_size:
        d_steps.append(D - window_size)

    with torch.no_grad():
        for h in h_steps:
            for w in w_steps:
                for d in d_steps:
                    h_end = h + window_size
                    w_end = w + window_size
                    d_end = d + window_size

                    mask_patch = brain_mask[h:h_end, w:w_end, d:d_end]
                    x_patch = vol[:, h:h_end, w:w_end, d:d_end] * mask_patch
                    
                    x_t = torch.from_numpy(x_patch).unsqueeze(0).to(device).float()

                    modality_mask = torch.tensor([[True, True, True, True]], device=device)
                    
                    output = model(x_t, mask=modality_mask)[0]
                    output = output.squeeze(0).cpu().numpy()

                    pred[:, h:h_end, w:w_end, d:d_end] += output
                    count[:, h:h_end, w:w_end, d:d_end] += 1

    pred = pred / (count + 1e-8)
    pred_softmax = pred
    pred_labels = np.argmax(pred_softmax, axis=0)

    return pred_softmax.transpose(1, 2, 3, 0), pred_labels

# --------------------------------------------------------------------
# 5. Dataset / DataLoader
# --------------------------------------------------------------------
class BraTSOnlineValidationDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for case in os.listdir(root_dir):
            case_dir = os.path.join(root_dir, case)
            if os.path.isdir(case_dir):
                mods = {
                    't1c': os.path.join(case_dir, f"{case}-t1c.nii.gz"),
                    't1n': os.path.join(case_dir, f"{case}-t1n.nii.gz"),
                    't2f': os.path.join(case_dir, f"{case}-t2f.nii.gz"),
                    't2w': os.path.join(case_dir, f"{case}-t2w.nii.gz")
                }
                if all(os.path.exists(p) for p in mods.values()):
                    self.samples.append((mods, case))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        modalities, case_id = self.samples[idx]
        return modalities, case_id

def custom_collate(batch):
    return batch[0]

# --------------------------------------------------------------------
# 6. Load Model
# --------------------------------------------------------------------
def load_model(checkpoint_path):
    model = Model(num_cls=4)
    model = torch.nn.DataParallel(model)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    return model

# --------------------------------------------------------------------
# 7. Crop Function
# --------------------------------------------------------------------
def crop(vol):
    """
    Determine crop bounds from the non-zero region.
    Expects vol shape (4, H, W, D).
    Returns (x_min, x_max, y_min, y_max, z_min, z_max).
    """
    vol_for_crop = np.amax(vol, axis=0)
    assert len(vol_for_crop.shape) == 3, "Expected 3D volume after collapsing channels."
    x_nonzeros, y_nonzeros, z_nonzeros = np.where(vol_for_crop != 0)

    x_min, x_max = np.amin(x_nonzeros), np.amax(x_nonzeros)
    y_min, y_max = np.amin(y_nonzeros), np.amax(y_nonzeros)
    z_min, z_max = np.amin(z_nonzeros), np.amax(z_nonzeros)

    def sup_128(xmin, xmax):
        if xmax - xmin < 128:
            ecart = int((128 - (xmax - xmin)) / 2)
            xmax = xmax + ecart + 1
            xmin = xmin - ecart
        if xmin < 0:
            xmax -= xmin
            xmin = 0
        return xmin, xmax

    x_min, x_max = sup_128(x_min, x_max)
    y_min, y_max = sup_128(y_min, y_max)
    z_min, z_max = sup_128(z_min, z_max)

    return x_min, x_max, y_min, y_max, z_min, z_max

# --------------------------------------------------------------------
# 8. Main Inference Pipeline
# --------------------------------------------------------------------
if __name__ == "__main__":
    model = load_model(CHECKPOINT_PATH)
    dataset = BraTSOnlineValidationDataset(BRATS_VAL_DIR)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)



    # === Call benchmark once before looping over all patients ===
    dummy_input = torch.randn(1, 4, 80, 80, 80)  # Match your typical cropped input size
    benchmark_model(model, dummy_input)




    # for (modalities, case_id) in tqdm(loader, desc="Processing Validation"):
    #     mod_order = ['t1c', 't1n', 't2f', 't2w']
    #     assert mod_order == ['t1c', 't1n', 't2f', 't2w'], "Modality order mismatch with training!"

    #     vol_full = np.stack([nib.load(modalities[m]).get_fdata() for m in mod_order], axis=0)
    #     full_shape = vol_full.shape[1:]
        
    #     vol_full = vol_full.astype(np.float32)
    #     brain_mask_full = (vol_full.sum(0) > 0).astype(np.float32)
        
    #     x_min, x_max, y_min, y_max, z_min, z_max = crop(vol_full)
    #     vol_cropped = vol_full[:, x_min:x_max, y_min:y_max, z_min:z_max]
    #     brain_mask_cropped = brain_mask_full[x_min:x_max, y_min:y_max, z_min:z_max]
        
    #     vol_cropped = normalize(vol_cropped)
        
    #     pred_softmax, pred_labels_cropped = sliding_window_inference(model, vol_cropped, brain_mask_cropped)
    #     cleaned_pred = remove_small_false_positives_3d(pred_labels_cropped, pred_softmax)
        
    #     full_pred = np.zeros(full_shape, dtype=np.uint8)
    #     full_pred[x_min:x_max, y_min:y_max, z_min:z_max] = cleaned_pred
        
    #     assert full_pred.shape == full_shape, f"Full segmentation shape mismatch: {full_pred.shape} vs {full_shape}"
        
    #     save_prediction(full_pred, modalities, OUTPUT_DIR)








