import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

from data.datasets_nii import Brats_loadall_test_nii
from RFNetAdapter import RFNetAdapter

# ── CONFIG ───────────────────────────────────────────────────────────
DATA_ROOT = "/scratch_net/maus/mrostamian/processed_BRATS2023"
SAVE_DIR = "/scratch_net/maus/mrostamian/project/compare/fold_0"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_CLASSES = {

    "RFNet_BSA": RFNetAdapter,
}

CHECKPOINTS = {
    "RFNet_BSA": "/scratch_net/maus/mrostamian/project/RFNet_main_org/checkpoints_org/model_300_adapter.pth"
}

WINDOW_SIZES = {
    "RFNet_org": 80,
    "RFNet_scaling": 80,
    "RFNet_BSA": 80,
}

ROW_LABELS = {
    "RFNet_BSA": "RFNet + BSA"
}

# Updated ablation levels to match 16 feature maps
ABLATION_LEVELS = [2, 4, 8]  # Number of channels to ablate

# ── GLOBAL HOOKS & ACTIVATION STORAGE ────────────────────────────────
model_hooks = {}
model_activations = {}
model_channels = {}  # Store total feature maps per model

# ── Hook Registration ─────────────────────────────────────────────────
def register_activation_hook(name, model):
    target = model.module.decoder_fuse.d1_out
    model_activations[name] = []
    def forward_hook(module, inp, outp):
        model_activations[name].append(outp.detach().cpu())
    handle = target.register_forward_hook(forward_hook)
    model_hooks[name] = handle

def compute_region_dice(pred, gt, k, smooth=1e-6):
    p = (pred == k)
    g = (gt == k)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, np.logical_not(g)).sum()
    fn = np.logical_and(g, np.logical_not(p)).sum()
    if (tp + fn) == 0:
        if (tp + fp + fn) == 0: return 1.0
        else: return 0.0
    return float((2 * tp + smooth) / (2 * tp + fp + fn + smooth))

def compute_composite_dice(pred, gt, labels, smooth=1e-6):
    """Compute Dice for composite regions (WT, TC, ET)"""
    pred_mask = np.isin(pred, labels)
    gt_mask = np.isin(gt, labels)
    
    if gt_mask.sum() == 0 and pred_mask.sum() == 0:
        return 1.0
    if gt_mask.sum() == 0:
        return 0.0
        
    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    fn = np.logical_and(gt_mask, ~pred_mask).sum()
    return float((2 * tp + smooth) / (2 * tp + fp + fn + smooth))

# ── UTILITIES ─────────────────────────────────────────────────────────
def normalize(vol):
    mask = vol.sum(0) > 0
    for k in range(4):
        x, vals = vol[k], vol[k][mask]
        if vals.size:
            vol[k] = (x - vals.mean()) / (vals.std() + 1e-8)
    return vol

def crop(vol):
    vol0 = np.amax(vol, axis=0)
    nz = np.where(vol0 != 0)
    if len(nz[0]) == 0:
        return (0, vol.shape[1], 0, vol.shape[2], 0, vol.shape[3])
    x0, x1 = nz[0].min(), nz[0].max()
    y0, y1 = nz[1].min(), nz[1].max()
    z0, z1 = nz[2].min(), nz[2].max()
    H, W, D = vol0.shape
    
    def pad(a, b, M):
        if (b - a) < 128:
            p = (128 - (b - a)) // 2
            a_new = max(0, a - p)
            b_new = min(M, a_new + 128)
            return a_new, b_new
        else:
            return a, b
    
    return (*pad(x0, x1, H), *pad(y0, y1, W), *pad(z0, z1, D))

# ── MODEL LOADER ──────────────────────────────────────────────────────
def load_model(ckpt, cls, name):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading {name}")
    model = cls(num_cls=4)
    checkpoint = torch.load(ckpt, map_location=dev)
    sd = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(sd, strict=False)
    model = torch.nn.DataParallel(model).to(dev).eval()
    # Register hook for all models
    register_activation_hook(name, model)
    return model



# ── INFERENCE PIPELINE ───────────────────────────────────────────────
def sliding_window_inference(model, name, vol, mask, window_size, overlap=0.5, ablate_channels=None):
    _, H, W, D = vol.shape
    stride = int(window_size * (1 - overlap))
    device = next(model.parameters()).device
    acc = np.zeros((4, H, W, D), dtype=np.float32)
    cnt = np.zeros((1, H, W, D), dtype=np.float32)
    feat_acc, feat_cnt = None, None
    
    # Clear activations
    model_activations[name] = []
    
    # Register ablation hook if needed
    if ablate_channels is not None:
        def ablate_hook(module, input):
            # Create a contiguous copy to fix negative stride error
            x = input[0].contiguous().clone()
            x[:, ablate_channels] = 0
            return (x,)
        handle = model.module.decoder_fuse.d1_out.register_forward_pre_hook(ablate_hook)
    else:
        handle = None
    
    def steps(L):
        s = list(range(0, L - window_size + 1, stride))
        if s and s[-1] != L - window_size:
            s.append(L - window_size)
        return s

    hs, ws, ds = steps(H), steps(W), steps(D)
    
    with torch.no_grad():
        for h in hs:
            for w in ws:
                for d in ds:
                    patch = vol[:, h:h+window_size, w:w+window_size, d:d+window_size]
                    msk = mask[h:h+window_size, w:w+window_size, d:d+window_size]
                    inp = torch.from_numpy((patch * msk).astype(np.float32)).unsqueeze(0).to(device)
                    
                    out = model(inp, mask=torch.ones((1, 4), dtype=torch.bool, device=device))
                    if isinstance(out, tuple): out = out[0]
                    
                    prob = out.squeeze(0).cpu().numpy()
                    acc[:, h:h+window_size, w:w+window_size, d:d+window_size] += prob
                    cnt[:, h:h+window_size, w:w+window_size, d:d+window_size] += 1
                    
                    # Only collect features if not ablating
                    if ablate_channels is None:
                        hook_out = model_activations[name].pop(0)
                        patch_feat = hook_out.abs().numpy()[0]
                        
                        if feat_acc is None:
                            C = patch_feat.shape[0]
                            feat_acc = np.zeros((C, H, W, D), dtype=np.float32)
                            feat_cnt = np.zeros((1, H, W, D), dtype=np.float32)
                        
                        feat_acc[:, h:h+window_size, w:w+window_size, d:d+window_size] += patch_feat
                        feat_cnt[:, h:h+window_size, w:w+window_size, d:d+window_size] += 1

    # Remove hook after inference
    if handle is not None:
        handle.remove()

    soft = acc / (cnt + 1e-8)
    lbl = np.argmax(soft, axis=0)
    
    # Only return features if not ablating
    if ablate_channels is None and feat_acc is not None:
        feat = feat_acc / (feat_cnt + 1e-8)
        return soft.transpose(1, 2, 3, 0), lbl.astype(np.uint8), feat
    else:
        return soft.transpose(1, 2, 3, 0), lbl.astype(np.uint8), None

# ── NEURON SCORING ───────────────────────────────────────────────────
def compute_neuron_scores(acts, lbl):
    scores = {}
    for k in (1, 2, 3):
        mask = (lbl == k)
        if mask.any():
            scores[k] = acts[:, mask].mean(axis=1)
        else:
            scores[k] = np.zeros(acts.shape[0], dtype=float)
    return scores

# ── MAIN EXECUTION ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Load dataset - all cases
    ds = Brats_loadall_test_nii(root=DATA_ROOT, test_file="test.txt", modal="all")
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    # Prepare all cases without filtering
    all_cases = []
    for x, y, name in loader:
        cid = name[0]
        vol = x.squeeze(0).numpy()
        gt = y.squeeze(0).numpy()
        zpos = int(np.argmax(gt.sum(axis=(0, 1)))) if (gt > 0).any() else gt.shape[2] // 2
        all_cases.append({"case_id": cid, "vol": vol, "gt": gt, "zpos": zpos})
    
    # Load models
    models = {nm: load_model(CHECKPOINTS[nm], MODEL_CLASSES[nm], nm) for nm in MODEL_CLASSES}
    
    # Initialize results storage
    ablation_results = []
    
    # Process all cases
    for case_idx, case in enumerate(all_cases):
        cid = case["case_id"]
        print(f"\nProcessing case {case_idx+1}/{len(all_cases)}: {cid}")
        
        # Preprocess volume
        vol_full = case["vol"]
        bm = (vol_full.sum(0) > 0).astype(np.float32)
        crop_coords = crop(vol_full)
        x0, x1, y0, y1, z0, z1 = crop_coords
        vc = normalize(vol_full[:, x0:x1, y0:y1, z0:z1].copy())
        gt_c = case["gt"][x0:x1, y0:y1, z0:z1]
        bm_c = bm[x0:x1, y0:y1, z0:z1]
        
        # Process each model
        for nm, mdl in models.items():
            print(f"  Running {ROW_LABELS[nm]}...")
            
            # Baseline inference
            soft, lbl, feat = sliding_window_inference(
                mdl, nm, vc, bm_c, window_size=WINDOW_SIZES[nm], overlap=0.5
            )
            
            # Store total feature maps if not already stored
            if feat is not None:
                if nm not in model_channels:
                    model_channels[nm] = feat.shape[0]
                    print(f"    Total feature maps: {model_channels[nm]}")
            
            # Compute baseline Dice scores
            baseline_composite = {
                'WT': compute_composite_dice(lbl, gt_c, [1, 2, 3]),
                'TC': compute_composite_dice(lbl, gt_c, [1, 3]),
                'ET': compute_composite_dice(lbl, gt_c, [3]),
            }
            
            # Compute neuron scores if features are available
            if feat is not None:
                scores = compute_neuron_scores(feat, lbl)
                top_channels = {k: np.argsort(scores[k])[::-1] for k in (1, 2, 3)}
            else:
                top_channels = {1: [], 2: [], 3: []}
            
            # Perform ablations at different levels
            for region in (1, 2, 3):
                for n_ablate in ABLATION_LEVELS:
                    # Skip if we don't have enough channels
                    if n_ablate > len(top_channels[region]):
                        continue
                    
                    # Select top channels to ablate
                    channels_to_ablate = top_channels[region][:n_ablate].tolist()
                    
                    # Run ablation inference
                    _, lbl_abl, _ = sliding_window_inference(
                        mdl, nm, vc, bm_c, 
                        window_size=WINDOW_SIZES[nm], 
                        overlap=0.5,
                        ablate_channels=channels_to_ablate
                    )
                    
                    # Compute ablated Dice scores
                    ablated_composite = {
                        'WT': compute_composite_dice(lbl_abl, gt_c, [1, 2, 3]),
                        'TC': compute_composite_dice(lbl_abl, gt_c, [1, 3]),
                        'ET': compute_composite_dice(lbl_abl, gt_c, [3]),
                    }
                    
                    # Calculate Dice drops
                    dice_drop = {
                        cr: baseline_composite[cr] - ablated_composite[cr]
                        for cr in ['WT', 'TC', 'ET']
                    }
                    
                    # Store results
                    ablation_results.append({
                        'case_id': cid,
                        'model': ROW_LABELS[nm],
                        'ablated_region': region,
                        'n_ablate': n_ablate,
                        'total_feature_maps': model_channels.get(nm, "Unknown"),
                        'baseline_WT': baseline_composite['WT'],
                        'ablated_WT': ablated_composite['WT'],
                        'drop_WT': dice_drop['WT'],
                        'baseline_TC': baseline_composite['TC'],
                        'ablated_TC': ablated_composite['TC'],
                        'drop_TC': dice_drop['TC'],
                        'baseline_ET': baseline_composite['ET'],
                        'ablated_ET': ablated_composite['ET'],
                        'drop_ET': dice_drop['ET'],
                    })
            
            # Print progress
            print(f"    Baseline Dice: WT={baseline_composite['WT']:.4f}, " +
                  f"TC={baseline_composite['TC']:.4f}, ET={baseline_composite['ET']:.4f}")

    # Convert results to DataFrame
    ablation_df = pd.DataFrame(ablation_results)
    
    # Calculate average drops per model, region, and ablation level
    summary = ablation_df.groupby(['model', 'ablated_region', 'n_ablate']).agg({
        'total_feature_maps': 'first',
        'drop_WT': ['mean', 'std'],
        'drop_TC': ['mean', 'std'],
        'drop_ET': ['mean', 'std'],
    }).reset_index()
    
    # Print comprehensive summary
    print("\n\n=== ABLATION STUDY RESULTS ===")
    print("Average Dice Drop (mean ± std) for Composite Regions")
    print("=" * 80)
    
    region_names = {1: "Non-enhancing", 2: "Edema", 3: "Enhancing"}
    
    for (model, region, n_ablate), group in summary.groupby(['model', 'ablated_region', 'n_ablate']):
        region_name = region_names[region]
        total_channels = group[('total_feature_maps', 'first')].iloc[0]
        percent_ablated = (n_ablate / total_channels) * 100
        
        print(f"\nModel: {model} | Total Feature Maps: {total_channels}")
        print(f"Ablated Region: {region_name} | Channels Removed: {n_ablate} ({percent_ablated:.1f}%)")
        print("-" * 60)
        print(f"  WT Drop: {group[('drop_WT', 'mean')].values[0]:.4f} ± {group[('drop_WT', 'std')].values[0]:.4f}")
        print(f"  TC Drop: {group[('drop_TC', 'mean')].values[0]:.4f} ± {group[('drop_TC', 'std')].values[0]:.4f}")
        print(f"  ET Drop: {group[('drop_ET', 'mean')].values[0]:.4f} ± {group[('drop_ET', 'std')].values[0]:.4f}")
    
    # Calculate and print overall ablation impact
    overall_summary = ablation_df.groupby(['model', 'n_ablate']).agg({
        'total_feature_maps': 'first',
        'drop_WT': 'mean',
        'drop_TC': 'mean',
        'drop_ET': 'mean',
    }).reset_index()
    
    print("\n\n=== OVERALL ABLATION IMPACT ===")
    print("Average Dice Drop Across All Ablated Regions")
    print("=" * 60)
    for (model, n_ablate), group in overall_summary.groupby(['model', 'n_ablate']):
        total_channels = group['total_feature_maps'].iloc[0]
        percent_ablated = (n_ablate / total_channels) * 100
        
        print(f"\nModel: {model} | Total Feature Maps: {total_channels}")
        print(f"Channels Removed: {n_ablate} ({percent_ablated:.1f}%)")
        print("-" * 40)
        print(f"  WT Drop: {group['drop_WT'].values[0]:.4f}")
        print(f"  TC Drop: {group['drop_TC'].values[0]:.4f}")
        print(f"  ET Drop: {group['drop_ET'].values[0]:.4f}")

    # Save detailed results to CSV
    results_path = os.path.join(SAVE_DIR, "ablation_results.csv")
    ablation_df.to_csv(results_path, index=False)
    print(f"\nDetailed results saved to: {results_path}")

    # Cleanup
    for h in model_hooks.values():
        h.remove()
    print("\nAll cases processed.")