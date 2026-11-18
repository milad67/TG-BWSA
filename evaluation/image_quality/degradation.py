"""Deterministic full-volume corruptions of already preprocessed RFNet inputs.

Input layout here is C,H,W,D. No segmentation is used by these transforms.
The clean arrays and the background values are preserved.
"""
import hashlib
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Condition:
    name: str
    modality: str = "none"
    noise_sigma: float = 0.0
    blur_sigma_vox: float = 0.0


CONDITIONS = (Condition("clean"),) + tuple(
    Condition("noise_%s_%03d" % (m, round(s * 100)), m, noise_sigma=s)
    for m in ("T1ce", "FLAIR") for s in (0.10, 0.20)
) + tuple(
    Condition("blur_%s_%03d" % (m, round(s * 100)), m, blur_sigma_vox=s)
    for m in ("T1ce", "FLAIR") for s in (0.50, 1.00)
)


def noise_seed(case_id, modality, base_seed=20261113):
    message = "%s|%s|%s" % (base_seed, case_id, modality)
    return int.from_bytes(hashlib.sha256(message.encode("utf-8")).digest()[:8], "big")


def to_chwd(array, layout):
    x = np.asarray(array)
    if x.ndim != 4 or layout not in ("HWDC", "CHWD"):
        raise ValueError("Use a 4D input and an explicit HWDC or CHWD layout.")
    if layout == "HWDC":
        x = np.moveaxis(x, -1, 0)
    if x.shape[0] != 4 or not np.isfinite(x).all():
        raise ValueError("Input must have four finite, already preprocessed channels.")
    if not np.issubdtype(x.dtype, np.floating):
        raise ValueError("Expected floating-point z-score input, not raw integer MRI.")
    return np.ascontiguousarray(x, dtype=np.float32)


def foreground_from_clean(x, method="constant_background", supplied=None):
    if supplied is not None:
        mask = np.asarray(supplied)
        if mask.shape != x.shape[1:] or not np.isin(mask, [0, 1]).all():
            raise ValueError("Foreground mask must be binary and match H,W,D.")
        mask = mask.astype(bool)
        info = {"method": "supplied_image_foreground"}
    elif method == "constant_background":
        # RFNet's public normalization may turn zero background into a nonzero
        # channel-specific constant. Do not treat all nonzero voxels as brain.
        corners = np.asarray([x[:, i, j, k]
                              for i in (0, x.shape[1]-1)
                              for j in (0, x.shape[2]-1)
                              for k in (0, x.shape[3]-1)])
        vectors, counts = np.unique(corners, axis=0, return_counts=True)
        chosen = int(counts.argmax())
        if counts[chosen] < 4:
            raise ValueError("No reliable constant background at the image corners. "
                             "Supply an IMAGE-derived foreground mask; do not use tumor labels.")
        bg = vectors[chosen]
        mask = np.any(np.abs(x - bg[:, None, None, None]) > 1e-6, axis=0)
        if (~mask).mean() < 0.005:
            raise ValueError("Insufficient constant background. Supply an image foreground mask.")
        info = {"method": method, "background_values_by_channel": bg.tolist(),
                "matching_corners": int(counts[chosen])}
    elif method == "zero_background":
        mask = np.any(x != 0, axis=0)
        if (~mask).mean() < 0.005:
            raise ValueError("Input does not have a reliable zero background.")
        info = {"method": method}
    else:
        raise ValueError("Unknown foreground method: " + method)
    if mask.mean() <= 0.001:
        raise ValueError("Empty or implausibly small foreground.")
    info["foreground_fraction"] = float(mask.mean())
    return mask, info


def apply_condition(clean, foreground, condition, channel_order, case_id, base_seed=20261113):
    if sorted(channel_order) != sorted(["FLAIR", "T1ce", "T1", "T2"]):
        raise ValueError("Declare the actual four-channel order explicitly.")
    if clean.shape[0] != 4 or foreground.shape != clean.shape[1:]:
        raise ValueError("Shape mismatch.")
    if condition not in CONDITIONS:
        raise ValueError("Condition is not one of the prespecified nine conditions.")
    out = clean.copy()
    if condition.name == "clean":
        return out
    channel = channel_order.index(condition.modality)
    if condition.noise_sigma:
        rng = np.random.default_rng(noise_seed(case_id, condition.modality, base_seed))
        # Model, checkpoint, execution order and severity are absent from the seed.
        # Reuse the same underlying noise field for the two severities.
        field = rng.standard_normal(foreground.shape).astype(np.float32)
        out[channel, foreground] += np.float32(condition.noise_sigma) * field[foreground]
    elif condition.blur_sigma_vox:
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(clean[channel], sigma=condition.blur_sigma_vox,
                                  mode="reflect", truncate=4.0)
        out[channel, foreground] = blurred[foreground]
    # All background voxels retain their original values, including nonzero ones.
    # No extra clipping, normalization, interpolation or cumulative transforms.
    return out
