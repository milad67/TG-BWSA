# coding=utf-8
import os
import numpy as np
import medpy.io as medio

# Set the source path to your raw BraTS2023 training data directory.
src_path = '/scratch_net/maus/mrostamian/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
# Set the target path where the processed volumes and segmentation files will be saved.
tar_path = '/scratch_net/maus/mrostamian/processed_BRATS2023'

# test.txt containing the names of test samples is located in the tar_path.
test_list_file = os.path.join(tar_path, 'test.txt')

# Load test sample names (each line should be a case name, e.g., "BraTS-GLI-00000-000").
with open(test_list_file, 'r') as f:
    test_list = [line.strip() for line in f if line.strip()]






def sup_128(xmin, xmax):
    """Ensure that the crop size in one dimension is at least 128 voxels."""
    if xmax - xmin < 128:
        print('#' * 100)
        ecart = int((128 - (xmax - xmin)) / 2)
        xmax = xmax + ecart + 1
        xmin = xmin - ecart
    if xmin < 0:
        xmax -= xmin
        xmin = 0
    return xmin, xmax

def crop(vol):
    """
    Determine crop bounds from the non-zero region.
    If the input volume has 4 dimensions (channels, H, W, D), collapse along the channel axis.
    """
    if len(vol.shape) == 4:
        vol = np.amax(vol, axis=0)
    assert len(vol.shape) == 3, "Expected 3D volume after collapsing channels."

    x_dim, y_dim, z_dim = vol.shape
    x_nonzeros, y_nonzeros, z_nonzeros = np.where(vol != 0)

    x_min, x_max = np.amin(x_nonzeros), np.amax(x_nonzeros)
    y_min, y_max = np.amin(y_nonzeros), np.amax(y_nonzeros)
    z_min, z_max = np.amin(z_nonzeros), np.amax(z_nonzeros)

    x_min, x_max = sup_128(x_min, x_max)
    y_min, y_max = sup_128(y_min, y_max)
    z_min, z_max = sup_128(z_min, z_max)

    return x_min, x_max, y_min, y_max, z_min, z_max













def normalize(vol):
    """
    Normalize each modality channel independently based on the non-zero region.
    Assumes vol has shape (4, H, W, D).
    """
    mask = vol.sum(0) > 0
    for k in range(4):
        channel = vol[k, ...]
        nonzero_voxels = channel[mask]
        std = nonzero_voxels.std() if nonzero_voxels.std() != 0 else 1.0
        channel = (channel - nonzero_voxels.mean()) / std
        vol[k, ...] = channel
    return vol

# Create target directories for volumes and segmentation if they do not exist.
vol_save_dir = os.path.join(tar_path, 'vol')
seg_save_dir = os.path.join(tar_path, 'seg')
os.makedirs(vol_save_dir, exist_ok=True)
os.makedirs(seg_save_dir, exist_ok=True)

for file_name in test_list:
    print("Processing test sample:", file_name)
    case_path = os.path.join(src_path, file_name)
    
    # Load four imaging modalities: t1c, t1n, t2f, t2w.
    t1c, _ = medio.load(os.path.join(case_path, file_name + '-t1c.nii.gz'))
    t1n, _ = medio.load(os.path.join(case_path, file_name + '-t1n.nii.gz'))
    t2f, _ = medio.load(os.path.join(case_path, file_name + '-t2f.nii.gz'))
    t2w, _ = medio.load(os.path.join(case_path, file_name + '-t2w.nii.gz'))
    
    # Stack modalities along a new axis; resulting shape will be (4, H, W, D)
    vol = np.stack((t1c, t1n, t2f, t2w), axis=0).astype(np.float32)



    x_min, x_max, y_min, y_max, z_min, z_max = crop(vol)
    
    # Crop and normalize the volume.
    vol_cropped = vol[:, x_min:x_max, y_min:y_max, z_min:z_max]




    
    # For test samples, we do not crop. Normalize the full volume.
    vol_normalized = normalize(vol_cropped)
    
    # Transpose to put channels in the last dimension: (H, W, D, 4)
    vol_normalized = vol_normalized.transpose(1, 2, 3, 0)
    print("Test volume shape (full volume):", vol_normalized.shape)
    
    # Save the processed volume.
    np.save(os.path.join(vol_save_dir, file_name + '_vol.npy'), vol_normalized)
    
    # Load segmentation. Since these test samples are chosen from training, segmentation labels are available.
    seg, _ = medio.load(os.path.join(case_path, file_name + '-seg.nii.gz'))
    seg = seg.astype(np.uint8)
    seg_cropped = seg[x_min:x_max, y_min:y_max, z_min:z_max]
    # Optional remapping: for example, if you wish to change label '4' to '3'.
    seg_cropped[seg_cropped == 4] = 3
    print("Segmentation shape (full volume):", seg_cropped.shape)
    
    # Save the processed segmentation.
    np.save(os.path.join(seg_save_dir, file_name + '_seg.npy'), seg_cropped)
