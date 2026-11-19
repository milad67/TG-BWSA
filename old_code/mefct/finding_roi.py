import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt

# Paths to the data
brats_data = "/cluster/work/cvl/mrostamian/"
brats_data_validation = os.path.join(brats_data, 'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData')
output_dir = '/cluster/home/mrostamian/log_ssl_flair/visualize_img'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def find_all_volume_paths(data_dir, modality_keyword="t2f"):
    """
    Find all .nii.gz files in the given directory that match the specified modality.
    """
    volume_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".nii.gz") and modality_keyword in file:
                volume_paths.append(os.path.join(root, file))
    return volume_paths

def find_largest_roi_in_volume(volume_path):
    """
    Find the largest ROI in a single volume and return the frame before and after cropping.
    """
    largest_x, largest_y, largest_w, largest_h = 0, 0, 0, 0
    largest_frame = None
    largest_frame_cropped = None
    largest_slice_index = -1

    volume_nifti = nib.load(volume_path)
    volume_data = volume_nifti.get_fdata()

    for i in range(volume_data.shape[2]):  # Loop through each slice
        img_slice = volume_data[:, :, i]

        # Create a binary mask for non-black (non-zero) pixels
        binary_mask = img_slice > 0

        # Find non-zero rows and columns
        rows = np.where(np.any(binary_mask, axis=1))[0]
        cols = np.where(np.any(binary_mask, axis=0))[0]

        if rows.size > 0 and cols.size > 0:
            # Get bounding box for the current slice
            x, y, w, h = cols[0], rows[0], cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1

            # Update if this bounding box is larger
            if w * h > largest_w * largest_h:
                largest_x, largest_y, largest_w, largest_h = x, y, w, h
                largest_frame = img_slice  # Save the original slice
                largest_frame_cropped = img_slice[y:y+h, x:x+w]  # Save the cropped slice
                largest_slice_index = i

    return largest_x, largest_y, largest_w, largest_h, largest_frame, largest_frame_cropped, largest_slice_index

def save_combined_image(original, cropped, output_path):
    """
    Save a combined image with both the original and cropped frames side by side.
    """
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original (Before Cropping)")
    plt.axis('off')

    # Cropped image
    plt.subplot(1, 2, 2)
    plt.imshow(cropped, cmap='gray')
    plt.title("Cropped (Largest ROI)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def crop_all_frames_in_volume(volume_path, roi_x, roi_y, crop_size=192, output_dir=None):
    """
    Crop all slices in a volume based on the global largest ROI and save the cropped volume.

    Parameters:
    - volume_path (str): Path to the volume file (.nii.gz).
    - roi_x (int): X-coordinate of the ROI top-left corner.
    - roi_y (int): Y-coordinate of the ROI top-left corner.
    - crop_size (int): Desired square crop size (default: 192).
    - output_dir (str): Directory to save the cropped volume.
    
    Returns:
    - None
    """
    volume_nifti = nib.load(volume_path)
    volume_data = volume_nifti.get_fdata()
    
    cropped_slices = []

    for i in range(volume_data.shape[2]):  # Loop through all slices
        img_slice = volume_data[:, :, i]
        
        # Calculate cropping bounds
        startx = max(0, roi_x)
        starty = max(0, roi_y)
        endx = startx + crop_size
        endy = starty + crop_size

        # Ensure cropping bounds are within image dimensions
        img_height, img_width = img_slice.shape[:2]
        if endx > img_width:
            startx = img_width - crop_size
            endx = img_width
        if endy > img_height:
            starty = img_height - crop_size
            endy = img_height

        # Crop the slice
        cropped_slice = img_slice[starty:endy, startx:endx]
        cropped_slices.append(cropped_slice)

    # Stack cropped slices and save as a new volume
    cropped_volume = np.stack(cropped_slices, axis=-1)
    cropped_nifti = nib.Nifti1Image(cropped_volume, affine=volume_nifti.affine)
    
    # Save the cropped volume
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{os.path.basename(volume_path).replace('.nii.gz', '_cropped.nii.gz')}")
        nib.save(cropped_nifti, output_path)
        print(f"Saved cropped volume: {output_path}")
    else:
        print("Output directory not provided; cropped volume not saved.")

def save_all_slices_as_png(volume_path, output_dir):
    """
    Save all slices of a volume as .png images.

    Parameters:
    - volume_path (str): Path to the volume file (.nii.gz).
    - output_dir (str): Directory to save the slices as .png images.

    Returns:
    - None
    """
    volume_nifti = nib.load(volume_path)
    volume_data = volume_nifti.get_fdata()

    # Create output directory for the volume
    volume_name = os.path.basename(volume_path).replace('.nii.gz', '')
    volume_output_dir = os.path.join(output_dir, f"{volume_name}_slices")
    os.makedirs(volume_output_dir, exist_ok=True)

    # Iterate through all slices and save as .png
    for i in range(volume_data.shape[2]):  # Loop through each slice
        img_slice = volume_data[:, :, i]
        output_path = os.path.join(volume_output_dir, f"slice_{i:03d}.png")
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img_slice, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f"Saved all slices as .png images in: {volume_output_dir}")

# Main script
if __name__ == "__main__":
    # Step 1: Get all volume paths for the FLAIR modality
    modality_keyword = "t2f"  # Keyword for FLAIR modality
    volume_paths = find_all_volume_paths(brats_data_validation, modality_keyword=modality_keyword)

    # Initialize variables to track the largest ROI globally
    global_largest_x, global_largest_y, global_largest_w, global_largest_h = 0, 0, 0, 0
    global_largest_frame = None
    global_largest_frame_cropped = None
    global_volume_name = None
    global_slice_index = None

    if not volume_paths:
        print(f"No volumes found for modality keyword '{modality_keyword}'.")
    else:
        print(f"Found {len(volume_paths)} volumes for modality '{modality_keyword}'.")
        for volume_path in volume_paths:
            largest_x, largest_y, largest_w, largest_h, largest_frame, largest_frame_cropped, largest_slice_index = find_largest_roi_in_volume(volume_path)

            # Update global largest ROI if this volume's ROI is larger
            if largest_w * largest_h > global_largest_w * global_largest_h:
                global_largest_x, global_largest_y, global_largest_w, global_largest_h = largest_x, largest_y, largest_w, largest_h
                global_largest_frame = largest_frame
                global_largest_frame_cropped = largest_frame_cropped
                global_volume_name = os.path.basename(volume_path).replace('.nii.gz', '')
                global_slice_index = largest_slice_index

        # Step 2: Save the global largest ROI combined image
        output_combined = os.path.join(output_dir, "global_largest_roi_combined.png")
        save_combined_image(global_largest_frame, global_largest_frame_cropped, output_combined)

        # Step 3: Print details of the global largest ROI
        print("\nOverall Largest ROI:")
        print(f"Volume: {global_volume_name}")
        print(f"Slice Index: {global_slice_index}")
        print(f"Largest ROI - x: {global_largest_x}, y: {global_largest_y}, width: {global_largest_w}, height: {global_largest_h}")
        print(f"Saved combined image: {output_combined}")

        # Step 4: Save all slices of the volume containing the global largest ROI as .png
        for volume_path in volume_paths:
            if os.path.basename(volume_path).replace('.nii.gz', '') == global_volume_name:
                save_all_slices_as_png(volume_path, output_dir)
