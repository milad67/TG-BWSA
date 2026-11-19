# changes when running on the cluster
# %% Imports
# changes when running on the cluster
import os
import tensorflow as tf
import logging

from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers.experimental import AdamW


# **Continue with Other Imports**
import time
import random
import re
import argparse
import shutil
import subprocess as sp
import configparser
import ctypes
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
import cv2
import pydicom as dicom
import nibabel as nib

from sklearn.model_selection import train_test_split
from medpy.metric.binary import hd, dc, hd95
import skimage.io as io
import skimage.transform as st
from sklearn.preprocessing import MinMaxScaler

import albumentations as A
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, MaxPooling1D, Dropout, UpSampling2D,
    concatenate, Reshape, Concatenate, BatchNormalization, Lambda, Dense, Layer
)
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback

# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')

# **TensorFlow Configurations**
# Disable layout autotuning
tf.config.optimizer.set_experimental_options({'autotune_layout': False})
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})


epsilon = 1e-5
smooth = 1

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, MaxPooling1D, Dropout, UpSampling2D,
    concatenate, Reshape, Concatenate, BatchNormalization, Lambda, Dense, Layer
)
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import albumentations as A

from tensorflow.keras.callbacks import Callback  # Add this import


# Disable layout autotuning
tf.config.optimizer.set_experimental_options({'autotune_layout': False})
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

TF_ENABLE_GPU_GARBAGE_COLLECTION = False

brats_data = "/scratch_net/maus/mrostamian/"
brats_data_train = brats_data + '/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
brats_data_test = brats_data + '/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'
brats_data_validation = brats_data + '/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'

img_cols, img_rows, col_channels = 224, 224, 155

n_classes = 4

dropout_rate = 0.3




def load_file_paths(data_dir, modalities=['t1c', 't1n', 't2f', 't2w']):
    sample_dict = {}

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_lower = file.lower()
            # Consider only NIfTI files
            if not (file_lower.endswith('.nii') or file_lower.endswith('.nii.gz')):
                continue
            
            # Remove extensions
            basename = file
            if basename.endswith('.gz'):
                basename = os.path.splitext(os.path.splitext(basename)[0])[0]
            else:
                basename = os.path.splitext(basename)[0]
            
            # Split filename using hyphens '-'
            parts = basename.split('-')
            if len(parts) < 2:
                continue  # Skip files not matching the pattern
            
            sample_id = '-'.join(parts[:-1])  # Extract patient ID
            modality = parts[-1]  # Extract modality
            
            # Collect paths if file corresponds to one of the desired modalities
            if modality in modalities:
                if sample_id not in sample_dict:
                    sample_dict[sample_id] = {}
                sample_dict[sample_id][modality] = os.path.join(root, file)

    # Build a list of samples that have all required modalities
    samples = []
    for sample_id, modalities_dict in sample_dict.items():
        # Only require the specified modalities, no mask requirement
        if all(mod in modalities_dict for mod in modalities):
            # Create an entry with paths for each modality
            sample_entry = {mod: modalities_dict[mod] for mod in modalities}
            samples.append(sample_entry)
        else:
            print(f"Sample {sample_id} is missing some modalities, skipping.")
    
    return samples



def load_all_slices_once(file_paths, input_size=(224, 224)):
    """
    Loads all data from disk exactly once, producing a list of slices in memory.

    Returns:
        all_slices: List of dictionaries, each containing 't1c', 't1n', 't2f', 't2w', 'mask'
                    for a single slice.
    """

    all_slices = []
    sample_count = 0

    # For debug or sample visualization
    save_sample_dir = '/scratch_net/maus/mrostamian/project/log_multi_v4_2GPU/visualize_img'
    os.makedirs(save_sample_dir, exist_ok=True)

    # Pre-load all slices
    for sample in file_paths:
        # Load images for all modalities
        imgs = {}
        for modality in ['t1c', 't1n', 't2f', 't2w']:
            img_path = sample[modality]
            img = nib.load(img_path).get_fdata()
            imgs[modality] = img

        # Load the mask
        mask_path = sample['mask']
        mask = nib.load(mask_path).get_fdata()

        # Assuming all modalities and the mask have the same dimensions
        num_slices = imgs['t1c'].shape[2]
        for idx in range(num_slices):
            img_slices = {}
            skip_slice = False
            for modality in ['t1c', 't1n', 't2f', 't2w']:
                # Crop and preprocess the image slice
                img_slice = crop_center(imgs[modality][:, :, idx], input_size[0], input_size[1])
                img_slice = np.expand_dims(img_slice, axis=-1)

                # Skip if the slice is empty
                if not np.any(img_slice):
                    skip_slice = True
                    break

                # Normalize the image slice
                img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-8)
                img_slices[modality] = img_slice

            if skip_slice:
                continue

            # Process the mask slice
            mask_slice = crop_center(mask[:, :, idx], input_size[0], input_size[1])
            mask_slice = np.expand_dims(mask_slice, axis=-1)

            # Append the slices to the all_slices list
            all_slices.append({
                't1c': img_slices['t1c'],
                't1n': img_slices['t1n'],
                't2f': img_slices['t2f'],
                't2w': img_slices['t2w'],
                'mask': mask_slice
            })

    print(f"[load_all_slices_once] Total slices loaded: {len(all_slices)}")

    # Shuffle all_slices once here
    random.shuffle(all_slices)

    return all_slices



def slice_generator_from_memory(all_slices, input_size=(224, 224), batch_size=32, augment=False, dset="training", single_pass=False):
    """
    Generator that operates on already loaded slices in memory.
    This function does NOT perform any disk I/O.

    Yields:
        (inputs, targets) per batch, matching the expected input_signature.
    """
    index = 0
    total_slices = len(all_slices)

    while True:
        if index >= total_slices:
            if single_pass:
                break
            # Reshuffle the data for the next epoch
            random.shuffle(all_slices)
            index = 0

        # Fetch the next batch of slices
        batch_slices = all_slices[index : index + batch_size]
        index += batch_size

        # If the last batch is smaller than batch_size, pad it by repeating some slices
        if len(batch_slices) < batch_size:
            batch_number = index // batch_size + 1  # Calculate the batch number
            print(f"\n[INFO] Batch {batch_number}: Original size is {len(batch_slices)}\n")
            
            num_to_pad = batch_size - len(batch_slices)

            # Handle edge case where the batch size is 1 or fewer slices are available
            if len(batch_slices) == 1:
                pad_indices = [0] * num_to_pad  # Repeat the only slice available
            else:
                pad_indices = random.choices(range(len(batch_slices)), k=num_to_pad)

            pad_slices = [batch_slices[i] for i in pad_indices]
            batch_slices.extend(pad_slices)

            print(f"\n[INFO] Batch {batch_number}: Padded size to {len(batch_slices)} with {num_to_pad} slices.\n")


        images_dict = {'t1c': [], 't1n': [], 't2f': [], 't2w': []}
        masks = []

        # Populate the images_dict and masks
        for slice_dict in batch_slices:
            try:
                for modality in ['t1c', 't1n', 't2f', 't2w']:
                    images_dict[modality].append(slice_dict[modality])
                masks.append(slice_dict['mask'])
            except KeyError as e:
                logging.error(f"Missing key {e} in slice_dict. Available keys: {list(slice_dict.keys())}")
                raise

        # Prepare the batch using the prepare_batch function
        try:
            inputs, targets = prepare_batch(images_dict, masks, augment, dset)
        except Exception as e:
            logging.error(f"Error preparing batch from indices {index - batch_size} to {index}: {e}")
            continue  # Skip this batch if an error occurs

        yield inputs, targets



def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=0, p=0.5),
        # A.Normalize(mean=(0.5,), std=(0.5,)),  # Include normalization
    ]
    additional_targets = {
        'image_t1n': 'image',
        'image_t2f': 'image',
        'image_t2w': 'image',
        'mask': 'mask',
    }
    return A.Compose(train_transform, additional_targets=additional_targets)

def get_validation_augmentation():
    return A.Compose([], additional_targets={
        'image_t1n': 'image',
        'image_t2f': 'image',
        'image_t2w': 'image',
        'mask': 'mask',
    })

def sep_gen(data, seed=42, batch_size=15, dset="training"):
    images, masks = data

    if dset == "training":
        augmentation = get_training_augmentation()
    elif dset in ["validation", "test"]:
        augmentation = A.Compose([A.Normalize(mean=(0.5,), std=(0.5,))], additional_targets={'mask': 'mask'})
    else:
        raise ValueError('The argument "dset" must be "training", "validation", or "test".')

    # Use the same seed for reproducibility
    np.random.seed(seed)

    while True:
        augmented_images, augmented_masks = [], []
        for i in range(len(images)):
            augmented = augmentation(image=images[i], mask=masks[i])
            augmented_images.append(augmented['image'])
            augmented_masks.append(augmented['mask'])

            # Yield a batch if it reaches the batch size
            if len(augmented_images) == batch_size:
                yield np.array(augmented_images), np.array(augmented_masks)
                augmented_images, augmented_masks = [], []

        # Handle the case when we need to process a single pass
        if dset in ["validation", "test"] and len(augmented_images) > 0:
            yield np.array(augmented_images), np.array(augmented_masks)
            break

def crop_center(img, cropx, cropy):
    y, x = img.shape[:2]
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty+cropy, startx:startx+cropx]


def prepare_batch(images_dict, masks, augment, dset):
    """
    Prepares a batch of data by applying augmentations, saving samples, and formatting inputs and targets.

    Parameters:
    - images_dict: Dictionary containing lists of images for each modality.
    - masks: List of mask slices.
    - augment: Boolean indicating whether to apply data augmentation.
    - dset: String indicating the dataset type ("training", "validation", or "test").

    Returns:
    - Tuple of (inputs_dict, targets_dict) ready for model consumption.
    """
    # Convert images and masks to NumPy arrays with correct data type
    images = {}
    for modality in ['t1c', 't1n', 't2f', 't2w']:
        images[modality] = np.array(images_dict[modality], dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)

    # Define augmentation
    if augment and dset == "training":
        augmentation = get_training_augmentation()
    else:
        # For validation and test sets, apply no augmentation
        augmentation = get_validation_augmentation()

    # Apply augmentation per sample
    augmented_images = {modality: [] for modality in ['t1c', 't1n', 't2f', 't2w']}
    augmented_masks = []
    sample_count = 0  # Counter for saving samples

    # save_sample_dir = '/scratch_net/maus/mrostamian/project/log_multi_v4_2GPU/visualize_img'  # Ensure this path is correct

    for i in range(len(masks)):
        data = {}
        # Designate 't1c' as the main 'image' for albumentations
        data['image'] = images['t1c'][i]  # Main image for augmentation
        # Add other modalities as additional targets
        data['image_t1n'] = images['t1n'][i]
        data['image_t2f'] = images['t2f'][i]
        data['image_t2w'] = images['t2w'][i]
        data['mask'] = masks[i]

        # Apply augmentation
        augmented = augmentation(**data)

        # Collect augmented images and masks
        augmented_images['t1c'].append(augmented['image'])
        augmented_images['t1n'].append(augmented['image_t1n'])
        augmented_images['t2f'].append(augmented['image_t2f'])
        augmented_images['t2w'].append(augmented['image_t2w'])
        augmented_masks.append(augmented['mask'])

        # # Save sample images and masks as a figure (first 50 slices)
        # if save_sample_dir and sample_count < 30:
        #     fig, axs = plt.subplots(1, 5, figsize=(15, 4))  # 1 row and 5 plots
        #     modalities_display = ['t1c', 't1n', 't2f', 't2w']
        #     for j, modality in enumerate(modalities_display):
        #         axs[j].imshow(np.squeeze(augmented_images[modality][-1]), cmap='gray')  # Display the image
        #         axs[j].set_title(modality.upper())  # Add the title of the image
        #         axs[j].axis('off')  # Turn off axis labels

        #     # Display Mask
        #     axs[4].imshow(np.squeeze(augmented_masks[-1]), cmap='gray')
        #     axs[4].set_title('MASK')
        #     axs[4].axis('off')

        #     plt.tight_layout()
        #     plt.savefig(os.path.join(save_sample_dir, f'sample_aug_{sample_count}.png'))
        #     plt.close(fig)
        #     sample_count += 1

    # Convert lists back to NumPy arrays with correct data type
    for modality in ['t1c', 't1n', 't2f', 't2w']:
        images[modality] = np.array(augmented_images[modality], dtype=np.float32)
    masks = np.array(augmented_masks, dtype=np.float32)

    # Convert masks to one-hot encoding
    masks_converted = convert_brats_masks(masks)  # Ensure this returns correct format and num_classes

    # Downsample masks as per your model's architecture
    y1 = masks_converted[:, ::4, ::4, :]  # For pred1 (56x56)
    y2 = masks_converted[:, ::2, ::2, :]  # For pred2 (112x112)
    y3 = masks_converted  # For final (224x224)

    # Prepare inputs as a dictionary with keys matching input layer names
    inputs = {
        't1n_input': images['t1n'],
        't1c_input': images['t1c'],
        't2w_input': images['t2w'],
        't2f_input': images['t2f']
    }

    # Prepare targets as a dictionary with keys matching model outputs
    targets = {
        'pred1': y1,
        'pred2': y2,
        'final': y3
    }

    return inputs, targets

def convert_brats_masks(y):

    # If y has more than 3 dimensions (e.g., batch size, channels), handle accordingly
    if len(y.shape) > 3:
        y = np.squeeze(y)  # Squeeze out extra dimensions (if they are 1)

    # Ensure that y now has 3 dimensions
    if len(y.shape) != 3:
        raise ValueError(f"Expected 3 dimensions for the mask, but got {len(y.shape)} dimensions")
    
    masks = np.zeros((*y.shape, 4), dtype=np.uint8)

    masks[:, :, :, 0] = (y == 0).astype(np.uint8)  # bg
    masks[:, :, :, 1] = (y == 1).astype(np.uint8)  
    masks[:, :, :, 2] = (y == 2).astype(np.uint8)  
    masks[:, :, :, 3] = (y == 3).astype(np.uint8)  

    return masks


#------------------------------------------------Model---------------------------------------------------------


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Layer, Input, Conv2D, MaxPooling2D, 
    UpSampling2D, Dropout, concatenate, Lambda, BatchNormalization, Activation, Reshape, Concatenate, DepthwiseConv2D
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization



# from tensorflow.keras.layers import Layer, Dense, Conv2D, Dropout, MultiHeadAttention, BatchNormalization, \
# DepthwiseConv2D, UpSampling2D, Flatten
# from tensorflow.keras.models import Sequential
# from tensorflow import Tensor, divide, concat, random, split, reshape, transpose, float32
# from typing import List, Union, Iterable
# import tensorflow as tf
# from tensorflow.keras import layers, Sequential
# import numpy as np

# Global training flag
training = True

# Adjusted number of filters for five encoder blocks
filters_list = [32, 64, 128, 256, 512]  # Encoder filters
filters_decoder = [256, 128, 64, 32]     # Decoder filters

# Combine filters for the entire network
filters = filters_list + filters_decoder

# Adjust attention heads for each block
att_heads = [2] * len(filters)

# Adjust blocks to include the five encoder blocks and decoder blocks
blocks = len(filters)

# Compute drop path rate for stochastic depth
stochastic_depth_rate = 0.0
dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]

# Number of classes
num_classes = 4  # Adjust based on your dataset

from tensorflow.keras.layers import Layer, MultiHeadAttention, Dropout, DepthwiseConv2D, LayerNormalization, Conv2D, Add
from tensorflow.keras import Sequential
import tensorflow as tf




#--------------------------------------------------------------------------------------------------------------


def create_pretrained_encoder(
    input_shape,
    filters,
    initializer='he_normal',
    act='relu',
    drp_out=0,
    name='encoder_model',
    name_prefix='encoder'
):
    """
    Reconstructs the pretrained encoder model without attention, retaining the original
    architecture and layer names (except for the attention layers), including collecting
    feature maps.

    Returns:
    - encoder_model: The encoder model
    - encoder_feature_map_model: Model for extracting feature maps
    """
    inputs = tf.keras.Input(shape=input_shape, name=f'{name_prefix}_input')
    x = inputs
    skip_connections = []
    encoder_feature_maps = []  # Collect feature maps at each block

    # Block 1
    x1 = layers.LayerNormalization(epsilon=1e-5, name=f'{name_prefix}_block1_norm')(x)
    x1 = layers.Lambda(
        lambda x: x[:, :, :, -1:],
        output_shape=lambda s: (s[0], s[1], s[2], 1),
        name=f'{name_prefix}_block1_lambda'
    )(x1)
    x1 = layers.Conv2D(
        filters[0], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_block1_conv1'
    )(x1)
    x1 = layers.Conv2D(
        filters[0], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_block1_conv2'
    )(x1)
    x1 = layers.Dropout(drp_out, name=f'{name_prefix}_block1_dropout')(x1)
    x1 = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_block1_pool')(x1)
    skip_connections.append(x1)
    encoder_feature_maps.append(x1)

    # Block 2
    x2 = layers.LayerNormalization(epsilon=1e-5, name=f'{name_prefix}_block2_norm')(x1)
    scale_img_2 = layers.AveragePooling2D(2, 2, name=f'{name_prefix}_block2_scale')(inputs)
    scale_conv2 = layers.Conv2D(
        filters[0], 3, padding='same', activation=act,
        name=f'{name_prefix}_block2_scale_conv'
    )(scale_img_2)
    x2 = layers.concatenate([scale_conv2, x2], axis=-1, name=f'{name_prefix}_block2_concat')
    x2 = layers.Conv2D(
        filters[1], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_block2_conv1'
    )(x2)
    x2 = layers.Conv2D(
        filters[1], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_block2_conv2'
    )(x2)
    x2 = layers.Dropout(drp_out, name=f'{name_prefix}_block2_dropout')(x2)
    x2 = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_block2_pool')(x2)
    skip_connections.append(x2)
    encoder_feature_maps.append(x2)

    # Block 3
    x3 = layers.LayerNormalization(epsilon=1e-5, name=f'{name_prefix}_block3_norm')(x2)
    scale_img_3 = layers.AveragePooling2D(2, 2, name=f'{name_prefix}_block3_scale')(scale_img_2)
    scale_conv3 = layers.Conv2D(
        filters[1], 3, padding='same', activation=act,
        name=f'{name_prefix}_block3_scale_conv'
    )(scale_img_3)
    x3 = layers.concatenate([scale_conv3, x3], axis=-1, name=f'{name_prefix}_block3_concat')
    x3 = layers.Conv2D(
        filters[2], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_block3_conv1'
    )(x3)
    x3 = layers.Conv2D(
        filters[2], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_block3_conv2'
    )(x3)
    x3 = layers.Dropout(drp_out, name=f'{name_prefix}_block3_dropout')(x3)
    x3 = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_block3_pool')(x3)
    skip_connections.append(x3)
    encoder_feature_maps.append(x3)

    # Block 4
    x4 = layers.LayerNormalization(epsilon=1e-5, name=f'{name_prefix}_block4_norm')(x3)
    scale_img_4 = layers.AveragePooling2D(2, 2, name=f'{name_prefix}_block4_scale')(scale_img_3)
    scale_conv4 = layers.Conv2D(
        filters[2], 3, padding='same', activation=act,
        name=f'{name_prefix}_block4_scale_conv'
    )(scale_img_4)
    x4 = layers.concatenate([scale_conv4, x4], axis=-1, name=f'{name_prefix}_block4_concat')
    x4 = layers.Conv2D(
        filters[3], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_block4_conv1'
    )(x4)
    x4 = layers.Conv2D(
        filters[3], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_block4_conv2'
    )(x4)
    x4 = layers.Dropout(drp_out, name=f'{name_prefix}_block4_dropout')(x4)
    x4 = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_block4_pool')(x4)
    skip_connections.append(x4)
    encoder_feature_maps.append(x4)

    # Bottleneck
    x5 = layers.LayerNormalization(epsilon=1e-5, name=f'{name_prefix}_bottleneck_norm')(x4)
    x5 = layers.Conv2D(
        filters[4], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_bottleneck_conv1'
    )(x5)
    x5 = layers.Conv2D(
        filters[4], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_bottleneck_conv2'
    )(x5)
    x5 = layers.Dropout(drp_out, name=f'{name_prefix}_bottleneck_dropout')(x5)
    x5 = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_bottleneck_pool')(x5)
    encoder_output = x5
    encoder_feature_maps.append(x5)

    # Create the encoder model
    outputs = [encoder_output] + skip_connections
    encoder_model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    # Create a model for extracting feature maps, if needed
    encoder_feature_map_model = tf.keras.Model(
        inputs=inputs, outputs=encoder_feature_maps, name=f'{name}_features'
    )

    return encoder_model, encoder_feature_map_model


def build_decoder(
    adapter_output,
    encoder_skips,
    filters_decoder,
    num_classes=4
):
    initializer = 'he_normal'
    act = 'relu'
    dropout_rate = 0.3  # Adjusted dropout rate

    x = adapter_output
    outputs = {}

    # Reverse the skip connections to match the decoder order
    encoder_skips = encoder_skips[::-1]

    # For capturing features at specific decoder levels for outputs
    skip7 = skip8 = skip9 = None

    num_blocks = len(filters_decoder)
    for block_idx in range(num_blocks):
        # Get the corresponding skip connection
        skip_connection = encoder_skips[block_idx]
        print(f"Block {block_idx}: Skip shape {skip_connection.shape}, X before upsample {x.shape}")

        # Upsample x by a factor of 2
        x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear',
                                name=f'decoder_upsample_{block_idx + 1}')(x)

        print(f"Block {block_idx}: X after upsample {x.shape}")

        # Concatenate skip connection with decoder output
        x = layers.Concatenate(axis=-1, name=f'decoder_concat_{block_idx + 1}')([x, skip_connection])

        # Decoder convolutions with adjustments
        x = layers.LayerNormalization(epsilon=1e-5, name=f'decoder_norm_{block_idx + 1}')(x)
        x = layers.Conv2D(
            filters_decoder[block_idx],
            kernel_size=3,
            padding='same',
            activation=None,  # Activation applied separately
            kernel_initializer=initializer,
            name=f'decoder_conv1_{block_idx + 1}'
        )(x)
        x = layers.BatchNormalization(name=f'decoder_bn1_{block_idx + 1}')(x)
        x = layers.Activation(act, name=f'decoder_act1_{block_idx + 1}')(x)
        x = layers.SpatialDropout2D(dropout_rate, name=f'decoder_dropout1_{block_idx + 1}')(x)

        x = layers.Conv2D(
            filters_decoder[block_idx],
            kernel_size=3,
            padding='same',
            activation=None,  # Activation applied separately
            kernel_initializer=initializer,
            name=f'decoder_conv2_{block_idx + 1}'
        )(x)
        x = layers.BatchNormalization(name=f'decoder_bn2_{block_idx + 1}')(x)
        x = layers.Activation(act, name=f'decoder_act2_{block_idx + 1}')(x)
        x = layers.SpatialDropout2D(dropout_rate, name=f'decoder_dropout2_{block_idx + 1}')(x)

        # Removed the attention layer from the decoder

        # Store the features at specific decoder levels for outputs
        if block_idx == num_blocks - 3:
            skip7 = x  # For pred1
        elif block_idx == num_blocks - 2:
            skip8 = x  # For pred2
        elif block_idx == num_blocks - 1:
            skip9 = x  # For final output

    # Now, generate the outputs using the collected features

    # Output at skip7 (pred1)
    x_pred1 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='upsample_pred1')(skip7)
    x_pred1 = layers.LayerNormalization(epsilon=1e-5, name='decoder_norm_pred1')(x_pred1)
    x_pred1 = layers.Conv2D(
        filters_decoder[-3],
        kernel_size=3,
        padding='same',
        activation=None,
        kernel_initializer=initializer,
        name='decoder_conv_pred1_1'
    )(x_pred1)
    x_pred1 = layers.BatchNormalization(name='decoder_bn_pred1_1')(x_pred1)
    x_pred1 = layers.Activation(act, name='decoder_act_pred1_1')(x_pred1)
    x_pred1 = layers.SpatialDropout2D(dropout_rate, name='decoder_dropout_pred1_1')(x_pred1)

    x_pred1 = layers.Conv2D(
        filters_decoder[-3],
        kernel_size=3,
        padding='same',
        activation=None,
        kernel_initializer=initializer,
        name='decoder_conv_pred1_2'
    )(x_pred1)
    x_pred1 = layers.BatchNormalization(name='decoder_bn_pred1_2')(x_pred1)
    x_pred1 = layers.Activation(act, name='decoder_act_pred1_2')(x_pred1)
    x_pred1 = layers.SpatialDropout2D(dropout_rate, name='decoder_dropout_pred1_2')(x_pred1)

    outputs['pred1'] = layers.Conv2D(
        num_classes,
        kernel_size=1,
        activation='softmax',
        name='pred1'
    )(x_pred1)

    # Output at skip8 (pred2)
    x_pred2 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='upsample_pred2')(skip8)
    x_pred2 = layers.LayerNormalization(epsilon=1e-5, name='decoder_norm_pred2')(x_pred2)
    x_pred2 = layers.Conv2D(
        filters_decoder[-2],
        kernel_size=3,
        padding='same',
        activation=None,
        kernel_initializer=initializer,
        name='decoder_conv_pred2_1'
    )(x_pred2)
    x_pred2 = layers.BatchNormalization(name='decoder_bn_pred2_1')(x_pred2)
    x_pred2 = layers.Activation(act, name='decoder_act_pred2_1')(x_pred2)
    x_pred2 = layers.SpatialDropout2D(dropout_rate, name='decoder_dropout_pred2_1')(x_pred2)

    x_pred2 = layers.Conv2D(
        filters_decoder[-2],
        kernel_size=3,
        padding='same',
        activation=None,
        kernel_initializer=initializer,
        name='decoder_conv_pred2_2'
    )(x_pred2)
    x_pred2 = layers.BatchNormalization(name='decoder_bn_pred2_2')(x_pred2)
    x_pred2 = layers.Activation(act, name='decoder_act_pred2_2')(x_pred2)
    x_pred2 = layers.SpatialDropout2D(dropout_rate, name='decoder_dropout_pred2_2')(x_pred2)

    outputs['pred2'] = layers.Conv2D(
        num_classes,
        kernel_size=1,
        activation='softmax',
        name='pred2'
    )(x_pred2)

    # Output at skip9 (final output)
    x_pred3 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='upsample_final')(skip9)
    x_pred3 = layers.LayerNormalization(epsilon=1e-5, name='decoder_norm_final')(x_pred3)
    x_pred3 = layers.Conv2D(
        filters_decoder[-1],
        kernel_size=3,
        padding='same',
        activation=None,
        kernel_initializer=initializer,
        name='decoder_conv_final_1'
    )(x_pred3)
    x_pred3 = layers.BatchNormalization(name='decoder_bn_final_1')(x_pred3)
    x_pred3 = layers.Activation(act, name='decoder_act_final_1')(x_pred3)
    x_pred3 = layers.SpatialDropout2D(dropout_rate, name='decoder_dropout_final_1')(x_pred3)

    x_pred3 = layers.Conv2D(
        filters_decoder[-1],
        kernel_size=3,
        padding='same',
        activation=None,
        kernel_initializer=initializer,
        name='decoder_conv_final_2'
    )(x_pred3)
    x_pred3 = layers.BatchNormalization(name='decoder_bn_final_2')(x_pred3)
    x_pred3 = layers.Activation(act, name='decoder_act_final_2')(x_pred3)
    x_pred3 = layers.SpatialDropout2D(dropout_rate, name='decoder_dropout_final_2')(x_pred3)

    outputs['final'] = layers.Conv2D(
        num_classes,
        kernel_size=1,
        activation='softmax',
        name='final'
    )(x_pred3)

    return outputs



#-------New Adapter--------------------------------------------------------------------------------------------------------------

from tensorflow.keras import layers, regularizers


class ScalingLayer(layers.Layer):
    """Dual scaling layer with regularization adjustments for early blocks"""
    def __init__(self, initial_alpha=0.5, initial_beta=1.0, **kwargs):
        # Adjust initial beta for Block 0 to preserve original features
        if "block0" in kwargs.get("name", ""):
            initial_alpha = .5
            initial_beta = 1.0  # Stronger preservation of original features
        
        super().__init__(**kwargs)
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta

    def build(self, input_shape):
        # Extract block index from layer name (e.g., "block0" -> 0)
        self.block_idx = int(self.name.split("_")[2][5:]) if "block" in self.name else 0
        
        # Disable L2 regularization for Block 0
        reg = tf.keras.regularizers.l2(1e-4) if self.block_idx > 0 else None
        
        self.alpha = self.add_weight(
            name='alpha',
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_alpha),
            regularizer=reg,  # No regularization for Block 0
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_beta),
            regularizer=reg,  # No regularization for Block 0
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        adapter_out, original = inputs
        # Add a small residual to prevent total feature collapse
        return self.alpha * adapter_out + self.beta * original + 1e-3 * original

def se_block(input_tensor, reduction_ratio=16, dropout_rate=0.1, name_prefix=''):
    """Original SE Block implementation"""
    filters = input_tensor.shape[-1]
    se_shape = (1, 1, filters)
    
    se = layers.GlobalAveragePooling2D(name=f'{name_prefix}_se_gap')(input_tensor)
    se = layers.Reshape(se_shape, name=f'{name_prefix}_se_reshape')(se)
    
    se = layers.Dense(filters // reduction_ratio, 
                      activation='relu', 
                      name=f'{name_prefix}_se_dense1')(se)
    se = layers.Dropout(rate=dropout_rate, name=f'{name_prefix}_se_dropout1')(se)
    se = layers.Dense(filters, 
                      activation='sigmoid', 
                      name=f'{name_prefix}_se_dense2')(se)
    
    return layers.Multiply(name=f'{name_prefix}_se_scale')([input_tensor, se])

def adapter_between_encoders(cnn_outputs_block, act='relu', compression_ratio=0.5, 
                             dropout_rate=0.1, block_idx=0):
    """Improved adapter with dual scaling and original SE block"""
    num_encoders = len(cnn_outputs_block)
    filters = cnn_outputs_block[0].shape[-1]
    compressed_filters = max(8, int(filters * compression_ratio))  # Minimum 8 channels

    # Compression path
    compressed_outputs = []
    for i, x in enumerate(cnn_outputs_block):
        x_compressed = layers.Conv2D(
            compressed_filters, (1, 1),
            kernel_initializer='he_normal',
            padding='same',
            name=f'adapter_comp_conv_block{block_idx}_{i}'
        )(x)
        x_compressed = layers.BatchNormalization(
            gamma_initializer='ones',
            name=f'adapter_comp_bn_block{block_idx}_{i}'
        )(x_compressed)
        x_compressed = layers.Activation(act, 
                                         name=f'adapter_comp_act_block{block_idx}_{i}')(x_compressed)
        x_compressed = layers.Dropout(rate=dropout_rate, 
                                      name=f'adapter_comp_dropout_block{block_idx}_{i}')(x_compressed)
        compressed_outputs.append(x_compressed)

    # Concatenate and apply SE block
    combined = layers.Concatenate(axis=-1, 
                                  name=f'adapter_combined_block{block_idx}')(compressed_outputs)
    se_processed = se_block(combined, 
                            reduction_ratio=16, 
                            dropout_rate=dropout_rate, 
                            name_prefix=f'adapter_se_block{block_idx}')

    # Lightweight processing: Separable convolution block
    attended = layers.SeparableConv2D(
        compressed_filters, (3, 3),
        padding='same',
        name=f'adapter_sep_conv_block{block_idx}'
    )(se_processed)
    attended = layers.BatchNormalization(
        name=f'adapter_sep_conv_bn_block{block_idx}')(attended)
    attended = layers.Activation(act, 
                                 name=f'adapter_sep_conv_act_block{block_idx}')(attended)
    attended = layers.Dropout(rate=dropout_rate, 
                              name=f'adapter_sep_conv_dropout_block{block_idx}')(attended)

    # Expansion and dual scaling
    adapted_outputs = []
    for i in range(num_encoders):
        x_expanded = layers.Conv2D(
            filters, (1, 1), 
            padding='same',
            name=f'adapter_exp_conv_block{block_idx}_{i}'
        )(attended)
        x_expanded = layers.BatchNormalization(
            name=f'adapter_exp_bn_block{block_idx}_{i}')(x_expanded)
        x_expanded = layers.Activation(act, 
                                      name=f'adapter_exp_act_block{block_idx}_{i}')(x_expanded)
        
        # Dual scaling with original features
        x_scaled = ScalingLayer(
            initial_alpha=0.5,  # Start with stronger adapter contribution
            initial_beta=1.0,   # Preserve original features
            name=f'scaling_layer_block{block_idx}_{i}'
        )([x_expanded, cnn_outputs_block[i]])
        
        adapted_outputs.append(x_scaled)
    
    return adapted_outputs


#--------------------------------------------------------------------------------------------------------------

def create_pretrained_encoder(
    input_shape,
    filters,
    initializer='he_normal',
    act='relu',
    drp_out=0,
    name='encoder_model',
    name_prefix='encoder'
):
    """
    Reconstructs the pretrained encoder model without attention, retaining the original
    architecture and layer names (except for the attention layers), including collecting
    feature maps.

    Returns:
    - encoder_model: The encoder model
    - encoder_feature_map_model: Model for extracting feature maps
    """
    inputs = tf.keras.Input(shape=input_shape, name=f'{name_prefix}_input')
    x = inputs
    skip_connections = []
    encoder_feature_maps = []  # Collect feature maps at each block

    # Block 1
    x1 = layers.LayerNormalization(epsilon=1e-5, name=f'{name_prefix}_block1_norm')(x)
    x1 = layers.Lambda(
        lambda x: x[:, :, :, -1:],
        output_shape=lambda s: (s[0], s[1], s[2], 1),
        name=f'{name_prefix}_block1_lambda'
    )(x1)
    x1 = layers.Conv2D(
        filters[0], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_block1_conv1'
    )(x1)
    x1 = layers.Conv2D(
        filters[0], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_block1_conv2'
    )(x1)
    x1 = layers.Dropout(drp_out, name=f'{name_prefix}_block1_dropout')(x1)
    x1 = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_block1_pool')(x1)
    skip_connections.append(x1)
    encoder_feature_maps.append(x1)

    # Block 2
    x2 = layers.LayerNormalization(epsilon=1e-5, name=f'{name_prefix}_block2_norm')(x1)
    scale_img_2 = layers.AveragePooling2D(2, 2, name=f'{name_prefix}_block2_scale')(inputs)
    scale_conv2 = layers.Conv2D(
        filters[0], 3, padding='same', activation=act,
        name=f'{name_prefix}_block2_scale_conv'
    )(scale_img_2)
    x2 = layers.concatenate([scale_conv2, x2], axis=-1, name=f'{name_prefix}_block2_concat')
    x2 = layers.Conv2D(
        filters[1], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_block2_conv1'
    )(x2)
    x2 = layers.Conv2D(
        filters[1], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_block2_conv2'
    )(x2)
    x2 = layers.Dropout(drp_out, name=f'{name_prefix}_block2_dropout')(x2)
    x2 = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_block2_pool')(x2)
    skip_connections.append(x2)
    encoder_feature_maps.append(x2)

    # Block 3
    x3 = layers.LayerNormalization(epsilon=1e-5, name=f'{name_prefix}_block3_norm')(x2)
    scale_img_3 = layers.AveragePooling2D(2, 2, name=f'{name_prefix}_block3_scale')(scale_img_2)
    scale_conv3 = layers.Conv2D(
        filters[1], 3, padding='same', activation=act,
        name=f'{name_prefix}_block3_scale_conv'
    )(scale_img_3)
    x3 = layers.concatenate([scale_conv3, x3], axis=-1, name=f'{name_prefix}_block3_concat')
    x3 = layers.Conv2D(
        filters[2], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_block3_conv1'
    )(x3)
    x3 = layers.Conv2D(
        filters[2], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_block3_conv2'
    )(x3)
    x3 = layers.Dropout(drp_out, name=f'{name_prefix}_block3_dropout')(x3)
    x3 = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_block3_pool')(x3)
    skip_connections.append(x3)
    encoder_feature_maps.append(x3)

    # Block 4
    x4 = layers.LayerNormalization(epsilon=1e-5, name=f'{name_prefix}_block4_norm')(x3)
    scale_img_4 = layers.AveragePooling2D(2, 2, name=f'{name_prefix}_block4_scale')(scale_img_3)
    scale_conv4 = layers.Conv2D(
        filters[2], 3, padding='same', activation=act,
        name=f'{name_prefix}_block4_scale_conv'
    )(scale_img_4)
    x4 = layers.concatenate([scale_conv4, x4], axis=-1, name=f'{name_prefix}_block4_concat')
    x4 = layers.Conv2D(
        filters[3], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_block4_conv1'
    )(x4)
    x4 = layers.Conv2D(
        filters[3], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_block4_conv2'
    )(x4)
    x4 = layers.Dropout(drp_out, name=f'{name_prefix}_block4_dropout')(x4)
    x4 = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_block4_pool')(x4)
    skip_connections.append(x4)
    encoder_feature_maps.append(x4)

    # Bottleneck
    x5 = layers.LayerNormalization(epsilon=1e-5, name=f'{name_prefix}_bottleneck_norm')(x4)
    x5 = layers.Conv2D(
        filters[4], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_bottleneck_conv1'
    )(x5)
    x5 = layers.Conv2D(
        filters[4], 3, padding='same', activation=act, kernel_initializer=initializer,
        name=f'{name_prefix}_bottleneck_conv2'
    )(x5)
    x5 = layers.Dropout(drp_out, name=f'{name_prefix}_bottleneck_dropout')(x5)
    x5 = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_bottleneck_pool')(x5)
    encoder_output = x5
    encoder_feature_maps.append(x5)

    # Create the encoder model
    outputs = [encoder_output] + skip_connections
    encoder_model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    # Create a model for extracting feature maps, if needed
    encoder_feature_map_model = tf.keras.Model(
        inputs=inputs, outputs=encoder_feature_maps, name=f'{name}_features'
    )

    return encoder_model, encoder_feature_map_model


def build_decoder(
    adapter_output,
    encoder_skips,
    filters_decoder,
    num_classes=4
):
    initializer = 'he_normal'
    act = 'relu'
    dropout_rate = 0.3  # Adjusted dropout rate

    x = adapter_output
    outputs = {}

    # Reverse the skip connections to match the decoder order
    encoder_skips = encoder_skips[::-1]

    # For capturing features at specific decoder levels for outputs
    skip7 = skip8 = skip9 = None

    num_blocks = len(filters_decoder)
    for block_idx in range(num_blocks):
        # Get the corresponding skip connection
        skip_connection = encoder_skips[block_idx]
        print(f"Block {block_idx}: Skip shape {skip_connection.shape}, X before upsample {x.shape}")

        # Upsample x by a factor of 2
        x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear',
                                name=f'decoder_upsample_{block_idx + 1}')(x)

        print(f"Block {block_idx}: X after upsample {x.shape}")

        # Concatenate skip connection with decoder output
        x = layers.Concatenate(axis=-1, name=f'decoder_concat_{block_idx + 1}')([x, skip_connection])

        # Decoder convolutions with adjustments
        x = layers.LayerNormalization(epsilon=1e-5, name=f'decoder_norm_{block_idx + 1}')(x)
        x = layers.Conv2D(
            filters_decoder[block_idx],
            kernel_size=3,
            padding='same',
            activation=None,  # Activation applied separately
            kernel_initializer=initializer,
            name=f'decoder_conv1_{block_idx + 1}'
        )(x)
        x = layers.BatchNormalization(name=f'decoder_bn1_{block_idx + 1}')(x)
        x = layers.Activation(act, name=f'decoder_act1_{block_idx + 1}')(x)
        x = layers.SpatialDropout2D(dropout_rate, name=f'decoder_dropout1_{block_idx + 1}')(x)

        x = layers.Conv2D(
            filters_decoder[block_idx],
            kernel_size=3,
            padding='same',
            activation=None,  # Activation applied separately
            kernel_initializer=initializer,
            name=f'decoder_conv2_{block_idx + 1}'
        )(x)
        x = layers.BatchNormalization(name=f'decoder_bn2_{block_idx + 1}')(x)
        x = layers.Activation(act, name=f'decoder_act2_{block_idx + 1}')(x)
        x = layers.SpatialDropout2D(dropout_rate, name=f'decoder_dropout2_{block_idx + 1}')(x)

        # Removed the attention layer from the decoder

        # Store the features at specific decoder levels for outputs
        if block_idx == num_blocks - 3:
            skip7 = x  # For pred1
        elif block_idx == num_blocks - 2:
            skip8 = x  # For pred2
        elif block_idx == num_blocks - 1:
            skip9 = x  # For final output

    # Now, generate the outputs using the collected features

    # Output at skip7 (pred1)
    x_pred1 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='upsample_pred1')(skip7)
    x_pred1 = layers.LayerNormalization(epsilon=1e-5, name='decoder_norm_pred1')(x_pred1)
    x_pred1 = layers.Conv2D(
        filters_decoder[-3],
        kernel_size=3,
        padding='same',
        activation=None,
        kernel_initializer=initializer,
        name='decoder_conv_pred1_1'
    )(x_pred1)
    x_pred1 = layers.BatchNormalization(name='decoder_bn_pred1_1')(x_pred1)
    x_pred1 = layers.Activation(act, name='decoder_act_pred1_1')(x_pred1)
    x_pred1 = layers.SpatialDropout2D(dropout_rate, name='decoder_dropout_pred1_1')(x_pred1)

    x_pred1 = layers.Conv2D(
        filters_decoder[-3],
        kernel_size=3,
        padding='same',
        activation=None,
        kernel_initializer=initializer,
        name='decoder_conv_pred1_2'
    )(x_pred1)
    x_pred1 = layers.BatchNormalization(name='decoder_bn_pred1_2')(x_pred1)
    x_pred1 = layers.Activation(act, name='decoder_act_pred1_2')(x_pred1)
    x_pred1 = layers.SpatialDropout2D(dropout_rate, name='decoder_dropout_pred1_2')(x_pred1)

    outputs['pred1'] = layers.Conv2D(
        num_classes,
        kernel_size=1,
        activation='softmax',
        name='pred1'
    )(x_pred1)

    # Output at skip8 (pred2)
    x_pred2 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='upsample_pred2')(skip8)
    x_pred2 = layers.LayerNormalization(epsilon=1e-5, name='decoder_norm_pred2')(x_pred2)
    x_pred2 = layers.Conv2D(
        filters_decoder[-2],
        kernel_size=3,
        padding='same',
        activation=None,
        kernel_initializer=initializer,
        name='decoder_conv_pred2_1'
    )(x_pred2)
    x_pred2 = layers.BatchNormalization(name='decoder_bn_pred2_1')(x_pred2)
    x_pred2 = layers.Activation(act, name='decoder_act_pred2_1')(x_pred2)
    x_pred2 = layers.SpatialDropout2D(dropout_rate, name='decoder_dropout_pred2_1')(x_pred2)

    x_pred2 = layers.Conv2D(
        filters_decoder[-2],
        kernel_size=3,
        padding='same',
        activation=None,
        kernel_initializer=initializer,
        name='decoder_conv_pred2_2'
    )(x_pred2)
    x_pred2 = layers.BatchNormalization(name='decoder_bn_pred2_2')(x_pred2)
    x_pred2 = layers.Activation(act, name='decoder_act_pred2_2')(x_pred2)
    x_pred2 = layers.SpatialDropout2D(dropout_rate, name='decoder_dropout_pred2_2')(x_pred2)

    outputs['pred2'] = layers.Conv2D(
        num_classes,
        kernel_size=1,
        activation='softmax',
        name='pred2'
    )(x_pred2)

    # Output at skip9 (final output)
    x_pred3 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='upsample_final')(skip9)
    x_pred3 = layers.LayerNormalization(epsilon=1e-5, name='decoder_norm_final')(x_pred3)
    x_pred3 = layers.Conv2D(
        filters_decoder[-1],
        kernel_size=3,
        padding='same',
        activation=None,
        kernel_initializer=initializer,
        name='decoder_conv_final_1'
    )(x_pred3)
    x_pred3 = layers.BatchNormalization(name='decoder_bn_final_1')(x_pred3)
    x_pred3 = layers.Activation(act, name='decoder_act_final_1')(x_pred3)
    x_pred3 = layers.SpatialDropout2D(dropout_rate, name='decoder_dropout_final_1')(x_pred3)

    x_pred3 = layers.Conv2D(
        filters_decoder[-1],
        kernel_size=3,
        padding='same',
        activation=None,
        kernel_initializer=initializer,
        name='decoder_conv_final_2'
    )(x_pred3)
    x_pred3 = layers.BatchNormalization(name='decoder_bn_final_2')(x_pred3)
    x_pred3 = layers.Activation(act, name='decoder_act_final_2')(x_pred3)
    x_pred3 = layers.SpatialDropout2D(dropout_rate, name='decoder_dropout_final_2')(x_pred3)

    outputs['final'] = layers.Conv2D(
        num_classes,
        kernel_size=1,
        activation='softmax',
        name='final'
    )(x_pred3)

    return outputs



def load_pretrained_encoder(modality, input_shape, filters, freeze=True, drp_out=0):
    """
    Loads pretrained encoder WITHOUT attention parameters.
    """
    # Reconstruct encoder without attention
    pretrained_encoder, _ = create_pretrained_encoder(
        input_shape=input_shape,
        filters=filters,
        drp_out=drp_out,
        name=f'pretrained_encoder_{modality}',
        name_prefix=modality
    )

    # # Load weights (assuming weights were saved without attention layers)
    # pretrained_weights_path = f'/scratch_net/maus/mrostamian/project/weights/best_encoder_weights_{modality}.weights.h5'
    # pretrained_encoder.load_weights(pretrained_weights_path, by_name=True)

    # # Freeze if needed
    # if freeze:
    #     for layer in pretrained_encoder.layers:
    #         layer.trainable = False

    return pretrained_encoder



def create_multi_encoder_unet(inputs, num_encoders=4, training=True, num_classes=4):
    """
    Creates the multi-encoder U-Net model WITHOUT attention mechanisms.
    """
    assert len(inputs) == num_encoders, "Number of inputs must match num_encoders"

    input_shape = inputs[0].shape[1:]
    modalities = ['t1c', 't1n', 't2f', 't2w']

    drp_out = 0.1  # Dropout rate

    filters_list = [32, 64, 128, 256, 512]
    filters_decoder = [256, 128, 64, 32]
    stochastic_depth_rate = 0.0

    # Load and prepare encoders (no attention parameters)
    pretrained_encoders = []
    for i, modality in enumerate(modalities[:num_encoders]):
        pretrained_encoder = load_pretrained_encoder(
            modality=modality,
            input_shape=input_shape,
            filters=filters_list,
            freeze=True,
            drp_out=drp_out,
        )
        pretrained_encoders.append(pretrained_encoder)

    # Initialize variables
    cnn_outputs_block = []
    scaled_images = []

    num_blocks = len(filters_list)
    encoder_skips_all = [[] for _ in range(num_blocks - 1)]
    # encoder_skips_all = [[] for _ in range(num_blocks - 2)]
    encoder_bottlenecks = []

    # Process Block 1 for each encoder
    for i in range(num_encoders):
        modality = modalities[i]
        x = inputs[i]
        scaled_images.append([x])

        # Block 1 processing
        x = pretrained_encoders[i].get_layer(f'{modality}_block1_norm')(x)
        lambda_layer_name = f'{modality}_block1_lambda'
        if lambda_layer_name in [layer.name for layer in pretrained_encoders[i].layers]:
            x = pretrained_encoders[i].get_layer(lambda_layer_name)(x)
        x = pretrained_encoders[i].get_layer(f'{modality}_block1_conv1')(x)
        x = pretrained_encoders[i].get_layer(f'{modality}_block1_conv2')(x)
        x = layers.Dropout(drp_out, name=f'{modality}_block1_dropout')(x)
        x = pretrained_encoders[i].get_layer(f'{modality}_block1_pool')(x)
        cnn_outputs_block.append(x)

    # Process Blocks 2-4
    for block_idx in range(num_blocks - 1):

        adapted_outputs = adapter_between_encoders(
            cnn_outputs_block=cnn_outputs_block,
            act='relu',
            compression_ratio=0.5,
            dropout_rate=0.1,
            block_idx=block_idx,
        )

        cnn_outputs_block_new = []
        for i in range(num_encoders):
            modality = modalities[i]
            x = adapted_outputs[i]

            encoder_skips_all[block_idx].append(x)

            next_block = block_idx + 2
            norm_layer_name = f'{modality}_block{next_block}_norm'
            if norm_layer_name in [layer.name for layer in pretrained_encoders[i].layers]:
                x = pretrained_encoders[i].get_layer(norm_layer_name)(x)
                scale_layer_name = f'{modality}_block{next_block}_scale'
                scaled_input = pretrained_encoders[i].get_layer(scale_layer_name)(scaled_images[i][-1])
                scaled_images[i].append(scaled_input)
                scale_conv_layer_name = f'{modality}_block{next_block}_scale_conv'
                scale_conv_output = pretrained_encoders[i].get_layer(scale_conv_layer_name)(scaled_input)
                x = layers.Concatenate(axis=-1, name=f'{modality}_block{next_block}_concat')([scale_conv_output, x])
                x = pretrained_encoders[i].get_layer(f'{modality}_block{next_block}_conv1')(x)
                x = pretrained_encoders[i].get_layer(f'{modality}_block{next_block}_conv2')(x)
                x = layers.Dropout(drp_out, name=f'{modality}_block{next_block}_dropout')(x)
                x = pretrained_encoders[i].get_layer(f'{modality}_block{next_block}_pool')(x)
                cnn_outputs_block_new.append(x)
            else:
                cnn_outputs_block_new.append(x)
        cnn_outputs_block = cnn_outputs_block_new

    # # Process Bottleneck (Block 5)
    # block_idx = num_blocks - 1

    # adapted_outputs = adapter_between_encoders(
    #     cnn_outputs_block=cnn_outputs_block,
    #     act='relu',
    #     compression_ratio=0.5,
    #     dropout_rate=0.1,
    #     block_idx=block_idx,
    # )

    encoder_bottlenecks = []
    for i in range(num_encoders):
        modality = modalities[i]
        x = cnn_outputs_block[i]
        x = pretrained_encoders[i].get_layer(f'{modality}_bottleneck_norm')(x)
        x = pretrained_encoders[i].get_layer(f'{modality}_bottleneck_conv1')(x)
        x = pretrained_encoders[i].get_layer(f'{modality}_bottleneck_conv2')(x)
        x = layers.Dropout(drp_out, name=f'{modality}_bottleneck_dropout')(x)
        x = pretrained_encoders[i].get_layer(f'{modality}_bottleneck_pool')(x)
        encoder_bottlenecks.append(x)

    # Combine outputs
    combined_bottleneck_output = layers.Concatenate(name='combined_bottleneck')(encoder_bottlenecks)
    combined_skips = [layers.Concatenate(name=f'combined_skip_{i+1}')(skips) for i, skips in enumerate(encoder_skips_all)]

    # Build decoder (no attention parameters)
    outputs = build_decoder(
        adapter_output=combined_bottleneck_output,
        encoder_skips=combined_skips,
        filters_decoder=filters_decoder,
        num_classes=num_classes
    )

    # Create final model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='multi_modal_unet')


    return model


#--------------------------------------------------------------------------------------------------------------
import os
from sklearn.model_selection import KFold
import tensorflow as tf
import pandas as pd
import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure
import matplotlib.pyplot as plt


# ------------------------
# Helper Functions
# ------------------------
def crop_center(img, cropx, cropy):
    # Define your crop_center function here if not already defined
    y, x = img.shape
    startx = x//2 - (cropx//2)
    starty = y//2 - (cropy//2)    
    return img[starty:starty+cropy, startx:startx+cropx]

def uncrop_center(cropped_img, original_shape, pad_value=0):
    # (Definition as provided earlier)
    if len(cropped_img.shape) == 3 and len(original_shape) == 3:
        cropped_height, cropped_width, channels = cropped_img.shape
        original_height, original_width, _ = original_shape
    else:
        cropped_height, cropped_width = cropped_img.shape[:2]
        original_height, original_width = original_shape[:2]
        channels = None if len(cropped_img.shape) == 2 else cropped_img.shape[2]
    
    if channels:
        result = np.full(original_shape, pad_value, dtype=cropped_img.dtype)
    else:
        result = np.full((original_height, original_width), pad_value, dtype=cropped_img.dtype)

    start_y = (original_height - cropped_height) // 2
    start_x = (original_width - cropped_width) // 2

    if channels:
        result[start_y:start_y+cropped_height, start_x:start_x+cropped_width, :] = cropped_img
    else:
        result[start_y:start_y+cropped_height, start_x:start_x+cropped_width] = cropped_img

    return result

def merge_binary_slices(prediction):
    merged_prediction = np.argmax(prediction, axis=-1).astype(np.uint8)
    return merged_prediction

# ------------------------
# Process Single Volume
# ------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[
                        logging.FileHandler("segmentation_pipeline.log"),
                        logging.StreamHandler()
                    ])


def normalize_volume(volume, lower_percentile=1, upper_percentile=99):
    """
    Normalizes the entire 3D volume based on percentile clipping.
    
    Args:
        volume (np.ndarray): 3D medical image volume.
        lower_percentile (float): Lower percentile for clipping.
        upper_percentile (float): Upper percentile for clipping.
    
    Returns:
        np.ndarray: Normalized 3D volume.
    """
    lower = np.percentile(volume, lower_percentile)
    upper = np.percentile(volume, upper_percentile)
    norm_volume = (volume - lower) / (upper - lower + 1e-8)
    norm_volume = np.clip(norm_volume, 0, 1)
    return norm_volume






def save_prediction(volume, sample_paths, save_dir):
    """
    Saves the segmentation prediction as a NIfTI file in the specified directory.

    Args:
        volume (np.ndarray): Segmentation volume to save.
        sample_paths (dict): Dictionary containing file paths for each modality.
        save_dir (str): Directory to save the NIfTI file.
    """
    nifti_output_dir = save_dir
    os.makedirs(nifti_output_dir, exist_ok=True)

    # Use the affine from one of the input images to maintain spatial consistency
    try:
        affine = nib.load(sample_paths['t1c']).affine
    except Exception as e:
        print(f"  Error loading affine from T1c modality: {e}. Using identity matrix.")
        affine = np.eye(4)

    # Extract case ID and timepoint from filename
    base_name = os.path.basename(sample_paths.get('t1c', ''))
    parts = base_name.split('-')
    if len(parts) >= 4:
        case_id = parts[2]
        timepoint = parts[3][:3]
    else:
        case_id = "unknown"
        timepoint = "000"
        print("Warning: Unexpected filename format.")

    output_filename = f"{case_id}-{timepoint}.nii.gz"
    output_filepath = os.path.join(nifti_output_dir, output_filename)

    # Save the NIfTI file
    nifti_img = nib.Nifti1Image(volume.astype(np.uint8), affine)
    nib.save(nifti_img, output_filepath)
    print(f"Saved NIfTI: {output_filepath}")




import logging
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import label, generate_binary_structure


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

            # 3) regular class threshold
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







def process_single_volume_and_save(model, sample_paths, output_dir, original_shape=(240, 240, 155), input_size=(224, 224, 1)):
    """
    Processes a single 3D medical image volume through model prediction and post-processing steps,
    and saves the results in specified directories.
    
    Args:
        model: Trained segmentation model.
        sample_paths (dict): Dictionary containing file paths for each modality.
        output_dir (str): Base directory to save the output NIfTI files.
        original_shape (tuple): Shape of the original volume (height, width, depth).
        input_size (tuple): Size to which each slice is cropped for model input.
    """
    import numpy as np
    from scipy.ndimage import label, generate_binary_structure, binary_closing
    import nibabel as nib
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

    # ------------------------
    # Step 1: Load Modalities
    # ------------------------
    print("\n=== Step 1: Loading Modalities ===")
    modalities = ['t1c', 't1n', 't2f', 't2w']
    modality_data = {}
    for modality in modalities:
        path = sample_paths.get(modality)
        if not path:
            print(f"Error: Missing modality {modality}. Skipping this sample.")
            return
        try:
            modality_data[modality] = nib.load(path).get_fdata()
            print(f"  Loaded {modality} from {path}.")
        except Exception as e:
            print(f"Error loading {modality} from {path}: {e}")
            return

    num_slices = modality_data['t1c'].shape[-1]
    print(f"  Total number of slices: {num_slices}")

    probability_slices = []
    image_slices = []

    # ------------------------
    # Step 2: Model Prediction per Slice
    # ------------------------
    print("\n=== Step 2: Model Prediction per Slice ===")
    for slice_idx in range(num_slices):
        if (slice_idx + 1) % 10 == 0 or slice_idx == 0:
            print(f"  Processing slice {slice_idx + 1}/{num_slices}")

        model_inputs = {}
        skip_slice = False
        original_slices = []

        for modality in modalities:
            slice_img = modality_data[modality][:, :, slice_idx]
            original_slices.append(slice_img)

            # Preprocess: Crop the slice
            cropped = crop_center(slice_img, input_size[0], input_size[1])

            # Check if the cropped slice is empty
            if not np.any(cropped):
                skip_slice = True
                print(f"    Slice {slice_idx + 1}: {modality} is empty after cropping. Skipping this slice.")
                break

            # Normalize the cropped slice
            min_val = np.min(cropped)
            max_val = np.max(cropped)
            if max_val - min_val > 1e-8:
                norm_slice = (cropped - min_val) / (max_val - min_val + 1e-8)
            else:
                norm_slice = np.zeros_like(cropped)
                print(f"    Slice {slice_idx + 1}: {modality} has constant intensity. Normalized to zeros.")

            # Expand dimensions to match model input
            input_slice = np.expand_dims(np.expand_dims(norm_slice, axis=0), axis=-1)
            model_inputs[f"{modality}_input"] = input_slice

        if skip_slice:
            # Append empty probabilities and image slices to maintain volume size
            empty_probs = np.zeros((original_shape[0], original_shape[1], 4))  # Including background
            probability_slices.append(empty_probs)
            empty_image_slice = np.zeros((original_shape[0], original_shape[1], len(modalities)))
            image_slices.append(empty_image_slice)
            continue

        try:
            # Make prediction
            predictions = model.predict(model_inputs)  # Dictionary output if model has named outputs
            if isinstance(predictions, dict):
                probabilities = predictions['final'][0]  # Shape: (224, 224, 4)
            else:
                probabilities = predictions[0]  # Adjust based on your model's output
            print(f"    Slice {slice_idx + 1}: Model prediction successful.")
        except Exception as e:
            print(f"    Slice {slice_idx + 1}: Model prediction failed with error: {e}. Skipping this slice.")
            empty_probs = np.zeros((original_shape[0], original_shape[1], 4))
            probability_slices.append(empty_probs)
            empty_image_slice = np.zeros((original_shape[0], original_shape[1], len(modalities)))
            image_slices.append(empty_image_slice)
            continue

        # Resize probabilities back to original slice size (240x240)
        probabilities_resized = np.zeros((original_shape[0], original_shape[1], 4))
        for c in range(4):
            probabilities_resized[..., c] = uncrop_center(probabilities[..., c], original_shape[:2])

        probability_slices.append(probabilities_resized)
        image_slice = np.stack(original_slices, axis=-1)  # Shape: (240, 240, num_modalities)
        image_slices.append(image_slice)

    print("\n  Completed model predictions for all slices.")

    # ------------------------
    # Step 3: Stack Volumes and Initial Prediction
    # ------------------------
    print("\n=== Step 3: Stacking Volumes and Initial Prediction ===")
    try:
        # Stack slices along the first axis (depth)
        probability_volume = np.stack(probability_slices, axis=0)  # Shape: (D, H, W, 4)
        image_volume = np.stack(image_slices, axis=0)              # Shape: (D, H, W, num_modalities)
        raw_prediction = np.argmax(probability_volume, axis=-1).astype(np.uint8)  # Shape: (D, H, W)
        print("  Stacked probability and image volumes successfully.")

        print(f"  probability_volume shape: {probability_volume.shape}")
        print(f"  image_volume shape: {image_volume.shape}")
    except Exception as e:
        print(f"Error stacking volumes: {e}")
        return

    # ------------------------
    # Step 4: Initialize Final Prediction Volume
    # ------------------------
    print("\n=== Step 4: Initializing Final Prediction Volume ===")
    final_prediction = np.zeros(original_shape, dtype=np.uint8)  # Shape: (240, 240, 155)
    print(f"  Initialized final_prediction with shape: {final_prediction.shape}")

    # Step 1: Apply label priorities — ET (3) > NCR (1) > ED (2)
    final_prediction_stack = np.zeros_like(raw_prediction, dtype=np.uint8)
    final_prediction_stack[raw_prediction == 2] = 2  # ED
    final_prediction_stack[raw_prediction == 1] = 1  # NCR
    final_prediction_stack[raw_prediction == 3] = 3  # ET overrides all

    # Step 2: Apply 3D false positive removal
    final_prediction_cleaned = remove_small_false_positives_3d(
        prediction=final_prediction_stack.transpose(1, 2, 0),        # now (H, W, D)
        probability_volume=probability_volume.transpose(1, 2, 0, 3)  # also (H, W, D, C)
    )


    save_prediction(final_prediction_cleaned, sample_paths, raw_dir)


  
# ------------------------
# Usage
# ------------------------


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral
import tensorflow as tf
from sklearn.model_selection import KFold


brats_data = "/scratch_net/maus/mrostamian/"
brats_data_validation = os.path.join(brats_data, 'ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData')

# Load all sample paths for the validation set
all_samples = load_file_paths(brats_data_validation)
print(f"Total samples loaded: {len(all_samples)}")

val_data = all_samples



# Load the trained model
checkpoint_filepath = "/scratch_net/maus/mrostamian/project/maus_second/log_multi_unet_new_adapter/best_weights_fold_1_epoch_111.weights.h5"

input_t1c = tf.keras.Input(shape=(224, 224, 1), name='t1c_input')
input_t1n = tf.keras.Input(shape=(224, 224, 1), name='t1n_input')
input_t2f = tf.keras.Input(shape=(224, 224, 1), name='t2f_input')
input_t2w = tf.keras.Input(shape=(224, 224, 1), name='t2w_input')
model_inputs = [input_t1c, input_t1n, input_t2f, input_t2w]

model = create_multi_encoder_unet(inputs=model_inputs, num_encoders=4, training=False, num_classes=4)
model.load_weights(checkpoint_filepath)
print("Loaded model with best weights.")

output_dir = '/scratch_net/maus/mrostamian/project/log_multi_unet_adapter_scaling/online_results_multi_unet_adapter_scaling_new_pp_v1'
os.makedirs(output_dir, exist_ok=True)  

# Initialize output directories
nifti_output_dir = os.path.join(output_dir, "nifti_predictions")
raw_dir = os.path.join(nifti_output_dir, "raw_multi_unet_adapter_scaling_new_pp_v1")


for directory in [raw_dir]:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

# Process each validation sample and save predictions
for idx, sample_paths in enumerate(val_data):
    print(f"\nProcessing sample {idx + 1}/{len(val_data)}:")
    print(f"  T1c Path: {sample_paths.get('t1c', 'Missing')}")


    # Process the single volume and get Dice scores (modified to save predictions instead)
    process_single_volume_and_save(model, sample_paths, output_dir)
