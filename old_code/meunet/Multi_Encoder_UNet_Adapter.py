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


def load_file_paths(data_dir, modalities=['t1c', 't1n', 't2f', 't2w'], mask_suffix='seg'):
    sample_dict = {}

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_lower = file.lower()
            # Consider only NIfTI files
            if not file_lower.endswith('.nii') and not file_lower.endswith('.nii.gz'):
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
            
            # Collect paths
            if modality in modalities or modality == mask_suffix:
                if sample_id not in sample_dict:
                    sample_dict[sample_id] = {}
                sample_dict[sample_id][modality] = os.path.join(root, file)

    # Build a list of samples that have all modalities and mask
    samples = []
    for sample_id, modalities_dict in sample_dict.items():
        if all(mod in modalities_dict for mod in modalities) and mask_suffix in modalities_dict:
            sample_entry = {mod: modalities_dict[mod] for mod in modalities}
            sample_entry['mask'] = modalities_dict[mask_suffix]
            samples.append(sample_entry)
        else:
            print(f"Sample {sample_id} is missing some modalities or mask, skipping.")
    
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
    save_sample_dir = '/scratch_net/maus_second/mrostamian/project/log_multi_unet_adapter_v2/visualize_img'
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

    # save_sample_dir = '/scratch_net/maus/mrostamian/project/log_multi_unet_adapter/visualize_img'  # Ensure this path is correct

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



#-------New Adapter--------------------------------------------------------------------------------------------------------------

from tensorflow.keras import layers, regularizers


# class ScalingLayer(layers.Layer):
#     """Dual scaling layer with regularization adjustments for early blocks"""
#     def __init__(self, initial_alpha=0.5, initial_beta=1.0, **kwargs):
#         # Adjust initial beta for Block 0 to preserve original features
#         if "block0" in kwargs.get("name", ""):
#             initial_alpha = .5
#             initial_beta = 1.0  # Stronger preservation of original features
        
#         super().__init__(**kwargs)
#         self.initial_alpha = initial_alpha
#         self.initial_beta = initial_beta

#     def build(self, input_shape):
#         # Extract block index from layer name (e.g., "block0" -> 0)
#         self.block_idx = int(self.name.split("_")[2][5:]) if "block" in self.name else 0
        
#         # Disable L2 regularization for Block 0
#         reg = tf.keras.regularizers.l2(1e-4) if self.block_idx > 0 else None
        
#         self.alpha = self.add_weight(
#             name='alpha',
#             shape=(),
#             initializer=tf.keras.initializers.Constant(self.initial_alpha),
#             regularizer=reg,  # No regularization for Block 0
#             trainable=True
#         )
#         self.beta = self.add_weight(
#             name='beta',
#             shape=(),
#             initializer=tf.keras.initializers.Constant(self.initial_beta),
#             regularizer=reg,  # No regularization for Block 0
#             trainable=True
#         )
#         super().build(input_shape)

#     def call(self, inputs):
#         adapter_out, original = inputs
#         # Add a small residual to prevent total feature collapse
#         return self.alpha * adapter_out + self.beta * original + 1e-3 * original

# def se_block(input_tensor, reduction_ratio=16, dropout_rate=0.1, name_prefix=''):
#     """Original SE Block implementation"""
#     filters = input_tensor.shape[-1]
#     se_shape = (1, 1, filters)
    
#     se = layers.GlobalAveragePooling2D(name=f'{name_prefix}_se_gap')(input_tensor)
#     se = layers.Reshape(se_shape, name=f'{name_prefix}_se_reshape')(se)
    
#     se = layers.Dense(filters // reduction_ratio, 
#                       activation='relu', 
#                       name=f'{name_prefix}_se_dense1')(se)
#     se = layers.Dropout(rate=dropout_rate, name=f'{name_prefix}_se_dropout1')(se)
#     se = layers.Dense(filters, 
#                       activation='sigmoid', 
#                       name=f'{name_prefix}_se_dense2')(se)
    
#     return layers.Multiply(name=f'{name_prefix}_se_scale')([input_tensor, se])

# def adapter_between_encoders(cnn_outputs_block, act='relu', compression_ratio=0.5, 
#                              dropout_rate=0.1, block_idx=0):
#     """Improved adapter with dual scaling and original SE block"""
#     num_encoders = len(cnn_outputs_block)
#     filters = cnn_outputs_block[0].shape[-1]
#     compressed_filters = max(8, int(filters * compression_ratio))  # Minimum 8 channels

#     # Compression path
#     compressed_outputs = []
#     for i, x in enumerate(cnn_outputs_block):
#         x_compressed = layers.Conv2D(
#             compressed_filters, (1, 1),
#             kernel_initializer='he_normal',
#             padding='same',
#             name=f'adapter_comp_conv_block{block_idx}_{i}'
#         )(x)
#         x_compressed = layers.BatchNormalization(
#             gamma_initializer='ones',
#             name=f'adapter_comp_bn_block{block_idx}_{i}'
#         )(x_compressed)
#         x_compressed = layers.Activation(act, 
#                                          name=f'adapter_comp_act_block{block_idx}_{i}')(x_compressed)
#         x_compressed = layers.Dropout(rate=dropout_rate, 
#                                       name=f'adapter_comp_dropout_block{block_idx}_{i}')(x_compressed)
#         compressed_outputs.append(x_compressed)

#     # Concatenate and apply SE block
#     combined = layers.Concatenate(axis=-1, 
#                                   name=f'adapter_combined_block{block_idx}')(compressed_outputs)
#     se_processed = se_block(combined, 
#                             reduction_ratio=16, 
#                             dropout_rate=dropout_rate, 
#                             name_prefix=f'adapter_se_block{block_idx}')

#     # Lightweight processing: Separable convolution block
#     attended = layers.SeparableConv2D(
#         compressed_filters, (3, 3),
#         padding='same',
#         name=f'adapter_sep_conv_block{block_idx}'
#     )(se_processed)
#     attended = layers.BatchNormalization(
#         name=f'adapter_sep_conv_bn_block{block_idx}')(attended)
#     attended = layers.Activation(act, 
#                                  name=f'adapter_sep_conv_act_block{block_idx}')(attended)
#     attended = layers.Dropout(rate=dropout_rate, 
#                               name=f'adapter_sep_conv_dropout_block{block_idx}')(attended)

#     # Expansion and dual scaling
#     adapted_outputs = []
#     for i in range(num_encoders):
#         x_expanded = layers.Conv2D(
#             filters, (1, 1), 
#             padding='same',
#             name=f'adapter_exp_conv_block{block_idx}_{i}'
#         )(attended)
#         x_expanded = layers.BatchNormalization(
#             name=f'adapter_exp_bn_block{block_idx}_{i}')(x_expanded)
#         x_expanded = layers.Activation(act, 
#                                       name=f'adapter_exp_act_block{block_idx}_{i}')(x_expanded)
        
#         # Dual scaling with original features
#         x_scaled = ScalingLayer(
#             initial_alpha=0.5,  # Start with stronger adapter contribution
#             initial_beta=1.0,   # Preserve original features
#             name=f'scaling_layer_block{block_idx}_{i}'
#         )([x_expanded, cnn_outputs_block[i]])
        
#         adapted_outputs.append(x_scaled)
    
#     return adapted_outputs


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


# first original adapter
def se_block(input_tensor, reduction_ratio=16, dropout_rate=0.1, name_prefix=''):
    """
    Squeeze-and-Excitation block for channel attention with optional Dropout after the first Dense layer.
    
    Args:
        input_tensor: Input tensor.
        reduction_ratio: Reduction ratio for channel compression.
        dropout_rate: Dropout rate for regularization.
        name_prefix: Prefix for layer naming.
    
    Returns:
        Tensor after applying SE block with channel attention.
    """
    filters = input_tensor.shape[-1]
    se_shape = (1, 1, filters)
    
    # Squeeze: Global Average Pooling
    se = layers.GlobalAveragePooling2D(name=f'{name_prefix}_se_gap')(input_tensor)
    se = layers.Reshape(se_shape, name=f'{name_prefix}_se_reshape')(se)
    
    # Excitation: Fully Connected with Reduction and Activation
    se = layers.Dense(filters // reduction_ratio, activation='relu', name=f'{name_prefix}_se_dense1')(se)
    # Optional Dropout after first Dense layer
    se = layers.Dropout(rate=dropout_rate, name=f'{name_prefix}_se_dropout1')(se)
    
    se = layers.Dense(filters, activation='sigmoid', name=f'{name_prefix}_se_dense2')(se)
    # Do not add dropout after sigmoid activation to preserve scaling factors
    
    # Scale: Multiply input by channel-wise weights
    x = layers.Multiply(name=f'{name_prefix}_se_scale')([input_tensor, se])
    
    return x

def adapter_between_encoders(cnn_outputs_block, act='relu', compression_ratio=0.25, dropout_rate=0.1, block_idx=0):
    """
    Adapter function to fuse features from multiple encoders with channel attention and Dropout.
    
    Args:
        cnn_outputs_block: List of feature maps from different encoders.
        act: Activation function to use.
        compression_ratio: Ratio to compress channels in the adapter.
        dropout_rate: Dropout rate for regularization.
        block_idx: Index of the block for naming layers uniquely.
    
    Returns:
        adapted_outputs: List of adapted feature maps for each encoder.
    """
    num_encoders = len(cnn_outputs_block)
    filters = cnn_outputs_block[0].shape[-1]  # Assuming all encoders have the same number of filters
    compressed_filters = max(1, int(filters * compression_ratio))
    
    # Compression, Activation, and Dropout
    compressed_outputs = []
    for i, x in enumerate(cnn_outputs_block):
        x_compressed = layers.Conv2D(
            compressed_filters, (1, 1), padding='same',
            name=f'adapter_comp_conv_block{block_idx}_{i}'
        )(x)
        x_compressed = layers.BatchNormalization(name=f'adapter_comp_bn_block{block_idx}_{i}')(x_compressed)
        x_compressed = layers.Activation(act, name=f'adapter_comp_act_block{block_idx}_{i}')(x_compressed)
        # Dropout after activation in compression stage
        x_compressed = layers.Dropout(rate=dropout_rate, name=f'adapter_comp_dropout_block{block_idx}_{i}')(x_compressed)
        compressed_outputs.append(x_compressed)
    
    # Concatenate compressed outputs
    combined = layers.Concatenate(axis=-1, name=f'adapter_combined_block{block_idx}')(compressed_outputs)
    
    # Channel Attention with SE Block
    combined = se_block(combined, reduction_ratio=16, dropout_rate=dropout_rate, name_prefix=f'adapter_se_block{block_idx}')
    
    # Lightweight Convolution Block with Dropout
    attended = layers.SeparableConv2D(
        compressed_filters,
        kernel_size=(3, 3),
        padding='same',
        name=f'adapter_sep_conv_block{block_idx}'
    )(combined)
    attended = layers.BatchNormalization(name=f'adapter_sep_conv_bn_block{block_idx}')(attended)
    attended = layers.Activation(act, name=f'adapter_sep_conv_act_block{block_idx}')(attended)
    # Dropout after activation in lightweight convolution block
    attended = layers.Dropout(rate=dropout_rate, name=f'adapter_sep_conv_dropout_block{block_idx}')(attended)
    
    # Expansion and Residual Connection
    adapted_outputs = []
    for i in range(num_encoders):
        # Expansion
        x_expanded = layers.Conv2D(
            filters, (1, 1), padding='same',
            name=f'adapter_exp_conv_block{block_idx}_{i}'
        )(attended)
        x_expanded = layers.BatchNormalization(name=f'adapter_exp_bn_block{block_idx}_{i}')(x_expanded)
        x_expanded = layers.Activation(act, name=f'adapter_exp_act_block{block_idx}_{i}')(x_expanded)
        # Optional: Add dropout after activation in expansion stage if needed
        # x_expanded = layers.Dropout(rate=dropout_rate, name=f'adapter_exp_dropout_block{block_idx}_{i}')(x_expanded)
        
        # Residual Connection
        x_residual = layers.Add(name=f'adapter_residual_block{block_idx}_{i}')([
            x_expanded, cnn_outputs_block[i]
        ])
        adapted_outputs.append(x_residual)
    
    return adapted_outputs



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

def weighted_binary_crossentropy(y_true, y_pred, pos_weights, neg_weights):
    epsilon_ = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon_, 1.0 - epsilon_)

    pos_weights = tf.reshape(pos_weights, [1, 1, 1, -1])
    neg_weights = tf.reshape(neg_weights, [1, 1, 1, -1])

    bce_pos = pos_weights * y_true * tf.math.log(y_pred + epsilon_)
    bce_neg = neg_weights * (1 - y_true) * tf.math.log(1 - y_pred + epsilon_)
    bce = - (bce_pos + bce_neg)

    bce = tf.reduce_sum(bce, axis=-1)
    return tf.reduce_mean(bce)

def dsc(y_true, y_pred, class_weights=None):
    smooth = 1e-5
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)

    if class_weights is not None:
        class_weights = tf.reshape(class_weights, [1, 1, 1, -1])
        y_true_f = y_true_f * class_weights
        y_pred_f = y_pred_f * class_weights

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denominator = tf.reduce_sum(y_true_f + y_pred_f)

    dice_score = (2. * intersection + smooth) / (denominator + smooth)
    dice_score = tf.clip_by_value(dice_score, 0.0, 1.0)  # Ensures dice_score is within [0, 1]
    return dice_score


def dice_loss(y_true, y_pred, class_weights=None):
    return 1.0 - dsc(y_true, y_pred, class_weights)


def bce_dice_loss(y_true, y_pred):
    """
    Combined loss function of weighted binary cross-entropy and Dice loss.

    Args:
        y_true: Ground truth tensor.
        y_pred: Predicted tensor.

    Returns:
        Scalar loss value.
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)


    # Example adjusted weights
    pos_weights = tf.constant([1.0, 5.0, 5.0, 5.0], dtype=tf.float32)
    neg_weights = tf.constant([1.0, 0.5, 0.5, 0.5], dtype=tf.float32)



    # Compute weighted binary cross-entropy loss
    bce_loss = weighted_binary_crossentropy(y_true, y_pred, pos_weights, neg_weights)

    # Compute weighted Dice loss
    class_weights = pos_weights  # Use positive weights as class weights for Dice loss
    dice_loss_val = dice_loss(y_true, y_pred, class_weights)

    # Combine BCE and Dice loss with specific weights
    bce_weight, dice_weight = 0.5, 1.0
    total_loss = bce_weight * bce_loss + dice_weight * dice_loss_val

    return total_loss



######################################################################################################################
def weighted_bce_loss(y_true, y_pred_logits, pos_weight=None):
    """
    Computes weighted binary cross-entropy loss with logits.

    Args:
        y_true: Ground truth tensor of shape (batch_size, height, width, num_classes).
        y_pred_logits: Predicted logits of the same shape as y_true.
        pos_weight: A list or numpy array of positive weights for each class (length = num_classes).

    Returns:
        A scalar tensor representing the weighted BCE loss.
    """
    y_true = tf.cast(y_true, dtype=tf.float32)

    if pos_weight is not None:
        pos_weight = tf.constant(pos_weight, dtype=tf.float32)
        pos_weight = tf.reshape(pos_weight, [1, 1, 1, -1])

        # Expand pos_weight to match y_pred_logits shape
        bce = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true,
            logits=y_pred_logits,
            pos_weight=pos_weight
        )
    else:
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred_logits)

    return tf.reduce_mean(bce)

def soft_dice_loss(y_true, y_pred_logits, smooth=1e-5, do_bg=True):
    """
    Computes the soft Dice loss with logits.

    Args:
        y_true: Ground truth tensor of shape (batch_size, height, width, num_classes).
        y_pred_logits: Predicted logits of the same shape as y_true.
        smooth: A smoothing constant to avoid division by zero.
        do_bg: Boolean indicating whether to include background class in the loss.

    Returns:
        A scalar tensor representing the Dice loss.
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.nn.sigmoid(y_pred_logits)

    # Exclude background class if do_bg is False
    if not do_bg:
        y_true = y_true[..., 1:]
        y_pred = y_pred[..., 1:]

    # Compute per-class Dice score
    axes = [0, 1, 2]  # Sum over batch and spatial dimensions
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    y_true_sum = tf.reduce_sum(y_true, axis=axes)
    y_pred_sum = tf.reduce_sum(y_pred, axis=axes)

    dice_score = (2. * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)

    # Average over classes
    dice_loss = 1 - tf.reduce_mean(dice_score)

    return dice_loss

def combined_dice_bce_loss(y_true, y_pred_logits, dice_weight=1.0, bce_weight=1.0, pos_weight=None, ignore_label=None, do_bg=True):
    """
    Combines Dice loss and weighted BCE loss, similar to nnU-Net's DC_and_BCE_loss.

    Args:
        y_true: Ground truth tensor of shape (batch_size, height, width, num_classes).
        y_pred_logits: Predicted logits of the same shape as y_true.
        dice_weight: Weight for the Dice loss component.
        bce_weight: Weight for the BCE loss component.
        pos_weight: Optional list or array of positive weights for each class.
        ignore_label: Optional label value to ignore during loss computation.
        do_bg: Boolean indicating whether to include background class in the loss.

    Returns:
        A scalar tensor representing the combined loss.
    """
    y_true = tf.cast(y_true, dtype=tf.float32)

    # Handle ignore label
    if ignore_label is not None:
        # Create a mask for valid labels
        mask = tf.not_equal(y_true, ignore_label)
        mask = tf.cast(mask, tf.float32)
        y_true = y_true * mask
        y_pred_logits = y_pred_logits * mask

    # Compute Dice loss
    dice = soft_dice_loss(y_true, y_pred_logits, smooth=1e-5, do_bg=do_bg)

    # Compute weighted BCE loss
    bce = weighted_bce_loss(y_true, y_pred_logits, pos_weight=pos_weight)

    # Combine losses with specified weights
    total_loss = dice_weight * dice + bce_weight * bce
    return total_loss


# Define positive class weights
pos_weight = [1.0, 5.0, 5.0, 5.0]  # Adjust as per your class imbalance

# Set the ignore label (if any)
ignore_label = None  # Set to an integer value if you have an ignore label

# Wrap the loss function with your specific parameters
def loss_function_wrapper(y_true, y_pred_logits):
    return combined_dice_bce_loss(
        y_true,
        y_pred_logits,
        dice_weight=1.0,
        bce_weight=0.5,
        pos_weight=pos_weight,
        ignore_label=ignore_label,
        do_bg=True  # Set to False if you want to exclude background in Dice loss
    )




######################################################################################################################
def weighted_categorical_crossentropy(y_true, y_pred, class_weights):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
    class_weights = tf.constant(class_weights, dtype=tf.float32)
    class_weights = tf.reshape(class_weights, [1, 1, 1, -1])
    cross_entropy = -tf.reduce_sum(y_true * tf.math.log(y_pred) * class_weights, axis=-1)
    return tf.reduce_mean(cross_entropy)

def dice_loss_categorical(y_true, y_pred, class_weights, smooth=1e-5):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_true_f = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
    y_pred_f = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    y_true_sum = tf.reduce_sum(y_true_f, axis=0)
    y_pred_sum = tf.reduce_sum(y_pred_f, axis=0)
    dice_numerator = 2.0 * intersection + smooth
    dice_denominator = y_true_sum + y_pred_sum + smooth
    dice_coefficients = dice_numerator / dice_denominator
    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
    dice_loss_per_class = (1.0 - dice_coefficients) * class_weights_tensor
    dice_loss_value = tf.reduce_sum(dice_loss_per_class) / tf.reduce_sum(class_weights_tensor)
    return dice_loss_value

def focal_loss(y_true, y_pred, alpha, gamma=2.0):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())

    alpha = tf.constant(alpha, dtype=tf.float32)
    alpha = tf.reshape(alpha, [1, 1, 1, -1])  # Shape compatible with y_true and y_pred

    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.pow(1 - y_pred, gamma)
    loss = weight * cross_entropy
    loss = tf.reduce_sum(loss, axis=-1)
    return tf.reduce_mean(loss)


def combined_loss(y_true, y_pred):

    # Background weight (example: 0.2 to de-emphasize background vs. tumor)
    # Then for classes 1, 2, 3, from the inverse frequency ratio
    # class_weights = # Slightly reduce ED's weight to prevent over-segmentation
    class_weights = [0.5, 3.0, 3.0, 3.0]  
  

    # Focal loss parameters
    # alpha = [1., 1., 1., 1.] 
    # gamma = 2.0               # Common starting choice

    # Loss combination weights
    lambda_wce = 0.5
    lambda_dice = 1.0
    # lambda_focal = 1.0

    # Compute losses
    loss_cce = weighted_categorical_crossentropy(y_true, y_pred, class_weights)
    loss_dice = dice_loss_categorical(y_true, y_pred, class_weights)
    # loss_focal = focal_loss(y_true, y_pred, alpha, gamma)

    # Combine losses (adjust weights as needed)
    # total_loss = lambda_wce * loss_cce + lambda_dice * loss_dice + lambda_focal * loss_focal
    total_loss = lambda_wce * loss_cce + lambda_dice * loss_dice

    return total_loss

######################################################################################################################


def confusion(y_true, y_pred):
    smooth=1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    prec = (tp + smooth)/(tp+fp+smooth)
    recall = (tp+smooth)/(tp+fn+smooth)
    return prec, recall

def tp(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth)
    return tp

def tn(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn

def tversky(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)


# def weighted_bce(y_true, y_pred, weight_background=0.1, weight_tumor=0.9):
#     bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#     weights = y_true * weight_tumor + (1 - y_true) * weight_background
#     return weights * bce(y_true, y_pred)


def weighted_bce(weight_background=0.1, weight_tumor=0.9):
    def loss(y_true, y_pred):
        # Initialize BinaryCrossentropy without logits (since y_pred is expected to be probabilities)
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        # Calculate the unweighted binary cross-entropy
        unweighted_loss = bce(y_true, y_pred)

        # Create the weight map (same shape as y_true/y_pred)
        weights = y_true * weight_tumor + (1 - y_true) * weight_background

        # Apply the weights to the binary cross-entropy loss
        weighted_loss = weights * unweighted_loss

        # Return the mean of the weighted loss
        return tf.reduce_mean(weighted_loss)
    
    return loss

def weighted_dice_bce_loss(bce_weight=0.2, dice_weight=0.8, foreground_weight=0.9):
    def loss(y_true, y_pred):
        # Ensure BCE is calculated per channel by preserving the last dimension
        # bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)  # Shape: [batch_size, height, width, channels]
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False, axis=-1)  # Retains channel dimension


        # Apply foreground weighting to each channel independently
        bce_loss = bce * tf.where(y_true == 1, foreground_weight, 1.0)  # Weighted per pixel in each channel

        # Compute Dice loss across channels
        dice_loss = dice_metric(y_true, y_pred)

        # Combine BCE and Dice loss with specified weights
        return bce_weight * tf.reduce_mean(bce_loss) + dice_weight * (1 - dice_loss)
    
    return loss


def generalized_dice_loss(smooth=1.0, class_weights=[0.1, 1.0, 1.0, 1.0]):
    class_weights_tensor = tf.convert_to_tensor(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # Reshape weights to be compatible for broadcasting
        class_weights_broadcast = tf.reshape(class_weights_tensor, [1, 1, 1, -1])

        # Calculate weighted intersection and union for each class
        intersection = tf.reduce_sum(class_weights_broadcast * y_true * y_pred, axis=[0, 1, 2])
        union = tf.reduce_sum(class_weights_broadcast * (y_true + y_pred), axis=[0, 1, 2])
        
        # Compute Generalized Dice Coefficient
        dice_score = (2. * intersection + smooth) / (union + smooth)

        # Return Generalized Dice Loss
        return 1.0 - dice_score

    return loss

#---------------------------------


#---------------------------------
def dice_metric(y_true, y_pred, threshold=0.5):
    """
    Computes the Dice coefficient per slice, handling empty cases as per nnU-Net (BraTS).

    Args:
        y_true: Ground truth tensor of shape (batch_size, height, width).
        y_pred: Predicted tensor of same shape.
        threshold: Threshold for binarizing predictions.

    Returns:
        Average Dice coefficient over the batch.
    """
    smooth = 1e-6
    y_true = tf.cast(y_true, tf.bool)
    y_pred_thresholded = tf.cast(y_pred > threshold, tf.bool)

    # Compute True Positives, False Positives, False Negatives per slice
    tp = tf.math.reduce_sum(tf.cast(y_true & y_pred_thresholded, tf.float32), axis=[1, 2])
    fp = tf.math.reduce_sum(tf.cast(~y_true & y_pred_thresholded, tf.float32), axis=[1, 2])
    fn = tf.math.reduce_sum(tf.cast(y_true & ~y_pred_thresholded, tf.float32), axis=[1, 2])

    # Calculate Dice per slice
    denominator = (2 * tp) + fp + fn + smooth
    dice_per_slice = (2 * tp + smooth) / denominator

    # Handle empty cases (nnU-Net style):
    # 1. If both ground truth and prediction are empty, Dice = 1
    # 2. If either is empty but not both, Dice = 0
    dice_per_slice = tf.where(tf.equal(tp + fp + fn, 0),  # Both empty?
                               tf.where(tf.equal(tp + fn, 0), 1.0, 0.0),  # Yes: 1 if also GT is empty, else 0
                               dice_per_slice)

    # Compute average Dice
    dice_value = tf.reduce_mean(dice_per_slice)
    return dice_value


def calculate_hausdorff95_2D(y_true, y_pred, threshold=0.5):
    """
    Approximate the 95% Hausdorff distance for 2D slices using TensorFlow operations.
    """
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred > threshold, tf.bool)

    def single_hausdorff(inputs):
        yt, yp = inputs  # Unpack the tuple of tensors
        # Calculate the distances between foreground pixels
        yt_indices = tf.cast(tf.where(yt), tf.float32)
        yp_indices = tf.cast(tf.where(yp), tf.float32)

        def max_distance():
            d_matrix = tf.norm(
                tf.expand_dims(yt_indices, axis=1) - tf.expand_dims(yp_indices, axis=0),
                axis=2
            )
            hd = tf.reduce_max(
                tf.minimum(
                    tf.reduce_min(d_matrix, axis=1),
                    tf.reduce_min(d_matrix, axis=0)
                )
            )
            return hd

        def zero_distance():
            return tf.constant(0.0, dtype=tf.float32)

        def max_possible_distance():
            # Diagonal of the image
            height = tf.cast(tf.shape(yt)[0], tf.float32)
            width = tf.cast(tf.shape(yt)[1], tf.float32)
            return tf.sqrt(height**2 + width**2)

        # Conditions
        yt_sum = tf.reduce_sum(tf.cast(yt, tf.int32))
        yp_sum = tf.reduce_sum(tf.cast(yp, tf.int32))
        both_empty = tf.logical_and(yt_sum == 0, yp_sum == 0)
        one_empty = tf.logical_or(yt_sum == 0, yp_sum == 0)

        hd = tf.case(
            [
                (both_empty, zero_distance),
                (one_empty, max_possible_distance)
            ],
            default=max_distance,
            exclusive=True
        )
        return hd

    hd_distances = tf.map_fn(
        fn=single_hausdorff,
        elems=(y_true, y_pred),
        fn_output_signature=tf.float32
    )

    return tf.reduce_mean(hd_distances)






def create_combined_channels_metric(metric_fn, class_indices, name, threshold=0.4):
    def metric(y_true, y_pred):
        # Combine the selected channels
        y_pred_combined = tf.reduce_max(tf.gather(y_pred, class_indices, axis=-1), axis=-1)
        y_true_combined = tf.reduce_max(tf.gather(y_true, class_indices, axis=-1), axis=-1)
        
        # Ensure dtype is float32
        y_true_f = tf.cast(y_true_combined, tf.float32)
        y_pred_f = tf.cast(y_pred_combined, tf.float32)
        
        # Call the modified metric function
        return metric_fn(y_true_f, y_pred_f, threshold=threshold)
    
    metric.__name__ = name
    return metric
#---------------------------------


def metrics(img_gt, img_pred, voxel_size, dset="brats"):
    """
    Function to compute the metrics between two segmentation maps given as input.
    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.
    img_pred: np.array
    Array of the predicted segmentation map.
    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.
    Return
    ------
    A list of metrics in this order, [Dice LV, Volume LV, Err LV(ml),
    Dice RV, Volume RV, Err RV(ml), Dice MYO, Volume MYO, Err MYO(ml)]
    """

    # Dice = (2xIntersection)/(Union+Intersection)
    # F1 score
    # https://www.youtube.com/watch?v=AZr64OxshLo



    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))
    if dset=="acdc":
        res = []
        # Loop on each classes of the input images
        for c in [3, 1, 2]:
            # Copy the gt image to not alterate the input
            gt_c_i = np.copy(img_gt)
            gt_c_i[gt_c_i != c] = 0

            # Copy the pred image to not alterate the input
            pred_c_i = np.copy(img_pred)
            pred_c_i[pred_c_i != c] = 0

            # Clip the value to compute the volumes
            gt_c_i = np.clip(gt_c_i, 0, 1)
            pred_c_i = np.clip(pred_c_i, 0, 1)

            # Compute the Dice
            dice = dc(gt_c_i, pred_c_i)

            # Compute volume
            volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
            volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.

            res += [dice, volpred, volpred-volgt]

        if voxel_size==0:
            res = [res[0], res[3], res[6]]

    elif dset=="synapse":
        res = []
        # Loop on each classes of the input images
        for c in [1,2,3,4,5,6,7,8]:
            # Copy the gt image to not alterate the input
            gt_c_i = np.copy(img_gt)
            gt_c_i[gt_c_i != c] = 0

            # Copy the pred image to not alterate the input
            pred_c_i = np.copy(img_pred)
            pred_c_i[pred_c_i != c] = 0

            # Clip the value to compute the volumes
            gt_c_i = np.clip(gt_c_i, 0, 1)
            pred_c_i = np.clip(pred_c_i, 0, 1)

            # Compute the Dice
            dice = dc(gt_c_i, pred_c_i)

            res += [dice]

    elif dset == "brats":
        res = []
        # Define the classes for the BraTS 2023 dataset
        # Update these class numbers based on the actual labels in the dataset
        classes = [1, 2, 3]  # Example: 1 - necrotic core, 2 - edema, etc.

        for c in classes:
            gt_c_i = np.copy(img_gt)
            gt_c_i[gt_c_i != c] = 0
            pred_c_i = np.copy(img_pred)
            pred_c_i[pred_c_i != c] = 0

            gt_c_i = np.clip(gt_c_i, 0, 1)
            pred_c_i = np.clip(pred_c_i, 0, 1)

            dice = dc(gt_c_i, pred_c_i)

            volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
            volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.

            res += [dice, volpred, volpred - volgt]

    return res


def count_non_empty_frames(data):
    """
    Counts various types of non-empty frames in the dataset.
    
    Parameters:
        data (list of dict): Each dict should have 't1c', 't1n', 't2f', 't2w', and 'mask' keys pointing to file paths.
    
    Returns:
        tuple: 
            total_non_empty_frames (int),
            total_non_empty_tumor (int),
            total_non_empty_class1 (int),
            total_non_empty_class2 (int),
            total_non_empty_class3 (int)
    """
    total_non_empty_frames = 0
    total_non_empty_tumor = 0
    total_non_empty_class1 = 0
    total_non_empty_class2 = 0
    total_non_empty_class3 = 0

    for sample_idx, sample in enumerate(data):
        try:
            img_paths = {
                't1c': sample['t1c'],
                't1n': sample['t1n'],
                't2f': sample['t2f'],
                't2w': sample['t2w']
            }
            mask_path = sample['mask']
        except KeyError as e:
            logging.error(f"Sample {sample_idx} is missing key: {e}. Skipping...")
            continue  # Skip this sample

        try:
            imgs = {modality: nib.load(path).get_fdata() for modality, path in img_paths.items()}
            mask = nib.load(mask_path).get_fdata()
        except Exception as e:
            logging.error(f"Error loading files for sample {sample_idx}: {e}. Skipping...")
            continue  # Skip this sample

        # Check if all modalities have the same number of slices as the mask
        num_slices = imgs['t1c'].shape[2]
        if not all(img.shape[2] == num_slices for img in imgs.values()) or mask.shape[2] != num_slices:
            logging.warning(f"Sample {sample_idx} has mismatched slice counts. Skipping...")
            continue  # Skip samples with mismatched slice counts

        for frame_idx in range(num_slices):
            # Check if **all four** modalities have non-zero values in this slice
            all_modalities_non_empty = True
            for modality in ['t1c', 't1n', 't2f', 't2w']:
                img_slice = imgs[modality][:, :, frame_idx]
                if not np.any(img_slice):
                    all_modalities_non_empty = False
                    break  # No need to check further modalities

            if not all_modalities_non_empty:
                continue  # Skip this slice as not all modalities have data

            # Increment the total non-empty frames
            total_non_empty_frames += 1

            # Count frames with any tumor
            mask_slice = mask[:, :, frame_idx]
            if np.any(mask_slice):
                total_non_empty_tumor += 1

            # Count frames for each tumor class
            if np.any(mask_slice == 1):
                total_non_empty_class1 += 1
            if np.any(mask_slice == 2):
                total_non_empty_class2 += 1
            if np.any(mask_slice == 3):
                total_non_empty_class3 += 1

    return (
        total_non_empty_frames,
        total_non_empty_tumor,
        total_non_empty_class1,
        total_non_empty_class2,
        total_non_empty_class3
    )

from tensorflow.keras.callbacks import Callback

class SaveSamplePredictions(Callback):
    def __init__(self, validation_data, save_dir, num_samples=3, num_non_tumor_samples=20, max_iter_resets=2, fold=1, threshold=0.5):
        super(SaveSamplePredictions, self).__init__()
        self.validation_data = validation_data
        self.save_dir = save_dir
        self.fold = fold
        self.num_samples = num_samples
        self.num_non_tumor_samples = num_non_tumor_samples
        self.max_iter_resets = max_iter_resets
        self.threshold = threshold

        # Create a directory for the current fold if it doesn't exist
        self.fold_dir = os.path.join(self.save_dir, f'fold_{self.fold}')
        if not os.path.exists(self.fold_dir):
            os.makedirs(self.fold_dir)

    def on_epoch_end(self, epoch, logs=None):
        # Create a new iterator from the dataset
        data_iterator = iter(self.validation_data)

        saved_tumor_samples = 0
        saved_non_tumor_samples = 0
        iter_resets = 0

        while (saved_tumor_samples < self.num_samples or saved_non_tumor_samples < self.num_non_tumor_samples) and iter_resets < self.max_iter_resets:
            try:
                # Get the next batch of validation data
                val_batch = next(data_iterator)
                if val_batch is None:
                    print("End of validation data sequence. Resetting iterator...")
                    data_iterator = iter(self.validation_data)
                    iter_resets += 1
                    continue

                val_images_dict, val_masks = val_batch  # val_images_dict is a dictionary

                # Predict using the model for the current batch
                predictions = self.model.predict(val_images_dict)

                # Handle model outputs (predictions)
                if isinstance(predictions, dict):
                    pred_masks = predictions['final']
                elif isinstance(predictions, list):
                    # Assuming 'final' output is the last one
                    pred_masks = predictions[-1]
                else:
                    pred_masks = predictions  # Single output

                # Get batch size
                batch_size = val_masks['final'].shape[0]

                for i in range(batch_size):
                    # Get the input images for all modalities
                    imgs = [val_images_dict[modality][i, :, :, 0] for modality in ['t1n_input', 't1c_input', 't2w_input', 't2f_input']]

                    # Get the true and predicted masks
                    true_mask = val_masks['final'][i]
                    pred_mask = pred_masks[i]

                    # Check if the ground truth mask contains any tumor (channels 1, 2, or 3 are non-zero)
                    if np.any(true_mask[..., 1:]) and saved_tumor_samples < self.num_samples:
                        self.save_sample(epoch, i, imgs, true_mask, pred_mask, sample_type="tumor", sample_index=saved_tumor_samples + 1)
                        saved_tumor_samples += 1

                    elif not np.any(true_mask[..., 1:]) and saved_non_tumor_samples < self.num_non_tumor_samples:
                        self.save_sample(epoch, i, imgs, true_mask, pred_mask, sample_type="non_tumor", sample_index=saved_non_tumor_samples + 1)
                        saved_non_tumor_samples += 1

                    if saved_tumor_samples >= self.num_samples and saved_non_tumor_samples >= self.num_non_tumor_samples:
                        break

            except StopIteration:
                print("End of validation data sequence. Resetting iterator...")
                data_iterator = iter(self.validation_data)
                iter_resets += 1

        print(f'Saved {saved_tumor_samples} tumor sample predictions for epoch {epoch + 1}')
        print(f'Saved {saved_non_tumor_samples} non-tumor sample predictions for epoch {epoch + 1}')

    def save_sample(self, epoch, i, imgs, true_mask, pred_mask, sample_type, sample_index):
        # Define modality names
        modality_names = ["T1N", "T1C", "T2W", "T2F"]
        mask_labels = ["Background", "Tumor Core (TC)", "Whole Tumor (WT)", "Enhancing Tumor (ET)"]
        num_modalities = len(imgs)

        fig, axes = plt.subplots(3, max(num_modalities, 4), figsize=(20, 15))
        fig.suptitle(f"Epoch {epoch + 1} - {sample_type.capitalize()} Sample {sample_index}")

        # Display the original input images for each modality
        for idx, img in enumerate(imgs):
            axes[0, idx].imshow(img, cmap='gray')
            axes[0, idx].set_title(f'{modality_names[idx]} Modality')
            axes[0, idx].axis('off')

        # Hide any unused axes in the first row
        for idx in range(num_modalities, axes.shape[1]):
            axes[0, idx].axis('off')

        # Display Ground Truth Masks
        for j in range(4):
            if j == 0:
                true_mask_class = true_mask[..., j]
            elif j == 1:
                # Tumor Core (TC): Labels 1 and 3
                true_mask_class = np.logical_or(true_mask[..., 1], true_mask[..., 3]).astype(np.float32)
            elif j == 2:
                # Whole Tumor (WT): Labels 1, 2, and 3
                true_mask_class = np.logical_or.reduce([true_mask[..., 1], true_mask[..., 2], true_mask[..., 3]]).astype(np.float32)
            elif j == 3:
                # Enhancing Tumor (ET): Label 3
                true_mask_class = true_mask[..., 3]

            axes[1, j].imshow(true_mask_class, cmap='jet')
            axes[1, j].set_title(f'Ground Truth: {mask_labels[j]}')
            axes[1, j].axis('off')

        # Hide any unused axes in the second row
        for idx in range(4, axes.shape[1]):
            axes[1, idx].axis('off')

        # Display Predicted Masks
        for j in range(4):
            if j == 0:
                pred_mask_class = (pred_mask[..., j] > self.threshold).astype(np.float32)
            elif j == 1:
                # Tumor Core (TC): Labels 1 and 3
                pred_mask_class = (np.maximum(pred_mask[..., 1], pred_mask[..., 3]) > self.threshold).astype(np.float32)
            elif j == 2:
                # Whole Tumor (WT): Labels 1, 2, and 3
                pred_mask_class = (np.maximum.reduce([pred_mask[..., 1], pred_mask[..., 2], pred_mask[..., 3]]) > self.threshold).astype(np.float32)
            elif j == 3:
                # Enhancing Tumor (ET): Label 3
                pred_mask_class = (pred_mask[..., 3] > self.threshold).astype(np.float32)

            axes[2, j].imshow(pred_mask_class, cmap='jet')
            axes[2, j].set_title(f'Predicted: {mask_labels[j]}')
            axes[2, j].axis('off')

        # Hide any unused axes in the third row
        for idx in range(4, axes.shape[1]):
            axes[2, idx].axis('off')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Create a subdirectory for the fold, epoch, and sample type if it doesn't exist
        fold_epoch_dir = os.path.join(self.fold_dir, f'epoch_{epoch + 1}', sample_type)
        os.makedirs(fold_epoch_dir, exist_ok=True)

        # Save figure to disk within the newly created subdirectory
        fig_path = os.path.join(fold_epoch_dir, f'sample_{sample_index}.png')
        fig.savefig(fig_path)
        plt.close(fig)



# class AlphaBetaTrackingCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         print(f"\nScaling parameters at epoch {epoch}:")
#         for layer in self.model.layers:
#             # Check if the layer is an instance that might have scaling parameters.
#             if isinstance(layer, ScalingLayer):
#                 s = ""
#                 if hasattr(layer, 'alpha'):
#                     alpha_value = layer.alpha.numpy()
#                     s += f"alpha: {alpha_value:.4f}"
#                 if hasattr(layer, 'beta'):
#                     beta_value = layer.beta.numpy()
#                     s += f", beta: {beta_value:.4f}"
#                 print(f"  {layer.name} -> {s}")



import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold

# Hyperparameters
EPOCHS = 150
BATCH_SIZE = 32
NUM_CLASSES = 4
INPUT_SHAPE = (224, 224, 1)
threshold = 0.5  # Make sure this is defined here


# Directory for logs and weights
save_dir = '/scratch_net/maus_second/mrostamian/project/log_multi_unet_adapter_v2'
os.makedirs(save_dir, exist_ok=True)

# Load file paths (user-defined)
data_samples = load_file_paths(brats_data_train)  # Your function to get training sample paths

# 5-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Use 4 GPUs via MirroredStrategy
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"])

for fold, (train_index, val_index) in enumerate(kf.split(data_samples)):
    fold_id = fold + 1
    print(f"\n=== Training fold {fold_id} ===")
    
    # Prepare training and validation data lists
    train_data = [data_samples[i] for i in train_index]
    val_data   = [data_samples[i] for i in val_index]

    # Count valid frames (your updated function)
    valid_train_frames, valid_train_with_tumor, valid_train_class1, valid_train_class2, valid_train_class3 = count_non_empty_frames(train_data)
    valid_validation_frames, valid_validation_with_tumor, valid_validation_class1, valid_validation_class2, valid_validation_class3 = count_non_empty_frames(val_data)

    # Print the results for better understanding
    print("Training Data:")
    print(f"Total non-empty frames: {valid_train_frames}")
    print(f"Frames with any tumor: {valid_train_with_tumor}")
    print(f"Frames with class 1 (e.g., NCR): {valid_train_class1}")
    print(f"Frames with class 2 (e.g., ED): {valid_train_class2}")
    print(f"Frames with class 3 (e.g., ET): {valid_train_class3}")

    print("\nValidation Data:")
    print(f"Total non-empty frames: {valid_validation_frames}")
    print(f"Frames with any tumor: {valid_validation_with_tumor}")
    print(f"Frames with class 1 (e.g., NCR): {valid_validation_class1}")
    print(f"Frames with class 2 (e.g., ED): {valid_validation_class2}")
    print(f"Frames with class 3 (e.g., ET): {valid_validation_class3}")

    # Log frames info
    msg = (f"Fold {fold_id}:\n"
           f"Valid train frames = {valid_train_frames}, "
           f"frames with tumor = {valid_train_with_tumor}, "
           f"frames with necrotic tumor core (NCR - label 1) = {valid_train_class1}, "
           f"frames with peritumoral edematous/invaded tissue (ED - label 2) = {valid_train_class2}, "
           f"frames with GD-enhancing tumor (ET - label 3) = {valid_train_class3}\n"
           f"Valid validation frames = {valid_validation_frames}, "
           f"frames with tumor = {valid_validation_with_tumor}, "
           f"frames with necrotic tumor core (NCR - label 1) = {valid_validation_class1}, "
           f"frames with peritumoral edematous/invaded tissue (ED - label 2) = {valid_validation_class2}, "
           f"frames with GD-enhancing tumor (ET - label 3) = {valid_validation_class3}\n")

    with open(os.path.join(save_dir, 'valid_frames_data.txt'), 'a') as f:
        f.write(msg)


    # Clear previous session
    tf.keras.backend.clear_session()
    
    with strategy.scope():
        # Define model inputs (one per modality)
        input_t1c = tf.keras.Input(shape=INPUT_SHAPE, name='t1c_input')
        input_t1n = tf.keras.Input(shape=INPUT_SHAPE, name='t1n_input')
        input_t2f = tf.keras.Input(shape=INPUT_SHAPE, name='t2f_input')
        input_t2w = tf.keras.Input(shape=INPUT_SHAPE, name='t2w_input')
        inputs = [input_t1c, input_t1n, input_t2f, input_t2w]
    
        # Build the multi-encoder U-Net model
        model = create_multi_encoder_unet(inputs=inputs, num_encoders=4, num_classes=NUM_CLASSES)
    
        # Define optimizer, loss, and metrics (adjust as needed)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        loss = {
            'pred1': combined_loss,  # your user-defined loss function
            'pred2': combined_loss,
            'final': combined_loss
        }
        loss_weights = {'pred1': 0.14, 'pred2': 0.29, 'final': 0.57}


        metrics = {
            'final': [
                create_combined_channels_metric(dice_metric, class_indices=[1,3], name='dice_tc', threshold=threshold),
                create_combined_channels_metric(dice_metric, class_indices=[1,2,3], name='dice_wt', threshold=threshold),
                create_combined_channels_metric(dice_metric, class_indices=[3], name='dice_et', threshold=threshold),
            ]
        }

        model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)
        model.summary()


        print("Trainable Variables (scaling parameters):")
        for var in model.trainable_weights:
            if 'alpha' in var.name or 'beta' in var.name:
                print(f"{var.name}: shape={var.shape}")


        # Steps per epoch
        steps_per_epoch   = int(np.ceil(valid_train_frames) / BATCH_SIZE)
        validation_steps  = int(np.ceil(valid_validation_frames) / BATCH_SIZE)
    
    # Define the output_signature for the data generator
    input_signature = (
        {
            't1c_input': tf.TensorSpec((None, INPUT_SHAPE[0], INPUT_SHAPE[1], 1), tf.float32),
            't1n_input': tf.TensorSpec((None, INPUT_SHAPE[0], INPUT_SHAPE[1], 1), tf.float32),
            't2f_input': tf.TensorSpec((None, INPUT_SHAPE[0], INPUT_SHAPE[1], 1), tf.float32),
            't2w_input': tf.TensorSpec((None, INPUT_SHAPE[0], INPUT_SHAPE[1], 1), tf.float32)
        },
        {
            'pred1': tf.TensorSpec((None, INPUT_SHAPE[0]//4, INPUT_SHAPE[1]//4, NUM_CLASSES), tf.float32),
            'pred2': tf.TensorSpec((None, INPUT_SHAPE[0]//2, INPUT_SHAPE[1]//2, NUM_CLASSES), tf.float32),
            'final': tf.TensorSpec((None, INPUT_SHAPE[0], INPUT_SHAPE[1], NUM_CLASSES), tf.float32)
        }
    )
    
    # Load slices into memory (your user-defined functions)
    train_slices = load_all_slices_once(train_data, input_size=INPUT_SHAPE[:2])
    val_slices = load_all_slices_once(val_data, input_size=INPUT_SHAPE[:2])
    
    # Create training and validation datasets
    train_gen = tf.data.Dataset.from_generator(
        lambda: slice_generator_from_memory(
            train_slices,
            input_size=INPUT_SHAPE[:2],
            batch_size=BATCH_SIZE,
            augment=True,
            dset="training",
            single_pass=False
        ),
        output_signature=input_signature
    ).repeat()
    
    val_gen = tf.data.Dataset.from_generator(
        lambda: slice_generator_from_memory(
            val_slices,
            input_size=INPUT_SHAPE[:2],
            batch_size=BATCH_SIZE,
            augment=False,
            dset="validation",
            single_pass=True
        ),
        output_signature=input_signature
    ).repeat()
    
    # SaveSamplePredictions callback (keep your provided callback)
    save_predictions_callback = SaveSamplePredictions(
        validation_data=val_gen,
        save_dir=os.path.join(save_dir, "sample_predictions_multi_encoder"),
        num_samples=50,
        num_non_tumor_samples=20,
        fold=fold_id,
        threshold=0.5  # adjust threshold if needed
    )
    
    # TensorBoard callback for logging
    log_dir = os.path.join(save_dir, "tensorboard_logs", f"fold_{fold_id}_{time.strftime('%Y%m%d-%H%M%S')}")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch')
    
    # ModelCheckpoint callback: save only the best weights.
    # The filename includes the epoch number and ends with ".weights.h5"
    best_weights_filepath = os.path.join(save_dir, f"best_weights_fold_{fold_id}_epoch_{{epoch:02d}}.weights.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_weights_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    # Use the two callbacks plus the sample predictions callback
    callbacks = [tensorboard_callback, save_predictions_callback, checkpoint_callback]
    
    # Train the model for 30 epochs
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # After training, save the final best weights with a filename ending in ".weights.h5"
    final_best_weights_path = os.path.join(save_dir, f"final_best_weights_fold_{fold_id}.weights.h5")
    model.save_weights(final_best_weights_path)
    print(f"Fold {fold_id}: Final best weights saved at {final_best_weights_path}")
    
    # Cleanup before the next fold
    del train_slices, val_slices, train_gen, val_gen
    import gc
    gc.collect()

print("All folds complete!")
