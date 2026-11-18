# changes when running on the cluster
# %% Imports
# changes when running on the cluster
import os
import tensorflow as tf
import logging

from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers.experimental import AdamW


from tqdm import tqdm 


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
from tensorflow.keras.optimizers import Adam


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






# def get_brats_generator(file_paths, input_size=(224, 224), batch_size=32, augment=False, dset="training", single_pass=False):
#     """
#     Generator that loads all data first, shuffles it, and yields batches.
    
#     Parameters:
#     - file_paths: List of dictionaries containing paths to modalities and masks.
#     - input_size: Tuple indicating the (height, width, channels) of input slices.
#     - batch_size: Number of samples per batch.
#     - augment: Boolean indicating whether to apply data augmentation.
#     - dset: String indicating the dataset type ("training", "validation", or "test").
#     - single_pass: Boolean indicating whether to iterate over the data once.
    
#     Yields:
#     - Tuple of (inputs_dict, targets_dict) matching the expected input_signature.
#     """
#     def load_images_and_masks(file_paths):
#         all_slices = []
#         sample_count = 0

#         save_sample_dir = '/scratch_net/maus/mrostamian/project/log_multi_v4_2GPU/visualize_img'
#         os.makedirs(save_sample_dir, exist_ok=True)

#         # Pre-load all slices
#         for sample in file_paths:
#             # Load images for all modalities
#             imgs = {}
#             for modality in ['t1c', 't1n', 't2f', 't2w']:
#                 img_path = sample[modality]
#                 img = nib.load(img_path).get_fdata()
#                 imgs[modality] = img

#             # Load the mask
#             mask_path = sample['mask']
#             mask = nib.load(mask_path).get_fdata()

#             # Assuming all modalities and the mask have the same dimensions
#             num_slices = imgs['t1c'].shape[2]
#             for idx in range(num_slices):
#                 img_slices = {}
#                 skip_slice = False
#                 for modality in ['t1c', 't1n', 't2f', 't2w']:
#                     # Crop and preprocess the image slice
#                     img_slice = crop_center(imgs[modality][:, :, idx], input_size[0], input_size[1])
#                     img_slice = np.expand_dims(img_slice, axis=-1)

#                     # Skip if the slice is empty
#                     if not np.any(img_slice):
#                         skip_slice = True
#                         break

#                     # Normalize the image slice
#                     img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-8)
#                     img_slices[modality] = img_slice

#                 if skip_slice:
#                     continue

#                 # Process the mask slice
#                 mask_slice = crop_center(mask[:, :, idx], input_size[0], input_size[1])
#                 mask_slice = np.expand_dims(mask_slice, axis=-1)

#                 # # Save sample images and masks as a figure (first 50 slices)
#                 # if save_sample_dir and sample_count < 30:
#                 #     fig, axs = plt.subplots(1, 5, figsize=(15, 4))  # 1 row and 5 plots
#                 #     modalities_display = ['t1c', 't1n', 't2f', 't2w']
#                 #     for i, modality in enumerate(modalities_display):
#                 #         axs[i].imshow(np.squeeze(img_slices[modality]), cmap='gray')  # Display the image
#                 #         axs[i].set_title(modality.upper())  # Add the title of the image
#                 #         axs[i].axis('off')  # Turn off axis labels

#                 #     # Display Mask
#                 #     axs[4].imshow(np.squeeze(mask_slice), cmap='gray')
#                 #     axs[4].set_title('MASK')
#                 #     axs[4].axis('off')

#                 #     plt.tight_layout()
#                 #     plt.savefig(os.path.join(save_sample_dir, f'sample_{sample_count}.png'))
#                 #     plt.close(fig)
#                 #     sample_count += 1

#                 # Append the slices to the all_slices list
#                 all_slices.append({
#                     't1c': img_slices['t1c'],
#                     't1n': img_slices['t1n'],
#                     't2f': img_slices['t2f'],
#                     't2w': img_slices['t2w'],
#                     'mask': mask_slice
#                 })

#         print(f"Total slices loaded: {len(all_slices)}")

#         # Shuffle all_slices
#         random.shuffle(all_slices)

#         # Yield batches
#         for i in range(0, len(all_slices), batch_size):
#             batch_slices = all_slices[i:i + batch_size]
#             images_dict = {'t1c': [], 't1n': [], 't2f': [], 't2w': []}
#             masks = []
#             for slice_dict in batch_slices:
#                 for modality in ['t1c', 't1n', 't2f', 't2w']:
#                     images_dict[modality].append(slice_dict[modality])
#                 masks.append(slice_dict['mask'])

#             # Prepare the batch
#             inputs, targets = prepare_batch(images_dict, masks, augment, dset)
#             yield inputs, targets

#             # Handle single_pass scenario
#             if single_pass and (i + batch_size) >= len(all_slices):
#                 break

#     return load_images_and_masks(file_paths)




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




class StochasticDepth(layers.Layer):
    def __init__(self, drop_prob=0.0, name=None, **kwargs):
        super(StochasticDepth, self).__init__(name=name, **kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if training and self.drop_prob > 0.0:
            keep_prob = 1 - self.drop_prob
            random_tensor = keep_prob + tf.random.uniform(tf.shape(x), 0, 1)
            random_tensor = tf.floor(random_tensor)
            output = tf.divide(x, keep_prob) * random_tensor
            return output
        else:
            return x


def wide_focus(x, filters, dropout_rate, name_prefix='wide_focus', training=None):
    x1 = layers.Conv2D(
        filters, 3, padding='same', activation=tf.nn.gelu,
        name=f'{name_prefix}_conv1'
    )(x)
    x1 = layers.Dropout(dropout_rate, name=f'{name_prefix}_dropout1')(x1, training=training)

    x2 = layers.Conv2D(
        filters, 3, padding='same', dilation_rate=2, activation=tf.nn.gelu,
        name=f'{name_prefix}_conv2'
    )(x)
    x2 = layers.Dropout(dropout_rate, name=f'{name_prefix}_dropout2')(x2, training=training)

    x3 = layers.Conv2D(
        filters, 3, padding='same', dilation_rate=3, activation=tf.nn.gelu,
        name=f'{name_prefix}_conv3'
    )(x)
    x3 = layers.Dropout(dropout_rate, name=f'{name_prefix}_dropout3')(x3, training=training)

    added = layers.Add(name=f'{name_prefix}_add')([x1, x2, x3])

    x_out = layers.Conv2D(
        filters, 3, padding='same', activation=tf.nn.gelu,
        name=f'{name_prefix}_conv4'
    )(added)
    x_out = layers.Dropout(dropout_rate, name=f'{name_prefix}_dropout4')(x_out, training=training)
    return x_out



class Attention(Layer):
    def __init__(self, dim_out, num_heads, proj_drop=0.0, kernel_size=3,
                 stride_kv=1, stride_q=1, padding_kv="same", padding_q="same",
                 attention_bias=True, name_prefix='attention', **kwargs):
        super().__init__(**kwargs)
        self.dim = dim_out
        self.num_heads = num_heads
        self.proj_drop = proj_drop
        self.name_prefix = name_prefix

        self.conv_proj_q = self._build_projection(
            kernel_size, stride_q, padding_q,
            name_prefix=f'{name_prefix}_conv_proj_q'
        )
        self.conv_proj_k = self._build_projection(
            kernel_size, stride_kv, padding_kv,
            name_prefix=f'{name_prefix}_conv_proj_k'
        )
        self.conv_proj_v = self._build_projection(
            kernel_size, stride_kv, padding_kv,
            name_prefix=f'{name_prefix}_conv_proj_v'
        )

        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.dim // self.num_heads,
            use_bias=attention_bias,
            name=f'{name_prefix}_mha'
        )
        self.proj_drop_layer = Dropout(proj_drop, name=f'{name_prefix}_proj_drop')

    def _build_projection(self, kernel_size, stride, padding, name_prefix):
        proj = Sequential([
            DepthwiseConv2D(
                kernel_size, padding=padding, strides=stride, use_bias=False,
                name=f'{name_prefix}_depthwise_conv'
            ),
            layers.LayerNormalization(name=f'{name_prefix}_layer_norm'),
        ], name=f'{name_prefix}_proj')
        return proj

    def call(self, inputs, mask=None, training=None):
        x = inputs  # [batch_size, H_q, W_q, C]
        batch_size = tf.shape(x)[0]
        H_q = tf.shape(x)[1]
        W_q = tf.shape(x)[2]

        # Apply projections
        q = self.conv_proj_q(x, training=training)
        k = self.conv_proj_k(x, training=training)
        v = self.conv_proj_v(x, training=training)

        # Flatten spatial dimensions
        q = tf.reshape(q, [batch_size, H_q * W_q, self.dim])
        k = tf.reshape(k, [batch_size, -1, self.dim])  # H_k * W_k computed automatically
        v = tf.reshape(v, [batch_size, -1, self.dim])

        # Apply attention
        attention_output = self.attention(
            query=q,
            value=v,
            key=k,
            # training=training
        )  # [batch_size, H_q * W_q, dim]

        # Reshape back to spatial dimensions
        attention_output = tf.reshape(attention_output, [batch_size, H_q, W_q, self.dim])

        # Apply dropout
        attention_output = self.proj_drop_layer(attention_output, training=training)

        return attention_output


def att(
    x_in,
    num_heads,
    dpr,
    proj_drop=0.0,
    attention_bias=True,
    padding_q="same",
    padding_kv="same",
    stride_kv=2,
    stride_q=1,
    training=None,
    name_prefix='att'
):
    c = x_in.shape[-1]

    attention_output = Attention(
        dim_out=c,
        num_heads=num_heads,
        proj_drop=proj_drop,
        attention_bias=attention_bias,
        padding_q=padding_q,
        padding_kv=padding_kv,
        stride_kv=stride_kv,
        stride_q=stride_q,
        name_prefix=f'{name_prefix}_attention'
    )(x_in, training=training)

    # Skip connection 1
    attention_output = StochasticDepth(
        dpr, name=f'{name_prefix}_stochastic_depth1'
    )(attention_output, training=training)

    attention_output = Conv2D(
        c, 3, padding="same", activation="relu",
        name=f'{name_prefix}_conv1'
    )(attention_output)

    x2 = layers.Add(name=f'{name_prefix}_add1')([attention_output, x_in])

    x3 = layers.LayerNormalization(epsilon=1e-5, name=f'{name_prefix}_layer_norm')(x2)

    x3 = wide_focus(
        x3, filters=c, dropout_rate=0.0,
        name_prefix=f'{name_prefix}_wide_focus',
        training=training  # Pass training to wide_focus
    )

    # Skip connection 2
    x3 = StochasticDepth(
        dpr, name=f'{name_prefix}_stochastic_depth2'
    )(x3, training=training)

    x3 = layers.Add(name=f'{name_prefix}_add2')([x3, x2])

    return x3


#--------------------------------------------------------------------------------------------------------------

import re

def set_trainable_by_name(model, patterns, exact_names):
    """
    Sets layers as trainable based on naming patterns and exact names.

    Parameters:
    - model: The Keras Model instance.
    - patterns: List of regex patterns to match layer names.
    - exact_names: List of exact layer names to set as trainable.
    """
    for layer in model.layers:
        # Check exact name matches
        if layer.name in exact_names:
            layer.trainable = True
            print(f"Layer '{layer.name}' set to trainable (exact match).")
        
        # Check pattern matches
        else:
            for pattern in patterns:
                if re.match(pattern, layer.name):
                    layer.trainable = True
                    print(f"Layer '{layer.name}' set to trainable (pattern match: '{pattern}').")
                    break  # Avoid multiple matches
        
        # If the layer has sublayers, recurse
        if hasattr(layer, 'layers') and len(layer.layers) > 0:
            set_trainable_by_name(layer, patterns, exact_names)



def print_layer_weights(layer, indent=0):
    """
    Recursively prints information about a layer and its sublayers if any.

    Parameters:
    - layer: A Keras layer instance.
    - indent: Indentation level for nested layers.
    """
    indent_str = "  " * indent
    layer_type = type(layer).__name__
    print(f"{indent_str}Layer: {layer.name} | Type: {layer_type}")
    
    weights = layer.get_weights()
    if weights:
        print(f"{indent_str}  Number of weights: {len(weights)}")
        for idx, weight in enumerate(weights):
            weight_sum = np.sum(weight)
            weight_mean = np.mean(weight)
            weight_std = np.std(weight)
            print(f"{indent_str}  Weight {idx + 1}: Shape: {weight.shape}, Sum: {weight_sum:.4f}, Mean: {weight_mean:.4f}, Std: {weight_std:.4f}")
    else:
        print(f"{indent_str}  No weights.")
    
    # If the layer has sublayers, recurse into them
    if hasattr(layer, 'layers') and isinstance(layer.layers, list):
        for sublayer in layer.layers:
            print_layer_weights(sublayer, indent + 1)

def verify_weights_loaded(model):
    """
    Verifies and prints detailed information about all layers and sublayers in the model.

    Parameters:
    - model: A Keras Model instance.
    """
    print("="*80)
    print("Verifying Weights Loaded in the Model")
    print("="*80)
    
    for layer in model.layers:
        print_layer_weights(layer)
        print("-"*80)
    
    print("Weight verification completed.")

def compare_specific_layers(pretrained_layer, new_layer, atol=1e-6):
    """
    Compares weights of two layers to ensure they are identical or within a tolerance.

    Parameters:
    - pretrained_layer: Layer from the pretrained encoder.
    - new_layer: Corresponding layer from the main model.
    - atol: Absolute tolerance for comparison.
    """
    pretrained_weights = pretrained_layer.get_weights()
    new_weights = new_layer.get_weights()
    
    if len(pretrained_weights) != len(new_weights):
        print(f"[Comparison Failed] Number of weights mismatch between '{pretrained_layer.name}' and '{new_layer.name}'.")
        return False
    
    for idx, (pw, nw) in enumerate(zip(pretrained_weights, new_weights)):
        if not np.allclose(pw, nw, atol=atol):
            print(f"[Comparison Failed] Weight {idx+1} mismatch in layer '{pretrained_layer.name}' and '{new_layer.name}'.")
            return False
    print(f"[Comparison Success] All weights match for layer '{pretrained_layer.name}' and '{new_layer.name}'.")
    return True

def load_pretrained_encoder(modality, input_shape, filters, att_heads, dpr, freeze=True, drp_out=0):
    """
    Reconstructs the pretrained encoder model (without adapters), loads the pretrained weights,
    optionally freezes the encoder layers, and returns the model.

    Parameters:
    - modality: The modality string (e.g., 't1n', 't1c', etc.)
    - input_shape: The shape of the input tensor (excluding batch size)
    - filters: List of filters for the convolutional layers
    - att_heads: List of attention heads for the attention mechanisms
    - dpr: List of drop path rates
    - freeze: Boolean flag to freeze the encoder layers after loading weights

    Returns:
    - pretrained_encoder: The pretrained encoder model with weights loaded (and optionally frozen)
    """

    # Reconstruct the pretrained encoder model without adapters
    pretrained_encoder, _ = create_pretrained_encoder(
        input_shape=input_shape,
        filters=filters,
        att_heads=att_heads,
        dpr=dpr,
        drp_out=drp_out,
        name=f'pretrained_encoder_{modality}',
        name_prefix=modality
    )

    # Path to the pretrained weights
    pretrained_weights_path = f'/scratch_net/maus/mrostamian/project/weights/best_encoder_weights_{modality}.weights.h5'

    # Load the pretrained weights
    pretrained_encoder.load_weights(pretrained_weights_path, by_name = True)

    # Verify weights loaded into the pretrained encoder
    print(f"\n--- Verifying Weights for Pretrained Encoder: {modality} ---")
    verify_weights_loaded(pretrained_encoder)

    # Freeze the encoder layers if specified

    print(f"Pretrained weights loaded successfully for {modality}")

    return pretrained_encoder


def generate_layer_name_mapping(pre_start_idx, pre_end_idx, new_start_idx):
    """
    Generates a dictionary mapping between pretrained layer names and new model layer names
    for the attention layers.

    Parameters:
    - pre_start_idx: The starting index of attention layers in the pretrained model
    - pre_end_idx: The ending index of attention layers in the pretrained model
    - new_start_idx: The starting index of attention layers in the new model

    Returns:
    - A dictionary mapping pretrained layer names to new model layer names
    """
    mapping = {}
    pre_indices = range(pre_start_idx, pre_end_idx + 1)
    new_indices = range(new_start_idx, new_start_idx + len(pre_indices))

    for pre_idx, new_idx in zip(pre_indices, new_indices):
        pre_name = 'attention' if pre_idx == 0 else f'attention_{pre_idx}'
        new_name = f'attention_{new_idx}'
        mapping[pre_name] = new_name
    return mapping



def create_pretrained_encoder(
    input_shape,
    filters,
    att_heads,
    dpr,
    initializer='he_normal',
    act='relu',
    drp_out=0,
    training=True,
    name='encoder_model',
    name_prefix='encoder'
):
    """
    Reconstructs the pretrained encoder model with the exact architecture
    and layer names used during pretraining, including collecting feature maps.

    Returns:
    - encoder_model: The pretrained encoder model used during pretraining
    - encoder_feature_map_model: (Optional) Model for extracting feature maps
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
    x1 = att(
        x1,
        num_heads=att_heads[0],
        dpr=dpr[0],
        training=training,
        name_prefix=f'{name_prefix}_block1_att'
    )
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
    x2 = att(
        x2,
        num_heads=att_heads[1],
        dpr=dpr[1],
        training=training,
        name_prefix=f'{name_prefix}_block2_att'
    )
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
    x3 = att(
        x3,
        num_heads=att_heads[2],
        dpr=dpr[2],
        training=training,
        name_prefix=f'{name_prefix}_block3_att'
    )
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
    x4 = att(
        x4,
        num_heads=att_heads[3],
        dpr=dpr[3],
        training=training,
        name_prefix=f'{name_prefix}_block4_att'
    )
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
    x5 = att(
        x5,
        num_heads=att_heads[4],
        dpr=dpr[4],
        training=training,
        name_prefix=f'{name_prefix}_bottleneck_att'
    )
    encoder_output = x5
    encoder_feature_maps.append(x5)  # Collect feature map from Bottleneck

    # Create the encoder model
    outputs = [encoder_output] + skip_connections
    encoder_model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    # Create a model for extracting feature maps, if needed
    encoder_feature_map_model = tf.keras.Model(
        inputs=inputs, outputs=encoder_feature_maps, name=f'{name}_features'
    )

    return encoder_model, encoder_feature_map_model



def create_encoder_with_adapters(
    pretrained_encoder,
    input_tensor,
    filters,
    att_heads,
    dpr,
    initializer='he_normal',
    act='relu',
    drp_out=0,
    training=True,
    name='encoder_model',
    name_prefix='encoder'
):
    """
    Creates the encoder model including adapters.

    Parameters:
        - pretrained_encoder: The pretrained encoder model.
        - input_tensor: Input tensor for this encoder.
        - filters: List of filters for the convolutional layers.
        - att_heads: List of attention heads for the attention mechanisms.
        - dpr: List of drop path rates.
        - initializer: Kernel initializer for convolutional layers.
        - act: Activation function to use.
        - drp_out: Dropout rate.
        - training: Boolean indicating training mode.
        - name: Name of the encoder model.
        - name_prefix: Prefix for layer names to ensure uniqueness.

    Returns:
        - encoder_model: The encoder model with adapters.
    """
    print("Pretrained encoder attention layers:")
    for layer in pretrained_encoder.layers:
        if 'attention' in layer.name:
            print(layer.name)

    inputs = input_tensor
    x = inputs
    skip_connections = []

    # Collect attention layer names from the pretrained encoder
    attention_layer_names = [layer.name for layer in pretrained_encoder.layers if 'attention' in layer.name] 
    # Sort the attention layers in the order of blocks 
    # attention_layer_names.sort(key=lambda name: int(name.split('_')[-1]) if '_' in name else 0)

    print("Pretrained encoder attention layers after sorting:")
    for layer in attention_layer_names:
        if 'attention' in layer:
            print(layer)
    ## Block 1 ##
    # LayerNormalization
    pretrained_layer = pretrained_encoder.get_layer(f'{name_prefix}_block1_norm')
    block1_norm_config = pretrained_layer.get_config()
    block1_norm = layers.LayerNormalization.from_config(block1_norm_config)
    block1_norm._name = f'{name_prefix}_block1_norm'
    x1 = block1_norm(x)  # Build the layer
    block1_norm.set_weights(pretrained_layer.get_weights())

    # Lambda layer (assuming the operation is intentional)
    x1 = layers.Lambda(
        lambda x: x[:, :, :, -1:],
        name=f'{name_prefix}_block1_lambda'
    )(x1)

    # Conv2D Layer 1
    pretrained_layer = pretrained_encoder.get_layer(f'{name_prefix}_block1_conv1')
    block1_conv1_config = pretrained_layer.get_config()
    block1_conv1 = layers.Conv2D.from_config(block1_conv1_config)
    block1_conv1._name = f'{name_prefix}_block1_conv1'
    x1 = block1_conv1(x1)  # Build the layer
    block1_conv1.set_weights(pretrained_layer.get_weights())

    # Conv2D Layer 2
    pretrained_layer = pretrained_encoder.get_layer(f'{name_prefix}_block1_conv2')
    block1_conv2_config = pretrained_layer.get_config()
    block1_conv2 = layers.Conv2D.from_config(block1_conv2_config)
    block1_conv2._name = f'{name_prefix}_block1_conv2'
    x1 = block1_conv2(x1)  # Build the layer
    block1_conv2.set_weights(pretrained_layer.get_weights())

    # Dropout and MaxPooling
    x1 = layers.Dropout(drp_out, name=f'{name_prefix}_block1_dropout')(x1)
    x1 = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_block1_pool')(x1)

    # Attention Layer
    # attention_layer_name = attention_layer_names[0] # use index to name attention block correctly
    x1 = att(
        x1,
        att_heads[0],
        dpr[0],
        training=training,
        name_prefix=f'{name_prefix}_block1_att'
    )
    # Override the name of the block to match the pretrained model's name
    # x1._name = attention_layer_names[0]

    skip_connections.append(x1)


    ## Block 2 ##
    # LayerNormalization
    pretrained_layer = pretrained_encoder.get_layer(f'{name_prefix}_block2_norm')
    block2_norm_config = pretrained_layer.get_config()
    block2_norm = layers.LayerNormalization.from_config(block2_norm_config)
    block2_norm._name = f'{name_prefix}_block2_norm'
    x2 = block2_norm(x1)  # Build the layer
    block2_norm.set_weights(pretrained_layer.get_weights())

    # Scaled Image
    scale_img_2 = layers.AveragePooling2D(2, 2, name=f'{name_prefix}_block2_scale')(inputs)

    # Scale Conv2D
    pretrained_layer = pretrained_encoder.get_layer(f'{name_prefix}_block2_scale_conv')
    block2_scale_conv_config = pretrained_layer.get_config()
    block2_scale_conv = layers.Conv2D.from_config(block2_scale_conv_config)
    block2_scale_conv._name = f'{name_prefix}_block2_scale_conv'
    scale_conv2 = block2_scale_conv(scale_img_2)  # Build the layer
    block2_scale_conv.set_weights(pretrained_layer.get_weights())

    # Concatenate
    x2 = layers.Concatenate(axis=-1, name=f'{name_prefix}_block2_concat')([scale_conv2, x2])

    # Conv2D Layer 1
    pretrained_layer = pretrained_encoder.get_layer(f'{name_prefix}_block2_conv1')
    block2_conv1_config = pretrained_layer.get_config()
    block2_conv1 = layers.Conv2D.from_config(block2_conv1_config)
    block2_conv1._name = f'{name_prefix}_block2_conv1'
    x2 = block2_conv1(x2)  # Build the layer
    block2_conv1.set_weights(pretrained_layer.get_weights())

    # Conv2D Layer 2
    pretrained_layer = pretrained_encoder.get_layer(f'{name_prefix}_block2_conv2')
    block2_conv2_config = pretrained_layer.get_config()
    block2_conv2 = layers.Conv2D.from_config(block2_conv2_config)
    block2_conv2._name = f'{name_prefix}_block2_conv2'
    x2 = block2_conv2(x2)  # Build the layer
    block2_conv2.set_weights(pretrained_layer.get_weights())

    # Dropout and MaxPooling
    x2 = layers.Dropout(drp_out, name=f'{name_prefix}_block2_dropout')(x2)
    x2 = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_block2_pool')(x2)

   # Attention Layer
    x2 = att(
        x2,
        att_heads[1],
        dpr[1],
        training=training,
        name_prefix=f'{name_prefix}_block2_att'
    )
    # Override the name of the block to match the pretrained model's name
    # x2._name = attention_layer_names[1]

    skip_connections.append(x2)

    ## Block 3 ##
    # LayerNormalization
    pretrained_layer = pretrained_encoder.get_layer(f'{name_prefix}_block3_norm')
    block3_norm_config = pretrained_layer.get_config()
    block3_norm = layers.LayerNormalization.from_config(block3_norm_config)
    block3_norm._name = f'{name_prefix}_block3_norm'
    x3 = block3_norm(x2)  # Build the layer
    block3_norm.set_weights(pretrained_layer.get_weights())

    # Scaled Image
    scale_img_3 = layers.AveragePooling2D(2, 2, name=f'{name_prefix}_block3_scale')(scale_img_2)

    # Scale Conv2D
    pretrained_layer = pretrained_encoder.get_layer(f'{name_prefix}_block3_scale_conv')
    block3_scale_conv_config = pretrained_layer.get_config()
    block3_scale_conv = layers.Conv2D.from_config(block3_scale_conv_config)
    block3_scale_conv._name = f'{name_prefix}_block3_scale_conv'
    scale_conv3 = block3_scale_conv(scale_img_3)  # Build the layer
    block3_scale_conv.set_weights(pretrained_layer.get_weights())

    # Concatenate
    x3 = layers.Concatenate(axis=-1, name=f'{name_prefix}_block3_concat')([scale_conv3, x3])

    # Conv2D Layer 1
    pretrained_layer = pretrained_encoder.get_layer(f'{name_prefix}_block3_conv1')
    block3_conv1_config = pretrained_layer.get_config()
    block3_conv1 = layers.Conv2D.from_config(block3_conv1_config)
    block3_conv1._name = f'{name_prefix}_block3_conv1'
    x3 = block3_conv1(x3)  # Build the layer
    block3_conv1.set_weights(pretrained_layer.get_weights())

    # Conv2D Layer 2
    pretrained_layer = pretrained_encoder.get_layer(f'{name_prefix}_block3_conv2')
    block3_conv2_config = pretrained_layer.get_config()
    block3_conv2 = layers.Conv2D.from_config(block3_conv2_config)
    block3_conv2._name = f'{name_prefix}_block3_conv2'
    x3 = block3_conv2(x3)  # Build the layer
    block3_conv2.set_weights(pretrained_layer.get_weights())

    # Dropout and MaxPooling
    x3 = layers.Dropout(drp_out, name=f'{name_prefix}_block3_dropout')(x3)
    x3 = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_block3_pool')(x3)

    # Attention Layer
    x3 = att(
        x3,
        att_heads[2],
        dpr[2],
        training=training,
        name_prefix=f'{name_prefix}_block3_att'
    )
     # Override the name of the block to match the pretrained model's name
    # x3._name = attention_layer_names[2]

    skip_connections.append(x3)

    ## Block 4 ##
    # LayerNormalization
    pretrained_layer = pretrained_encoder.get_layer(f'{name_prefix}_block4_norm')
    block4_norm_config = pretrained_layer.get_config()
    block4_norm = layers.LayerNormalization.from_config(block4_norm_config)
    block4_norm._name = f'{name_prefix}_block4_norm'
    x4 = block4_norm(x3)  # Build the layer
    block4_norm.set_weights(pretrained_layer.get_weights())

    # Scaled Image
    scale_img_4 = layers.AveragePooling2D(2, 2, name=f'{name_prefix}_block4_scale')(scale_img_3)

    # Scale Conv2D
    pretrained_layer = pretrained_encoder.get_layer(f'{name_prefix}_block4_scale_conv')
    block4_scale_conv_config = pretrained_layer.get_config()
    block4_scale_conv = layers.Conv2D.from_config(block4_scale_conv_config)
    block4_scale_conv._name = f'{name_prefix}_block4_scale_conv'
    scale_conv4 = block4_scale_conv(scale_img_4)  # Build the layer
    block4_scale_conv.set_weights(pretrained_layer.get_weights())

    # Concatenate
    x4 = layers.Concatenate(axis=-1, name=f'{name_prefix}_block4_concat')([scale_conv4, x4])

    # Conv2D Layer 1
    pretrained_layer = pretrained_encoder.get_layer(f'{name_prefix}_block4_conv1')
    block4_conv1_config = pretrained_layer.get_config()
    block4_conv1 = layers.Conv2D.from_config(block4_conv1_config)
    block4_conv1._name = f'{name_prefix}_block4_conv1'
    x4 = block4_conv1(x4)  # Build the layer
    block4_conv1.set_weights(pretrained_layer.get_weights())

    # Conv2D Layer 2
    pretrained_layer = pretrained_encoder.get_layer(f'{name_prefix}_block4_conv2')
    block4_conv2_config = pretrained_layer.get_config()
    block4_conv2 = layers.Conv2D.from_config(block4_conv2_config)
    block4_conv2._name = f'{name_prefix}_block4_conv2'
    x4 = block4_conv2(x4)  # Build the layer
    block4_conv2.set_weights(pretrained_layer.get_weights())

    # Dropout and MaxPooling
    x4 = layers.Dropout(drp_out, name=f'{name_prefix}_block4_dropout')(x4)
    x4 = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_block4_pool')(x4)

    # Attention Layer
    x4 = att(
        x4,
        att_heads[3],
        dpr[3],
        training=training,
        name_prefix=f'{name_prefix}_block4_att'
    )
    # Override the name of the block to match the pretrained model's name
    # x4._name = attention_layer_names[3]

    skip_connections.append(x4)

    ## Bottleneck ##
    # LayerNormalization
    pretrained_layer = pretrained_encoder.get_layer(f'{name_prefix}_bottleneck_norm')
    bottleneck_norm_config = pretrained_layer.get_config()
    bottleneck_norm = layers.LayerNormalization.from_config(bottleneck_norm_config)
    bottleneck_norm._name = f'{name_prefix}_bottleneck_norm'
    x5 = bottleneck_norm(x4)  # Build the layer
    bottleneck_norm.set_weights(pretrained_layer.get_weights())

    # Conv2D Layer 1
    pretrained_layer = pretrained_encoder.get_layer(f'{name_prefix}_bottleneck_conv1')
    bottleneck_conv1_config = pretrained_layer.get_config()
    bottleneck_conv1 = layers.Conv2D.from_config(bottleneck_conv1_config)
    bottleneck_conv1._name = f'{name_prefix}_bottleneck_conv1'
    x5 = bottleneck_conv1(x5)  # Build the layer
    bottleneck_conv1.set_weights(pretrained_layer.get_weights())

    # Conv2D Layer 2
    pretrained_layer = pretrained_encoder.get_layer(f'{name_prefix}_bottleneck_conv2')
    bottleneck_conv2_config = pretrained_layer.get_config()
    bottleneck_conv2 = layers.Conv2D.from_config(bottleneck_conv2_config)
    bottleneck_conv2._name = f'{name_prefix}_bottleneck_conv2'
    x5 = bottleneck_conv2(x5)  # Build the layer
    bottleneck_conv2.set_weights(pretrained_layer.get_weights())

    # Dropout and MaxPooling
    x5 = layers.Dropout(drp_out, name=f'{name_prefix}_bottleneck_dropout')(x5)
    x5 = layers.MaxPooling2D((2, 2), name=f'{name_prefix}_bottleneck_pool')(x5)

    # Attention Layer
    x5 = att(
        x5,
        att_heads[4],
        dpr[4],
        training=training,
        name_prefix=f'{name_prefix}_bottleneck_att'
    )
     # Override the name of the block to match the pretrained model's name
    # x5._name = attention_layer_names[4]

    encoder_output = x5
    
    # Create the encoder model
    encoder_outputs = [encoder_output] + skip_connections
    encoder_model = tf.keras.Model(inputs=inputs, outputs=encoder_outputs, name=name)

    return encoder_model



def transfer_pretrained_weights(pretrained_model, new_model, layer_name_mapping=None):
    """
    Transfers weights from the pretrained encoder model to the new encoder model using a layer name mapping.
    
    Parameters:
    - pretrained_model: The pretrained encoder model without adapters
    - new_model: The new encoder model with adapters
    - layer_name_mapping: Optional dictionary mapping pretrained layer names to new model layer names
    """
    pretrained_layers_dict = {layer.name: layer for layer in pretrained_model.layers}
    new_layers_dict = {layer.name: layer for layer in new_model.layers}

    transferred_layers = []

    # First, transfer weights for layers specified in the mapping
    if layer_name_mapping:
        for pre_name, new_name in layer_name_mapping.items():
            pretrained_layer = pretrained_layers_dict.get(pre_name)
            new_layer = new_layers_dict.get(new_name)
            if pretrained_layer and new_layer:
                try:
                    new_layer.set_weights(pretrained_layer.get_weights())
                    transferred_layers.append(new_name)
                    print(f"Weights transferred from {pre_name} to {new_name}")
                except Exception as e:
                    print(f"Could not transfer weights from {pre_name} to {new_name}: {e}")
            else:
                print(f"Layer {pre_name} or {new_name} not found.")
                
    # Then, transfer weights for layers with matching names, excluding those already transferred
    for layer_name in new_layers_dict.keys():
        # Skip adapter layers
        if 'adapter' in layer_name:
            continue
        # Skip layers already transferred via mapping
        if layer_name in transferred_layers:
            new_layers_dict[layer_name].trainable = False
            continue
        pretrained_layer = pretrained_layers_dict.get(layer_name)
        if pretrained_layer:
            try:
                new_layers_dict[layer_name].set_weights(pretrained_layer.get_weights())
                transferred_layers.append(layer_name)
                print(f"Weights transferred for layer: {layer_name}")
            except Exception as e:
                print(f"Could not transfer weights for layer {layer_name}: {e}")
        else:
            print(f"Layer {layer_name} not found in pretrained model.")

        new_layers_dict[layer_name].trainable = False
            

    # Optional: Print summary of transferred layers
    print(f"\nTotal layers with weights transferred: {len(transferred_layers)}")
    not_transferred_layers = [layer.name for layer in new_model.layers 
                              if layer.name not in transferred_layers and 'adapter' not in layer.name]
    if not_transferred_layers:
        print("\nLayers that did not receive weights (excluding adapters):")
        for name in not_transferred_layers:
            print(name)


def build_decoder(
    adapter_output,
    encoder_skips,
    filters_decoder,
    att_heads,
    dpr,
    training=None,
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

        # Upsample x by a factor of 2
        x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear',
                                name=f'decoder_upsample_{block_idx + 1}')(x)

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

        # Optional: Add attention layer
        x = att(
            x,
            num_heads=att_heads[block_idx],
            dpr=dpr[block_idx],
            training=training,  # Pass training parameter if required
            name_prefix=f'decoder_att_{block_idx + 1}'
        )

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
        activation=None,  # Activation applied separately
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
        activation=None,  # Activation applied separately
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
        activation=None,  # Activation applied separately
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
        activation=None,  # Activation applied separately
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
        activation=None,  # Activation applied separately
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
        activation=None,  # Activation applied separately
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


import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def create_multi_encoder_unet(inputs, num_encoders=4, training=True, num_classes=4):
    """
    Creates the multi-encoder U-Net model with pretrained and adapted encoders.
    """
    assert len(inputs) == num_encoders, "Number of inputs must match num_encoders"

    input_shape = inputs[0].shape[1:]  # Assuming all inputs have the same shape
    modalities = ['t1c', 't1n', 't2f', 't2w']  # Updated to match data generator


    drp_out = 0.1  # Dropout rate

    filters_list = [32, 64, 128, 256, 512]
    filters_decoder = [256, 128, 64, 32]  # Decoder filters
    filters = filters_list + filters_decoder
    att_heads = [2] * len(filters)
    stochastic_depth_rate = 0.0
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, len(filters))]

    # Split att_heads and dpr for encoder and decoder
    att_heads_encoder = att_heads[:len(filters_list)]
    dpr_encoder = dpr[:len(filters_list)]
    att_heads_decoder = att_heads[len(filters_list):]
    dpr_decoder = dpr[len(filters_list):]

    # Load and prepare encoders
    pretrained_encoders = []  # Store the loaded pretrained encoders

    for i, modality in enumerate(modalities[:num_encoders]):
        # Load pretrained encoder
        pretrained_encoder = load_pretrained_encoder(
            modality=modality,
            input_shape=input_shape,
            filters=filters_list,
            att_heads=att_heads_encoder,
            dpr=dpr_encoder,
            freeze=True,
            drp_out=drp_out,
        )

        pretrained_encoders.append(pretrained_encoder)

    # Initialize variables
    cnn_outputs_block = []
    scaled_images = []  # To keep track of scaled images for each encoder

    # Collect skip connections
    num_blocks = len(filters_list)  # Total number of blocks including bottleneck (should be 5)
    encoder_skips_all = [[] for _ in range(num_blocks - 1)]  # For skip connections from attention layers 1 to 4
    encoder_bottlenecks = []

    # Process the initial block (Block 1) for each encoder
    for i in range(num_encoders):
        modality = modalities[i]
        x = inputs[i]
        scaled_images.append([x])

        # Block 1 CNN layers
        x = pretrained_encoders[i].get_layer(f'{modality}_block1_norm')(x)
        lambda_layer_name = f'{modality}_block1_lambda'
        if lambda_layer_name in [layer.name for layer in pretrained_encoders[i].layers]:
            x = pretrained_encoders[i].get_layer(lambda_layer_name)(x)
        x = pretrained_encoders[i].get_layer(f'{modality}_block1_conv1')(x)
        x = pretrained_encoders[i].get_layer(f'{modality}_block1_conv2')(x)
        x = layers.Dropout(drp_out, name=f'{modality}_block1_dropout')(x)
        x = pretrained_encoders[i].get_layer(f'{modality}_block1_pool')(x)

        cnn_outputs_block.append(x)

    # Define start indices for attention layers in pretrained models if necessary
    # This depends on your pretrained model's layer naming convention
    # For illustration, assuming attention layers are named uniquely per modality

    # Loop through Blocks 1 to 3
    for block_idx in range(num_blocks - 2):  # Blocks 1 to 3
        # Apply adapter between encoders at this block
        adapted_outputs = adapter_between_encoders(
            cnn_outputs_block,
            act='relu',
            compression_ratio=0.5,
            dropout_rate=0.1,
            block_idx=block_idx
        )

        cnn_outputs_block_new = []

        for i in range(num_encoders):
            modality = modalities[i]
            x = adapted_outputs[i]

            # Define attention layer name
            attention_layer_name = f'{modality}_block{block_idx + 1}_att'

            # Apply attention using the 'att' function
            x = att(
                x,
                att_heads_encoder[block_idx],
                dpr_encoder[block_idx],
                training=training,
                name_prefix=attention_layer_name
            )

            # Collect skip connections
            encoder_skips_all[block_idx].append(x)

            # Prepare for next block
            next_block = block_idx + 2  # Blocks are 1-indexed

            norm_layer_name = f'{modality}_block{next_block}_norm'
            if norm_layer_name in [layer.name for layer in pretrained_encoders[i].layers]:
                x = pretrained_encoders[i].get_layer(norm_layer_name)(x)

                # Scaled Image
                scale_layer_name = f'{modality}_block{next_block}_scale'
                scale_layer = pretrained_encoders[i].get_layer(scale_layer_name)
                scaled_input = scale_layer(scaled_images[i][-1])
                scaled_images[i].append(scaled_input)

                # Scale Conv2D
                scale_conv_layer_name = f'{modality}_block{next_block}_scale_conv'
                scale_conv_layer = pretrained_encoders[i].get_layer(scale_conv_layer_name)
                scale_conv_output = scale_conv_layer(scaled_input)

                # Concatenate
                concat_layer_name = f'{modality}_block{next_block}_concat'
                x = layers.Concatenate(axis=-1, name=concat_layer_name)([scale_conv_output, x])

                # Conv2D Layers
                x = pretrained_encoders[i].get_layer(f'{modality}_block{next_block}_conv1')(x)
                x = pretrained_encoders[i].get_layer(f'{modality}_block{next_block}_conv2')(x)

                # Dropout and MaxPooling
                x = layers.Dropout(drp_out, name=f'{modality}_block{next_block}_dropout')(x)
                x = pretrained_encoders[i].get_layer(f'{modality}_block{next_block}_pool')(x)

                # Append to new block outputs
                cnn_outputs_block_new.append(x)
            else:
                # No further blocks; end of encoder blocks
                cnn_outputs_block_new.append(x)

        # Update cnn_outputs_block for the next iteration
        cnn_outputs_block = cnn_outputs_block_new

    # Process Block 4 Separately (Assuming Block 4 is index 3)
    block_idx = num_blocks - 2  # Block 4 (0-indexed)
    adapted_outputs = adapter_between_encoders(
        cnn_outputs_block,
        act='relu',
        compression_ratio=0.5,
        dropout_rate=0.1,
        block_idx=block_idx
    )

    cnn_outputs_block_new = []

    for i in range(num_encoders):
        modality = modalities[i]
        x = adapted_outputs[i]

        # Define attention layer name
        attention_layer_name = f'{modality}_block{block_idx + 1}_att'

        # Apply attention using the 'att' function
        x = att(
            x,
            att_heads_encoder[block_idx],
            dpr_encoder[block_idx],
            training=training,
            name_prefix=attention_layer_name
        )

        # Collect skip connections
        encoder_skips_all[block_idx].append(x)

        # Prepare for bottleneck (No further blocks)
        cnn_outputs_block_new.append(x)

    # Update cnn_outputs_block for the bottleneck processing
    cnn_outputs_block = cnn_outputs_block_new

    # Process the Bottleneck Separately
    block_idx = num_blocks - 1  # Bottleneck (Block 5)
    adapted_outputs = adapter_between_encoders(
        cnn_outputs_block,
        act='relu',
        compression_ratio=0.5,
        dropout_rate=0.1,
        block_idx=block_idx
    )

    encoder_bottlenecks = []

    for i in range(num_encoders):
        modality = modalities[i]
        x = adapted_outputs[i]

        # LayerNormalization
        norm_layer_name = f'{modality}_bottleneck_norm'
        norm_layer = pretrained_encoders[i].get_layer(norm_layer_name)
        x = norm_layer(x)

        # Conv2D Layers
        conv1_layer_name = f'{modality}_bottleneck_conv1'
        x = pretrained_encoders[i].get_layer(conv1_layer_name)(x)
        conv2_layer_name = f'{modality}_bottleneck_conv2'
        x = pretrained_encoders[i].get_layer(conv2_layer_name)(x)

        # Dropout and MaxPooling
        dropout_layer_name = f'{modality}_bottleneck_dropout'
        pool_layer_name = f'{modality}_bottleneck_pool'
        x = layers.Dropout(drp_out, name=dropout_layer_name)(x)
        x = pretrained_encoders[i].get_layer(pool_layer_name)(x)

        # Attention Layer for Bottleneck
        attention_layer_name = f'{modality}_bottleneck_att'
        x = att(
            x,
            att_heads_encoder[block_idx],
            dpr_encoder[block_idx],
            training=training,
            name_prefix=attention_layer_name
        )

        # Collect bottleneck outputs
        encoder_bottlenecks.append(x)

    # Combine Bottleneck Outputs
    combined_bottleneck_output = layers.Concatenate(name='combined_bottleneck')(encoder_bottlenecks)

    # Combine Skip Connections from Each Level
    combined_skips = []
    for level, skips_at_level in enumerate(encoder_skips_all):
        combined_skip = layers.Concatenate(name=f'combined_skip_{level + 1}')(skips_at_level)
        combined_skips.append(combined_skip)

    # Build the Decoder
    outputs = build_decoder(
        adapter_output=combined_bottleneck_output,
        encoder_skips=combined_skips,
        filters_decoder=filters_decoder,
        att_heads=att_heads_decoder,
        dpr=dpr_decoder,
        training=training,
        num_classes=num_classes
    )

    # Create the Final Model with Inputs and Outputs
    model_inputs = inputs
    model_outputs = outputs

    model = tf.keras.Model(inputs=model_inputs, outputs=model_outputs, name='multi_modal_unet')

    pre_attention_start_indices = [0, 5, 10, 15]  # Starting indices in pretrained model



    # Load Pretrained Weights into the Attention Layers
    for i in range(num_encoders):
        modality = modalities[i]
        for block_idx in range(num_blocks):  # Blocks 1 to 5 (including bottleneck)
            
            # Calculate the unique identifier for the attention layer in the new model
            num = i + 20 + 4 * (block_idx)
            attention_layer_name = f'attention_{num}'

            # Attempt to retrieve the new attention layer from the model
            try:
                new_attention_layer = model.get_layer(attention_layer_name)
            except ValueError:
                print(f"[Error] Attention layer '{attention_layer_name}' not found in the model.")
                continue  # Skip to the next iteration if the layer is not found

            # Determine the corresponding pretrained attention layer's name
            pre_start_idx = pre_attention_start_indices[i]
            attention_layer_idx = pre_start_idx + block_idx
            attention_layer_name_pre = f'attention_{attention_layer_idx}'
            if attention_layer_idx == 0:
                attention_layer_name_pre = 'attention'  # Special case for the first attention layer

            # Attempt to retrieve the pretrained attention layer
            try:
                pretrained_attention_layer = pretrained_encoders[i].get_layer(attention_layer_name_pre)
            except ValueError:
                print(f"[Error] Pretrained attention layer '{attention_layer_name_pre}' not found in encoder {i}.")
                continue  # Skip to the next iteration if the pretrained layer is not found

            # Retrieve weights from both layers
            pretrained_weights = pretrained_attention_layer.get_weights()
            new_weights = new_attention_layer.get_weights()

            # Verify that both layers have the same number of weights
            if len(pretrained_weights) != len(new_weights):
                print(f"[Mismatch] Number of weights mismatch for layer '{attention_layer_name_pre}' and '{attention_layer_name}'.")
                print(f"  Pretrained layer has {len(pretrained_weights)} weights; New layer has {len(new_weights)} weights.")
                continue  # Skip if there's a mismatch in the number of weights

            # Check if each corresponding weight has the same shape
            shape_mismatch = False
            for w_idx, (pw, nw) in enumerate(zip(pretrained_weights, new_weights)):
                if pw.shape != nw.shape:
                    print(f"[Mismatch] Weight shape mismatch for layer '{attention_layer_name_pre}' and '{attention_layer_name}', weight index {w_idx}:")
                    print(f"  Pretrained weight shape: {pw.shape}")
                    print(f"  New weight shape: {nw.shape}")
                    shape_mismatch = True
            if shape_mismatch:
                print(f"[Skipped] Skipping weight transfer for layer '{attention_layer_name}' due to shape mismatch.")
                continue  # Skip if any weight shape doesn't match

            # Transfer weights from the pretrained layer to the new layer
            new_attention_layer.set_weights(pretrained_weights)

            # Confirm that weights have been loaded
            print(f"[Success] Weights loaded for layer '{attention_layer_name}' from pretrained layer '{attention_layer_name_pre}'.")

            # Print detailed information about the weights
            print(f"  [Layer Information] '{attention_layer_name}':")
            print(f"    - Number of weights: {len(pretrained_weights)}")
            for idx, weight in enumerate(pretrained_weights):
                print(f"    - Weight {idx + 1}: shape {weight.shape}")

            # Print information about sublayers within the attention layer
            print(f"  [Sublayer Information] Sublayers within '{attention_layer_name}':")
            # Iterate through all attributes of the attention layer to find sublayers
            for attr_name in dir(new_attention_layer):
                # Skip private and non-layer attributes
                if attr_name.startswith('_'):
                    continue
                sublayer = getattr(new_attention_layer, attr_name)
                if isinstance(sublayer, tf.keras.layers.Layer):
                    print(f"    - Sublayer '{sublayer.name}':")
                    # Iterate through weights of the sublayer
                    for w in sublayer.weights:
                        print(f"        * Weight '{w.name}': shape {w.shape}")
            
            print("-" * 80)  # Separator for readability

    # --- Print Output Shapes ---
    if isinstance(outputs, dict):
        for output_name, output_tensor in outputs.items():
            print(f"[Output] '{output_name}' output shape: {output_tensor.shape}")
    else:
        print(f"[Output] Model output shape: {outputs.shape}")
    # ---

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

import tensorflow.keras.backend as K

# ------------------------
# Loss Functions
# ------------------------


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



# ------------------------
# Helper Functions
# ------------------------
def dice_metric_3d(y_true, y_pred, class_labels, smooth=1e-6):
    """
    Computes the Dice coefficient for a composite class in 3D volumes.
    
    Args:
        y_true (numpy.ndarray): Ground truth mask of shape (H, W, D).
        y_pred (numpy.ndarray): Predicted mask of same shape as y_true.
        class_labels (list or set): List of class labels to include in the composite class.
        smooth (float): Smoothing factor to avoid division by zero.
    
    Returns:
        float: Dice coefficient for the composite class.
    """
    # Create binary masks for the composite class
    y_true_f = np.isin(y_true, class_labels).astype(np.float32)
    y_pred_f = np.isin(y_pred, class_labels).astype(np.float32)
    
    intersection = np.sum(y_true_f * y_pred_f)
    sum_masks = np.sum(y_true_f) + np.sum(y_pred_f)
    
    if sum_masks == 0:
        # If both y_true and y_pred are empty for this composite class, define Dice as 1
        return 1.0
    else:
        dice = (2. * intersection + smooth) / (sum_masks + smooth)
        return dice

def crop_center(img, cropx, cropy):
    # Define your crop_center function here if not already defined
    y, x = img.shape
    startx = x//2 - (cropx//2)
    starty = y//2 - (cropy//2)    
    return img[starty:starty+cropy, startx:startx+cropx]

def crop_center_in_xy(volume_3d, cropx, cropy):
    """
    Crops the center (cropx, cropy) in the x- and y-dimensions
    for every slice in a (H, W, D) volume, returning a new 3D volume 
    of shape (cropy, cropx, D).
    
    Args:
        volume_3d (numpy.ndarray): A 3D array of shape (H, W, D).
        cropx (int): The target width of each cropped slice.
        cropy (int): The target height of each cropped slice.
        
    Returns:
        numpy.ndarray: Cropped volume of shape (cropy, cropx, D).
    """
    H, W, D = volume_3d.shape
    startx = W // 2 - (cropx // 2)
    starty = H // 2 - (cropy // 2)

    cropped_slices = []
    for z in range(D):
        slice_2d = volume_3d[..., z]  # shape (H, W)
        # Crop the slice along y (rows) and x (columns):
        cropped_2d = slice_2d[starty : starty + cropy, startx : startx + cropx]
        cropped_slices.append(cropped_2d)

    # Stack them back into a 3D volume along the last dimension
    cropped_volume_3d = np.stack(cropped_slices, axis=-1)  # shape (cropy, cropx, D)
    return cropped_volume_3d



def uncrop_center(cropped_img, original_hw, pad_value=0):
    """
    Uncrop a 2D or 3D 'cropped_img' back to 'original_hw' (H, W),
    preserving channels if present.
    """
    cropped_shape = cropped_img.shape
    
    # If cropped_img is 2D => shape (h, w)
    # If cropped_img is 3D => shape (h, w, c)
    channels = None
    if len(cropped_shape) == 3:
        h, w, channels = cropped_shape
    else:
        h, w = cropped_shape
    
    oh, ow = original_hw  # (240, 240)
    
    # Build a result shape of (oh, ow, c) if channels, else (oh, ow)
    if channels:
        result = np.full((oh, ow, channels), pad_value, dtype=cropped_img.dtype)
    else:
        result = np.full((oh, ow), pad_value, dtype=cropped_img.dtype)
    
    start_y = (oh - h) // 2
    start_x = (ow - w) // 2
    
    if channels:
        result[start_y:start_y+h, start_x:start_x+w, :] = cropped_img
    else:
        result[start_y:start_y+h, start_x:start_x+w] = cropped_img
    
    return result

def uncrop_center_3d(cropped_img, original_shape_3d, pad_value=0):
    """
    Uncrops (pads) a 3D array (H, W, C) back to original_shape_3d (OH, OW, OC),
    centering the cropped_img. Useful for going from (224,224,4) to (240,240,4).

    Args:
        cropped_img (np.ndarray): shape (h, w, c)
        original_shape_3d (tuple): (OH, OW, OC), e.g. (240, 240, 4)
        pad_value (int or float): fill value for the padded area

    Returns:
        np.ndarray: shape (OH, OW, OC) with cropped_img centered.
    """
    # Extract shapes
    h, w, c = cropped_img.shape
    oh, ow, oc = original_shape_3d  # e.g. (240, 240, 4)

    if c != oc:
        raise ValueError(f"Channel mismatch: cropped has {c} channels, but original_shape_3d has {oc} channels.")

    # Create an empty result
    result = np.full((oh, ow, oc), pad_value, dtype=cropped_img.dtype)

    # Compute start indices to center the cropped area
    start_y = (oh - h) // 2
    start_x = (ow - w) // 2

    # Place cropped_img in the center of result
    result[start_y:start_y + h, start_x:start_x + w, :] = cropped_img
    return result


def merge_binary_slices(prediction):
    merged_prediction = np.argmax(prediction, axis=-1).astype(np.uint8)
    return merged_prediction

def dice_coefficient(y_true, y_pred, smooth=1):
    """
    Calculates the Dice Coefficient.
    
    Args:
        y_true (tensor): Ground truth one-hot encoded labels.
        y_pred (tensor): Predicted probabilities for each class.
        smooth (float): Smoothing factor to prevent division by zero.
    
    Returns:
        dice (tensor): Dice coefficient for each class.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

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


def normalize_volume(img):
    """
    Normalizes the image volume to have zero mean and unit variance.
    Adjust as per your normalization strategy.

    Args:
        img (numpy.ndarray): Image volume.

    Returns:
        numpy.ndarray: Normalized image volume.
    """
    img = img.astype(np.float32)
    mean = np.mean(img)
    std = np.std(img)
    if std < 1e-8:
        std = 1e-8  # Prevent division by zero
    normalized = (img - mean) / std
    return normalized

    
def histogram_matching_func(modality_data, reference_modality='t1c'):
    """
    Matches the histogram of all modalities to the reference modality.
    
    Args:
        modality_data (dict): Dictionary containing modality images.
        reference_modality (str): The modality to match histograms to.
    
    Returns:
        dict: Histogram-matched modality data.
    """
    reference = modality_data[reference_modality]
    matched_data = {}
    for modality, data in modality_data.items():
        if modality == reference_modality:
            matched_data[modality] = data
        else:
            matched = match_histograms(data, reference, multichannel=False)
            matched_data[modality] = matched
            print(f"  Histogram matched {modality} to {reference_modality}.")
    return matched_data


# ------------------------
# Model Definitions
# ------------------------

from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, MaxPooling3D, Dropout, UpSampling3D, Concatenate, Activation, add, Multiply
from tensorflow.keras.models import Model

def attention_block(x, g, inter_channels):
    """
    Attention block for 3D U-Net.

    Args:
        x (tensor): Skip-connection feature map from the encoder path 
                    (already the same resolution as 'g').
        g (tensor): Decoder feature map (the gating signal).
        inter_channels (int): Number of intermediate channels.

    Returns:
        tensor: Attention-weighted output, same shape as x and g.
    """
    # Instead of kernel_size=2, strides=2, we keep the same resolution:
    theta_x = Conv3D(inter_channels, kernel_size=1, strides=1, padding='same')(x)
    phi_g = Conv3D(inter_channels, kernel_size=1, strides=1, padding='same')(g)

    f = Activation('relu')(Add()([theta_x, phi_g]))  # Combine, shape is (D, H, W, inter_channels)
    psi_f = Conv3D(1, kernel_size=1, padding='same')(f)
    rate = Activation('sigmoid')(psi_f)

    # No further UpSampling, since we didn't downsample x.
    att_x = Multiply()([x, rate])  # Element-wise multiply to apply attention
    return att_x

def create_3d_unet(input_shape=(240, 240, 155, 5), num_classes=4):
    """
    Creates a 3D U-Net model with attention mechanisms.

    Args:
        input_shape (tuple): Shape of the input tensor (D, H, W, C).
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: 3D U-Net model with attention blocks.
    """
    inputs = Input(shape=input_shape, name='3d_input')  # (D, H, W, C)

    # ---- Encoder Path ----
    c1 = Conv3D(32, (3,3,3), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv3D(32, (3,3,3), activation='relu', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling3D((2,2,2))(c1)
    p1 = Dropout(0.1)(p1)

    c2 = Conv3D(64, (3,3,3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv3D(64, (3,3,3), activation='relu', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling3D((2,2,2))(c2)
    p2 = Dropout(0.1)(p2)

    # ---- Bottleneck ----
    c3 = Conv3D(128, (3,3,3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv3D(128, (3,3,3), activation='relu', padding='same')(c3)
    c3 = BatchNormalization()(c3)

    # ---- Decoder Path with Attention ----
    # Up-sample from c3 => same scale as c2
    u4 = UpSampling3D((2,2,2))(c3)
    attn_1 = attention_block(c2, u4, inter_channels=64)
    u4 = Concatenate()([u4, attn_1])
    c4 = Conv3D(64, (3,3,3), activation='relu', padding='same')(u4)
    c4 = BatchNormalization()(c4)
    c4 = Conv3D(64, (3,3,3), activation='relu', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.1)(c4)

    # Up-sample => same scale as c1
    u5 = UpSampling3D((2,2,2))(c4)
    attn_2 = attention_block(c1, u5, inter_channels=32)
    u5 = Concatenate()([u5, attn_2])
    c5 = Conv3D(32, (3,3,3), activation='relu', padding='same')(u5)
    c5 = BatchNormalization()(c5)
    c5 = Conv3D(32, (3,3,3), activation='relu', padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.1)(c5)

    # ---- Output Layer ----
    outputs = Conv3D(num_classes, (1,1,1), activation='softmax', name='final')(c5)

    model = Model(inputs=[inputs], outputs=[outputs], name='3D_U-Net')
    return model


# ------------------------
# Processing Functions
# ------------------------

import logging
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral
from scipy.ndimage import binary_closing, label, generate_binary_structure


def post_process_volume_with_3d_closing(volume, close_iterations=1, remove_small_lesions=True, min_voxel_size=50, structuring_elements=None):
    """
    Applies 3D morphological closing followed by small region removal for post-processing,
    using class-specific structuring elements.

    Args:
        volume (np.ndarray): Segmentation volume (H, W, D), containing labels {0,1,2,3}.
        close_iterations (int): Number of iterations for closing operation.
        remove_small_lesions (bool): Whether to remove small connected components post-closing.
        min_voxel_size (int): Minimum voxel count to retain a connected component.
        structuring_elements (dict): Dictionary mapping class labels to their structuring elements.
                                    e.g., {1: struct_3d_tc, 2: struct_3d_wt}

    Returns:
        np.ndarray: Post-processed volume with morphological closing and small region removal applied.
    """
    if structuring_elements is None:
        # Default structuring element if none provided
        structuring_elements = {
            1: np.ones((3, 3, 3), dtype=bool),  # TC
            2: np.ones((3, 3, 3), dtype=bool)   # WT (adjusted structuring element size)
        }

    final_volume = np.zeros_like(volume, dtype=np.uint8)

    for class_label, struct_3d in structuring_elements.items():
        # Ensure structuring element is 3D
        if struct_3d.ndim != 3:
            raise ValueError(f"Structuring element for class {class_label} must be 3D, got {struct_3d.ndim}D.")

        # Extract the class mask and ensure it's boolean
        class_mask = (volume == class_label).astype(bool)
        print(f"\nProcessing class {class_label} with structuring element shape: {struct_3d.shape}")
        print(f"  Class {class_label} mask shape: {class_mask.shape}")

        # Add padding to the class mask to handle border effects
        pad_width = [(s // 2, s // 2) for s in struct_3d.shape]  # Assuming cubic structuring element
        padded_class_mask = np.pad(class_mask, pad_width=pad_width, mode='constant', constant_values=0)
        print(f"  Padded class mask shape: {padded_class_mask.shape}")

        # 1) 3D Morphological Closing
        closed_mask = binary_closing(
            padded_class_mask, 
            structure=struct_3d, 
            iterations=close_iterations
        )
        print(f"  Closed mask shape: {closed_mask.shape}")

        # Remove padding to return to original volume size
        closed_mask = closed_mask[
            pad_width[0][0]:-pad_width[0][1], 
            pad_width[1][0]:-pad_width[1][1], 
            pad_width[2][0]:-pad_width[2][1]
        ]
        print(f"  Closed mask after removing padding: {closed_mask.shape}")

        # 2) Optionally Remove Small Components
        if remove_small_lesions:
            labeled_mask, num_features = label(closed_mask, structure=struct_3d)
            print(f"  Labeled mask shape: {labeled_mask.shape}, Number of features: {num_features}")
            cleaned_mask = np.zeros_like(closed_mask, dtype=bool)

            for feat_id in range(1, num_features + 1):
                feat_voxels = (labeled_mask == feat_id)
                size = np.sum(feat_voxels)

                if size >= min_voxel_size:
                    cleaned_mask = cleaned_mask | feat_voxels  # Retain this component
                    print(f"    Retained feature {feat_id} with size {size}")
                else:
                    print(f"    Discarded feature {feat_id} with size {size}")
        else:
            cleaned_mask = closed_mask
            print("  Small lesion removal skipped.")

        # 3) Combine with Final Volume
        final_volume[cleaned_mask] = class_label
        print(f"  Combined class {class_label} into final volume.")

    # Preserve the ET class (class 3) by copying it directly
    final_volume[volume == 3] = 3
    print("  Preserved ET class (class 3) in final volume.")

    return final_volume

# ------------------------
# Training the 3D U-Net
# ------------------------
import os
import numpy as np
import nibabel as nib
from tensorflow.keras.models import Model
from tqdm import tqdm  # For progress bars

def is_slice_empty(modality_slices):
    return all(np.sum(slice_img) == 0 for slice_img in modality_slices)

# def generate_3d_training_data(train_data, model, predictions_dir, original_shape=(240, 240, 155), input_size=(224, 224), batch_size=16):
#     """
#     Generates training data for the 3D U-Net by producing 3D predictions from the 2D model.
#     Skips 2D predictions for slices empty across all modalities and handles them appropriately.

#     Args:
#         train_data (list of dict): List where each dict contains paths for modalities and mask.
#         model (tf.keras.Model): Trained 2D segmentation model (outputs only 'final').
#         predictions_dir (str): Directory to save or load slice predictions.
#         original_shape (tuple): Shape of the original volume (H, W, D).
#         input_size (tuple): Size to which each slice is cropped for model input (H, W).
#         batch_size (int): Number of slices to process in a batch.

#     Returns:
#         Tuple of numpy.ndarrays: (input_volumes, prediction_volumes, ground_truth_volumes)
#     """
#     input_volumes = []
#     prediction_volumes = []
#     ground_truth_volumes = []

#     os.makedirs(predictions_dir, exist_ok=True)

#     logging.info("=== Generating 3D Training Data ===")

#     # Phase 1: Generate and save predictions
#     for idx, sample_paths in enumerate(tqdm(train_data, desc="Processing Samples")):
#         sample_id = os.path.basename(sample_paths.get('mask', 'sample_unknown')).split('.')[0]
#         prediction_file = os.path.join(predictions_dir, f"{sample_id}.npy")

#         # Check if prediction already exists
#         if os.path.exists(prediction_file):
#             logging.info(f"Loading saved predictions for sample '{sample_id}' from '{prediction_file}'.")
#             try:
#                 probability_volume = np.load(prediction_file)  # Shape: (D, H, W, 4)
#                 logging.info(f"Loaded prediction volume shape: {probability_volume.shape}")
#                 continue  # Skip prediction generation if already exists
#             except Exception as e:
#                 logging.error(f"Error loading predictions for sample '{sample_id}': {e}. Recomputing predictions.")
#                 probability_volume = None
#         else:
#             probability_volume = None

#         if probability_volume is None:
#             # Proceed to compute predictions
#             logging.info(f"Processing training sample {idx + 1}/{len(train_data)}: '{sample_id}'")
#             modalities = ['t1c', 't1n', 't2f', 't2w']
#             original_slices = []
#             skip_sample = False

#             # Load and preprocess modalities
#             for modality in modalities:
#                 path = sample_paths.get(modality)
#                 if not path:
#                     logging.error(f"Missing modality '{modality}'. Skipping sample '{sample_id}'.")
#                     skip_sample = True
#                     break
#                 try:
#                     img = nib.load(path).get_fdata()
#                     if img.ndim != 3:
#                         logging.error(f"Modality '{modality}' does not have 3 dimensions. Skipping sample '{sample_id}'.")
#                         skip_sample = True
#                         break
#                     original_slices.append(img)
#                     logging.info(f"Loaded '{modality}' from '{path}'.")
#                 except Exception as e:
#                     logging.error(f"Error loading '{modality}' from '{path}': {e}. Skipping sample '{sample_id}'.")
#                     skip_sample = True
#                     break

#             if skip_sample:
#                 continue

#             # Normalize slices
#             for modality_idx, modality in enumerate(modalities):
#                 original_slices[modality_idx] = normalize_volume(original_slices[modality_idx])
#                 logging.info(f"Normalized '{modality}'.")

#             num_slices = original_slices[0].shape[-1]
#             logging.info(f"Number of slices: {num_slices}")

#             # Initialize arrays to store per-slice predictions and image slices
#             probability_slices = []
#             image_slices = []

#             # Batch processing
#             for batch_start in range(0, num_slices, batch_size):
#                 batch_end = min(batch_start + batch_size, num_slices)
#                 current_batch_size = batch_end - batch_start
#                 batch_model_inputs = {}
#                 batch_images = []
#                 valid_slices = []  # To keep track of which slices are valid

#                 for slice_idx in range(batch_start, batch_end):
#                     model_inputs_slice = {}
#                     slice_images = []

#                     # Extract slices for all modalities
#                     modality_slices = [original_slices[m_idx][:, :, slice_idx] for m_idx in range(len(modalities))]

#                     # Check if slice is empty across all modalities
#                     if is_slice_empty(modality_slices):
#                         # Append default values
#                         probability_slices.append(np.zeros((original_shape[0], original_shape[1], 4)))
#                         image_slices.append(np.zeros((original_shape[0], original_shape[1], len(modalities))))
#                         continue

#                     skip_slice = False
#                     for modality_idx, modality in enumerate(modalities):
#                         slice_img = original_slices[modality_idx][:, :, slice_idx]
#                         # Preprocess: Crop
#                         cropped = crop_center(slice_img, input_size[0], input_size[1])

#                         # Check if the cropped slice is empty
#                         if not np.any(cropped):
#                             skip_slice = True
#                             logging.warning(f"Slice {slice_idx + 1}: '{modality}' is empty after cropping. Skipping this slice.")
#                             break

#                         # Normalize
#                         min_val = np.min(cropped)
#                         max_val = np.max(cropped)
#                         if max_val - min_val > 1e-8:
#                             norm_slice = (cropped - min_val) / (max_val - min_val + 1e-8)
#                         else:
#                             norm_slice = np.zeros_like(cropped)
#                             logging.warning(f"Slice {slice_idx + 1}: '{modality}' has constant intensity. Normalized to zeros.")

#                         # Expand dimensions
#                         input_slice = np.expand_dims(np.expand_dims(norm_slice, axis=0), axis=-1)  # Shape: (1, H, W, 1)
#                         model_inputs_slice[f"{modality}_input"] = input_slice
#                         slice_images.append(cropped)

#                     if skip_slice:
#                         # Append default values for skipped slice
#                         probability_slices.append(np.zeros((original_shape[0], original_shape[1], 4)))
#                         image_slices.append(np.zeros((original_shape[0], original_shape[1], len(modalities))))
#                         continue

#                     # Collect batch inputs
#                     for key, value in model_inputs_slice.items():
#                         if key not in batch_model_inputs:
#                             batch_model_inputs[key] = []
#                         batch_model_inputs[key].append(value)
#                     batch_images.append(slice_images)
#                     valid_slices.append(slice_idx)

#                 # Convert lists to numpy arrays for batch prediction
#                 if not batch_model_inputs:
#                     logging.info(f"Batch {batch_start + 1}-{batch_end}: No valid slices to predict.")
#                     continue

#                 for key in batch_model_inputs:
#                     batch_model_inputs[key] = np.vstack(batch_model_inputs[key])  # Shape: (num_valid_slices, H, W, 1)

#                 # Predict batch using the 'final' output
#                 try:
#                     # 'model' is the final_model with only 'final' output
#                     batch_probabilities = model.predict(batch_model_inputs)  # Shape: (num_valid_slices, 224, 224, 4)
#                     logging.info(f"Batch {batch_start + 1}-{batch_end}: 2D Model batch prediction successful.")
#                 except Exception as e:
#                     logging.error(f"Batch {batch_start + 1}-{batch_end}: 2D Model batch prediction failed with error: {e}. Using zeros.")
#                     num_valid_slices = batch_model_inputs[list(batch_model_inputs.keys())[0]].shape[0]
#                     batch_probabilities = np.zeros((num_valid_slices, original_shape[0], original_shape[1], 4))

#                 # Assign batch predictions to corresponding slices
#                 for prob in batch_probabilities:
#                     probabilities_resized = np.zeros((original_shape[0], original_shape[1], 4))
#                     for c in range(4):
#                         probabilities_resized[..., c] = uncrop_center(prob[..., c], original_shape[:2])
#                     probability_slices.append(probabilities_resized)
#                     # Find the corresponding image slice
#                     # Assuming batch_images are in the same order
#                     image_slice = np.stack(batch_images.pop(0), axis=-1)  # Shape: (240, 240, 4)
#                     image_slices.append(image_slice)

#             # Stack into 3D volumes
#             if len(probability_slices) == 0:
#                 logging.warning(f"No valid slices found for sample '{sample_id}'. Skipping.")
#                 continue

#             probability_volume = np.stack(probability_slices, axis=0)  # Shape: (D, H, W, 4)
#             prediction_volume = np.argmax(probability_volume, axis=-1).astype(np.uint8)  # Shape: (D, H, W)

#             # Save predictions to disk
#             try:
#                 np.save(prediction_file, probability_volume)
#                 logging.info(f"Saved prediction volume to '{prediction_file}'.")
#             except Exception as e:
#                 logging.error(f"Error saving predictions for sample '{sample_id}': {e}.")
#                 # Proceeding without saving is acceptable; however, ensure to append data below

#     # Phase 2: Append loaded predictions
#     logging.info("=== Appending Loaded Predictions ===")
#     for idx, sample_paths in enumerate(tqdm(train_data, desc="Appending Loaded Predictions")):
#         sample_id = os.path.basename(sample_paths.get('mask', 'sample_unknown')).split('.')[0]
#         prediction_file = os.path.join(predictions_dir, f"{sample_id}.npy")

#         if not os.path.exists(prediction_file):
#             logging.warning(f"Prediction file '{prediction_file}' does not exist. Skipping sample '{sample_id}'.")
#             continue

#         try:
#             probability_volume = np.load(prediction_file)  # Shape: (D, H, W, 4)
#             logging.info(f"Loading prediction for sample '{sample_id}' from '{prediction_file}'.")
#         except Exception as e:
#             logging.error(f"Error loading prediction for sample '{sample_id}': {e}. Skipping.")
#             continue

#         # Load and preprocess input modalities
#         modalities = ['t1c', 't1n', 't2f', 't2w']
#         input_slices = []
#         skip_sample = False
#         for modality in modalities:
#             path = sample_paths.get(modality)
#             if not path:
#                 logging.error(f"Missing modality '{modality}' for sample '{sample_id}'. Skipping.")
#                 skip_sample = True
#                 break
#             try:
#                 img = nib.load(path).get_fdata()
#                 if img.ndim != 3:
#                     logging.error(f"Modality '{modality}' does not have 3 dimensions for sample '{sample_id}'. Skipping.")
#                     skip_sample = True
#                     break
#                 img = normalize_volume(img)
#                 # Crop or resize as needed to match `input_shape`
#                 cropped = crop_center_in_xy(img, input_size[0], input_size[1])
#                 input_slices.append(cropped)
#             except Exception as e:
#                 logging.error(f"Error loading modality '{modality}' for sample '{sample_id}': {e}. Skipping.")
#                 skip_sample = True
#                 break

#         if skip_sample:
#             continue

#         input_volume = np.stack(input_slices, axis=-1)  # Shape: (D, H, W, 4)

#         # Load ground truth mask
#         ground_truth_path = sample_paths.get('mask')
#         if not ground_truth_path:
#             logging.error(f"Missing ground truth mask for sample '{sample_id}'. Skipping.")
#             continue
#         try:
#             ground_truth = nib.load(ground_truth_path).get_fdata().astype(np.uint8)
#             if ground_truth.ndim != 3:
#                 logging.error(f"Ground truth mask for sample '{sample_id}' does not have 3 dimensions. Skipping.")
#                 continue
#             ground_truth = np.transpose(ground_truth, (2, 0, 1))  # Assuming (H, W, D) to (D, H, W)
#             logging.info(f"Loaded ground truth mask from '{ground_truth_path}'.")
#         except Exception as e:
#             logging.error(f"Error loading ground truth mask from '{ground_truth_path}': {e}. Skipping.")
#             continue

#         # Append to the lists
#         input_volumes.append(input_volume)  # Shape: (D, H, W, 4)
#         prediction_volumes.append(np.argmax(probability_volume, axis=-1).astype(np.uint8))  # Shape: (D, H, W)
#         ground_truth_volumes.append(ground_truth)  # Shape: (D, H, W)

#     # Convert lists to numpy arrays
#     input_volumes = np.array(input_volumes)  # Shape: (N, D, H, W, 4)
#     prediction_volumes = np.array(prediction_volumes)  # Shape: (N, D, H, W)
#     ground_truth_volumes = np.array(ground_truth_volumes)  # Shape: (N, D, H, W)

#     logging.info("=== Data Generation Complete ===")
#     logging.info(f"Total Input Volumes: {input_volumes.shape[0]}")
#     logging.info(f"Total Prediction Volumes: {prediction_volumes.shape[0]}")
#     logging.info(f"Total Ground Truth Volumes: {ground_truth_volumes.shape[0]}")

#     return input_volumes, prediction_volumes, ground_truth_volumes


def generate_3d_training_data_one_sample(train_data,
                                         model,
                                         predictions_dir,
                                         original_shape=(240, 240, 155),
                                         input_size=(224, 224),
                                         batch_size=16):
    """
    A generator that yields (input_volume, prediction_volume, ground_truth_volume)
    for each sample in `train_data`, thereby avoiding storing large arrays
    for all samples in memory.

    Args:
        train_data (list of dict): List where each dict contains paths for modalities and mask.
        model (tf.keras.Model): Trained 2D segmentation model (outputs only 'final').
        predictions_dir (str): Directory to save or load slice predictions.
        original_shape (tuple): Shape of the original volume (H, W, D).
        input_size (tuple): Size to which each slice is cropped for model input (H, W).
        batch_size (int): Number of slices to process in a batch.

    Yields:
        (input_volume, prediction_volume, ground_truth_volume): All are np.ndarray for a single case.
            - input_volume: Shape (D, H, W, num_modalities) (e.g., 4)
            - prediction_volume: Shape (D, H, W)
            - ground_truth_volume: Shape (D, H, W)
    """
    os.makedirs(predictions_dir, exist_ok=True)
    logging.info("=== Generating 3D Training Data (One Sample at a Time) ===")

    for idx, sample_paths in enumerate(tqdm(train_data, desc="Processing Samples")):
        sample_id = os.path.basename(sample_paths.get('mask', 'sample_unknown')).split('.')[0]
        prediction_file = os.path.join(predictions_dir, f"{sample_id}.npy")

        # -----------------------------------------------------
        # PHASE 1: Generate (or load) Probability Volume
        # -----------------------------------------------------
        # Check if prediction file already exists
        if os.path.exists(prediction_file):
            logging.info(f"Loading saved predictions for sample '{sample_id}' from '{prediction_file}'.")
            try:
                probability_volume = np.load(prediction_file)  # shape: (D, H, W, 4)
                logging.info(f"Loaded prediction volume shape: {probability_volume.shape}")
            except Exception as e:
                logging.error(f"Error loading predictions for sample '{sample_id}': {e}. Recomputing predictions.")
                probability_volume = None
        else:
            probability_volume = None

        # If probability_volume not on disk, compute from 2D model
        if probability_volume is None:
            modalities = ['t1c', 't1n', 't2f', 't2w']
            original_slices = []
            skip_sample = False

            # Load & preprocess modalities
            for modality in modalities:
                path = sample_paths.get(modality)
                if not path:
                    logging.error(f"Missing modality '{modality}' for sample '{sample_id}'. Skipping.")
                    skip_sample = True
                    break
                try:
                    img = nib.load(path).get_fdata()
                    if img.ndim != 3:
                        logging.error(f"Modality '{modality}' is not 3D. Skipping sample '{sample_id}'.")
                        skip_sample = True
                        break
                    # Optionally normalize slices
                    img = normalize_volume(img)
                    original_slices.append(img)
                    logging.info(f"Loaded and normalized '{modality}' from '{path}'.")
                except Exception as e:
                    logging.error(f"Error loading '{modality}' from '{path}': {e}. Skipping.")
                    skip_sample = True
                    break

            if skip_sample:
                # Skip yielding this sample
                continue

            num_slices = original_slices[0].shape[-1]
            logging.info(f"[Sample {sample_id}] Number of slices: {num_slices}")

            # Prepare storage for slice-level predictions
            probability_slices = []
            image_slices = []

            # Batch slicing
            for batch_start in range(0, num_slices, batch_size):
                batch_end = min(batch_start + batch_size, num_slices)
                current_batch_size = batch_end - batch_start
                batch_model_inputs = {}
                batch_images = []

                valid_slice_indices = []

                # Build batch
                for slice_idx in range(batch_start, batch_end):
                    # Extract slice across all modalities
                    modality_slices = [original_slices[m_idx][:, :, slice_idx]
                                       for m_idx in range(len(modalities))]

                    # skip empty slices across all modalities
                    if is_slice_empty(modality_slices):
                        # append blank predictions
                        prob_zeros = np.zeros((original_shape[0], original_shape[1], 4))
                        img_zeros = np.zeros((original_shape[0], original_shape[1], len(modalities)))
                        probability_slices.append(prob_zeros)
                        image_slices.append(img_zeros)
                        continue

                    # For each modality, crop & normalize
                    skip_this_slice = False
                    slice_images = []
                    slice_model_inputs = {}
                    for modality_idx, single_slice in enumerate(modality_slices):
                        cropped = crop_center(single_slice, input_size[0], input_size[1])
                        if not np.any(cropped):
                            # skip
                            logging.warning(f"[Sample {sample_id} Slice {slice_idx}] empty after cropping. Skipped.")
                            skip_this_slice = True
                            break

                        # scale intensities [optional: already normalized above?]
                        min_val, max_val = cropped.min(), cropped.max()
                        if max_val - min_val > 1e-8:
                            norm_slice = (cropped - min_val) / (max_val - min_val + 1e-8)
                        else:
                            norm_slice = np.zeros_like(cropped)

                        # expand dims for model input: (1, H, W, 1)
                        norm_slice = np.expand_dims(np.expand_dims(norm_slice, axis=0), axis=-1)
                        slice_model_inputs[f"{modalities[modality_idx]}_input"] = norm_slice
                        slice_images.append(cropped)

                    if skip_this_slice:
                        # appended blank above, so skip
                        continue

                    # Accumulate into batch_model_inputs
                    for key, arr in slice_model_inputs.items():
                        if key not in batch_model_inputs:
                            batch_model_inputs[key] = []
                        batch_model_inputs[key].append(arr)

                    batch_images.append(slice_images)
                    valid_slice_indices.append(slice_idx)

                # If batch_model_inputs is empty, skip
                if not batch_model_inputs:
                    continue

                # Convert to np arrays
                for key in batch_model_inputs:
                    # shape: (num_valid_slices, 224, 224, 1)
                    batch_model_inputs[key] = np.vstack(batch_model_inputs[key])

                # Predict in batch
                try:
                    batch_pred = model.predict(batch_model_inputs)  # shape: (num_valid_slices, 224, 224, 4)
                    logging.info(f"[Sample {sample_id}] 2D model batch {batch_start+1}-{batch_end} predicted.")
                except Exception as e:
                    logging.error(f"Batch prediction failed for sample '{sample_id}' batch {batch_start}-{batch_end}: {e}")
                    batch_pred = np.zeros((len(valid_slice_indices), original_shape[0], original_shape[1], 4))

                # For each predicted slice in this batch
                for i_slice in range(len(valid_slice_indices)):
                    pred_2d = batch_pred[i_slice]  # shape (224,224,4)
                    # upsample to (240,240)
                    prob_240 = np.zeros((original_shape[0], original_shape[1], 4))
                    for c in range(4):
                        prob_240[..., c] = uncrop_center(pred_2d[..., c], original_shape[:2])

                    probability_slices.append(prob_240)

                    # retrieve the corresponding slice_images
                    single_slice_images = batch_images[i_slice]  # list of cropped [224,224], one per modality
                    # upsample each cropped image to 240x240
                    upsampled_slices = []
                    for ch_idx, cimg in enumerate(single_slice_images):
                        up_img_240 = np.zeros(original_shape[:2])  # (240,240)
                        up_img_240 = uncrop_center(cimg, original_shape[:2])
                        upsampled_slices.append(up_img_240)

                    # stack channel dimension => shape (240,240,len(modalities))
                    image_slices.append(np.stack(upsampled_slices, axis=-1))

            # after building all slices
            if len(probability_slices) == 0:
                logging.warning(f"No valid slices found for sample '{sample_id}'. Skipping.")
                continue

            # shape: (D,240,240,4)
            probability_volume = np.stack(probability_slices, axis=0)
            # shape: (D,240,240,len(modalities))
            # e.g. (155,240,240,4)
            _ = np.stack(image_slices, axis=0)  # Not needed for final output but you could keep it

            # Save to disk
            try:
                np.save(prediction_file, probability_volume)
                logging.info(f"[Sample {sample_id}] Probability volume saved to '{prediction_file}'.")
            except Exception as e:
                logging.error(f"Error saving predictions for '{sample_id}': {e}")

        # -----------------------------------------------------
        # PHASE 2: Build final input/prediction/GT for this sample
        # -----------------------------------------------------
        # Load the probability volume from disk
        try:
            probability_volume = np.load(prediction_file)  # shape: (D,240,240,4)
            logging.info(f"[Sample {sample_id}] Probability volume reloaded from disk, shape={probability_volume.shape}.")
        except Exception as e:
            logging.error(f"Error loading final prob volume for '{sample_id}': {e}. Skipping.")
            continue

        # Build input_volume: shape (D,240,240,num_modalities)
        # e.g. re-load the modalities, normalizing them as needed
        # (Alternatively, you could have stored image_slices from above to disk.)
        # For demonstration, let's do it from scratch:
        modalities = ['t1c','t1n','t2f','t2w']
        input_slices = []
        skip_this_case = False
        for modality in modalities:
            path = sample_paths.get(modality)
            if not path:
                logging.error(f"[Sample {sample_id}] Missing '{modality}'. Skipping.")
                skip_this_case = True
                break
            try:
                img = nib.load(path).get_fdata()  # shape: (240,240,D) or (H,W,D)
                # if it's (H,W,D) but your code expects (D,H,W) you might do a transpose
                # but let's assume it's consistent with how we've used it
                # We'll assume shape is (240,240,D). We can stack along the 4th axis
                # but let's do shape => (D,240,240)
                # So do a transpose => (D,H,W)
                # then gather them
                # We'll skip big details; just note we want final shape (D,240,240,num_mod)
                img = np.transpose(img,(2,0,1))  # now shape (D,240,240)
                # optionally normalization
                img = normalize_volume(img)
                input_slices.append(img)  # shape: (D,240,240)
            except Exception as e:
                logging.error(f"[Sample {sample_id}] Error loading '{modality}': {e}")
                skip_this_case = True
                break

        if skip_this_case:
            continue

        # shape => (D,240,240,4) if len(modalities)=4
        input_volume = np.stack(input_slices, axis=-1)

        # Load ground truth
        ground_truth_path = sample_paths.get('mask')
        if not ground_truth_path:
            logging.error(f"[Sample {sample_id}] Missing ground truth. Skipping.")
            continue
        try:
            gt_data = nib.load(ground_truth_path).get_fdata().astype(np.uint8)
            if gt_data.ndim != 3:
                logging.error(f"[Sample {sample_id}] GT is not 3D. Skipping.")
                continue
            # transpose to shape => (D,H,W)
            ground_truth_volume = np.transpose(gt_data,(2,0,1))
        except Exception as e:
            logging.error(f"[Sample {sample_id}] Error loading GT: {e}")
            continue

        # Build prediction_volume => argmax(prob_volume, axis=-1)
        # shape => (D,240,240)
        prediction_volume = np.argmax(probability_volume, axis=-1).astype(np.uint8)

        # Finally, yield these volumes for the calling function
        yield (input_volume, prediction_volume, ground_truth_volume)


def generate_3d_training_data(train_data,
                               model,
                               predictions_dir,
                               original_shape=(240, 240, 155),
                               input_size=(224, 224),
                               batch_size=16):
    """
    A generator that yields batches of (input_volume, prediction_volume, ground_truth_volume)
    for the training data, to be used in model training.

    Args:
        train_data (list of dict): List where each dict contains paths for modalities and mask.
        model (tf.keras.Model): Trained 2D segmentation model (outputs only 'final').
        predictions_dir (str): Directory to save or load slice predictions.
        original_shape (tuple): Shape of the original volume (H, W, D).
        input_size (tuple): Size to which each slice is cropped for model input (H, W).
        batch_size (int): Number of slices to process in a batch.

    Yields:
        (input_volume, prediction_volume, ground_truth_volume): A batch of training data.
            - input_volume: Shape (D, H, W, num_modalities) (e.g., 4)
            - prediction_volume: Shape (D, H, W)
            - ground_truth_volume: Shape (D, H, W)
    """
    os.makedirs(predictions_dir, exist_ok=True)
    logging.info("=== Generating 3D Training Data (Batch Mode) ===")

    batch_input_volumes = []
    batch_prediction_volumes = []
    batch_ground_truth_volumes = []

    for idx, sample_paths in enumerate(tqdm(train_data, desc="Processing Samples")):
        sample_id = os.path.basename(sample_paths.get('mask', 'sample_unknown')).split('.')[0]
        prediction_file = os.path.join(predictions_dir, f"{sample_id}.npy")

        # -----------------------------------------------------
        # PHASE 1: Generate (or load) Probability Volume
        # -----------------------------------------------------
        # Check if prediction file already exists
        if os.path.exists(prediction_file):
            logging.info(f"Loading saved predictions for sample '{sample_id}' from '{prediction_file}'.")
            try:
                probability_volume = np.load(prediction_file)  # shape: (D, H, W, 4)
                logging.info(f"Loaded prediction volume shape: {probability_volume.shape}")
            except Exception as e:
                logging.error(f"Error loading predictions for sample '{sample_id}': {e}. Recomputing predictions.")
                probability_volume = None
        else:
            probability_volume = None

        # If probability_volume not on disk, compute from 2D model
        if probability_volume is None:
            modalities = ['t1c', 't1n', 't2f', 't2w']
            original_slices = []
            skip_sample = False

            # Load & preprocess modalities
            for modality in modalities:
                path = sample_paths.get(modality)
                if not path:
                    logging.error(f"Missing modality '{modality}' for sample '{sample_id}'. Skipping.")
                    skip_sample = True
                    break
                try:
                    img = nib.load(path).get_fdata()
                    if img.ndim != 3:
                        logging.error(f"Modality '{modality}' is not 3D. Skipping sample '{sample_id}'.")
                        skip_sample = True
                        break
                    # Optionally normalize slices
                    img = normalize_volume(img)
                    original_slices.append(img)
                    logging.info(f"Loaded and normalized '{modality}' from '{path}'.")
                except Exception as e:
                    logging.error(f"Error loading '{modality}' from '{path}': {e}. Skipping.")
                    skip_sample = True
                    break

            if skip_sample:
                # Skip yielding this sample
                continue

            num_slices = original_slices[0].shape[-1]
            logging.info(f"[Sample {sample_id}] Number of slices: {num_slices}")

            # Prepare storage for slice-level predictions
            probability_slices = []
            image_slices = []

            # Batch slicing
            for batch_start in range(0, num_slices, batch_size):
                batch_end = min(batch_start + batch_size, num_slices)
                current_batch_size = batch_end - batch_start
                batch_model_inputs = {}
                batch_images = []

                valid_slice_indices = []

                # Build batch
                for slice_idx in range(batch_start, batch_end):
                    # Extract slice across all modalities
                    modality_slices = [original_slices[m_idx][:, :, slice_idx]
                                       for m_idx in range(len(modalities))]

                    # skip empty slices across all modalities
                    if is_slice_empty(modality_slices):
                        # append blank predictions
                        prob_zeros = np.zeros((original_shape[0], original_shape[1], 4))
                        img_zeros = np.zeros((original_shape[0], original_shape[1], len(modalities)))
                        probability_slices.append(prob_zeros)
                        image_slices.append(img_zeros)
                        continue

                    # For each modality, crop & normalize
                    skip_this_slice = False
                    slice_images = []
                    slice_model_inputs = {}
                    for modality_idx, single_slice in enumerate(modality_slices):
                        cropped = crop_center(single_slice, input_size[0], input_size[1])
                        if not np.any(cropped):
                            # skip
                            logging.warning(f"[Sample {sample_id} Slice {slice_idx}] empty after cropping. Skipped.")
                            skip_this_slice = True
                            break

                        # scale intensities [optional: already normalized above?]
                        min_val, max_val = cropped.min(), cropped.max()
                        if max_val - min_val > 1e-8:
                            norm_slice = (cropped - min_val) / (max_val - min_val + 1e-8)
                        else:
                            norm_slice = np.zeros_like(cropped)

                        # expand dims for model input: (1, H, W, 1)
                        norm_slice = np.expand_dims(np.expand_dims(norm_slice, axis=0), axis=-1)
                        slice_model_inputs[f"{modalities[modality_idx]}_input"] = norm_slice
                        slice_images.append(cropped)

                    if skip_this_slice:
                        # appended blank above, so skip
                        continue

                    # Accumulate into batch_model_inputs
                    for key, arr in slice_model_inputs.items():
                        if key not in batch_model_inputs:
                            batch_model_inputs[key] = []
                        batch_model_inputs[key].append(arr)

                    batch_images.append(slice_images)
                    valid_slice_indices.append(slice_idx)

                # If batch_model_inputs is empty, skip
                if not batch_model_inputs:
                    continue

                # Convert to np arrays
                for key in batch_model_inputs:
                    # shape: (num_valid_slices, 224, 224, 1)
                    batch_model_inputs[key] = np.vstack(batch_model_inputs[key])

                # Predict in batch
                try:
                    batch_pred = model.predict(batch_model_inputs)  # shape: (num_valid_slices, 224, 224, 4)
                    logging.info(f"[Sample {sample_id}] 2D model batch {batch_start+1}-{batch_end} predicted.")
                except Exception as e:
                    logging.error(f"Batch prediction failed for sample '{sample_id}' batch {batch_start}-{batch_end}: {e}")
                    batch_pred = np.zeros((len(valid_slice_indices), original_shape[0], original_shape[1], 4))

                # For each predicted slice in this batch
                for i_slice in range(len(valid_slice_indices)):
                    pred_2d = batch_pred[i_slice]  # shape (224,224,4)
                    # upsample to (240,240)
                    prob_240 = np.zeros((original_shape[0], original_shape[1], 4))
                    for c in range(4):
                        prob_240[..., c] = uncrop_center(pred_2d[..., c], original_shape[:2])

                    probability_slices.append(prob_240)

                    # retrieve the corresponding slice_images
                    single_slice_images = batch_images[i_slice]  # list of cropped [224,224], one per modality
                    # upsample each cropped image to 240x240
                    upsampled_slices = []
                    for ch_idx, cimg in enumerate(single_slice_images):
                        up_img_240 = np.zeros(original_shape[:2])  # (240,240)
                        up_img_240 = uncrop_center(cimg, original_shape[:2])
                        upsampled_slices.append(up_img_240)

                    # stack channel dimension => shape (240,240,len(modalities))
                    image_slices.append(np.stack(upsampled_slices, axis=-1))

            # after building all slices
            if len(probability_slices) == 0:
                logging.warning(f"No valid slices found for sample '{sample_id}'. Skipping.")
                continue

            # shape: (D,240,240,4)
            probability_volume = np.stack(probability_slices, axis=0)
            # shape: (D,240,240,len(modalities))
            # e.g. (155,240,240,4)
            _ = np.stack(image_slices, axis=0)  # Not needed for final output but you could keep it

            # Save to disk
            try:
                np.save(prediction_file, probability_volume)
                logging.info(f"[Sample {sample_id}] Probability volume saved to '{prediction_file}'.")
            except Exception as e:
                logging.error(f"Error saving predictions for '{sample_id}': {e}")

        # -----------------------------------------------------
        # PHASE 2: Build final input/prediction/GT for this sample
        # -----------------------------------------------------
        # Load the probability volume from disk
        try:
            probability_volume = np.load(prediction_file)  # shape: (D,240,240,4)
            logging.info(f"[Sample {sample_id}] Probability volume reloaded from disk, shape={probability_volume.shape}.")
        except Exception as e:
            logging.error(f"Error loading final prob volume for '{sample_id}': {e}. Skipping.")
            continue

        # Build input_volume: shape (D,240,240,num_modalities)
        modalities = ['t1c','t1n','t2f','t2w']
        input_slices = []
        skip_this_case = False
        for modality in modalities:
            path = sample_paths.get(modality)
            if not path:
                logging.error(f"[Sample {sample_id}] Missing '{modality}'. Skipping.")
                skip_this_case = True
                break
            try:
                img = nib.load(path).get_fdata()  # shape: (240,240,D) or (H,W,D)
                img = np.transpose(img,(2,0,1))  # now shape (D,240,240)
                img = normalize_volume(img)
                input_slices.append(img)  # shape: (D,240,240)
            except Exception as e:
                logging.error(f"[Sample {sample_id}] Error loading '{modality}': {e}")
                skip_this_case = True
                break

        if skip_this_case:
            continue

        # shape => (D,240,240,4) if len(modalities)=4
        input_volume = np.stack(input_slices, axis=-1)

        # Load ground truth
        ground_truth_path = sample_paths.get('mask')
        if not ground_truth_path:
            logging.error(f"[Sample {sample_id}] Missing ground truth. Skipping.")
            continue
        try:
            gt_data = nib.load(ground_truth_path).get_fdata().astype(np.uint8)
            ground_truth_volume = np.transpose(gt_data,(2,0,1))
        except Exception as e:
            logging.error(f"[Sample {sample_id}] Error loading GT: {e}")
            continue

        # Build prediction_volume => argmax(prob_volume, axis=-1)
        prediction_volume = np.argmax(probability_volume, axis=-1).astype(np.uint8)

        # Yield batch of data
        yield (input_volume, prediction_volume, ground_truth_volume)


# Testing and Post-processing
# ------------------------

def apply_crf_3d(prob_volume, img_volume, max_schan=20, num_iterations=10, compat_gaussian=3, compat_bilateral=4, sdims=(5, 5, 5)):
    """
    Applies 3D Conditional Random Fields (CRF) for segmentation refinement.

    Args:
        prob_volume (numpy.ndarray): Probability volume with shape (D, H, W, C).
        img_volume (numpy.ndarray): Image volume with shape (D, H, W, M).
        max_schan (int, optional): Maximum standard deviation for channel features to prevent extreme values.
        num_iterations (int, optional): Number of CRF inference iterations.
        compat_gaussian (int, optional): Compatibility coefficient for Gaussian pairwise potentials.
        compat_bilateral (int, optional): Compatibility coefficient for bilateral pairwise potentials.
        sdims (tuple, optional): Spatial standard deviations for Gaussian pairwise potentials.

    Returns:
        numpy.ndarray: Refined segmentation prediction with shape (D, H, W).
    """
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

    logging.info("Starting 3D CRF post-processing...")

    try:
        d, h, w, num_classes = prob_volume.shape
        num_modalities = img_volume.shape[-1]

        # Ensure spatial dimensions match
        assert img_volume.shape[:3] == (d, h, w), "Mismatch in spatial dimensions between prob_volume and img_volume"

        # Ensure C-contiguous arrays
        prob_volume = np.ascontiguousarray(prob_volume)
        img_volume = np.ascontiguousarray(img_volume)

        # Flatten and transpose probabilities
        probs = prob_volume.reshape(d * h * w, num_classes).transpose(1, 0).copy(order='C')  # Shape: (C, D*H*W)

        # Create unary potentials
        U = unary_from_softmax(probs)  # Shape: (C, D*H*W)

        # Initialize CRF model
        dcrf_model = dcrf.DenseCRF(d * h * w, num_classes)
        dcrf_model.setUnaryEnergy(U)

        # Create Gaussian pairwise potentials
        feats_gaussian = create_pairwise_gaussian(sdims=sdims, shape=(d, h, w))
        dcrf_model.addPairwiseEnergy(feats_gaussian, compat=compat_gaussian)

        # Prepare appearance features
        schan_values = []
        for modality_idx in range(num_modalities):
            modality_data = img_volume[..., modality_idx]
            intensity_std = np.std(modality_data)
            intensity_std_capped = min(intensity_std, max_schan)
            schan_values.append(intensity_std_capped)

        # Create bilateral pairwise features
        feats_bilateral = create_pairwise_bilateral(
            sdims=sdims,
            schan=tuple(schan_values),
            img=img_volume.astype(np.float32),
            chdim=3  # Channels are in the last dimension
        )
        dcrf_model.addPairwiseEnergy(feats_bilateral, compat=compat_bilateral)

        # Run inference
        logging.info(f"Running CRF inference for {num_iterations} iterations...")
        Q = dcrf_model.inference(num_iterations)
        preds = np.argmax(Q, axis=0).reshape(d, h, w).astype(np.uint8)
        preds = np.ascontiguousarray(preds)  # Ensure C-contiguous

        logging.info("CRF post-processing completed successfully.")
        return preds  # Shape: (D, H, W)

    except Exception as e:
        logging.error("Error during CRF post-processing", exc_info=True)
        return None

def compute_dice_scores(ground_truth, prediction, composite_classes):
    """
    Computes Dice scores for all composite classes.

    Args:
        ground_truth (numpy.ndarray): Ground truth mask (H, W, D).
        prediction (numpy.ndarray): Predicted mask (H, W, D).
        composite_classes (dict): Dictionary mapping class names to label lists.

    Returns:
        dict: Dice scores for each class and the average.
    """
    dice_scores = {}
    for class_name, class_labels in composite_classes.items():
        dice = dice_metric_3d(ground_truth, prediction, class_labels)
        dice_scores[class_name] = dice
    dice_scores['average'] = np.mean(list(dice_scores.values()))
    return dice_scores


def apply_post_processing(prediction, probability_volume, image_volume, ground_truth, thresholds, min_voxel_counts, min_mean_probs, composite_classes):
    """
    Applies post-processing steps including CRF and morphological operations.
    Computes Dice scores after each step.

    Args:
        prediction (numpy.ndarray): Initial 3D prediction (from 3D U-Net).
        probability_volume (numpy.ndarray): Probability volume from 3D U-Net.
        image_volume (numpy.ndarray): Original image volume (D, H, W, Channels).
        ground_truth (numpy.ndarray): Ground truth segmentation labels.
        thresholds (dict): Thresholds for each class label.
        min_voxel_counts (dict): Minimum voxel counts to retain a lesion for each class.
        min_mean_probs (dict): Minimum mean probability to retain a lesion for each class.
        composite_classes (dict): Mapping from class names to label lists for Dice computation.

    Returns:
        Tuple:
            - numpy.ndarray: Post-processed final prediction.
            - dict: Contains predictions and Dice scores at each step.
    """
    results = {}

    # Compute Dice scores for the initial prediction
    dice_scores_initial = compute_dice_scores(ground_truth, prediction, composite_classes)
    results['Initial'] = {'prediction': prediction, 'dice_scores': dice_scores_initial}

    # Apply CRF
    crf_prediction = apply_crf_3d(
        prob_volume=probability_volume, 
        img_volume=normalize_volume(image_volume), 
        max_schan=10, 
        num_iterations=30, 
        compat_gaussian=5, 
        compat_bilateral=6, 
        sdims=(3, 3, 3)
    )

    if crf_prediction is not None:
        print("  Applied CRF post-processing successfully.")
    else:
        print("  CRF post-processing failed. Using raw prediction instead.")
        crf_prediction = prediction

    # Compute Dice scores after CRF
    dice_scores_crf = compute_dice_scores(ground_truth, crf_prediction, composite_classes)
    results['CRF'] = {'prediction': crf_prediction, 'dice_scores': dice_scores_crf}

    # Morphological Gap Filling
    structuring_elements = {
        1: np.ones((3, 3, 3), dtype=bool),  # Class 1
        2: np.ones((3, 3, 3), dtype=bool),  # Class 2
        3: np.ones((3, 3, 3), dtype=bool)   # Class 3
    }

    try:
        prediction_after_gap_filling = post_process_volume_with_3d_closing(
            volume=crf_prediction, 
            close_iterations=1, 
            remove_small_lesions=False, 
            min_voxel_size=50,
            structuring_elements=structuring_elements
        )
        print("  3D morphological gap filling completed successfully.")
    except Exception as e:
        print(f"  Error during morphological gap filling: {e}")
        prediction_after_gap_filling = crf_prediction  # Fallback to CRF prediction

    # Compute Dice scores after gap filling
    dice_scores_gap_filling = compute_dice_scores(ground_truth, prediction_after_gap_filling, composite_classes)
    results['Gap_Filling'] = {'prediction': prediction_after_gap_filling, 'dice_scores': dice_scores_gap_filling}

    # Thresholding and Removing Small False Positives
    structure = generate_binary_structure(3, 2)  # 26-connectivity
    final_prediction = np.zeros_like(prediction_after_gap_filling, dtype=np.uint8)

    for class_label in [1, 2, 3]:
        print(f"\n  Processing class {class_label} for small false positive removal.")
        binary_mask = (prediction_after_gap_filling == class_label).astype(bool)
        labeled_mask, num_features = label(binary_mask, structure=structure)
        print(f"    Found {num_features} connected components for class {class_label}.")

        for lesion_idx in range(1, num_features + 1):
            lesion_mask = (labeled_mask == lesion_idx)
            voxel_count = np.sum(lesion_mask)
            mean_prob = np.mean(probability_volume[..., class_label][lesion_mask])

            if voxel_count >= min_voxel_counts[class_label] and mean_prob >= min_mean_probs[class_label]:
                final_prediction[lesion_mask] = class_label
                print(f"      Retained lesion {lesion_idx}: Voxels={voxel_count}, Mean Prob={mean_prob:.3f}")
            else:
                print(f"      Discarded lesion {lesion_idx}: Voxels={voxel_count}, Mean Prob={mean_prob:.3f}")

    # Preserve the ET class (class 3) by copying it directly if needed
    final_prediction[prediction_after_gap_filling == 3] = 3
    print("  Preserved ET class (class 3) in final volume.")

    # Compute Dice scores after removal
    dice_scores_removal = compute_dice_scores(ground_truth, final_prediction, composite_classes)
    results['Removal'] = {'prediction': final_prediction, 'dice_scores': dice_scores_removal}

    # Store the final prediction and Dice scores
    results['Final'] = {'prediction': final_prediction, 'dice_scores': dice_scores_removal}

    return final_prediction, results


def create_combined_channels_metric(base_metric, class_indices, name, threshold=0.5):
    """
    Creates a custom metric that computes the base_metric (e.g., Dice) over a subset of classes.
    
    Args:
        base_metric (function): The base metric function to apply (e.g., dice_coefficient).
        class_indices (list of int): List of class indices to include in the metric.
        name (str): Name of the metric.
        threshold (float): Threshold for binarizing predictions.
    
    Returns:
        function: A metric function to be used in model.compile.
    """
    def metric_fn(y_true, y_pred):
        # Select the relevant channels for the composite class
        y_true_subset = tf.gather(y_true, class_indices, axis=-1)
        y_pred_subset = tf.gather(y_pred, class_indices, axis=-1)
        
        # Binarize predictions based on the threshold
        y_pred_subset = tf.cast(y_pred_subset > threshold, tf.float32)
        
        # Binarize ground truth (assuming it's already one-hot encoded)
        y_true_subset = tf.cast(y_true_subset > 0, tf.float32)
        
        # Compute the base metric (Dice) over the selected channels
        return base_metric(y_true_subset, y_pred_subset)
    
    # Assign the custom name to the metric function
    metric_fn.__name__ = name
    return metric_fn


# ------------------------
# Aggregate and Save Dice Scores
# ------------------------

def aggregate_and_save_dice_scores(all_dice_scores, output_dir):
    """
    Aggregates Dice scores from all samples, computes per-class averages and statistics
    for each post-processing step, and saves the results to CSV files.

    Args:
        all_dice_scores (list of dict): List where each dict contains Dice scores for a sample.
        output_dir (str): Directory where the CSV files will be saved.
    """
    if not all_dice_scores:
        print("No Dice scores to aggregate.")
        return

    # Convert the list of dictionaries to a DataFrame
    records = []
    for entry in all_dice_scores:
        record = {'Sample_ID': entry.get('Sample_ID', 'Unknown')}
        for step in ['Raw', 'Refinement', 'CRF', 'Gap_Filling', 'Removal', 'Final']:
            step_scores = entry.get(step, {})
            for class_name in ['WT', 'TC', 'ET']:
                key = f"{step}_{class_name}"
                record[key] = step_scores.get(class_name, np.nan)
            # Compute average Dice score for the step
            valid_scores = [score for score in step_scores.values() if not np.isnan(score)]
            if valid_scores:
                average_dice = sum(valid_scores) / len(valid_scores)
            else:
                average_dice = np.nan
            record[f"{step}_average"] = average_dice
        records.append(record)

    dice_df = pd.DataFrame(records)

    # Save detailed results to a CSV file
    detailed_csv = os.path.join(output_dir, 'dice_scores_detailed.csv')
    dice_df.to_csv(detailed_csv, index=False, float_format='%.4f')
    print(f"\nDetailed Dice scores have been saved to '{detailed_csv}'.")

    # Compute and save summarized statistics
    summary_records = []
    for step in ['Raw', 'Refinement', 'CRF', 'Gap_Filling', 'Removal', 'Final']:
        for class_name in ['WT', 'TC', 'ET', 'average']:
            col_name = f"{step}_{class_name}"
            if col_name in dice_df.columns:
                class_scores = dice_df[col_name].dropna()
                summary_records.append({
                    'Step': step,
                    'Class': class_name,
                    'Mean': class_scores.mean(),
                    'Std': class_scores.std(),
                    'Min': class_scores.min(),
                    'Max': class_scores.max()
                })

    summary_df = pd.DataFrame(summary_records)
    summary_csv = os.path.join(output_dir, 'dice_scores_summary.csv')
    summary_df.to_csv(summary_csv, index=False, float_format='%.4f')
    print(f"Summarized Dice statistics have been saved to '{summary_csv}'.")

# ------------------------
# Usage
# ------------------------
import os
import numpy as np
import nibabel as nib
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import KFold
import pandas as pd
import tensorflow as tf
from tqdm import tqdm  # For progress bars
import datetime
from tensorflow.keras import backend as K


def get_final_model(original_model):
    """
    Creates a new model that outputs only the 'final' layer of the original multi-output model.

    Args:
        original_model (tf.keras.Model): The original multi-output model.

    Returns:
        tf.keras.Model: A new model with the same inputs but only the 'final' output.
    """
    final_output = original_model.get_layer('final').output
    final_model = tf.keras.Model(inputs=original_model.input, outputs=final_output)
    return final_model

# def main():
#     # ------------------------
#     # Initialize Distribution Strategy
#     # ------------------------
#     strategy = tf.distribute.MirroredStrategy()  # Automatically detects all available GPUs
#     print(f"Number of GPUs detected: {strategy.num_replicas_in_sync}")

#     # ------------------------
#     # Load your data samples
#     # ------------------------
#     save_dir = '/scratch_net/maus_second/mrostamian/project/log_3D_result'
#     os.makedirs(save_dir, exist_ok=True)    

#     # Placeholder for loading file paths
#     # Replace `load_file_paths` and `brats_data_train` with your actual data loading function and dataset
#     data_samples = load_file_paths(brats_data_train)  # Replace with your actual data loading function

#     # Initialize K-Fold Cross-Validation
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     fold = 0  # Fold 1

#     # Generate the train and validation indices for the first fold
#     train_index, val_index = next(kf.split(data_samples))

#     # Extract training and validation data based on the indices
#     train_data = [data_samples[i] for i in train_index]
#     val_data = [data_samples[i] for i in val_index]

#     # For testing, select first 10 samples
#     # test_train_data = train_data[:10] if len(train_data) >= 10 else train_data
#     # test_val_data = val_data[:10] if len(val_data) >= 10 else val_data


#     test_train_data = train_data
#     test_val_data = val_data

#     print(f"Fold {fold + 1}:")
#     print(f"  Training samples: {len(test_train_data)}")
#     print(f"  Validation samples: {len(test_val_data)}")

#     # ------------------------
#     # Load the trained 2D segmentation model
#     # ------------------------
#     checkpoint_filepath = "/scratch_net/maus/mrostamian/project/log_multi_v4_2GPU/checkpoints_fold_1/brats_myunet_weights_fold_1_phase_2.weights.h5"

#     # Assuming `create_multi_encoder_unet` is defined elsewhere and correctly
#     input_t1c = tf.keras.Input(shape=(224, 224, 1), name='t1c_input')
#     input_t1n = tf.keras.Input(shape=(224, 224, 1), name='t1n_input')
#     input_t2f = tf.keras.Input(shape=(224, 224, 1), name='t2f_input')
#     input_t2w = tf.keras.Input(shape=(224, 224, 1), name='t2w_input')
#     model_inputs = [input_t1c, input_t1n, input_t2f, input_t2w]

#     # Replace `create_multi_encoder_unet` with your actual function to create the 2D model
#     model = create_multi_encoder_unet(inputs=model_inputs, num_encoders=4, training=False, num_classes=4)
#     model.load_weights(checkpoint_filepath)
#     print("Loaded 2D segmentation model with best weights.")

#     # Create a new model that outputs only the 'final' predictions
#     final_model = get_final_model(model)
#     print("Created final_model with only 'final' output.")

#     # ------------------------
#     # Generate 3D Training Data
#     # ------------------------
#     predictions_dir = os.path.join(save_dir, "predictions")
#     input_volumes, prediction_volumes, ground_truth_volumes = generate_3d_training_data(
#         test_train_data, 
#         final_model,  # Use final_model instead of original model
#         predictions_dir
#     )
#     print("\nGenerated 3D Training Data:")
#     print(f"  Input Volumes Shape: {input_volumes.shape}")          # Expected: (N, D, H, W, 5)
#     print(f"  Prediction Volumes Shape: {prediction_volumes.shape}")  # Expected: (N, D, H, W)
#     print(f"  Ground Truth Volumes Shape: {ground_truth_volumes.shape}")  # Expected: (N, D, H, W)")

#     # ------------------------
#     # Prepare Inputs for 3D U-Net
#     # ------------------------
#     try:
#         # 1) Transpose input_volumes to (N, D, 224, 224, 4)
#         print("\nPreparing to upsample the input volumes to 240x240...")
#         print("Original input_volumes shape:", input_volumes.shape)
#         # e.g. (N, 224,224, D, 4) -> (N, D,224,224, 4)
#         input_volumes = np.transpose(input_volumes, (0, 3, 1, 2, 4))
#         print("After transpose, input_volumes shape:", input_volumes.shape)
#         # Now input_volumes is (N, D, 224,224, 4)

#         # 2) Allocate a new array for 240x240
#         N, D, oldH, oldW, C = input_volumes.shape
#         newH, newW = 240, 240
#         bigger_input = np.zeros((N, D, newH, newW, C), dtype=input_volumes.dtype)

#         # 3) Uncrop each slice from 224->240
#         for i in range(N):
#             for z in range(D):
#                 slice_224 = input_volumes[i, z]  # shape (224,224,4)
#                 # uncrop_center_3d expects shape arguments like (240,240,4)
#                 slice_240 = uncrop_center_3d(slice_224, (240,240,4), pad_value=0)
#                 bigger_input[i, z] = slice_240

#         # Replace input_volumes
#         input_volumes = bigger_input
#         print("After uncropping, input_volumes shape:", input_volumes.shape)
#         # => (N, D, 240,240, 4)

#         # 4) Expand dims on prediction_volumes => (N, D, 240,240, 1)
#         print("prediction_volumes shape:", prediction_volumes.shape)
#         prediction_volumes_expanded = np.expand_dims(prediction_volumes, axis=-1)
#         print("prediction_volumes_expanded shape:", prediction_volumes_expanded.shape)

#         # 5) Concatenate (N, D, 240,240, 4) + (N, D, 240,240, 1) => (N, D, 240,240, 5)
#         three_d_inputs = np.concatenate((input_volumes, prediction_volumes_expanded), axis=-1)
#         print("three_d_inputs shape:", three_d_inputs.shape)

#     except ValueError as e:
#         print(f"Error during concatenation or uncropping: {e}")
#         print(f"input_volumes shape: {input_volumes.shape}")
#         raise


#     # One-hot encode ground truth masks
#     ground_truth_one_hot = to_categorical(ground_truth_volumes, num_classes=4)  # Shape: (N, D, H, W, 4)

#     # Normalize inputs if not already normalized
#     three_d_inputs = normalize_volume(three_d_inputs)
#     # Note: One-hot encoding already outputs binary values; normalization might not be necessary
#     # ground_truth_one_hot = normalize_volume(ground_truth_one_hot)  # Optional

#     # Validate Data Shapes
#     print(f"three_d_inputs shape: {three_d_inputs.shape}")        # Expected: (N, D, H, W, 5)
#     print(f"ground_truth_one_hot shape: {ground_truth_one_hot.shape}")  # Expected: (N, D, H, W, 4)")

#     if three_d_inputs.ndim != 5:
#         raise ValueError(f"Expected three_d_inputs to be 5D, but got {three_d_inputs.ndim}D.")
#     if ground_truth_one_hot.ndim != 5:
#         raise ValueError(f"Expected ground_truth_one_hot to be 5D, but got {ground_truth_one_hot.ndim}D.")

#     # ------------------------
#     # Create and Train the 3D U-Net within Strategy Scope
#     # ------------------------
#     print("\n=== Training 3D U-Net ===")

#     with strategy.scope():

#         # three_d_inputs.shape == (N, 155, 240, 240, 5)
#         # ground_truth_one_hot.shape == (N, 155, 240, 240, 4)

#         # 1) Decide your desired depth
#         desired_depth = 160
#         current_depth = three_d_inputs.shape[1]

#         if current_depth < desired_depth:
#             pad_total = desired_depth - current_depth  # e.g. 5
#             pad_before = pad_total // 2               # e.g. 2
#             pad_after  = pad_total - pad_before       # e.g. 3

#             # 2) Pad the inputs to (N, 160, 240, 240, 5)
#             three_d_inputs = np.pad(
#                 three_d_inputs,
#                 pad_width=((0, 0), (pad_before, pad_after), (0, 0), (0, 0), (0, 0)),
#                 mode="constant",
#                 constant_values=0
#             )

#             # 3) Pad the ground truth masks identically
#             ground_truth_one_hot = np.pad(
#                 ground_truth_one_hot,
#                 pad_width=((0, 0), (pad_before, pad_after), (0, 0), (0, 0), (0, 0)),
#                 mode="constant",
#                 constant_values=0
#             )

#         elif current_depth > desired_depth:
#             # optional cropping logic if your data is too large
#             pass

#         # Now (N, 160, 240, 240, 5) for inputs and (N, 160, 240, 240, 4) for GT
#         print("After depth-padding:")
#         print(" three_d_inputs.shape:", three_d_inputs.shape)
#         print(" ground_truth_one_hot.shape:", ground_truth_one_hot.shape)


#         # Define the 3D U-Net model
#         three_d_unet = create_3d_unet(input_shape=three_d_inputs.shape[1:], num_classes=4)  # (D, H, W, 5)

#         # Define your threshold value
#         threshold = 0.5  # Adjust as needed

#         # Compile the model with custom Dice metrics
#         three_d_unet.compile(
#             optimizer='adam',  # Replace with your optimizer if different
#             loss=combined_loss,  # Ensure combined_loss is defined appropriately
#             metrics={
#                 'final': [
#                     create_combined_channels_metric(
#                         base_metric=dice_coefficient,
#                         class_indices=[1, 3],
#                         name='dice_tc',
#                         threshold=threshold
#                     ),
#                     create_combined_channels_metric(
#                         base_metric=dice_coefficient,
#                         class_indices=[1, 2, 3],
#                         name='dice_wt',
#                         threshold=threshold
#                     ),
#                     create_combined_channels_metric(
#                         base_metric=dice_coefficient,
#                         class_indices=[3],
#                         name='dice_et',
#                         threshold=threshold
#                     ),
#                 ]
#             }
#         )

#         three_d_unet.summary()

#     # Define training parameters
#     batch_size = 1  # Adjust based on GPU memory and number of GPUs
#     epochs = 30  # Adjust as needed

#     # Define callbacks (updated to monitor Dice metrics instead of accuracy)
#     log_dir = os.path.join("logs", "dice_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#     tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#     checkpoint_callback = ModelCheckpoint(
#         filepath=os.path.join(save_dir, "best_3d_unet_test.h5"),
#         monitor='val_loss',  # Monitor the Whole Tumor Dice score
#         save_best_only=True,
#         verbose=1
#     )

#     early_stopping_callback = EarlyStopping(
#         monitor='val_loss',
#         patience=10,
#         verbose=1,
#         restore_best_weights=True
#     )

#     reduce_lr_callback = ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.5,
#         patience=5,
#         verbose=1
#     )

#     # Train the 3D U-Net model
#     history = three_d_unet.fit(
#         three_d_inputs, 
#         ground_truth_one_hot,
#         batch_size=batch_size,
#         epochs=epochs,
#         validation_split=0.1,  # Adjust as needed
#         callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_callback, tensorboard_callback],
#         verbose=1
#     )

#     # Save the trained 3D U-Net model
#     three_d_unet.save(os.path.join(save_dir, "3d_unet_refinement_model_test.h5"))
#     print("Trained 3D U-Net model saved.")

#     # ------------------------
#     # Testing the Model on Validation Data
#     # ------------------------
#     print("\n=== Testing on Validation Data ===")
#     # Load the trained 3D U-Net model

#     three_d_unet = tf.keras.models.load_model(
#         os.path.join(save_dir, "best_3d_unet_test.h5"),
#         custom_objects={
#             'dice_tc': create_combined_channels_metric(
#                 base_metric=dice_coefficient,
#                 class_indices=[1,3],
#                 name='dice_tc',
#                 threshold=0.5
#             ),
#             'dice_wt': create_combined_channels_metric(
#                 base_metric=dice_coefficient,
#                 class_indices=[1,2,3],
#                 name='dice_wt',
#                 threshold=0.5
#             ),
#             'dice_et': create_combined_channels_metric(
#                 base_metric=dice_coefficient,
#                 class_indices=[3],
#                 name='dice_et',
#                 threshold=0.5
#             ),
#             'combined_loss': combined_loss,
#             'dice_coefficient': dice_coefficient,
#         }
#     )
    

#     print("Loaded trained 3D U-Net refinement model.")

#     # Initialize a list to store Dice scores for all validation samples
#     all_dice_scores = []

#     # Define composite class mappings
#     composite_classes = {'WT': [1, 2, 3], 'TC': [1, 3], 'ET': [3]}

#     original_shape = (240, 240)  # Define original shape

#     for idx, sample_paths in enumerate(tqdm(test_val_data, desc="Validation Samples")):
#         sample_id = os.path.basename(sample_paths.get('mask', 'sample_unknown')).split('.')[0]
#         print(f"\nProcessing validation sample {idx + 1}/{len(test_val_data)}: '{sample_id}'")

#         # ------------------------
#         # Step 1: Load Modalities
#         # ------------------------
#         modalities = ['t1c', 't1n', 't2f', 't2w']
#         modality_data = {}
#         skip_sample = False
#         for modality in modalities:
#             path = sample_paths.get(modality)
#             if not path:
#                 print(f"  Error: Missing modality '{modality}'. Skipping this sample.")
#                 skip_sample = True
#                 break
#             try:
#                 img = nib.load(path).get_fdata()
#                 if img.ndim != 3:
#                     print(f"  Error: Modality '{modality}' does not have 3 dimensions. Skipping this sample.")
#                     skip_sample = True
#                     break
#                 modality_data[modality] = normalize_volume(img)
#                 print(f"  Loaded and normalized '{modality}' from '{path}'.")
#             except Exception as e:
#                 print(f"  Error loading '{modality}' from '{path}': {e}. Skipping this sample.")
#                 skip_sample = True
#                 break

#         if skip_sample:
#             continue

#         num_slices = modality_data['t1c'].shape[-1]
#         print(f"  Total number of slices: {num_slices}")

#         probability_slices = []
#         image_slices = []

#         for slice_idx in range(num_slices):
#             if (slice_idx + 1) % 10 == 0 or slice_idx == 0:
#                 print(f"    Processing slice {slice_idx + 1}/{num_slices}")

#             model_inputs = {}
#             skip_slice = False
#             original_slices = []

#             for modality in modalities:
#                 slice_img = modality_data[modality][:, :, slice_idx]
#                 original_slices.append(slice_img)

#                 # Preprocess: Crop the slice
#                 cropped = crop_center(slice_img, 224, 224)

#                 # Check if the cropped slice is empty
#                 if not np.any(cropped):
#                     skip_slice = True
#                     print(f"      Slice {slice_idx + 1}: '{modality}' is empty after cropping. Skipping this slice.")
#                     break

#                 # Expand dimensions to match model input
#                 input_slice = np.expand_dims(np.expand_dims(cropped, axis=0), axis=-1)  # Shape: (1, H, W, 1)
#                 model_inputs[f"{modality}_input"] = input_slice

#             if skip_slice:
#                 # Append default values for skipped slice
#                 probability_slices.append(np.zeros((224, 224, 4)))  # Including background
#                 image_slices.append(np.zeros((224, 224, len(modalities))))
#                 continue

#             try:
#                 # Make prediction with final_model
#                 probabilities = final_model.predict(model_inputs)[0]  # Shape: (224, 224, 4)
#                 print(f"      Slice {slice_idx + 1}: 2D Model prediction successful.")
#             except Exception as e:
#                 print(f"      Slice {slice_idx + 1}: 2D Model prediction failed with error: {e}. Skipping this slice.")
#                 probability_slices.append(np.zeros((224, 224, 4)))
#                 image_slices.append(np.zeros((224, 224, len(modalities))))
#                 continue

#             # Resize probabilities back to original slice size (240x240)
#             probabilities_resized = np.zeros((original_shape[0], original_shape[1], 4))
#             for c in range(4):
#                 probabilities_resized[..., c] = uncrop_center(probabilities[..., c], original_shape[:2])

#             probability_slices.append(probabilities_resized)
#             image_slice = np.stack(original_slices, axis=-1)  # Shape: (240, 240, 4)
#             image_slices.append(image_slice)


#         print("\n  Completed 2D model predictions for all slices.")

#         # Stack into 3D volumes
#         probability_volume = np.stack(probability_slices, axis=0)  # Shape: (D, H, W, 4)
#         raw_prediction = np.argmax(probability_volume, axis=-1).astype(np.uint8)  # Shape: (D, H, W)

#         # Load ground truth mask
#         ground_truth_path = sample_paths.get('mask')
#         if not ground_truth_path:
#             print("  Error: Missing ground truth mask. Skipping this sample.")
#             continue
#         try:
#             ground_truth = nib.load(ground_truth_path).get_fdata().astype(np.uint8)
#             if ground_truth.ndim != 3:
#                 print("  Error: Ground truth mask does not have 3 dimensions. Skipping this sample.")
#                 continue
#             ground_truth = np.transpose(ground_truth, (2, 0, 1))  # Assuming (H, W, D) to (D, H, W)
#             print(f"  Loaded ground truth mask from '{ground_truth_path}'.")
#         except Exception as e:
#             print(f"  Error loading ground truth mask from '{ground_truth_path}': {e}. Skipping this sample.")
#             continue

#         # Compute initial Dice scores for raw prediction
#         dice_scores_raw = compute_dice_scores(ground_truth, raw_prediction, composite_classes)
#         print("\n  Raw Prediction Dice Scores:")
#         for class_name, dice in dice_scores_raw.items():
#             print(f"    {class_name}: {dice:.4f}")

#         # Apply 3D U-Net Refinement
#         # Prepare input for 3D U-Net: concatenate image_volume with raw_prediction
#         image_volume = np.stack(image_slices, axis=0)  # Shape: (D, H, W, Channels)

#         prediction_volume_expanded = np.expand_dims(raw_prediction, axis=-1)  # Shape: (D, H, W, 1)
#         three_d_input = np.concatenate((image_volume, prediction_volume_expanded), axis=-1)  # Shape: (D, H, W, 5)
#         three_d_input = np.expand_dims(three_d_input, axis=0)  # Shape: (1, D, H, W, 5)
#         three_d_input = normalize_volume(three_d_input)

#         desired_depth = 160
#         current_depth = three_d_input.shape[1]  # e.g., 155
#         if current_depth < desired_depth:
#             pad_total = desired_depth - current_depth
#             pad_before = pad_total // 2
#             pad_after = pad_total - pad_before
#             three_d_input = np.pad(
#                 three_d_input,
#                 pad_width=((0, 0), (pad_before, pad_after), (0, 0), (0, 0), (0, 0)),
#                 mode='constant',
#                 constant_values=0
#             )
#         elif current_depth > desired_depth:
#             # Crop the excess slices from the center
#             crop_total = current_depth - desired_depth
#             crop_before = crop_total // 2
#             crop_after = crop_total - crop_before
#             three_d_input = three_d_input[:, crop_before:-crop_after, :, :, :]

#         # Predict with 3D U-Net
#         try:
#             refined_prediction_prob = three_d_unet.predict(three_d_input)[0]  # Shape: (D, H, W, 4)
#             print("  3D U-Net prediction successful.")
#         except Exception as e:
#             print(f"  3D U-Net prediction failed with error: {e}. Skipping this sample.")
#             continue

#         # After obtaining refined_prediction from the model
#         refined_prediction = np.argmax(refined_prediction_prob, axis=-1).astype(np.uint8)  # Shape: (160, H, W)

#         # Remove padding to match ground truth depth
#         pad_total = refined_prediction.shape[0] - ground_truth.shape[0]
#         if pad_total > 0:
#             pad_before = pad_total // 2
#             pad_after = pad_total - pad_before
#             refined_prediction = refined_prediction[pad_before:-pad_after if pad_after > 0 else None, :, :]
#             refined_prediction_prob = refined_prediction_prob[pad_before:-pad_after if pad_after > 0 else None, :, :]

#         elif pad_total < 0:
#             raise ValueError("Refined prediction depth is smaller than ground truth depth.")

#         # Now refined_prediction and ground_truth have matching shapes
#         dice_scores_refined = compute_dice_scores(ground_truth, refined_prediction, composite_classes)


#         # Compute Dice scores after refinement (before post-processing)
#         dice_scores_refined = compute_dice_scores(ground_truth, refined_prediction, composite_classes)
#         print("\n  Refined Prediction Dice Scores:")
#         for class_name, dice in dice_scores_refined.items():
#             print(f"    {class_name}: {dice:.4f}")

#         # Apply post-processing
#         # Define post-processing thresholds and parameters
#         thresholds = {'WT': 0.5, 'TC': 0.5, 'ET': 0.5}
#         min_voxel_counts = {1: 50, 2: 50, 3: 20}  # Class labels as integers
#         min_mean_probs = {1: 0.8, 2: 0.8, 3: 0.8}

#         final_prediction, post_processing_results = apply_post_processing(
#             refined_prediction,
#             refined_prediction_prob,
#             image_volume,  # Pass the image volume
#             ground_truth,  # Pass ground truth for Dice computation
#             thresholds=thresholds,
#             min_voxel_counts=min_voxel_counts,
#             min_mean_probs=min_mean_probs,
#             composite_classes=composite_classes
#         )

#         # Compute Dice scores after final prediction
#         dice_scores_final = post_processing_results['Final']['dice_scores']
#         print("\n  Final Prediction Dice Scores:")
#         for class_name, dice in dice_scores_final.items():
#             print(f"    {class_name}: {dice:.4f}")




#         # Display results for the sample
#         results_df = pd.DataFrame({
#             'Stage': ['Raw', 'Refinement', 'CRF', 'Gap_Filling', 'Removal', 'Final'],
#             'WT': [
#                 dice_scores_raw.get('WT', np.nan), 
#                 dice_scores_refined.get('WT', np.nan), 
#                 post_processing_results['CRF']['dice_scores'].get('WT', np.nan),
#                 post_processing_results['Gap_Filling']['dice_scores'].get('WT', np.nan),
#                 post_processing_results['Removal']['dice_scores'].get('WT', np.nan),
#                 dice_scores_final.get('WT', np.nan)
#             ],
#             'TC': [
#                 dice_scores_raw.get('TC', np.nan), 
#                 dice_scores_refined.get('TC', np.nan), 
#                 post_processing_results['CRF']['dice_scores'].get('TC', np.nan),
#                 post_processing_results['Gap_Filling']['dice_scores'].get('TC', np.nan),
#                 post_processing_results['Removal']['dice_scores'].get('TC', np.nan),
#                 dice_scores_final.get('TC', np.nan)
#             ],
#             'ET': [
#                 dice_scores_raw.get('ET', np.nan), 
#                 dice_scores_refined.get('ET', np.nan), 
#                 post_processing_results['CRF']['dice_scores'].get('ET', np.nan),
#                 post_processing_results['Gap_Filling']['dice_scores'].get('ET', np.nan),
#                 post_processing_results['Removal']['dice_scores'].get('ET', np.nan),
#                 dice_scores_final.get('ET', np.nan)
#             ],
#             'Average': [
#                 dice_scores_raw.get('average', np.nan), 
#                 dice_scores_refined.get('average', np.nan), 
#                 post_processing_results['CRF']['dice_scores'].get('average', np.nan),
#                 post_processing_results['Gap_Filling']['dice_scores'].get('average', np.nan),
#                 post_processing_results['Removal']['dice_scores'].get('average', np.nan),
#                 dice_scores_final.get('average', np.nan)
#             ]
#         })

#         print(f"\nSample ID: {sample_id}")
#         print(results_df.to_string(index=False))
#         print("\n--- End of Sample ---\n")



#         # Append to all_dice_scores
#         all_dice_scores.append({
#             'Sample_ID': sample_id,
#             'Raw': dice_scores_raw,
#             'Refinement': dice_scores_refined,
#             'CRF': post_processing_results['CRF']['dice_scores'],
#             'Gap_Filling': post_processing_results['Gap_Filling']['dice_scores'],
#             'Removal': post_processing_results['Removal']['dice_scores'],
#             'Final': dice_scores_final
#         })
        

#         # ------------------------
#         # Execute Aggregation and Saving
#         # ------------------------
#         # Define the output directory for test results
#         output_dir = os.path.join(save_dir, "test_results")
#         os.makedirs(output_dir, exist_ok=True)

#         if all_dice_scores:
#             aggregate_and_save_dice_scores(all_dice_scores, output_dir)
#         else:
#             print("No Dice scores to aggregate.")







def main():
    # ------------------------
    # Initialize Distribution Strategy
    # ------------------------
    strategy = tf.distribute.MirroredStrategy()  # Automatically detects all available GPUs
    print(f"Number of GPUs detected: {strategy.num_replicas_in_sync}")

    # ------------------------
    # Load your data samples
    # ------------------------
    save_dir = '/scratch_net/maus_second/mrostamian/project/log_3D_result'
    os.makedirs(save_dir, exist_ok=True)   

    # Replace `load_file_paths` and `brats_data_train` with your actual data loading function and dataset
    data_samples = load_file_paths(brats_data_train)  # Your actual data loading function

    # Initialize K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 0  # Fold 1

    # Generate the train and validation indices for the first fold
    train_index, val_index = next(kf.split(data_samples))

    # Extract training and validation data based on the indices
    train_data = [data_samples[i] for i in train_index]
    val_data = [data_samples[i] for i in val_index]

    test_val_data = val_data

    print(f"Fold {fold + 1}:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")

    # ------------------------
    # Load the trained 2D segmentation model
    # ------------------------
    checkpoint_filepath = "/scratch_net/maus/mrostamian/project/log_multi_v4_2GPU/checkpoints_fold_1/brats_myunet_weights_fold_1_phase_2.weights.h5"

    # Define inputs for the multi-encoder U-Net
    input_t1c = tf.keras.Input(shape=(224, 224, 1), name='t1c_input')
    input_t1n = tf.keras.Input(shape=(224, 224, 1), name='t1n_input')
    input_t2f = tf.keras.Input(shape=(224, 224, 1), name='t2f_input')
    input_t2w = tf.keras.Input(shape=(224, 224, 1), name='t2w_input')
    model_inputs = [input_t1c, input_t1n, input_t2f, input_t2w]

    # Create and load the 2D segmentation model
    model = create_multi_encoder_unet(inputs=model_inputs, num_encoders=4, training=False, num_classes=4)
    model.load_weights(checkpoint_filepath)
    print("Loaded 2D segmentation model with best weights.")

    # Create a model that outputs only the 'final' predictions
    final_model = get_final_model(model)
    print("Created final_model with only 'final' output.")

    # ------------------------
    # Prepare Data Generators and Datasets
    # ------------------------

    # Define helper function to prepare data for a single sample
        
    def prepare_sample(input_volume, prediction_volume, ground_truth_volume, desired_depth=160):
        # Transpose input_volume to (D, H, W, C)
        input_volume = np.transpose(input_volume, (0, 2, 1, 3))  # From (D, H, W, C)
        ground_truth_volume = np.transpose(ground_truth_volume, (0, 2, 1))  # From (D, H, W)

        current_depth = input_volume.shape[0]

        # Adjust depth for input_volume, ground_truth_volume, and prediction_volume
        if current_depth < desired_depth:
            pad_total = desired_depth - current_depth  # Should be 5
            pad_before = pad_total // 2               # e.g., 2
            pad_after = pad_total - pad_before        # e.g., 3

            # Pad input_volume along the depth dimension
            input_volume = np.pad(
                input_volume,
                pad_width=((pad_before, pad_after), (0, 0), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0
            )

            # Pad ground_truth_volume similarly
            ground_truth_volume = np.pad(
                ground_truth_volume,
                pad_width=((pad_before, pad_after), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0
            )

            # Pad prediction_volume similarly
            prediction_volume = np.pad(
                prediction_volume,
                pad_width=((pad_before, pad_after), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0
            )

        elif current_depth > desired_depth:
            # Crop from the center if depth is greater than desired
            crop_total = current_depth - desired_depth
            crop_before = crop_total // 2
            crop_after = crop_total - crop_before

            input_volume = input_volume[crop_before:-crop_after, :, :, :]
            ground_truth_volume = ground_truth_volume[crop_before:-crop_after, :, :]
            prediction_volume = prediction_volume[crop_before:-crop_after, :, :]

        else:
            # No padding or cropping needed
            pass

        # Expand dims on prediction_volume to match input_volume shape
        prediction_volume_expanded = np.expand_dims(prediction_volume, axis=-1)  # Shape: (D, H, W, 1)

        # Concatenate input_volume and prediction_volume
        three_d_input = np.concatenate((input_volume, prediction_volume_expanded), axis=-1)  # Shape: (D, H, W, 5)

        # One-hot encode ground_truth_volume
        ground_truth_one_hot = to_categorical(ground_truth_volume, num_classes=4)  # Shape: (D, H, W, 4)

        # Normalize inputs
        three_d_input = normalize_volume(three_d_input)

        return three_d_input, ground_truth_one_hot


    # Generator function for training data
    def train_data_generator():
        for input_volume, prediction_volume, ground_truth_volume in generate_3d_training_data_one_sample(
            train_data, 
            final_model,
            predictions_dir=os.path.join(save_dir, "predictions"),
            original_shape=(240, 240, 155),
            input_size=(224, 224),
            batch_size=16
        ):
            try:
                three_d_input, ground_truth_one_hot = prepare_sample(
                    input_volume, prediction_volume, ground_truth_volume
                )
                yield three_d_input, ground_truth_one_hot
            except Exception as e:
                print(f"Error preparing training sample: {e}")
                continue  # Skip sample on error

    # Generator function for validation data
    def val_data_generator():
        for input_volume, prediction_volume, ground_truth_volume in generate_3d_training_data_one_sample(
            val_data, 
            final_model,
            predictions_dir=os.path.join(save_dir, "predictions_val"),
            original_shape=(240, 240, 155),
            input_size=(224, 224),
            batch_size=16
        ):
            try:
                three_d_input, ground_truth_one_hot = prepare_sample(
                    input_volume, prediction_volume, ground_truth_volume
                )
                yield three_d_input, ground_truth_one_hot
            except Exception as e:
                print(f"Error preparing validation sample: {e}")
                continue  # Skip sample on error

    # Create tf.data.Datasets from generators
    batch_size = 1  # Adjust based on GPU memory

    train_dataset = tf.data.Dataset.from_generator(
        train_data_generator,
        output_signature=(
            tf.TensorSpec(shape=(160, 240, 240, 5), dtype=tf.float32),
            tf.TensorSpec(shape=(160, 240, 240, 4), dtype=tf.float32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        val_data_generator,
        output_signature=(
            tf.TensorSpec(shape=(160, 240, 240, 5), dtype=tf.float32),
            tf.TensorSpec(shape=(160, 240, 240, 4), dtype=tf.float32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Set sharding options to suppress warnings
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_dataset = train_dataset.with_options(options)
    val_dataset = val_dataset.with_options(options)


    # ------------------------
    # Create and Train the 3D U-Net within Strategy Scope
    # ------------------------
    print("\n=== Training 3D U-Net ===")

    with strategy.scope():
        # Define the 3D U-Net model
        three_d_unet = create_3d_unet(input_shape=(160, 240, 240, 5), num_classes=4)

        # Define your threshold value
        threshold = 0.5  # Adjust as needed

        three_d_unet.compile(
            optimizer=Adam(learning_rate=0.0001),  # Set lr to 0.0001
            loss=combined_loss,  
            metrics={
                'final': [
                    create_combined_channels_metric(
                        base_metric=dice_coefficient,
                        class_indices=[1, 3],
                        name='dice_tc',
                        threshold=threshold
                    ),
                    create_combined_channels_metric(
                        base_metric=dice_coefficient,
                        class_indices=[1, 2, 3],
                        name='dice_wt',
                        threshold=threshold
                    ),
                    create_combined_channels_metric(
                        base_metric=dice_coefficient,
                        class_indices=[3],
                        name='dice_et',
                        threshold=threshold
                    ),
                ]
            }
        )


        three_d_unet.summary()

    # Define training parameters
    epochs = 50  # Adjust as needed

    # Define callbacks
    log_dir = os.path.join("logs", "dice_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(save_dir, "best_3d_unet_test.h5"),
        monitor='val_loss',  # Monitor validation loss
        save_best_only=True,
        verbose=1
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=1,
        restore_best_weights=True
    )

    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1
    )

    # Train the 3D U-Net model using the datasets
    history = three_d_unet.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_callback, tensorboard_callback],
        verbose=1
    )

    # Save the trained 3D U-Net model
    three_d_unet.save(os.path.join(save_dir, "3d_unet_refinement_model_test.h5"))
    print("Trained 3D U-Net model saved.")

    # ------------------------
    # Testing the Model on Validation Data
    # ------------------------

    print("\n=== Testing on Validation Data ===")
    # Load the trained 3D U-Net model

    three_d_unet = tf.keras.models.load_model(
        os.path.join(save_dir, "best_3d_unet_test.h5"),
        custom_objects={
            'dice_tc': create_combined_channels_metric(
                base_metric=dice_coefficient,
                class_indices=[1,3],
                name='dice_tc',
                threshold=0.5
            ),
            'dice_wt': create_combined_channels_metric(
                base_metric=dice_coefficient,
                class_indices=[1,2,3],
                name='dice_wt',
                threshold=0.5
            ),
            'dice_et': create_combined_channels_metric(
                base_metric=dice_coefficient,
                class_indices=[3],
                name='dice_et',
                threshold=0.5
            ),
            'combined_loss': combined_loss,
            'dice_coefficient': dice_coefficient,
        }
    )
    

    print("Loaded trained 3D U-Net refinement model.")

    # Initialize a list to store Dice scores for all validation samples
    all_dice_scores = []

    # Define composite class mappings
    composite_classes = {'WT': [1, 2, 3], 'TC': [1, 3], 'ET': [3]}

    original_shape = (240, 240)  # Define original shape

    for idx, sample_paths in enumerate(tqdm(test_val_data, desc="Validation Samples")):
        sample_id = os.path.basename(sample_paths.get('mask', 'sample_unknown')).split('.')[0]
        print(f"\nProcessing validation sample {idx + 1}/{len(test_val_data)}: '{sample_id}'")

        # ------------------------
        # Step 1: Load Modalities
        # ------------------------
        modalities = ['t1c', 't1n', 't2f', 't2w']
        modality_data = {}
        skip_sample = False
        for modality in modalities:
            path = sample_paths.get(modality)
            if not path:
                print(f"  Error: Missing modality '{modality}'. Skipping this sample.")
                skip_sample = True
                break
            try:
                img = nib.load(path).get_fdata()
                if img.ndim != 3:
                    print(f"  Error: Modality '{modality}' does not have 3 dimensions. Skipping this sample.")
                    skip_sample = True
                    break
                modality_data[modality] = normalize_volume(img)
                print(f"  Loaded and normalized '{modality}' from '{path}'.")
            except Exception as e:
                print(f"  Error loading '{modality}' from '{path}': {e}. Skipping this sample.")
                skip_sample = True
                break

        if skip_sample:
            continue

        num_slices = modality_data['t1c'].shape[-1]
        print(f"  Total number of slices: {num_slices}")

        probability_slices = []
        image_slices = []

        for slice_idx in range(num_slices):
            if (slice_idx + 1) % 10 == 0 or slice_idx == 0:
                print(f"    Processing slice {slice_idx + 1}/{num_slices}")

            model_inputs = {}
            skip_slice = False
            original_slices = []

            for modality in modalities:
                slice_img = modality_data[modality][:, :, slice_idx]
                original_slices.append(slice_img)

                # Preprocess: Crop the slice
                cropped = crop_center(slice_img, 224, 224)

                # Check if the cropped slice is empty
                if not np.any(cropped):
                    skip_slice = True
                    print(f"      Slice {slice_idx + 1}: '{modality}' is empty after cropping. Skipping this slice.")
                    break

                # Expand dimensions to match model input
                input_slice = np.expand_dims(np.expand_dims(cropped, axis=0), axis=-1)  # Shape: (1, H, W, 1)
                model_inputs[f"{modality}_input"] = input_slice

            if skip_slice:
                # Append default values for skipped slice
                probability_slices.append(np.zeros((224, 224, 4)))  # Including background
                image_slices.append(np.zeros((224, 224, len(modalities))))
                continue

            try:
                # Make prediction with final_model
                probabilities = final_model.predict(model_inputs)[0]  # Shape: (224, 224, 4)
                print(f"      Slice {slice_idx + 1}: 2D Model prediction successful.")
            except Exception as e:
                print(f"      Slice {slice_idx + 1}: 2D Model prediction failed with error: {e}. Skipping this slice.")
                probability_slices.append(np.zeros((224, 224, 4)))
                image_slices.append(np.zeros((224, 224, len(modalities))))
                continue

            # Resize probabilities back to original slice size (240x240)
            probabilities_resized = np.zeros((original_shape[0], original_shape[1], 4))
            for c in range(4):
                probabilities_resized[..., c] = uncrop_center(probabilities[..., c], original_shape[:2])

            probability_slices.append(probabilities_resized)
            image_slice = np.stack(original_slices, axis=-1)  # Shape: (240, 240, 4)
            image_slices.append(image_slice)


        print("\n  Completed 2D model predictions for all slices.")

        # Stack into 3D volumes
        probability_volume = np.stack(probability_slices, axis=0)  # Shape: (D, H, W, 4)
        raw_prediction = np.argmax(probability_volume, axis=-1).astype(np.uint8)  # Shape: (D, H, W)

        # Load ground truth mask
        ground_truth_path = sample_paths.get('mask')
        if not ground_truth_path:
            print("  Error: Missing ground truth mask. Skipping this sample.")
            continue
        try:
            ground_truth = nib.load(ground_truth_path).get_fdata().astype(np.uint8)
            if ground_truth.ndim != 3:
                print("  Error: Ground truth mask does not have 3 dimensions. Skipping this sample.")
                continue
            ground_truth = np.transpose(ground_truth, (2, 0, 1))  # Assuming (H, W, D) to (D, H, W)
            print(f"  Loaded ground truth mask from '{ground_truth_path}'.")
        except Exception as e:
            print(f"  Error loading ground truth mask from '{ground_truth_path}': {e}. Skipping this sample.")
            continue

        # Compute initial Dice scores for raw prediction
        dice_scores_raw = compute_dice_scores(ground_truth, raw_prediction, composite_classes)
        print("\n  Raw Prediction Dice Scores:")
        for class_name, dice in dice_scores_raw.items():
            print(f"    {class_name}: {dice:.4f}")

        # Apply 3D U-Net Refinement
        # Prepare input for 3D U-Net: concatenate image_volume with raw_prediction
        image_volume = np.stack(image_slices, axis=0)  # Shape: (D, H, W, Channels)

        prediction_volume_expanded = np.expand_dims(raw_prediction, axis=-1)  # Shape: (D, H, W, 1)
        three_d_input = np.concatenate((image_volume, prediction_volume_expanded), axis=-1)  # Shape: (D, H, W, 5)
        three_d_input = np.expand_dims(three_d_input, axis=0)  # Shape: (1, D, H, W, 5)
        three_d_input = normalize_volume(three_d_input)

        desired_depth = 160
        current_depth = three_d_input.shape[1]  # e.g., 155
        if current_depth < desired_depth:
            pad_total = desired_depth - current_depth
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            three_d_input = np.pad(
                three_d_input,
                pad_width=((0, 0), (pad_before, pad_after), (0, 0), (0, 0), (0, 0)),
                mode='constant',
                constant_values=0
            )
        elif current_depth > desired_depth:
            # Crop the excess slices from the center
            crop_total = current_depth - desired_depth
            crop_before = crop_total // 2
            crop_after = crop_total - crop_before
            three_d_input = three_d_input[:, crop_before:-crop_after, :, :, :]

        # Predict with 3D U-Net
        try:
            refined_prediction_prob = three_d_unet.predict(three_d_input)[0]  # Shape: (D, H, W, 4)
            print("  3D U-Net prediction successful.")
        except Exception as e:
            print(f"  3D U-Net prediction failed with error: {e}. Skipping this sample.")
            continue

        # After obtaining refined_prediction from the model
        refined_prediction = np.argmax(refined_prediction_prob, axis=-1).astype(np.uint8)  # Shape: (160, H, W)

        # Remove padding to match ground truth depth
        pad_total = refined_prediction.shape[0] - ground_truth.shape[0]
        if pad_total > 0:
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            refined_prediction = refined_prediction[pad_before:-pad_after if pad_after > 0 else None, :, :]
            refined_prediction_prob = refined_prediction_prob[pad_before:-pad_after if pad_after > 0 else None, :, :]

        elif pad_total < 0:
            raise ValueError("Refined prediction depth is smaller than ground truth depth.")

        # Now refined_prediction and ground_truth have matching shapes
        dice_scores_refined = compute_dice_scores(ground_truth, refined_prediction, composite_classes)


        # Compute Dice scores after refinement (before post-processing)
        dice_scores_refined = compute_dice_scores(ground_truth, refined_prediction, composite_classes)
        print("\n  Refined Prediction Dice Scores:")
        for class_name, dice in dice_scores_refined.items():
            print(f"    {class_name}: {dice:.4f}")

        # Apply post-processing
        # Define post-processing thresholds and parameters
        thresholds = {'WT': 0.5, 'TC': 0.5, 'ET': 0.5}
        min_voxel_counts = {1: 50, 2: 50, 3: 20}  # Class labels as integers
        min_mean_probs = {1: 0.8, 2: 0.8, 3: 0.8}

        final_prediction, post_processing_results = apply_post_processing(
            refined_prediction,
            refined_prediction_prob,
            image_volume,  # Pass the image volume
            ground_truth,  # Pass ground truth for Dice computation
            thresholds=thresholds,
            min_voxel_counts=min_voxel_counts,
            min_mean_probs=min_mean_probs,
            composite_classes=composite_classes
        )

        # Compute Dice scores after final prediction
        dice_scores_final = post_processing_results['Final']['dice_scores']
        print("\n  Final Prediction Dice Scores:")
        for class_name, dice in dice_scores_final.items():
            print(f"    {class_name}: {dice:.4f}")




        # Display results for the sample
        results_df = pd.DataFrame({
            'Stage': ['Raw', 'Refinement', 'CRF', 'Gap_Filling', 'Removal', 'Final'],
            'WT': [
                dice_scores_raw.get('WT', np.nan), 
                dice_scores_refined.get('WT', np.nan), 
                post_processing_results['CRF']['dice_scores'].get('WT', np.nan),
                post_processing_results['Gap_Filling']['dice_scores'].get('WT', np.nan),
                post_processing_results['Removal']['dice_scores'].get('WT', np.nan),
                dice_scores_final.get('WT', np.nan)
            ],
            'TC': [
                dice_scores_raw.get('TC', np.nan), 
                dice_scores_refined.get('TC', np.nan), 
                post_processing_results['CRF']['dice_scores'].get('TC', np.nan),
                post_processing_results['Gap_Filling']['dice_scores'].get('TC', np.nan),
                post_processing_results['Removal']['dice_scores'].get('TC', np.nan),
                dice_scores_final.get('TC', np.nan)
            ],
            'ET': [
                dice_scores_raw.get('ET', np.nan), 
                dice_scores_refined.get('ET', np.nan), 
                post_processing_results['CRF']['dice_scores'].get('ET', np.nan),
                post_processing_results['Gap_Filling']['dice_scores'].get('ET', np.nan),
                post_processing_results['Removal']['dice_scores'].get('ET', np.nan),
                dice_scores_final.get('ET', np.nan)
            ],
            'Average': [
                dice_scores_raw.get('average', np.nan), 
                dice_scores_refined.get('average', np.nan), 
                post_processing_results['CRF']['dice_scores'].get('average', np.nan),
                post_processing_results['Gap_Filling']['dice_scores'].get('average', np.nan),
                post_processing_results['Removal']['dice_scores'].get('average', np.nan),
                dice_scores_final.get('average', np.nan)
            ]
        })

        print(f"\nSample ID: {sample_id}")
        print(results_df.to_string(index=False))
        print("\n--- End of Sample ---\n")



        # Append to all_dice_scores
        all_dice_scores.append({
            'Sample_ID': sample_id,
            'Raw': dice_scores_raw,
            'Refinement': dice_scores_refined,
            'CRF': post_processing_results['CRF']['dice_scores'],
            'Gap_Filling': post_processing_results['Gap_Filling']['dice_scores'],
            'Removal': post_processing_results['Removal']['dice_scores'],
            'Final': dice_scores_final
        })
        

        # ------------------------
        # Execute Aggregation and Saving
        # ------------------------
        # Define the output directory for test results
        output_dir = os.path.join(save_dir, "test_results")
        os.makedirs(output_dir, exist_ok=True)

        if all_dice_scores:
            aggregate_and_save_dice_scores(all_dice_scores, output_dir)
        else:
            print("No Dice scores to aggregate.")







# ------------------------
# Entry Point
# ------------------------

if __name__ == "__main__":
    main()
