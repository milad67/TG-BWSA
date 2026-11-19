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
    Creates a multi-encoder U-Net model by directly connecting pretrained encoders to the decoder.
    """
    assert len(inputs) == num_encoders, "Number of inputs must match num_encoders"

    input_shape = inputs[0].shape[1:]  # Assuming all inputs have the same shape
    modalities = ['t1c', 't1n', 't2f', 't2w'][:num_encoders]

    filters_list = [32, 64, 128, 256, 512]  # Encoder filters
    filters_decoder = [256, 128, 64, 32]  # Decoder filters
    att_heads = [2] * len(filters_list)
    dpr = [x for x in np.linspace(0, 0.0, len(filters_list))]  # Stochastic depth rate

    # Load pretrained encoders
    pretrained_encoders = []
    encoder_outputs = []
    encoder_skip_connections = [[] for _ in range(len(filters_list) - 1)]  # Collect skip connections

    for i, modality in enumerate(modalities):
        pretrained_encoder = load_pretrained_encoder(
            modality=modality,
            input_shape=input_shape,
            filters=filters_list,
            att_heads=att_heads,
            dpr=dpr,
            freeze=True
        )
        pretrained_encoders.append(pretrained_encoder)

        # Extract outputs (bottleneck and skip connections)
        outputs = pretrained_encoder(inputs[i])
        bottleneck = outputs[0]
        skips = outputs[1:]

        # Add bottleneck and skip connections to respective lists
        encoder_outputs.append(bottleneck)
        for level, skip in enumerate(skips):
            encoder_skip_connections[level].append(skip)

    # Combine bottlenecks
    combined_bottleneck = layers.Concatenate(name="combined_bottleneck")(encoder_outputs)

    # Combine skip connections for each level
    combined_skips = [
        layers.Concatenate(name=f"combined_skip_{level + 1}")(skips_at_level)
        for level, skips_at_level in enumerate(encoder_skip_connections)
    ]

    # Build the decoder
    outputs = build_decoder(
        adapter_output=combined_bottleneck,
        encoder_skips=combined_skips,
        filters_decoder=filters_decoder,
        att_heads=[2] * len(filters_decoder),
        dpr=[0.0] * len(filters_decoder),
        training=training,
        num_classes=num_classes,
    )

    # Create the final model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="multi_modal_unet")

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

    # ------------------------
    # Step 5: Save Raw Prediction
    # ------------------------
    print("\n=== Step 5: Saving Raw Prediction ===")
    # Transpose raw_prediction to (H, W, D)
    raw_prediction_transposed = np.transpose(raw_prediction, (1, 2, 0))  # From (D, H, W) to (H, W, D)
    # Initialize final_prediction_raw with shape (240, 240, 155)
    final_prediction_raw = np.zeros(original_shape, dtype=np.uint8)
    # Apply priority-based combining: ET (3) > NCR/NET (1) > ED (2)
    final_prediction_raw[raw_prediction_transposed == 2] = 2
    final_prediction_raw[raw_prediction_transposed == 1] = 1
    final_prediction_raw[raw_prediction_transposed == 3] = 3
    # Save Raw Prediction
    save_prediction(final_prediction_raw, sample_paths, raw_dir)

    # ------------------------
    # Step 6: CRF Post-processing
    # ------------------------
    print("\n=== Step 6: CRF Post-processing ===")

    def apply_crf_3d(prob_volume, img_volume, max_schan=20, num_iterations=10, compat_gaussian=3, compat_bilateral=4, sdims=(5, 5, 5)):
        """
        Applies 3D Conditional Random Fields (CRF) for segmentation refinement.

        Args:
            prob_volume (np.ndarray): Probability volume with shape (D, H, W, C).
            img_volume (np.ndarray): Image volume with shape (D, H, W, M).
            max_schan (int, optional): Maximum standard deviation for channel features to prevent extreme values.
            num_iterations (int, optional): Number of CRF inference iterations.
            compat_gaussian (int, optional): Compatibility coefficient for Gaussian pairwise potentials.
            compat_bilateral (int, optional): Compatibility coefficient for bilateral pairwise potentials.
            sdims (tuple, optional): Spatial standard deviations for Gaussian pairwise potentials.

        Returns:
            np.ndarray: Refined segmentation prediction with shape (D, H, W).
        """
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral
        import logging

        logging.basicConfig(level=logging.INFO)

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

            # Prepare spatial features
            feats_gaussian = create_pairwise_gaussian(sdims=sdims, shape=(d, h, w))
            dcrf_model.addPairwiseEnergy(feats_gaussian, compat=compat_gaussian)

            # Prepare appearance features
            schan_values = []
            for modality_idx in range(num_modalities):
                modality_data = img_volume[..., modality_idx]
                intensity_std = np.std(modality_data)
                intensity_std_capped = min(intensity_std, max_schan)
                schan_values.append(intensity_std_capped)

            # Create pairwise bilateral features
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

    try:
        crf_prediction = apply_crf_3d(
            probability_volume, 
            normalize_volume(image_volume), 
            max_schan=10, 
            num_iterations=30, 
            compat_gaussian=5, 
            compat_bilateral=6, 
            sdims=(3, 3, 3)
        )
            
        if crf_prediction is not None:
            print("  Applied CRF post-processing successfully.")
            crf_prediction_transposed = np.transpose(crf_prediction, (1, 2, 0))  # From (D, H, W) to (H, W, D)
            # Initialize final_prediction_crf with shape (240, 240, 155)
            final_prediction_crf = np.zeros(original_shape, dtype=np.uint8)
            # Apply priority-based combining: ET (3) > NCR/NET (1) > ED (2)
            final_prediction_crf[crf_prediction_transposed == 2] = 2
            final_prediction_crf[crf_prediction_transposed == 1] = 1
            final_prediction_crf[crf_prediction_transposed == 3] = 3
            # Save CRF post-processed prediction
            save_prediction(final_prediction_crf, sample_paths, crf_dir)
        else:
            print("  CRF post-processing failed. Using raw prediction instead.")
            crf_prediction_transposed = raw_prediction_transposed
            final_prediction_crf = final_prediction_raw.copy()
    except Exception as e:
        print(f"  Error during CRF post-processing: {e}")
        crf_prediction_transposed = raw_prediction_transposed
        final_prediction_crf = final_prediction_raw.copy()

    # ------------------------
    # Step 7: 3D Morphological Gap Filling
    # ------------------------
    print("\n=== Step 7: 3D Morphological Gap Filling ===")

    def post_process_volume_with_3d_closing(volume, close_iterations=1, remove_small_lesions=False, min_voxel_size=50, structuring_elements=None):
        """
        Applies 3D morphological closing followed by small region removal for post-processing.

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
                2: np.ones((3, 3, 3), dtype=bool)   # WT
            }

        final_volume = np.zeros_like(volume, dtype=np.uint8)

        for class_label, struct_3d in structuring_elements.items():
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

    # Define class-specific structuring elements
    struct_3d_tc = np.ones((3, 3, 3), dtype=bool)  # TC
    struct_3d_wt = np.ones((3, 3, 3), dtype=bool)  # WT (adjusted structuring element size)
    structuring_elements = {
        1: struct_3d_tc,  # TC
        2: struct_3d_wt   # WT
    }

    print("\n=== Step 8: 3D Morphological Gap Filling ===")
    try:
        # Apply morphological closing with class-specific structuring elements
        prediction_after_gap_filling = post_process_volume_with_3d_closing(
            volume=crf_prediction_transposed, 
            close_iterations=1, 
            remove_small_lesions=False, 
            min_voxel_size=50,
            structuring_elements=structuring_elements
        )
        print("  3D morphological gap filling completed successfully.")
        
        # Initialize final_prediction_gap_filling with shape (240, 240, 155)
        final_prediction_gap_filling = np.zeros(original_shape, dtype=np.uint8)
        # Apply priority-based combining: ET (3) > NCR/NET (1) > ED (2)
        final_prediction_gap_filling[prediction_after_gap_filling == 2] = 2
        final_prediction_gap_filling[prediction_after_gap_filling == 1] = 1
        final_prediction_gap_filling[prediction_after_gap_filling == 3] = 3
        # Save Gap Filling prediction
        save_prediction(final_prediction_gap_filling, sample_paths, gap_filling_dir)
    except Exception as e:
        prediction_after_gap_filling = crf_prediction_transposed  # Fallback to CRF prediction
        final_prediction_gap_filling = final_prediction_crf.copy()
        print(f"  3D Morphological Gap Filling failed with error: {e}. Using CRF prediction instead.")

    # ------------------------
    # Step 9: Remove Small False Positives
    # ------------------------
    print("\n=== Step 9: Remove Small False Positives ===")

    def remove_small_false_positives(volume, probability_volume, thresholds, min_voxel_counts, min_mean_probs):
        """
        Removes small false positives based on thresholds and voxel counts.

        Args:
            volume (np.ndarray): Segmentation volume (H, W, D), containing labels {0,1,2,3}.
            probability_volume (np.ndarray): Probability volume with shape (H, W, D, C).
            thresholds (dict): Thresholds for each class label.
            min_voxel_counts (dict): Minimum voxel counts to retain a lesion for each class.
            min_mean_probs (dict): Minimum mean probability to retain a lesion for each class.

        Returns:
            np.ndarray: Post-processed volume with small false positives removed.
        """
        structure = generate_binary_structure(3, 2)  # 26-connectivity
        class_masks_removal = {class_label: (volume == class_label).astype(bool) for class_label in [1, 2, 3]}
        final_prediction_after_removal = np.zeros_like(volume, dtype=np.uint8)

        for class_label in [1, 2, 3]:
            print(f"\n  Processing class {class_label} for small false positive removal.")
            binary_mask = class_masks_removal[class_label]
            labeled_mask, num_features = label(binary_mask, structure=structure)
            print(f"    Found {num_features} connected components for class {class_label}.")

            final_mask = np.zeros_like(binary_mask, dtype=np.uint8)
            prob_map = probability_volume[..., class_label]

            retained = 0
            discarded = 0

            for lesion_idx in range(1, num_features + 1):
                lesion_mask = (labeled_mask == lesion_idx)
                voxel_count = np.sum(lesion_mask)
                mean_prob = np.mean(prob_map[lesion_mask])

                if voxel_count >= min_voxel_counts[class_label] and mean_prob >= min_mean_probs[class_label]:
                    final_mask[lesion_mask] = 1
                    retained += 1
                    print(f"      Retained lesion {lesion_idx}: Voxels={voxel_count}, Mean Prob={mean_prob:.3f}")
                else:
                    discarded += 1
                    print(f"      Discarded lesion {lesion_idx}: Voxels={voxel_count}, Mean Prob={mean_prob:.3f}")

            print(f"    Total retained lesions for class {class_label}: {retained}")
            print(f"    Total discarded lesions for class {class_label}: {discarded}")

            final_prediction_after_removal[final_mask == 1] = class_label

        return final_prediction_after_removal

    # Define thresholds and minimums
    thresholds = {1: 0.45, 2: 0.5, 3: 0.3}
    min_voxel_counts = {1: 50, 2: 50, 3: 50}  # Adjust as needed
    min_mean_probs = {1: 0.9, 2: 0.9, 3: 0.9}  # Adjust as needed


    # min_voxel_counts = {1: 500, 2: 1000, 3: 300}  # Adjust as needed
    # min_mean_probs = {1: 0.8, 2: 0.8, 3: 0.8}  # Adjust as needed

    try:
        final_prediction_after_removal = remove_small_false_positives(
            volume=final_prediction_gap_filling, 
            probability_volume=probability_volume.transpose(1, 2, 0, 3),  # From (D, H, W, C) to (H, W, D, C)
            thresholds=thresholds, 
            min_voxel_counts=min_voxel_counts, 
            min_mean_probs=min_mean_probs
        )
        print("  Removal of small false positives completed successfully.")

        # Initialize final_prediction_after_removal_final with shape (240, 240, 155)
        final_prediction_after_removal_final = np.zeros(original_shape, dtype=np.uint8)
        # Apply priority-based combining: ET (3) > NCR/NET (1) > ED (2)
        final_prediction_after_removal_final[final_prediction_after_removal == 2] = 2
        final_prediction_after_removal_final[final_prediction_after_removal == 1] = 1
        final_prediction_after_removal_final[final_prediction_after_removal == 3] = 3

        # Save Removal prediction
        save_prediction(final_prediction_after_removal_final, sample_paths, removal_dir)
    except Exception as e:
        final_prediction_after_removal = final_prediction_gap_filling  # Fallback to Gap Filling prediction
        final_prediction_after_removal_final = final_prediction_gap_filling.copy()
        print(f"  Removal of small false positives failed with error: {e}. Using Gap Filling prediction instead.")



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
# checkpoint_filepath = "/scratch_net/maus/mrostamian/project/log_multi_v4_2GPU/checkpoints_fold_1/brats_myunet_weights_fold_1_phase_2.weights.h5"
checkpoint_filepath = "/scratch_net/maus/mrostamian/project/maus_second/log_multi_v4_2GPU_softmax_no_adapter/checkpoints_fold_1/brats_myunet_weights_fold_1_phase_1.weights.h5"

input_t1c = tf.keras.Input(shape=(224, 224, 1), name='t1c_input')
input_t1n = tf.keras.Input(shape=(224, 224, 1), name='t1n_input')
input_t2f = tf.keras.Input(shape=(224, 224, 1), name='t2f_input')
input_t2w = tf.keras.Input(shape=(224, 224, 1), name='t2w_input')
model_inputs = [input_t1c, input_t1n, input_t2f, input_t2w]

model = create_multi_encoder_unet(inputs=model_inputs, num_encoders=4, training=False, num_classes=4)
model.load_weights(checkpoint_filepath)
print("Loaded model with best weights.")

output_dir = '/scratch_net/maus_second/mrostamian/project/online_results_no_adapter'
os.makedirs(output_dir, exist_ok=True)  

# Initialize output directories
nifti_output_dir = os.path.join(output_dir, "nifti_predictions")
raw_dir = os.path.join(nifti_output_dir, "raw")
crf_dir = os.path.join(nifti_output_dir, "crf")
gap_filling_dir = os.path.join(nifti_output_dir, "3D_Morphological_Gap_Filling")
removal_dir = os.path.join(nifti_output_dir, "Remove_Small_False_Positives")

for directory in [raw_dir, crf_dir, gap_filling_dir, removal_dir]:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

# Process each validation sample and save predictions
for idx, sample_paths in enumerate(val_data):
    print(f"\nProcessing sample {idx + 1}/{len(val_data)}:")
    print(f"  T1c Path: {sample_paths.get('t1c', 'Missing')}")


    # Process the single volume and get Dice scores (modified to save predictions instead)
    process_single_volume_and_save(model, sample_paths, output_dir)
