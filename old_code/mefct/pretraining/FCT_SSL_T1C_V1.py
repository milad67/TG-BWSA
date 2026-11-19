# changes when running on the cluster
# %% Imports
import os
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


import numpy as np
import nibabel as nib
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
import tensorflow as tf
import logging

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

brats_data = "/cluster/work/cvl/mrostamian/"
brats_data_train = brats_data + '/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
brats_data_test = brats_data + '/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'
brats_data_validation = brats_data + '/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'

img_cols, img_rows, col_channels = 224, 224, 155

n_classes = 4

dropout_rate = 0.3

# batch_size = 4

def visualize_samples(file_paths, input_size, num_samples=50):
    # Ensure necessary functions are available
    def load_nib_image(file_path):
        img = nib.load(file_path).get_fdata()
        img = img.astype(np.float32)
        return img

    def crop_center(img, cropx, cropy):
        y, x = img.shape[:2]
        startx = x // 2 - cropx // 2
        starty = y // 2 - cropy // 2
        return img[starty:starty + cropy, startx:startx + cropx]

    def create_masked_image(img_slice, mask_ratio=0.15, patch_size=16):
        h, w = img_slice.shape[:2]
        mask = np.ones((h, w), dtype=bool)
        non_black_threshold = patch_size * patch_size * 0.1
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = img_slice[i:i + patch_size, j:j + patch_size]
                if np.count_nonzero(patch) > non_black_threshold:
                    if np.random.rand() < mask_ratio:
                        mask[i:i + patch_size, j:j + patch_size] = False
        masked_img_slice = np.copy(img_slice)
        masked_img_slice[~mask] = 0
        return masked_img_slice

    def apply_ssl_augmentation(img_slice, augmentation):
        augmented = augmentation(image=img_slice.squeeze())
        return np.expand_dims(augmented['image'], axis=-1)

    # Instantiate the augmentation
    ssl_augmentation = get_ssl_augmentation()

    # Visualization directory
    visualization_dir = '/cluster/home/mrostamian/log_ssl_t1c/visualize_img'
    os.makedirs(visualization_dir, exist_ok=True)

    samples_visualized = 0

    for file_path in file_paths:
        if samples_visualized >= num_samples:
            break

        # Load the image file
        img = load_nib_image(file_path)
        num_slices = img.shape[2]

        for idx in range(num_slices):
            if samples_visualized >= num_samples:
                break

            img_slice = img[:, :, idx]
            img_slice = np.expand_dims(img_slice, axis=-1)  # Shape: (height, width, 1)

            # Crop the center
            img_slice_cropped = crop_center(img_slice, input_size[0], input_size[1])

            # Skip completely black slices
            if not np.any(img_slice_cropped):
                continue

            # Proceed with normalization and augmentation

            # Normalize the image slice
            img_min = np.min(img_slice_cropped)
            img_max = np.max(img_slice_cropped)
            img_slice_norm = (img_slice_cropped - img_min) / (img_max - img_min + 1e-5)

            # Create a random mask for SSL (Masked Image Modeling)
            masked_img_slice = create_masked_image(img_slice_norm)

            # Generate augmentations for contrastive learning
            seed = np.random.randint(0, 1000)
            np.random.seed(seed)
            view_1 = apply_ssl_augmentation(img_slice_norm, ssl_augmentation)
            np.random.seed(seed)
            view_2 = apply_ssl_augmentation(img_slice_norm, ssl_augmentation)

            # Visualization code
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            axs[0].imshow(img_slice_norm.squeeze(), cmap='gray')
            axs[0].set_title('Original Slice')
            axs[0].axis('off')

            axs[1].imshow(masked_img_slice.squeeze(), cmap='gray')
            axs[1].set_title('Masked Slice')
            axs[1].axis('off')

            axs[2].imshow(view_1.squeeze(), cmap='gray')
            axs[2].set_title('Augmented View 1')
            axs[2].axis('off')

            axs[3].imshow(view_2.squeeze(), cmap='gray')
            axs[3].set_title('Augmented View 2')
            axs[3].axis('off')

            # Save the figure
            save_path = os.path.join(visualization_dir, f'sample_{samples_visualized + 1}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)

            samples_visualized += 1

            if samples_visualized >= num_samples:
                break



from sklearn.model_selection import train_test_split

import os

def load_file_paths(data_dir, image_suffix='t1c'):
    image_paths = []

    # Traverse the directory to gather file paths
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if image_suffix in file:  # Identify image files based on suffix
                img_path = os.path.join(root, file)
                image_paths.append(img_path)

    # Sort paths to maintain consistency
    image_paths.sort()

    return image_paths



# Example usage:
# image_paths = load_file_paths(data_dir, image_suffix='t2f')

def create_dataset_from_file_paths(file_paths, input_size, batch_size, ssl_augmentation, shuffle=False, seed=None):
    # Create a dataset from file paths
    ds = tf.data.Dataset.from_tensor_slices(file_paths)

    # Shuffle the file paths if requested
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths), seed=seed)

    # Map the file processing function
    ds = ds.interleave(
        lambda file_path: process_file(file_path, input_size, ssl_augmentation),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Batch and prefetch
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds



def crop_center_tf(img, cropx, cropy):
    y = tf.shape(img)[0]
    x = tf.shape(img)[1]
    startx = tf.cast(x // 2 - (cropx // 2), tf.int32)
    starty = tf.cast(y // 2 - (cropy // 2), tf.int32)
    return img[starty:starty + cropy, startx:startx + cropx, :]


def get_ssl_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=0, p=0.5),
    ]
    return A.Compose(train_transform)


def create_masked_image_tf(img_slice, mask_ratio=0.15, patch_size=16):
    h = tf.shape(img_slice)[0]
    w = tf.shape(img_slice)[1]

    # Calculate the number of patches
    num_patches_h = tf.cast(tf.math.ceil(h / patch_size), tf.int32)
    num_patches_w = tf.cast(tf.math.ceil(w / patch_size), tf.int32)
    total_patches = num_patches_h * num_patches_w

    # Create a mask of ones
    mask = tf.ones((num_patches_h, num_patches_w), dtype=tf.bool)
    mask = tf.reshape(mask, [-1])

    # Randomly select patches to mask
    num_masked_patches = tf.cast(tf.cast(total_patches, tf.float32) * mask_ratio, tf.int32)
    indices = tf.random.shuffle(tf.range(total_patches))[:num_masked_patches]
    mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indices, 1), tf.zeros_like(indices, dtype=tf.bool))
    mask = tf.reshape(mask, (num_patches_h, num_patches_w))

    # Ensure mask has 3 dimensions
    mask = tf.expand_dims(mask, axis=-1)  # Shape: (num_patches_h, num_patches_w, 1)
    mask = tf.cast(mask, tf.float32)

    # Upsample the mask to the image size
    mask = tf.image.resize(mask, [h, w], method='nearest')
    mask = tf.cast(mask >= 0.5, tf.bool)

    # Apply mask to the image slice
    masked_img_slice = tf.where(mask, img_slice, tf.zeros_like(img_slice))

    return masked_img_slice


# Function to apply augmentations for SSL (e.g., contrastive learning)

def apply_ssl_augmentation_tf(img_slice, seed, ssl_augmentation):
    def aug_fn(image, seed_value):
        np.random.seed(seed_value)
        augmented = ssl_augmentation(image=image.squeeze())
        return np.expand_dims(augmented['image'], axis=-1).astype(np.float32)

    seed_value = tf.cast(seed, tf.int32)
    augmented_img = tf.numpy_function(aug_fn, [img_slice, seed_value], tf.float32)
    augmented_img.set_shape(img_slice.shape)

    return augmented_img


def load_nib_image(file_path):
    file_path = file_path.decode('utf-8')
    img = nib.load(file_path).get_fdata()
    img = img.astype(np.float32)
    return img




# def process_slice(img, idx, input_size, ssl_augmentation):
#     idx = tf.cast(idx, tf.int32)
#     img_slice = img[:, :, idx]
#     img_slice = tf.expand_dims(img_slice, axis=-1)

#     # Extract input dimensions as tensors
#     input_height = tf.constant(input_size[0], dtype=tf.int32)
#     input_width = tf.constant(input_size[1], dtype=tf.int32)

#     # Crop the center
#     img_slice_cropped = crop_center_tf(img_slice, input_height, input_width)

#     # Check if the image slice is completely black
#     condition = tf.reduce_any(tf.cast(img_slice_cropped, tf.bool))

#     def slice_is_not_black():
#         # Normalize the image slice
#         img_min = tf.reduce_min(img_slice_cropped)
#         img_max = tf.reduce_max(img_slice_cropped)
#         img_slice_norm = (img_slice_cropped - img_min) / (img_max - img_min + 1e-5)

#         # Create a random mask for SSL (Masked Image Modeling)
#         masked_img_slice = create_masked_image_tf(img_slice_norm)

#         # Generate augmentations for contrastive learning
#         seed = tf.random.uniform(shape=(), maxval=1000, dtype=tf.int32)
#         view_1 = apply_ssl_augmentation_tf(img_slice_norm, seed, ssl_augmentation)
#         view_2 = apply_ssl_augmentation_tf(img_slice_norm, seed, ssl_augmentation)

#         # Create targets
#         y3 = img_slice_norm  # Original resolution
#         y2 = tf.image.resize(y3, (112, 112), method='nearest')
#         y1 = tf.image.resize(y3, (56, 56), method='nearest')

#         # Prepare the inputs and outputs
#         inputs = (masked_img_slice, view_1, view_2)
#         outputs = {
#             'pred1': y1,
#             'pred2': y2,
#             'final': y3,
#             'projection_view1': tf.zeros((64,), dtype=tf.float32),
#             'projection_view2': tf.zeros((64,), dtype=tf.float32)
#         }

#         is_valid = tf.constant(1, dtype=tf.int32)
#         return inputs, outputs, is_valid

#     def slice_is_black():
#         # Return zero tensors
#         zero_input = tf.zeros((input_height, input_width, 1), dtype=tf.float32)
#         zero_pred1 = tf.zeros((56, 56, 1), dtype=tf.float32)
#         zero_pred2 = tf.zeros((112, 112, 1), dtype=tf.float32)
#         zero_final = tf.zeros((input_height, input_width, 1), dtype=tf.float32)

#         inputs = (zero_input, zero_input, zero_input)
#         outputs = {
#             'pred1': zero_pred1,
#             'pred2': zero_pred2,
#             'final': zero_final,
#             'projection_view1': tf.zeros((64,), dtype=tf.float32),
#             'projection_view2': tf.zeros((64,), dtype=tf.float32)
#         }

#         is_valid = tf.constant(0, dtype=tf.int32)
#         return inputs, outputs, is_valid

#     inputs, outputs, is_valid = tf.cond(condition, slice_is_not_black, slice_is_black)
#     return inputs, outputs, is_valid





# def process_file(file_path, input_size, ssl_augmentation):
#     # Load the image file
#     img = tf.numpy_function(load_nib_image, [file_path], tf.float32)
#     img.set_shape([None, None, None])  # Update shape according to your data

#     # Get the number of slices
#     num_slices = tf.shape(img)[2]

#     # Create a dataset of slice indices
#     slices_dataset = tf.data.Dataset.from_tensor_slices(tf.range(num_slices))

#     # Map the slice processing function
#     slices_dataset = slices_dataset.map(
#         lambda idx: process_slice(img, idx, input_size, ssl_augmentation),
#         num_parallel_calls=tf.data.AUTOTUNE
#     )

#     # Filter out invalid slices
#     slices_dataset = slices_dataset.filter(
#         lambda inputs, outputs, is_valid: tf.equal(is_valid, 1)
#     )

#     # Remove the is_valid from the output
#     slices_dataset = slices_dataset.map(
#         lambda inputs, outputs, is_valid: (inputs, outputs),
#         num_parallel_calls=tf.data.AUTOTUNE
#     )

#     return slices_dataset



def process_slice(img, idx, input_size, ssl_augmentation):
    idx = tf.cast(idx, tf.int32)
    img_slice = img[:, :, idx]
    img_slice = tf.expand_dims(img_slice, axis=-1)

    # Extract input dimensions as tensors
    input_height = tf.constant(input_size[0], dtype=tf.int32)
    input_width = tf.constant(input_size[1], dtype=tf.int32)

    # Crop the center
    img_slice_cropped = crop_center_tf(img_slice, input_height, input_width)

    # Check if the image slice is completely black
    condition = tf.reduce_any(tf.cast(img_slice_cropped, tf.bool))

    def slice_is_not_black():
        # Normalize the image slice
        img_min = tf.reduce_min(img_slice_cropped)
        img_max = tf.reduce_max(img_slice_cropped)
        img_slice_norm = (img_slice_cropped - img_min) / (img_max - img_min + 1e-5)

        # Create a random mask for SSL (Masked Image Modeling)
        masked_img_slice = create_masked_image_tf(img_slice_norm)

        # Generate augmentations for contrastive learning
        seed = tf.random.uniform(shape=(), maxval=1000, dtype=tf.int32)
        view_1 = apply_ssl_augmentation_tf(img_slice_norm, seed, ssl_augmentation)
        view_2 = apply_ssl_augmentation_tf(img_slice_norm, seed, ssl_augmentation)

        # Create targets
        y3 = img_slice_norm  # Original resolution
        y2 = tf.image.resize(y3, (112, 112), method='nearest')
        y1 = tf.image.resize(y3, (56, 56), method='nearest')

        # Prepare the inputs and outputs
        inputs = (masked_img_slice, view_1, view_2)
        outputs = {
            'pred1': y1,
            'pred2': y2,
            'final': y3,
            'projection_view1': tf.zeros((64,), dtype=tf.float32),
            'projection_view2': tf.zeros((64,), dtype=tf.float32)
        }

        is_valid = tf.constant(1, dtype=tf.int32)
        return inputs, outputs, is_valid

    def slice_is_black():
        # Return zero tensors
        zero_input = tf.zeros((input_height, input_width, 1), dtype=tf.float32)
        zero_pred1 = tf.zeros((56, 56, 1), dtype=tf.float32)
        zero_pred2 = tf.zeros((112, 112, 1), dtype=tf.float32)
        zero_final = tf.zeros((input_height, input_width, 1), dtype=tf.float32)

        inputs = (zero_input, zero_input, zero_input)
        outputs = {
            'pred1': zero_pred1,
            'pred2': zero_pred2,
            'final': zero_final,
            'projection_view1': tf.zeros((64,), dtype=tf.float32),
            'projection_view2': tf.zeros((64,), dtype=tf.float32)
        }

        is_valid = tf.constant(0, dtype=tf.int32)
        return inputs, outputs, is_valid

    inputs, outputs, is_valid = tf.cond(
        condition,
        lambda: slice_is_not_black(),
        lambda: slice_is_black()
    )

    return inputs, outputs, is_valid




def process_file(file_path, input_size, ssl_augmentation):
    # Load the image file
    img = tf.numpy_function(load_nib_image, [file_path], tf.float32)
    img.set_shape([None, None, None])  # Update shape according to your data

    # Get the number of slices
    num_slices = tf.shape(img)[2]

    # Create a dataset of slice indices
    slices_dataset = tf.data.Dataset.from_tensor_slices(tf.range(num_slices))

    # Map the slice processing function
    slices_dataset = slices_dataset.map(
        lambda idx: process_slice(img, idx, input_size, ssl_augmentation),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Filter out invalid slices
    slices_dataset = slices_dataset.filter(
        lambda inputs, outputs, is_valid: tf.equal(is_valid, 1)
    )

    # Remove the is_valid from the output
    slices_dataset = slices_dataset.map(
        lambda inputs, outputs, is_valid: (inputs, outputs),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return slices_dataset




from tensorflow.keras.layers import Layer, Dense, Conv2D, Dropout, MultiHeadAttention, BatchNormalization, \
DepthwiseConv2D, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow import Tensor, divide, concat, random, split, reshape, transpose, float32
from typing import List, Union, Iterable


def myunet(input_shapes=None, training=True):
    # Model configuration parameters
    if input_shapes is None:
        input_shapes = {
            'masked_images': (224, 224, 1),
            'view1': (224, 224, 1),
            'view2': (224, 224, 1),
        }
    filters = [32, 64, 128, 256, 512, 256, 128, 64, 32]
    att_heads = [2] * len(filters)
    blocks = len(filters)
    stochastic_depth_rate = 0.0

    dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]


    # Initializer, activation, and dropout rate
    initializer = 'he_normal'
    act = 'relu'
    drp_out = 0.3

    # Utility classes and functions within myunet
    # ... [Include StochasticDepth, wide_focus, Attention, att, add_projection_head]
    # [Ensure all these functions are properly defined as in previous answers, adjusted if necessary]

    class StochasticDepth(layers.Layer):
        def __init__(self, drop_prob=0.0, **kwargs):
            super(StochasticDepth, self).__init__(**kwargs)
            self.drop_prob = drop_prob

        def call(self, x, training=None):
            if training:
                keep_prob = 1 - self.drop_prob
                random_tensor = keep_prob + tf.random.uniform(tf.shape(x), 0, 1)
                random_tensor = tf.floor(random_tensor)
                output = tf.multiply(x, random_tensor) / keep_prob
                return output
            else:
                return x

    def wide_focus(x, filters, dropout_rate):
        x1 = layers.Conv2D(filters, 3, padding='same', activation=tf.nn.gelu)(x)
        x1 = layers.Dropout(dropout_rate)(x1)
        x2 = layers.Conv2D(filters, 3, padding='same', dilation_rate=2, activation=tf.nn.gelu)(x)
        x2 = layers.Dropout(dropout_rate)(x2)
        x3 = layers.Conv2D(filters, 3, padding='same', dilation_rate=3, activation=tf.nn.gelu)(x)
        x3 = layers.Dropout(dropout_rate)(x3)
        added = layers.Add()([x1, x2, x3])
        x_out = layers.Conv2D(filters, 3, padding='same', activation=tf.nn.gelu)(added)
        x_out = layers.Dropout(dropout_rate)(x_out)
        return x_out

    class Attention(Layer):
            def __init__(self,
                        dim_out,
                        num_heads,
                        proj_drop=0.0,
                        kernel_size=3,
                        stride_kv=1,
                        stride_q=1,
                        padding_kv="same",
                        padding_q="same",
                        attention_bias=True):
                super().__init__()
                self.stride_kv = stride_kv
                self.stride_q = stride_q
                self.dim = dim_out
                self.num_heads = num_heads

                self.conv_proj_q = self._build_projection(kernel_size, stride_q, padding_q)
                self.conv_proj_k = self._build_projection(kernel_size, stride_kv, padding_kv)
                self.conv_proj_v = self._build_projection(kernel_size, stride_kv, padding_kv)

                self.attention = MultiHeadAttention(self.num_heads, dim_out, use_bias=attention_bias)
                self.proj_drop = Dropout(proj_drop)


            @staticmethod
            def _build_projection(kernel_size, stride, padding):
                proj = Sequential([DepthwiseConv2D(kernel_size, padding=padding, strides=stride, use_bias=False),
                    layers.LayerNormalization(),
                ])
                return proj


            def call_conv(self, x, h, w):
                q = self.conv_proj_q(x)
                k = self.conv_proj_k(x)
                v = self.conv_proj_v(x)
                return q, k, v

            def call(self, inputs, mask=None, training=training, h=1, w=1):
                x = inputs
                q, k, v = self.call_conv(x, h, w)
                x = self.attention(q, v, key=k)
                if training:
                    x = self.proj_drop(x)
                return x

    def att(x_in,
            num_heads,
            dpr,
            proj_drop=0.0,
            attention_bias=True,
            padding_q="same",
            padding_kv="same",
            stride_kv=2,
            stride_q=1, training=True):

        b, h, w, c = x_in.shape

        attention_output = Attention(dim_out=c,
                                      num_heads=num_heads,
                                      proj_drop=proj_drop,
                                      attention_bias=attention_bias,
                                      padding_q=padding_q,
                                      padding_kv=padding_kv,
                                      stride_kv=stride_kv,
                                      stride_q=stride_q,
                                      )(x_in, h=h, w=w, training=training, mask=None)


        # Skip connection 1.
        attention_output = StochasticDepth(dpr)(attention_output)
        attention_output = Conv2D(x_in.shape[-1], 3, 1, padding="same", activation="relu")(attention_output)
        x2 = layers.Add()([attention_output, x_in])
        x3 = layers.LayerNormalization(epsilon=1e-5)(x2)
        x3 = wide_focus(x3, filters=c, dropout_rate=0.0)
        # Skip connection 2.
        x3 = StochasticDepth(dpr)(x3)
        x3 = layers.Add()([x3, x2])

        return x3


    def add_projection_head(x, name=None):
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation=None, name=name)(x)  # No activation for the output layer
        return x


    # Define inputs
    masked_images_input_shape = input_shapes['masked_images']
    view1_input_shape = input_shapes['view1']
    view2_input_shape = input_shapes['view2']

    masked_images_input = tf.keras.Input(shape=masked_images_input_shape, name='masked_images_input')
    view1_input = tf.keras.Input(shape=view1_input_shape, name='view1_input')
    view2_input = tf.keras.Input(shape=view2_input_shape, name='view2_input')

    # Preprocess inputs (if needed)
    inputs = masked_images_input  # For simplicity, we can rename masked_images_input to inputs

    # Build downsampled versions of the inputs
    scale_img_2 = layers.AveragePooling2D(2, 2)(inputs)
    scale_img_3 = layers.AveragePooling2D(2, 2)(scale_img_2)
    scale_img_4 = layers.AveragePooling2D(2, 2)(scale_img_3)

    # Build encoder
    x = inputs
    skip_connections = []

    # Block 1
    x1 = layers.LayerNormalization(epsilon=1e-5)(x)
    x1 = layers.Lambda(lambda x: tf.expand_dims(x[:, :, :, -1], -1))(x1)
    x1 = layers.Conv2D(filters[0], 3, padding='same', activation=act, kernel_initializer=initializer)(x1)
    x1 = layers.Conv2D(filters[0], 3, padding='same', activation=act, kernel_initializer=initializer)(x1)
    x1 = layers.Dropout(drp_out)(x1)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x1 = att(x1, att_heads[0], dpr[0], training=training)
    skip1 = x1  # Save for decoder
    #print(f"Encoder Block 1 output shape: {x1.shape}")

    # Block 2
    x2 = layers.LayerNormalization(epsilon=1e-5)(x1)
    scale_conv2 = layers.Conv2D(filters[0], 3, padding='same', activation=act)(scale_img_2)
    x2 = layers.concatenate([scale_conv2, x2], axis=-1)
    x2 = layers.Conv2D(filters[1], 3, padding='same', activation=act, kernel_initializer=initializer)(x2)
    x2 = layers.Conv2D(filters[1], 3, padding='same', activation=act, kernel_initializer=initializer)(x2)
    x2 = layers.Dropout(drp_out)(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x2 = att(x2, att_heads[1], dpr[1], training=training)
    skip2 = x2
    #print(f"Encoder Block 2 output shape: {x2.shape}")

    # Block 3
    x3 = layers.LayerNormalization(epsilon=1e-5)(x2)
    scale_conv3 = layers.Conv2D(filters[1], 3, padding='same', activation=act)(scale_img_3)
    x3 = layers.concatenate([scale_conv3, x3], axis=-1)
    x3 = layers.Conv2D(filters[2], 3, padding='same', activation=act, kernel_initializer=initializer)(x3)
    x3 = layers.Conv2D(filters[2], 3, padding='same', activation=act, kernel_initializer=initializer)(x3)
    x3 = layers.Dropout(drp_out)(x3)
    x3 = layers.MaxPooling2D((2, 2))(x3)
    x3 = att(x3, att_heads[2], dpr[2], training=training)
    skip3 = x3
    #print(f"Encoder Block 3 output shape: {x3.shape}")

    # Block 4
    x4 = layers.LayerNormalization(epsilon=1e-5)(x3)
    scale_conv4 = layers.Conv2D(filters[2], 3, padding='same', activation=act)(scale_img_4)
    x4 = layers.concatenate([scale_conv4, x4], axis=-1)
    x4 = layers.Conv2D(filters[3], 3, padding='same', activation=act, kernel_initializer=initializer)(x4)
    x4 = layers.Conv2D(filters[3], 3, padding='same', activation=act, kernel_initializer=initializer)(x4)
    x4 = layers.Dropout(drp_out)(x4)
    x4 = layers.MaxPooling2D((2, 2))(x4)
    x4 = att(x4, att_heads[3], dpr[3], training=training)
    skip4 = x4
    #print(f"Encoder Block 4 output shape: {x4.shape}")

    # Block 5 (Bottleneck)
    x5 = layers.LayerNormalization(epsilon=1e-5)(x4)
    x5 = layers.Conv2D(filters[4], 3, padding='same', activation=act, kernel_initializer=initializer)(x5)
    x5 = layers.Conv2D(filters[4], 3, padding='same', activation=act, kernel_initializer=initializer)(x5)
    x5 = layers.Dropout(drp_out)(x5)
    x5 = layers.MaxPooling2D((2, 2))(x5)
    x5 = att(x5, att_heads[4], dpr[4], training=training)
    encoder_output = x5
    #print(f"Bottleneck output shape: {x5.shape}")

    # Build lightweight decoder
    # Block 6
    x6 = layers.LayerNormalization(epsilon=1e-5)(x5)
    x6 = layers.UpSampling2D(size=(2, 2))(x6)
    x6 = layers.Conv2D(filters[5] // 2, 3, padding='same', activation=act, kernel_initializer=initializer)(x6)
    x6 = layers.concatenate([skip4, x6], axis=-1)
    x6 = layers.Conv2D(filters[5] // 2, 3, padding='same', activation=act, kernel_initializer=initializer)(x6)

    # Block 7
    x7 = layers.LayerNormalization(epsilon=1e-5)(x6)
    x7 = layers.UpSampling2D(size=(2, 2))(x7)
    x7 = layers.Conv2D(filters[6] // 2, 3, padding='same', activation=act, kernel_initializer=initializer)(x7)
    x7 = layers.concatenate([skip3, x7], axis=-1)
    x7 = layers.Conv2D(filters[6] // 2, 3, padding='same', activation=act, kernel_initializer=initializer)(x7)
    skip7 = x7  # Save for output

    # Block 8
    x8 = layers.LayerNormalization(epsilon=1e-5)(x7)
    x8 = layers.UpSampling2D(size=(2, 2))(x8)
    x8 = layers.Conv2D(filters[7] // 2, 3, padding='same', activation=act, kernel_initializer=initializer)(x8)
    x8 = layers.concatenate([skip2, x8], axis=-1)
    x8 = layers.Conv2D(filters[7] // 2, 3, padding='same', activation=act, kernel_initializer=initializer)(x8)
    skip8 = x8  # Save for output

    # Block 9
    x9 = layers.LayerNormalization(epsilon=1e-5)(x8)
    x9 = layers.UpSampling2D(size=(2, 2))(x9)
    x9 = layers.Conv2D(filters[8] // 2, 3, padding='same', activation=act, kernel_initializer=initializer)(x9)
    x9 = layers.concatenate([skip1, x9], axis=-1)
    x9 = layers.Conv2D(filters[8] // 2, 3, padding='same', activation=act, kernel_initializer=initializer)(x9)
    skip9 = x9  # Save for output

    # Create outputs from skips
    # Output 1
    out7 = layers.UpSampling2D(size=(2, 2))(skip7)
    out7 = layers.LayerNormalization(epsilon=1e-5)(out7)
    out7 = layers.Conv2D(filters[6] // 2, 3, padding='same', activation=act, kernel_initializer=initializer)(out7)
    out7 = layers.Conv2D(filters[6] // 2, 3, padding='same', activation=act, kernel_initializer=initializer)(out7)
    out7 = layers.Conv2D(1, (1, 1), activation='sigmoid', name='pred1')(out7)

    # Output 2
    out8 = layers.UpSampling2D(size=(2, 2))(skip8)
    out8 = layers.LayerNormalization(epsilon=1e-5)(out8)
    out8 = layers.Conv2D(filters[7] // 2, 3, padding='same', activation=act, kernel_initializer=initializer)(out8)
    out8 = layers.Conv2D(filters[7] // 2, 3, padding='same', activation=act, kernel_initializer=initializer)(out8)
    out8 = layers.Conv2D(1, (1, 1), activation='sigmoid', name='pred2')(out8)

    # Output 3
    out9 = layers.UpSampling2D(size=(2, 2))(skip9)
    out9 = layers.LayerNormalization(epsilon=1e-5)(out9)
    out9 = layers.Conv2D(filters[8] // 2, 3, padding='same', activation=act, kernel_initializer=initializer)(out9)
    out9 = layers.Conv2D(filters[8] // 2, 3, padding='same', activation=act, kernel_initializer=initializer)(out9)
    out9 = layers.Conv2D(1, (1, 1), activation='sigmoid', name='final')(out9)

    # Build the encoder model for augmented views (shared encoder)
    encoder_model = keras.Model(inputs=inputs, outputs=encoder_output, name="encoder")

    # Build the unified model
    # For masked_images_input, we use 'inputs'
    # For view1_input and view2_input, we need to process them through the encoder
    # Pass augmented views through the shared encoder
    encoder_output_view1 = encoder_model(view1_input)
    encoder_output_view2 = encoder_model(view2_input)

    # Global Average Pooling
    encoder_output_view1_pooled = layers.GlobalAveragePooling2D()(encoder_output_view1)
    encoder_output_view2_pooled = layers.GlobalAveragePooling2D()(encoder_output_view2)

    # Projection head
    projection_view1 = add_projection_head(encoder_output_view1_pooled, name='projection_view1')
    projection_view2 = add_projection_head(encoder_output_view2_pooled, name='projection_view2')

    # Define model inputs and outputs
    model_inputs = [masked_images_input, view1_input, view2_input]
    model_outputs = {
        'pred1': out7,
        'pred2': out8,
        'final': out9,
        'projection_view1': projection_view1,
        'projection_view2': projection_view2,
    }

    # Create the unified model
    unified_model = tf.keras.Model(inputs=model_inputs, outputs=model_outputs, name='unified_model')

    return unified_model, encoder_model


# Function to provide dummy loss for projections
def dummy_loss(y_true, y_pred):
    return tf.reduce_mean(tf.zeros_like(y_pred))

def mim_loss_function(original, reconstructed):
    return tf.reduce_mean(tf.square(original - reconstructed))

# def ssim_loss_function(original, reconstructed):
#     return 1 - tf.reduce_mean(tf.image.ssim(original, reconstructed, max_val=1.0))

def ssim_loss_function(original, reconstructed):
    """
    Computes the SSIM-based loss with optional edge regularization.
    Returns loss per example in the batch.
    """
    # Compute per-example SSIM loss
    ssim_loss = 1 - tf.image.ssim(original, reconstructed, max_val=1.0)  # Shape: (batch_size,)

    # Compute per-example Edge Loss
    sobel_reconstructed = tf.image.sobel_edges(reconstructed)  # Shape: (batch_size, height, width, 1, 2)
    sobel_original = tf.image.sobel_edges(original)
    edge_diff = tf.abs(sobel_original - sobel_reconstructed)

    # Reduce over spatial and edge dimensions, keep batch dimension
    edge_loss = tf.reduce_mean(edge_diff, axis=[1, 2, 3, 4])  # Shape: (batch_size,)

    # Combine SSIM and Edge Loss per example
    combined_loss = ssim_loss + 0.1 * edge_loss  # Shape: (batch_size,)

    return combined_loss  # Returns tensor of shape (batch_size,)


def contrastive_loss_function(z_i, z_j, temperature=0.5):
    # L2 normalization
    z_i = tf.math.l2_normalize(z_i, axis=1)  # Shape: (local_batch_size, embedding_dim)
    z_j = tf.math.l2_normalize(z_j, axis=1)

    # Gather embeddings from all replicas
    z_i_large = concat_all_gather(z_i)  # Shape: (global_batch_size, embedding_dim)
    z_j_large = concat_all_gather(z_j)

    # Compute logits
    logits_ab = tf.matmul(z_i, z_j_large, transpose_b=True) / temperature  # Shape: (local_batch_size, global_batch_size)
    logits_ba = tf.matmul(z_j, z_i_large, transpose_b=True) / temperature

    # Labels are indices from 0 to local_batch_size - 1
    batch_size = tf.shape(z_i)[0]
    labels = tf.range(batch_size)

    # Cross entropy loss between positive pairs and all other pairs in the batch
    loss_a = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits_ab)  # Shape: (local_batch_size,)
    loss_b = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits_ba)

    # Combine losses
    loss = (loss_a + loss_b) / 2.0  # Shape: (local_batch_size,)

    return loss


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensor.
    The input tensor is concatenated along the first dimension across all replicas.
    """
    # Get the current strategy scope
    gathered = tf.distribute.get_replica_context().all_gather(tensor, axis=0)
    return gathered


class WarmUpLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, total_steps, target_lr, initial_lr=0.0, verbose=0):
        super(WarmUpLearningRateScheduler, self).__init__()
        self.total_steps = total_steps
        self.target_lr = target_lr
        self.initial_lr = initial_lr
        self.verbose = verbose

    def get_current_lr(self, step):
        if step < self.total_steps:
            return self.initial_lr + (self.target_lr - self.initial_lr) * (step / self.total_steps)
        else:
            return self.target_lr

    def on_batch_begin(self, batch, logs=None):
        lr = self.get_current_lr(self.model.optimizer.iterations)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print(f'Batch {batch+1}: LearningRateScheduler setting learning rate to {lr}.')


import pandas as pd
logging.basicConfig(level=logging.INFO)
class SaveSamplePredictions(Callback):
    def __init__(
        self,
        validation_data: tf.data.Dataset,
        save_dir: str,
        num_samples: int = 30,
        fold: int = 1,
        threshold: float = 0.5
    ):
        super(SaveSamplePredictions, self).__init__()
        self.validation_data = validation_data
        self.save_dir = save_dir
        self.fold = fold
        self.num_samples = num_samples
        self.threshold = threshold

        # Create a directory for the current fold if it doesn't exist
        self.fold_dir = os.path.join(self.save_dir, f'fold_{self.fold}')
        os.makedirs(self.fold_dir, exist_ok=True)

        # A placeholder for cosine similarity values collected during the epoch
        self.cosine_similarities = []

    def on_epoch_end(self, epoch: int, logs=None):
        data_iterator = iter(self.validation_data)
        saved_samples = 0
        iter_resets = 0
        individual_cosine_sims = []

        while saved_samples < self.num_samples and iter_resets < 5:
            try:
                val_inputs, val_targets = next(data_iterator)
                masked_images, view_1, view_2 = val_inputs
                y3 = val_targets['final']

                predictions = self.model.predict([masked_images, view_1, view_2])
                pred_masks = predictions.get('final', None)
                projection_view1 = predictions.get('projection_view1', None)
                projection_view2 = predictions.get('projection_view2', None)

                if pred_masks is None or projection_view1 is None or projection_view2 is None:
                    logging.error("The model prediction does not contain 'final' or projection outputs.")
                    return

                batch_cosine_sim = self.compute_cosine_similarity(projection_view1, projection_view2)
                self.cosine_similarities.extend(batch_cosine_sim)
                individual_cosine_sims.extend(batch_cosine_sim.tolist())

                logging.info(f"Batch cosine similarities: {batch_cosine_sim}")

                for i in range(len(masked_images)):
                    img = masked_images[i, :, :, 0]
                    true_mask = y3[i, :, :, :]
                    pred_mask = pred_masks[i, :, :, :]
                    sample_cosine_sim = batch_cosine_sim[i]

                    # Pass the full list of cosine similarities to save_sample for histogram
                    self.save_sample(
                        epoch,
                        i,
                        img,
                        true_mask,
                        pred_mask,
                        sample_cosine_sim,
                        self.cosine_similarities,  # Full list for histogram
                        sample_type="sample",
                        sample_index=saved_samples + 1
                    )
                    saved_samples += 1

                    if saved_samples >= self.num_samples:
                        break

            except StopIteration:
                logging.info("End of validation data sequence. Resetting iterator...")
                data_iterator = iter(self.validation_data)
                iter_resets += 1

            except Exception as e:
                logging.error(f"An error occurred during sample prediction saving: {e}")
                continue

        # Save cosine similarities to a CSV
        cosine_sim_file = os.path.join(self.fold_dir, f'epoch_{epoch + 1}', 'cosine_similarities.csv')
        os.makedirs(os.path.dirname(cosine_sim_file), exist_ok=True)

        pd.DataFrame({'SampleIndex': range(1, len(individual_cosine_sims) + 1),
                      'CosineSimilarity': individual_cosine_sims}).to_csv(cosine_sim_file, index=False)

        logging.info(f"Saved cosine similarity file at {cosine_sim_file}")

    def compute_cosine_similarity(self, a, b):
        dot_product = np.sum(a * b, axis=-1)
        norm_a = np.linalg.norm(a, axis=-1)
        norm_b = np.linalg.norm(b, axis=-1)
        return dot_product / (norm_a * norm_b)

    def save_sample(self, epoch: int, i: int, img, true_mask, pred_mask, cosine_sim, cosine_sims, sample_type: str, sample_index: int):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 4 subplots: Original, GT Mask, Pred Mask, Histogram
        fig.suptitle(f"Epoch {epoch + 1} - {sample_type.capitalize()} Sample {sample_index} - Cosine Similarity: {cosine_sim:.4f}")

        if isinstance(img, tf.Tensor):
            img = img.numpy()
        if isinstance(true_mask, tf.Tensor):
            true_mask = true_mask.numpy()
        if isinstance(pred_mask, tf.Tensor):
            pred_mask = pred_mask.numpy()

        img, true_mask, pred_mask = map(np.squeeze, [img, true_mask, pred_mask])

        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(true_mask, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')

        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')

        # Plot the histogram of cosine similarities
        axes[3].hist(cosine_sims, bins=20, color='blue', alpha=0.7, edgecolor='black')
        axes[3].set_title('Cosine Similarity Distribution')
        axes[3].set_xlabel('Cosine Similarity')
        axes[3].set_ylabel('Frequency')
        axes[3].grid(axis='y', alpha=0.75)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fold_epoch_dir = os.path.join(self.fold_dir, f'epoch_{epoch + 1}', sample_type)
        os.makedirs(fold_epoch_dir, exist_ok=True)
        save_path = os.path.join(fold_epoch_dir, f'sample_{sample_index}.png')
        logging.info(f"Saving figure to: {save_path}")
        fig.savefig(save_path)
        plt.close(fig)


def count_non_empty_frames(data):
    total_non_empty_frames = 0

    for img_path in data:
        # Load the image using nibabel (assuming .nii format)
        img = nib.load(img_path).get_fdata()  # Load the image data
        
        # Iterate over each frame (slice) in the image
        for frame_idx in range(img.shape[2]):  # Assuming the 3rd dimension is the frame axis
            img_slice = img[:, :, frame_idx]

            # Count frames with non-zero values
            if np.any(img_slice):  # If the slice contains any non-zero values, count it
                total_non_empty_frames += 1

    return total_non_empty_frames


@tf.function
def train_step(inputs, global_batch_size):
    def step_fn(inputs):
        inputs_batch, targets_batch = inputs
        masked_images, view_1, view_2 = inputs_batch

        y1 = targets_batch['pred1']  # Shape: (batch_size, H, W, C)
        y2 = targets_batch['pred2']
        y3 = targets_batch['final']

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = unified_model([masked_images, view_1, view_2], training=True)
            reconstructed_pred1 = outputs['pred1']  # Shape: (batch_size, H, W, C)
            reconstructed_pred2 = outputs['pred2']
            reconstructed_final = outputs['final']
            projection_view1 = outputs['projection_view1']  # Shape: (batch_size, embedding_dim)
            projection_view2 = outputs['projection_view2']

            # Compute per-example SSIM losses
            ssim_loss_pred1 = ssim_loss_function(y1, reconstructed_pred1)  # Shape: (batch_size,)
            ssim_loss_pred2 = ssim_loss_function(y2, reconstructed_pred2)
            ssim_loss_final = ssim_loss_function(y3, reconstructed_final)
            total_ssim_loss = (
                0.1 * ssim_loss_pred1 +
                0.3 * ssim_loss_pred2 +
                0.6 * ssim_loss_final
            )  # Shape: (batch_size,)

            # Compute contrastive loss
            contrastive_loss = contrastive_loss_function(projection_view1, projection_view2)  # Shape: (batch_size,)

            # Total loss per example
            total_loss = 0.5 * total_ssim_loss + 0.5 * contrastive_loss  # Shape: (batch_size,)

            # Compute scaled loss
            scaled_loss = tf.nn.compute_average_loss(
                total_loss, 
                global_batch_size=global_batch_size
            )

        # Compute gradients
        grads = tape.gradient(scaled_loss, unified_model.trainable_variables)

        # Gradient clipping
        clipped_grads = [
            tf.clip_by_value(grad, -1.0, 1.0) if grad is not None else None 
            for grad in grads
        ]

        # Apply gradients
        opt.apply_gradients(zip(clipped_grads, unified_model.trainable_variables))

        return scaled_loss

    # Run the step function on each replica
    per_replica_losses = strategy.run(step_fn, args=(inputs,))

    # Aggregate the per-replica losses
    mean_loss = strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
    )
    return mean_loss


@tf.function
def val_step(inputs, global_batch_size):
    def step_fn(inputs):
        inputs_batch, targets_batch = inputs
        masked_images, view_1, view_2 = inputs_batch

        y1 = targets_batch['pred1']
        y2 = targets_batch['pred2']
        y3 = targets_batch['final']

        # Forward pass
        outputs = unified_model([masked_images, view_1, view_2], training=False)
        reconstructed_pred1 = outputs['pred1']
        reconstructed_pred2 = outputs['pred2']
        reconstructed_final = outputs['final']
        projection_view1 = outputs['projection_view1']
        projection_view2 = outputs['projection_view2']

        # Compute per-example SSIM losses
        ssim_loss_pred1 = ssim_loss_function(y1, reconstructed_pred1)  # Shape: (batch_size,)
        ssim_loss_pred2 = ssim_loss_function(y2, reconstructed_pred2)
        ssim_loss_final = ssim_loss_function(y3, reconstructed_final)
        total_ssim_loss = (
            0.1 * ssim_loss_pred1 +
            0.3 * ssim_loss_pred2 +
            0.6 * ssim_loss_final
        )  # Shape: (batch_size,)

        # Compute contrastive loss
        contrastive_loss = contrastive_loss_function(projection_view1, projection_view2)  # Shape: (batch_size,)

        # Total loss per example
        total_loss = 0.5 * total_ssim_loss + 0.5 * contrastive_loss  # Shape: (batch_size,)

        # Compute scaled loss
        scaled_loss = tf.nn.compute_average_loss(
            total_loss, 
            global_batch_size=global_batch_size
        )

        return scaled_loss

    # Run the step function on each replica
    per_replica_losses = strategy.run(step_fn, args=(inputs,))

    # Aggregate the per-replica losses
    mean_loss = strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
    )
    return mean_loss



import os
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Import necessary functions and classes (assuming these are defined in codebase)
# from your_module import (
#     load_file_paths,
#     create_dataset_from_file_paths,
#     ssl_augmentation,
#     get_ssl_augmentation,
#     myunet,
#     train_step,
#     val_step,
#     WarmUpLearningRateScheduler,
#     SaveSamplePredictions
# )

# If the above functions/classes are not imported, define them here or ensure they are available

# Distributed Strategy
strategy = tf.distribute.MirroredStrategy()

# Set random seeds for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)


# Load image file paths from the directory
image_paths = load_file_paths(brats_data_validation)
data_pairs = list(image_paths)

# Define KFold with a fixed random_state
kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

# Directory to save logs and weights
save_dir = '/cluster/home/mrostamian/log_ssl_t1c'
os.makedirs(save_dir, exist_ok=True)

# Initialize custom metrics
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

# Define model hyperparameters
input_shapes = {
    'masked_images': (224, 224, 1),
    'view1': (224, 224, 1),
    'view2': (224, 224, 1),
}
batch_size = 16
global_batch_size = batch_size * strategy.num_replicas_in_sync
warmup_run_epochs = 1
normal_run_epochs = 5

# Instantiate the SSL augmentation
ssl_augmentation = get_ssl_augmentation()

# K-Fold Cross-Validation Loop
for fold, (train_index, val_index) in enumerate(kf.split(data_pairs)):
    print(f"Training fold {fold + 1}/{kf.n_splits}")

    # Clear session and reset states for each fold
    tf.keras.backend.clear_session()

    # Prepare training and validation data
    train_file_paths = [data_pairs[i] for i in train_index]
    val_file_paths = [data_pairs[i] for i in val_index]


    # Visualize random samples from the training dataset
    visualize_samples(train_file_paths, input_size=input_shapes['masked_images'][:2], num_samples=50)


    # Ensure reproducibility in data generators
    generator_random_seed = random_seed + fold  # Different seed for each fold

    # Set up TensorBoard logs for this fold
    fold_log_dir = os.path.join(save_dir, f"tensorboard_logs/fold_{fold + 1}")
    os.makedirs(fold_log_dir, exist_ok=True)

    train_summary_writer = tf.summary.create_file_writer(os.path.join(fold_log_dir, 'train'))
    val_summary_writer = tf.summary.create_file_writer(os.path.join(fold_log_dir, 'val'))

    # Calculate the number of valid frames for each epoch
    # valid_train_frames = count_non_empty_frames(train_file_paths)
    # valid_validation_frames = count_non_empty_frames(val_file_paths)
    valid_train_frames = 1000
    valid_validation_frames = 1000

    # Log the number of valid frames
    file_path = os.path.join(save_dir, 'valid_frames_data.txt')
    message = f'Fold {fold + 1}: The number of valid train data is {valid_train_frames} and the number of valid validation data is {valid_validation_frames}\n'
    with open(file_path, 'a') as file:
        file.write(message)

    # Calculate steps per epoch
    steps_per_epoch = max(int(np.floor(valid_train_frames / batch_size)), 1)
    validation_steps = max(int(np.floor(valid_validation_frames / batch_size)), 1)

    # Instantiate the learning rate scheduler inside the fold loop
    total_warmup_steps = warmup_run_epochs * steps_per_epoch
    warm_up_lr = WarmUpLearningRateScheduler(
        total_steps=total_warmup_steps,
        target_lr=1e-3,
        initial_lr=1e-4,
        verbose=1
    )

    # Create the datasets
    train_ds = create_dataset_from_file_paths(
        file_paths=train_file_paths,
        input_size=input_shapes['masked_images'][:2],
        batch_size=batch_size,
        ssl_augmentation=ssl_augmentation,
        shuffle=True,
        seed=generator_random_seed
    )

    val_ds = create_dataset_from_file_paths(
        file_paths=val_file_paths,
        input_size=input_shapes['masked_images'][:2],
        batch_size=batch_size,
        ssl_augmentation=ssl_augmentation,
        shuffle=False
    )

    # Adjust datasets for distributed strategy
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_ds = train_ds.with_options(options)
    val_ds = val_ds.with_options(options)

    dist_train_gen = strategy.experimental_distribute_dataset(train_ds)
    dist_val_gen = strategy.experimental_distribute_dataset(val_ds)

    # Model Definition: Using strategy.scope to create and compile the model
    with strategy.scope():
        # Build the model
        unified_model, encoder = myunet(input_shapes=input_shapes, training=True)
        unified_model.summary()

        # Define optimizer
        lr_schedule = 1e-4
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=1e-2, amsgrad=True)

        # Compile the model without freezing any layers
        unified_model.compile(optimizer=opt)

    # Training Loop for Warm-up Phase
    current_step = 0  # Initialize step counter
    for epoch in range(warmup_run_epochs):
        print(f"Warm-up Epoch {epoch + 1}/{warmup_run_epochs}")

        total_loss = 0.0
        num_batches = 0
        train_loss_metric.reset_state()

        train_iterator = iter(dist_train_gen)
        for step in range(steps_per_epoch):
            inputs = next(train_iterator)
            # Adjust learning rate for warm-up phase
            current_lr = warm_up_lr.get_current_lr(current_step)
            opt.learning_rate.assign(current_lr)
            current_step += 1

            loss = train_step(inputs, global_batch_size)
            total_loss += loss.numpy()
            num_batches += 1

            # Update metrics
            train_loss_metric.update_state(loss)

            # Log training metrics to TensorBoard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=epoch * steps_per_epoch + step)

        # Compute average epoch loss
        epoch_loss = total_loss / num_batches
        print(f"Warm-up Epoch {epoch + 1}, Training Loss: {epoch_loss}")

    # Training Loop for Main Training Phase
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(normal_run_epochs):
        print(f"Main Epoch {epoch + 1}/{normal_run_epochs}")

        total_loss = 0.0
        num_batches = 0
        train_loss_metric.reset_state()

        train_iterator = iter(dist_train_gen)
        for step in range(steps_per_epoch):
            inputs = next(train_iterator)
            loss = train_step(inputs, global_batch_size)
            total_loss += loss.numpy()
            num_batches += 1

            # Update metrics
            train_loss_metric.update_state(loss)

            # Log training metrics to TensorBoard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=(warmup_run_epochs * steps_per_epoch) + epoch * steps_per_epoch + step)

        # Compute average epoch loss
        epoch_loss = total_loss / num_batches
        print(f"Main Epoch {epoch + 1}, Training Loss: {epoch_loss}")

        # Validation steps
        val_total_loss = 0.0
        val_num_batches = 0
        val_loss_metric.reset_state()

        val_iterator = iter(dist_val_gen)
        for step in range(validation_steps):
            val_inputs = next(val_iterator)
            val_loss = val_step(val_inputs, global_batch_size)
            val_total_loss += val_loss.numpy()
            val_num_batches += 1

            # Update validation metrics
            val_loss_metric.update_state(val_loss)


        # Assuming `strategy` is your distributed strategy instance
        save_predictions_callback = SaveSamplePredictions(
            validation_data=val_ds,
            save_dir=os.path.join(save_dir, "sample_predictions_t1c"),
            num_samples=30,
            fold=fold + 1,
            threshold=0.5
        )    

        save_predictions_callback.set_model(unified_model)
        save_predictions_callback.on_epoch_end(epoch, logs={'loss': epoch_loss})

        # Compute average validation loss
        val_epoch_loss = val_total_loss / val_num_batches
        print(f"Main Epoch {epoch + 1}, Validation Loss: {val_epoch_loss}")

        # Early stopping logic
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            patience_counter = 0
            # Save best model
            checkpoint_filepath = os.path.join(save_dir, f'self_supervised_best_weights_fold_{fold + 1}.weights.h5')
            unified_model.save_weights(checkpoint_filepath)
            print(f"Best model saved for fold {fold + 1} with validation loss: {best_loss}")
        else:
            patience_counter += 1

        # Check if patience exceeded
        if patience_counter >= 2:  # Same value as `patience` in the EarlyStopping callback
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break

    # End of fold loop






