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


from sklearn.model_selection import train_test_split

import os

def load_file_paths(data_dir, image_suffix='t2f'):

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


def crop_center(img, cropx, cropy):
    y, x = img.shape[:2]
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty+cropy, startx:startx+cropx]

def crop_center_based_on_roi(img, roi_x, roi_y, roi_width, roi_height, crop_size=192):
    """
    Crop a square region of size `crop_size` centered around the ROI.
    
    Parameters:
    - img (ndarray): Input image to crop.
    - roi_x (int): X-coordinate of the ROI top-left corner.
    - roi_y (int): Y-coordinate of the ROI top-left corner.
    - roi_width (int): Width of the ROI.
    - roi_height (int): Height of the ROI.
    - crop_size (int): Desired square crop size (default: 192).
    
    Returns:
    - ndarray: Cropped image of size (crop_size, crop_size).
    """
    # Calculate the center of the ROI
    roi_center_x = roi_x + roi_width // 2
    roi_center_y = roi_y + roi_height // 2

    # Calculate cropping bounds
    startx = max(0, roi_center_x - crop_size // 2)
    starty = max(0, roi_center_y - crop_size // 2)
    endx = startx + crop_size
    endy = starty + crop_size

    # Ensure cropping bounds are within image dimensions
    img_height, img_width = img.shape[:2]
    if endx > img_width:
        startx = img_width - crop_size
        endx = img_width
    if endy > img_height:
        starty = img_height - crop_size
        endy = img_height

    return img[starty:endy, startx:endx]


def get_ssl_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=0, p=0.5),

    ]
    return A.Compose(train_transform)

# Function to generate masked images for Masked Image Modeling
def create_masked_image(img_slice, mask_ratio=0.15, patch_size=16):
    h, w = img_slice.shape[:2]
    mask = np.ones((h, w), dtype=bool)

    # Threshold to define whether a patch belongs to the ROI (avoiding black areas)
    non_black_threshold = patch_size * patch_size * 0.1  # Only mask patches with sufficient non-black pixels

    # Randomly select patches to mask
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img_slice[i:i+patch_size, j:j+patch_size]
            
            # Check if the patch is within ROI (i.e., not a black area)
            if np.count_nonzero(patch) > non_black_threshold:
                if np.random.rand() < mask_ratio:
                    mask[i:i+patch_size, j:j+patch_size] = False

    masked_img_slice = np.copy(img_slice)
    masked_img_slice[~mask] = 0
    return masked_img_slice


# Function to apply augmentations for SSL (e.g., contrastive learning)

def apply_ssl_augmentation(img_slice, augmentation):
    augmented = augmentation(image=img_slice.squeeze())
    return np.expand_dims(augmented['image'], axis=-1)

    

def get_brats_ssl_generator(file_paths, input_size=(224, 224, 1), batch_size=32, single_pass=False):
    def load_images_for_ssl(file_paths):
        images, masked_images, view_1_list, view_2_list = [], [], [], []
        y1_list, y2_list, y3_list = [], [], []  # Dummy targets to match model output
        file_count = 0

        ssl_augmentation = get_ssl_augmentation()

        # Create the visualization directory if it doesn't exist 
        visualization_dir = '/cluster/home/mrostamian/log_ssl_flair/visualize_img' 
        os.makedirs(visualization_dir, exist_ok=True)

        while True:
            for img_path in file_paths:
                img = nib.load(img_path).get_fdata()
                file_count += 1

                for idx in range(img.shape[2]):
                    img_slice = crop_center(img[:, :, idx], input_size[0], input_size[1])
                    img_slice = np.expand_dims(img_slice, axis=-1)

                    # Skip completely black slices
                    if not np.any(img_slice):
                        continue

                    # Create a random mask for SSL (Masked Image Modeling)
                    masked_img_slice = create_masked_image(img_slice)

                    # Normalize the image slice
                    # img_slice = (img_slice - np.mean(img_slice)) / (np.std(img_slice) + 1e-5)
                    img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))

                    # Generate augmentations for contrastive learning
                    seed = np.random.randint(0, 1000) # Ensure same random seed for both views 
                    np.random.seed(seed) 
                    view_1 = apply_ssl_augmentation(img_slice, ssl_augmentation) 
                    np.random.seed(seed) 
                    view_2 = apply_ssl_augmentation(img_slice, ssl_augmentation)

                    # Create dummy targets to match the model's expected outputs
                    y3 = img_slice  # Original resolution (224x224)
                    y2 = tf.image.resize(y3, (112, 112), method='nearest').numpy()
                    y1 = tf.image.resize(y3, (56, 56), method='nearest').numpy()

                    # --- Visualization Code Start ---
                    if file_count == 1 and idx < 50:
                        # Only save the first three slices from the first file
                        fig, axs = plt.subplots(1, 4, figsize=(16, 4))

                        axs[0].imshow(img_slice.squeeze(), cmap='gray')
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
                        save_path = os.path.join(visualization_dir, f'file{file_count}_slice{idx}.png')
                        plt.savefig(save_path, bbox_inches='tight')
                        plt.close(fig)
                    # --- Visualization Code End ---
                    

                    # Append data to lists
                    images.append(img_slice)
                    masked_images.append(masked_img_slice)
                    view_1_list.append(view_1)
                    view_2_list.append(view_2)
                    y1_list.append(y1)
                    y2_list.append(y2)
                    y3_list.append(y3)

                    # Yield a batch once the required batch size is reached
                    if len(images) == batch_size:
                        yield (
                            (
                                np.array(masked_images, dtype=np.float32), 
                                np.array(view_1_list, dtype=np.float32), 
                                np.array(view_2_list, dtype=np.float32)
                            ),
                            {
                                'pred1': np.array(y1_list, dtype=np.float32), 
                                'pred2': np.array(y2_list, dtype=np.float32), 
                                'final': np.array(y3_list, dtype=np.float32), 
                                'projection_view1': np.zeros((batch_size, 64), dtype=np.float32),
                                'projection_view2': np.zeros((batch_size, 64), dtype=np.float32)
                            }
                        )

                        images, masked_images, view_1_list, view_2_list = [], [], [], []
                        y1_list, y2_list, y3_list = [], [], []

                # Handle the end of a single pass if specified
                if single_pass and file_count >= len(file_paths):
                    # Handling padding to return a complete batch
                    pad_size = batch_size - len(images)
                    pad_indices = np.random.choice(len(images), pad_size, replace=True)
                    images = np.concatenate([images, [images[idx] for idx in pad_indices]], axis=0)
                    masked_images = np.concatenate([masked_images, [masked_images[idx] for idx in pad_indices]], axis=0)
                    view_1_list = np.concatenate([view_1_list, [view_1_list[idx] for idx in pad_indices]], axis=0)
                    view_2_list = np.concatenate([view_2_list, [view_2_list[idx] for idx in pad_indices]], axis=0)
                    y1_list = np.concatenate([y1_list, [y1_list[idx] for idx in pad_indices]], axis=0)
                    y2_list = np.concatenate([y2_list, [y2_list[idx] for idx in pad_indices]], axis=0)
                    y3_list = np.concatenate([y3_list, [y3_list[idx] for idx in pad_indices]], axis=0)

                    yield (
                        (
                            np.array(masked_images, dtype=np.float32), 
                            np.array(view_1_list, dtype=np.float32), 
                            np.array(view_2_list, dtype=np.float32)
                        ),
                        {
                            'pred1': np.array(y1_list, dtype=np.float32), 
                            'pred2': np.array(y2_list, dtype=np.float32), 
                            'final': np.array(y3_list, dtype=np.float32), 
                            'projection_view1': np.zeros((batch_size, 64), dtype=np.float32),
                            'projection_view2': np.zeros((batch_size, 64), dtype=np.float32)
                        }
                    )

                    return

            if single_pass:
                break

    return load_images_for_ssl(file_paths)


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

    # Build decoder
    # Block 6
    x6 = layers.LayerNormalization(epsilon=1e-5)(x5)
    x6 = layers.UpSampling2D(size=(2, 2))(x6)
    x6 = layers.Conv2D(filters[5], 2, padding='same', activation=act, kernel_initializer=initializer)(x6)
    x6 = layers.concatenate([skip4, x6], axis=-1)
    x6 = layers.Conv2D(filters[5], 3, padding='same', activation=act, kernel_initializer=initializer)(x6)
    x6 = layers.Conv2D(filters[5], 3, padding='same', activation=act, kernel_initializer=initializer)(x6)
    x6 = layers.Dropout(drp_out)(x6)
    x6 = att(x6, att_heads[5], dpr[5], training=training)
    # No skip saved here
    #print(f"Decoder Block 6 output shape: {x6.shape}")

    # Block 7
    x7 = layers.LayerNormalization(epsilon=1e-5)(x6)
    x7 = layers.UpSampling2D(size=(2, 2))(x7)
    x7 = layers.Conv2D(filters[6], 2, padding='same', activation=act, kernel_initializer=initializer)(x7)
    x7 = layers.concatenate([skip3, x7], axis=-1)
    x7 = layers.Conv2D(filters[6], 3, padding='same', activation=act, kernel_initializer=initializer)(x7)
    x7 = layers.Conv2D(filters[6], 3, padding='same', activation=act, kernel_initializer=initializer)(x7)
    x7 = layers.Dropout(drp_out)(x7)
    x7 = att(x7, att_heads[6], dpr[6], training=training)
    skip7 = x7  # Save for output
    #print(f"Decoder Block 7 output shape: {x7.shape}")

    # Block 8
    x8 = layers.LayerNormalization(epsilon=1e-5)(x7)
    x8 = layers.UpSampling2D(size=(2, 2))(x8)
    x8 = layers.Conv2D(filters[7], 2, padding='same', activation=act, kernel_initializer=initializer)(x8)
    x8 = layers.concatenate([skip2, x8], axis=-1)
    x8 = layers.Conv2D(filters[7], 3, padding='same', activation=act, kernel_initializer=initializer)(x8)
    x8 = layers.Conv2D(filters[7], 3, padding='same', activation=act, kernel_initializer=initializer)(x8)
    x8 = layers.Dropout(drp_out)(x8)
    x8 = att(x8, att_heads[7], dpr[7], training=training)
    skip8 = x8  # Save for output
    #print(f"Decoder Block 8 output shape: {x8.shape}")

    # Block 9
    x9 = layers.LayerNormalization(epsilon=1e-5)(x8)
    x9 = layers.UpSampling2D(size=(2, 2))(x9)
    x9 = layers.Conv2D(filters[8], 2, padding='same', activation=act, kernel_initializer=initializer)(x9)
    x9 = layers.concatenate([skip1, x9], axis=-1)
    x9 = layers.Conv2D(filters[8], 3, padding='same', activation=act, kernel_initializer=initializer)(x9)
    x9 = layers.Conv2D(filters[8], 3, padding='same', activation=act, kernel_initializer=initializer)(x9)
    x9 = layers.Dropout(drp_out)(x9)
    x9 = att(x9, att_heads[8], dpr[8], training=training)
    skip9 = x9  # Save for output
    #print(f"Decoder Block 9 output shape: {x9.shape}")

    # Create outputs from skips
    # Output 1
    out7 = layers.UpSampling2D(size=(2, 2))(skip7)
    out7 = layers.LayerNormalization(epsilon=1e-5)(out7)
    out7 = layers.Conv2D(filters[6], 3, padding='same', activation=act, kernel_initializer=initializer)(out7)
    out7 = layers.Conv2D(filters[6], 3, padding='same', activation=act, kernel_initializer=initializer)(out7)
    out7 = layers.Conv2D(1, (1, 1), activation='sigmoid', name='pred1')(out7)

    # Output 2
    out8 = layers.UpSampling2D(size=(2, 2))(skip8)
    out8 = layers.LayerNormalization(epsilon=1e-5)(out8)
    out8 = layers.Conv2D(filters[7], 3, padding='same', activation=act, kernel_initializer=initializer)(out8)
    out8 = layers.Conv2D(filters[7], 3, padding='same', activation=act, kernel_initializer=initializer)(out8)
    out8 = layers.Conv2D(1, (1, 1), activation='sigmoid', name='pred2')(out8)

    # Output 3
    out9 = layers.UpSampling2D(size=(2, 2))(skip9)
    out9 = layers.LayerNormalization(epsilon=1e-5)(out9)
    out9 = layers.Conv2D(filters[8], 3, padding='same', activation=act, kernel_initializer=initializer)(out9)
    out9 = layers.Conv2D(filters[8], 3, padding='same', activation=act, kernel_initializer=initializer)(out9)
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

def ssim_loss_function(original, reconstructed):
    """
    Computes the SSIM-based loss with optional edge regularization.
    The SSIM loss is defined as 1 - SSIM to minimize dissimilarity.
    """
    # Compute SSIM
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(original, reconstructed, max_val=1.0))

    # Optional: Compute Edge Loss
    sobel_filter = tf.image.sobel_edges(reconstructed)  # Gradient magnitude of reconstructed image
    sobel_original = tf.image.sobel_edges(original)     # Gradient magnitude of original image
    edge_loss = tf.reduce_mean(tf.abs(sobel_original - sobel_filter))  # L1 difference of edge maps

    # Combine SSIM and Edge Loss
    combined_loss = ssim_loss + 0.1 * edge_loss  # Weight the edge loss (0.1 can be tuned)

    return combined_loss

def contrastive_loss_function(z_i, z_j, temperature=0.5):
    # L2 normalization to ensure all vectors are unit length
    z_i = tf.math.l2_normalize(z_i, axis=1)
    z_j = tf.math.l2_normalize(z_j, axis=1)

    # Compute positive similarity (dot product between z_i and z_j)
    positive_similarity = tf.reduce_sum(z_i * z_j, axis=1) / temperature  # Shape: (batch_size,)

    # Compute logits: similarity between each pair of (z_i, z_j) over the batch
    # Note: This creates a similarity matrix where each row of z_i compares to each row of z_j
    logits = tf.matmul(z_i, tf.transpose(z_j)) / temperature  # Shape: (batch_size, batch_size)

    # Labels are indices from 0 to batch_size - 1, because each (z_i, z_j) pair is a positive
    labels = tf.range(tf.shape(z_i)[0])

    # Cross entropy loss between positive pairs and all other pairs in the batch
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

    # Return the mean loss over the batch
    return tf.reduce_mean(loss)

class EagleLoss(tf.keras.losses.Loss):
    def __init__(self, patch_size=3, cutoff=0.5):
        super(EagleLoss, self).__init__()
        self.patch_size = patch_size
        self.cutoff = cutoff
        
        # Scharr kernel for the gradient map calculation (adjusted for single-channel input)
        kernel_values = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
        self.kernel_x = tf.constant(kernel_values, dtype=tf.float32)
        self.kernel_x = self.kernel_x[:, :, tf.newaxis, tf.newaxis]
        self.kernel_y = tf.transpose(self.kernel_x, perm=[1, 0, 2, 3])

    def calculate_gradient(self, img):
        # Perform convolution to calculate gradients in x and y directions
        gx = tf.nn.conv2d(img, self.kernel_x, strides=[1, 1, 1, 1], padding='SAME')
        gy = tf.nn.conv2d(img, self.kernel_y, strides=[1, 1, 1, 1], padding='SAME')
        return gx, gy

    def calculate_patch_loss(self, output_gradient, target_gradient):
        # Extract patches from gradient maps
        patches_output = tf.image.extract_patches(images=output_gradient,
                                                  sizes=[1, self.patch_size, self.patch_size, 1],
                                                  strides=[1, self.patch_size, self.patch_size, 1],
                                                  rates=[1, 1, 1, 1],
                                                  padding='VALID')
        
        patches_target = tf.image.extract_patches(images=target_gradient,
                                                  sizes=[1, self.patch_size, self.patch_size, 1],
                                                  strides=[1, self.patch_size, self.patch_size, 1],
                                                  rates=[1, 1, 1, 1],
                                                  padding='VALID')

        # Calculate variance for each patch
        var_output = tf.math.reduce_variance(patches_output, axis=-1, keepdims=True)
        var_target = tf.math.reduce_variance(patches_target, axis=-1, keepdims=True)

        return self.fft_loss(var_target, var_output)

    def gaussian_highpass_weights2d(self, size):
        # Generate frequencies using meshgrid
        freq_x = tf.linspace(-0.5, 0.5, num=size[0])
        freq_y = tf.linspace(-0.5, 0.5, num=size[1])
        freq_x, freq_y = tf.meshgrid(freq_x, freq_y)
        
        freq_mag = tf.sqrt(freq_x ** 2 + freq_y ** 2)
        weights = tf.exp(-0.5 * ((freq_mag - self.cutoff) ** 2))
        return 1 - weights  # Inverted for high pass

    def fft_loss(self, pred, gt):
        # Compute 2D FFT for both predicted and ground-truth images
        pred_fft = tf.signal.fft2d(tf.cast(pred, tf.complex64))
        gt_fft = tf.signal.fft2d(tf.cast(gt, tf.complex64))
        
        # Calculate magnitude spectrums
        pred_mag = tf.abs(pred_fft)
        gt_mag = tf.abs(gt_fft)
        
        # Apply Gaussian high-pass filter
        weights = self.gaussian_highpass_weights2d(tf.shape(pred)[1:3])
        weights = tf.cast(weights, tf.complex64)  # Ensure weights are in the complex domain
        weighted_pred_fft = weights * pred_fft
        weighted_gt_fft = weights * gt_fft

        # Convert back to magnitude after filtering
        weighted_pred_mag = tf.abs(weighted_pred_fft)
        weighted_gt_mag = tf.abs(weighted_gt_fft)

        # Return L1 loss of the filtered magnitude spectrum
        return tf.reduce_mean(tf.abs(weighted_pred_mag - weighted_gt_mag))

    def call(self, output, target):
        # Ensure the input images are grayscale
        if output.shape[-1] != 1 or target.shape[-1] != 1:
            raise ValueError("Input 'output' and 'target' should be grayscale images (1 channel)")

        # Calculate gradient maps
        gx_output, gy_output = self.calculate_gradient(output)
        gx_target, gy_target = self.calculate_gradient(target)

        # Compute the variance loss using gradient maps
        eagle_loss = self.calculate_patch_loss(gx_output, gx_target) + self.calculate_patch_loss(gy_output, gy_target)

        return eagle_loss



class WarmUpLearningRateScheduler(keras.callbacks.Callback):
    """Warmup learning rate scheduler"""

    def __init__(self, warmup_batches, target_lr, initial_lr=1e-6, verbose=0):
        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_batches = warmup_batches
        self.target_lr = target_lr
        self.initial_lr = initial_lr
        self.verbose = verbose
        self.batch_count = 0

    def get_current_lr(self, current_batch):
        """Calculate the current learning rate for warm-up."""
        if current_batch <= self.warmup_batches:
            return self.initial_lr + current_batch * (self.target_lr - self.initial_lr) / self.warmup_batches
        else:
            return self.target_lr

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


logging.basicConfig(level=logging.INFO)
import pandas as pd
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


# Temporarily remove @tf.function for easier debugging
@tf.function
def train_step(inputs):
    def step_fn(inputs):
        inputs_batch, targets_batch = inputs
        masked_images, view_1, view_2 = inputs_batch
        y1, y2, y3 = targets_batch['pred1'], targets_batch['pred2'], targets_batch['final']

        # Debugging input shapes
        tf.debugging.assert_rank(masked_images, 4, message="masked_images should be rank 4")
        tf.debugging.assert_rank(view_1, 4, message="view_1 should be rank 4")
        tf.debugging.assert_rank(view_2, 4, message="view_2 should be rank 4")
        tf.debugging.assert_rank(y1, 4, message="y1 should be rank 4")
        tf.debugging.assert_rank(y2, 4, message="y2 should be rank 4")
        tf.debugging.assert_rank(y3, 4, message="y3 should be rank 4")

        with tf.GradientTape() as tape:
            # Forward pass
            outputs = unified_model([masked_images, view_1, view_2], training=True)
            reconstructed_pred1 = outputs['pred1']
            reconstructed_pred2 = outputs['pred2']
            reconstructed_final = outputs['final']
            projection_view1 = outputs['projection_view1']
            projection_view2 = outputs['projection_view2']

            # Debugging output shapes
            tf.debugging.assert_rank(reconstructed_pred1, 4, message="reconstructed_pred1 should be rank 4")
            tf.debugging.assert_rank(reconstructed_pred2, 4, message="reconstructed_pred2 should be rank 4")
            tf.debugging.assert_rank(reconstructed_final, 4, message="reconstructed_final should be rank 4")
            tf.debugging.assert_rank(projection_view1, 2, message="projection_view1 should be rank 2")
            tf.debugging.assert_rank(projection_view2, 2, message="projection_view2 should be rank 2")

            # Compute MIM loss

            # Instantiate the EagleLoss class
            eagle_loss_function = EagleLoss()

            ssim_loss_pred1 = eagle_loss_function(y1, reconstructed_pred1)
            ssim_loss_pred2 = eagle_loss_function(y2, reconstructed_pred2)
            ssim_loss_final = eagle_loss_function(y3, reconstructed_final)
            total_ssim_loss = 0.1 * ssim_loss_pred1 + 0.3 * ssim_loss_pred2 + 0.6 * ssim_loss_final

            # Compute Contrastive Loss
            contrastive_loss = contrastive_loss_function(projection_view1, projection_view2)

            # Total Loss for Backpropagation
            total_loss = 0.6 * total_ssim_loss + 0.4 * contrastive_loss

            # Scaling loss for multiple replicas
            scaled_loss = total_loss / tf.cast(strategy.num_replicas_in_sync, tf.float32)

        # Compute gradients
        grads = tape.gradient(scaled_loss, unified_model.trainable_variables)

        # Gradient clipping (if gradients are not None)
        clipped_grads = [
            tf.clip_by_value(grad, -1.0, 1.0) if grad is not None else None
            for grad in grads
        ]

        # Apply gradients using the optimizer
        opt.apply_gradients(zip(clipped_grads, unified_model.trainable_variables))

        return total_loss

    # Run step function on each replica
    per_replica_losses = strategy.run(step_fn, args=(inputs,))

    # Aggregate the losses from all replicas
    mean_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    return mean_loss



# Define the validation step function
@tf.function
def val_step(inputs):
    def step_fn(inputs):
        inputs_batch, targets_batch = inputs
        masked_images, view_1, view_2 = inputs_batch
        y1, y2, y3 = targets_batch['pred1'], targets_batch['pred2'], targets_batch['final']

        outputs = unified_model([masked_images, view_1, view_2], training=False)
        reconstructed_pred1 = outputs['pred1']
        reconstructed_pred2 = outputs['pred2']
        reconstructed_final = outputs['final']
        projection_view1 = outputs['projection_view1']
        projection_view2 = outputs['projection_view2']

        # Compute MIM loss
        eagle_loss_function = EagleLoss()

        ssim_loss_pred1 = eagle_loss_function(y1, reconstructed_pred1)
        ssim_loss_pred2 = eagle_loss_function(y2, reconstructed_pred2)
        ssim_loss_final = eagle_loss_function(y3, reconstructed_final)
        total_ssim_loss = 0.1 * ssim_loss_pred1 + 0.3 * ssim_loss_pred2 + 0.6 * ssim_loss_final

        # Compute Contrastive Loss
        contrastive_loss = contrastive_loss_function(projection_view1, projection_view2)

        # Total validation loss
        total_val_loss = 0.6 * total_ssim_loss + 0.4 * contrastive_loss

        # Scale the loss by the number of replicas
        scaled_loss = total_val_loss / strategy.num_replicas_in_sync

        return scaled_loss

    per_replica_losses = strategy.run(step_fn, args=(inputs,))
    # Aggregate the losses from all replicas
    mean_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    return mean_loss


# Assuming you've already defined everything up to the training loop, including strategy, datasets, and model.

# Import necessary libraries
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

# Distributed Strategy
strategy = tf.distribute.MirroredStrategy()

# strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"])

# Load image file paths from the directory
image_paths = load_file_paths(brats_data_validation)
data_pairs = list(image_paths)

# Define KFold
kf = KFold(n_splits=5, shuffle=True)

# Directory to save logs and weights
save_dir = '/cluster/home/mrostamian/log_ssl_flair'
os.makedirs(save_dir, exist_ok=True)

# Initialize TensorBoard log directory
log_dir = os.path.join(save_dir, "tensorboard_logs")
os.makedirs(log_dir, exist_ok=True)

# Initialize summary writers for TensorBoard
train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
val_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'val'))

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
warmup_run_epochs = 10
normal_run_epochs = 50

# K-Fold Cross-Validation Loop
for fold, (train_index, val_index) in enumerate(kf.split(data_pairs)):
    print(f"Training fold {fold + 1}/{kf.n_splits}")

    tf.keras.backend.clear_session()

    # Prepare training and validation data
    train_data = [data_pairs[i] for i in train_index]
    val_data = [data_pairs[i] for i in val_index]

    # Calculate the number of valid frames for each epoch
    valid_train_frames = count_non_empty_frames(train_data)
    valid_validation_frames = count_non_empty_frames(val_data)
    # valid_train_frames = 100
    # valid_validation_frames = 100

    # Log the number of valid frames
    file_path = os.path.join(save_dir, 'valid_frames_data.txt')
    message = f'Fold {fold + 1}: The number of valid train data is {valid_train_frames} and the number of valid validation data is {valid_validation_frames}\n'
    with open(file_path, 'a') as file:
        file.write(message)

    # Calculate steps per epoch
    steps_per_epoch = max(int(np.floor(valid_train_frames / batch_size)), 1)
    validation_steps = max(int(np.floor(valid_validation_frames / batch_size)), 1)

    # Instantiate the warm-up learning rate scheduler
    warm_up_lr = WarmUpLearningRateScheduler(
        warmup_run_epochs * steps_per_epoch,
        target_lr=1e-3,
        initial_lr=1e-4,
        verbose=1
    )

    # Modify Generators for SSL tasks
    train_gen = tf.data.Dataset.from_generator(
        lambda: get_brats_ssl_generator(train_data, input_size=input_shapes['masked_images'], batch_size=batch_size),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),  # masked_images_input
                tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),  # view1_input
                tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),  # view2_input
            ),
            {
                'pred1': tf.TensorSpec(shape=(None, 56, 56, 1), dtype=tf.float32),
                'pred2': tf.TensorSpec(shape=(None, 112, 112, 1), dtype=tf.float32),
                'final': tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),
                'projection_view1': tf.TensorSpec(shape=(None, 64), dtype=tf.float32),
                'projection_view2': tf.TensorSpec(shape=(None, 64), dtype=tf.float32),
            }
        )
    ).shuffle(buffer_size=1000).prefetch(tf.data.AUTOTUNE).repeat()

    val_gen = tf.data.Dataset.from_generator(
        lambda: get_brats_ssl_generator(val_data, input_size=input_shapes['masked_images'], batch_size=batch_size),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),  # masked_images_input
                tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),  # view1_input
                tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),  # view2_input
            ),
            {
                'pred1': tf.TensorSpec(shape=(None, 56, 56, 1), dtype=tf.float32),
                'pred2': tf.TensorSpec(shape=(None, 112, 112, 1), dtype=tf.float32),
                'final': tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),
                'projection_view1': tf.TensorSpec(shape=(None, 64), dtype=tf.float32),
                'projection_view2': tf.TensorSpec(shape=(None, 64), dtype=tf.float32),
            }
        )
    ).prefetch(tf.data.AUTOTUNE).repeat()

    # Add options for distributed strategy
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_gen = train_gen.with_options(options)
    val_gen = val_gen.with_options(options)

    # Distribute the dataset
    dist_train_gen = strategy.experimental_distribute_dataset(train_gen)
    dist_val_gen = strategy.experimental_distribute_dataset(val_gen)

    # Model Definition: Using strategy.scope to create and compile the model
    with strategy.scope():
        # Build the model
        unified_model, encoder = myunet(input_shapes=input_shapes, training=True)
        unified_model.summary()

        # Define optimizer and compile the model
        lr_schedule = 1e-4
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=1e-2, amsgrad=True)

        # Warm-up Phase Compilation (Freeze encoder layers)
        # for layer in encoder.layers:
        #     layer.trainable = False
        unified_model.compile(optimizer=opt)

    # Training Loop for Warm-up Phase
    for epoch in range(warmup_run_epochs):
        print(f"Warm-up Epoch {epoch + 1}/{warmup_run_epochs}")

        # Adjust learning rate for warm-up phase
        current_lr = warm_up_lr.get_current_lr(epoch * steps_per_epoch)
        opt.learning_rate.assign(current_lr)

        total_loss = 0.0
        num_batches = 0
        train_loss_metric.reset_state()

        train_iterator = iter(dist_train_gen)
        for step in range(steps_per_epoch):
            inputs = next(train_iterator)
            loss = train_step(inputs)
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

    # Reset the training generator before starting the main training phase
    train_gen = tf.data.Dataset.from_generator(
        lambda: get_brats_ssl_generator(train_data, input_size=input_shapes['masked_images'], batch_size=batch_size),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),  # masked_images_input
                tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),  # view1_input
                tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),  # view2_input
            ),
            {
                'pred1': tf.TensorSpec(shape=(None, 56, 56, 1), dtype=tf.float32),
                'pred2': tf.TensorSpec(shape=(None, 112, 112, 1), dtype=tf.float32),
                'final': tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),
                'projection_view1': tf.TensorSpec(shape=(None, 64), dtype=tf.float32),
                'projection_view2': tf.TensorSpec(shape=(None, 64), dtype=tf.float32),
            }
        )
    ).shuffle(buffer_size=1000).prefetch(tf.data.AUTOTUNE).repeat()

    # Distribute the updated dataset
    dist_train_gen = strategy.experimental_distribute_dataset(train_gen)

    # Add Early Stopping Callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor the validation loss
        patience=12,          # Number of epochs with no improvement to wait before stopping
        restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
        verbose=1
    )

    # Main Training Phase: Unfreeze encoder layers and recompile the model
    # with strategy.scope():
        # for layer in encoder.layers:
        #     layer.trainable = True

        # unified_model.compile(optimizer=opt)

    # Training Loop for Main Training Phase
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(normal_run_epochs):
        print(f"Main Epoch {epoch + 1}/{normal_run_epochs}")

        # Set learning rate for the main training phase
        opt.learning_rate.assign(lr_schedule)

        total_loss = 0.0
        num_batches = 0
        train_loss_metric.reset_state()

        train_iterator = iter(dist_train_gen)
        for step in range(steps_per_epoch):
            inputs = next(train_iterator)
            loss = train_step(inputs)
            total_loss += loss.numpy()
            num_batches += 1

            # Update metrics
            train_loss_metric.update_state(loss)

            # Log training metrics to TensorBoard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=(warmup_run_epochs + epoch) * steps_per_epoch + step)

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
            val_loss = val_step(val_inputs)
            val_total_loss += val_loss.numpy()
            val_num_batches += 1

            # Update validation metrics
            val_loss_metric.update_state(val_loss)

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
        if patience_counter >= 12:  # Same value as `patience` in the EarlyStopping callback
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break    

        # Call the callback to save validation samples at the end of each epoch
        save_predictions_callback = SaveSamplePredictions(
            validation_data=val_gen,
            save_dir=os.path.join(save_dir, "sample_predictions_Flair"),
            num_samples=30,  # Save exactly 30 samples
            fold=fold + 1,
            threshold=0.5
        )

        save_predictions_callback.set_model(unified_model)
        save_predictions_callback.on_epoch_end(epoch, logs={'loss': epoch_loss})

    # End of fold loop




    

# %%
