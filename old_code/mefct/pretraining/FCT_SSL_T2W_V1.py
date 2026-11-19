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

def load_file_paths(data_dir, image_suffix='t2w'):

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
    # Ensure cropx and cropy are not larger than x and y
    cropx = min(cropx, x)
    cropy = min(cropy, y)
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
    augmented_image = augmented['image']
    # Ensure the augmented image has the correct shape
    augmented_image = np.expand_dims(augmented_image, axis=-1)
    return tf.convert_to_tensor(augmented_image, dtype=tf.float32)

def get_brats_ssl_generator(file_paths, input_size=(224, 224, 1), batch_size=32, single_pass=False):
    def load_images_for_ssl(file_paths):
        images, masked_images, view_1_list, view_2_list = [], [], [], []
        y1_list, y2_list, y3_list = [], [], []
        file_count = 0

        ssl_augmentation = get_ssl_augmentation()

        visualization_dir = '/cluster/home/mrostamian/log_ssl_t2w/visualize_img' 
        os.makedirs(visualization_dir, exist_ok=True)

        while True:
            for img_path in file_paths:
                img_data = nib.load(img_path)
                if img_data is None:
                    print(f"Failed to load image at path: {img_path}")
                    continue
                img = img_data.get_fdata()

                file_count += 1

                for idx in range(img.shape[2]):
                    img_slice = crop_center(img[:, :, idx], input_size[0], input_size[1])

                    # Handle empty slices or slices with unexpected shapes
                    if img_slice.size == 0 or np.count_nonzero(img_slice) == 0:
                        print(f"Skipping empty or invalid slice at index {idx} in image at path: {img_path}")
                        continue

                    if img_slice.shape != input_size[:2]:
                        # Resize the slice to match input_size
                        img_slice = cv2.resize(img_slice, input_size[:2], interpolation=cv2.INTER_NEAREST)

                    img_slice = np.expand_dims(img_slice, axis=-1)

                    # Normalize the image slice
                    img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-8)

                    # Create a random mask for SSL
                    masked_img_slice = create_masked_image(img_slice)

                    # Generate augmentations for contrastive learning
                    seed = np.random.randint(0, 1000)
                    np.random.seed(seed)
                    view_1 = apply_ssl_augmentation(img_slice, ssl_augmentation)
                    np.random.seed(seed)
                    view_2 = apply_ssl_augmentation(img_slice, ssl_augmentation)

                    # Convert to tensors
                    y3 = tf.convert_to_tensor(img_slice, dtype=tf.float32)
                    y2 = tf.image.resize(y3, [112, 112], method='nearest')
                    y1 = tf.image.resize(y3, [56, 56], method='nearest')

                    # Visualization (optional)
                    if file_count == 1 and idx < 50:
                        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                        axs[0].imshow(tf.squeeze(img_slice), cmap='gray')
                        # ... (rest of visualization code)
                        plt.close(fig)

                    # Append data to lists
                    images.append(y3)
                    masked_images.append(tf.convert_to_tensor(masked_img_slice, dtype=tf.float32))
                    view_1_list.append(tf.convert_to_tensor(view_1, dtype=tf.float32))
                    view_2_list.append(tf.convert_to_tensor(view_2, dtype=tf.float32))
                    y1_list.append(y1)
                    y2_list.append(y2)
                    y3_list.append(y3)

                    # Yield a batch
                    if len(images) == batch_size:
                        yield (
                            (
                                tf.stack(masked_images),
                                tf.stack(view_1_list),
                                tf.stack(view_2_list)
                            ),
                            {
                                'pred1': tf.stack(y1_list),
                                'pred2': tf.stack(y2_list),
                                'final': tf.stack(y3_list),
                                'projection_view1': tf.zeros([batch_size, 64], dtype=tf.float32),
                                'predicted_projection_view1': tf.zeros([batch_size, 64], dtype=tf.float32)
                            }
                        )
                        images, masked_images, view_1_list, view_2_list = [], [], [], []
                        y1_list, y2_list, y3_list = [], [], []

                if single_pass and file_count >= len(file_paths):
                    # Pad the final batch if necessary
                    if len(images) > 0:
                        pad_size = batch_size - len(images)
                        pad_indices = np.random.choice(len(images), pad_size, replace=True)
                        for idx in pad_indices:
                            images.append(images[idx])
                            masked_images.append(masked_images[idx])
                            view_1_list.append(view_1_list[idx])
                            view_2_list.append(view_2_list[idx])
                            y1_list.append(y1_list[idx])
                            y2_list.append(y2_list[idx])
                            y3_list.append(y3_list[idx])

                        # Yield the final batch
                        yield (
                            (
                                tf.stack(masked_images),
                                tf.stack(view_1_list),
                                tf.stack(view_2_list)
                            ),
                            {
                                'pred1': tf.stack(y1_list),
                                'pred2': tf.stack(y2_list),
                                'final': tf.stack(y3_list),
                                'projection_view1': tf.zeros([batch_size, 64], dtype=tf.float32),
                                'predicted_projection_view1': tf.zeros([batch_size, 64], dtype=tf.float32)
                            }
                        )
                    return

            if single_pass:
                break

    return load_images_for_ssl(file_paths)


from tensorflow.keras.layers import Layer, Dense, Conv2D, Dropout, MultiHeadAttention, BatchNormalization, \
DepthwiseConv2D, UpSampling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow import Tensor, divide, concat, random, split, reshape, transpose, float32
from typing import List, Union, Iterable

#--------------------------------------------Model-------------------------------------------
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import numpy as np

# Custom StochasticDepth Layer
class StochasticDepth(layers.Layer):
    def __init__(self, drop_prob=0.0, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
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

# Wide Focus Function
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
    def __init__(self, dim_out, num_heads, proj_drop=0.0, kernel_size=3,
                 stride_kv=1, stride_q=1, padding_kv="same", padding_q="same",
                 attention_bias=True):
        super().__init__()
        self.dim = dim_out
        self.num_heads = num_heads
        self.proj_drop = proj_drop

        self.conv_proj_q = self._build_projection(kernel_size, stride_q, padding_q)
        self.conv_proj_k = self._build_projection(kernel_size, stride_kv, padding_kv)
        self.conv_proj_v = self._build_projection(kernel_size, stride_kv, padding_kv)

        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.dim // self.num_heads,
            use_bias=attention_bias,
        )
        self.proj_drop = Dropout(proj_drop)

    @staticmethod
    def _build_projection(kernel_size, stride, padding):
        proj = Sequential([
            DepthwiseConv2D(kernel_size, padding=padding, strides=stride, use_bias=False),
            layers.LayerNormalization(),
        ])
        return proj

    def call(self, inputs, mask=None, training=True):
        x = inputs  # [batch_size, H_q, W_q, C]
        batch_size = tf.shape(x)[0]
        H_q = tf.shape(x)[1]
        W_q = tf.shape(x)[2]

        # Apply projections
        q = self.conv_proj_q(x)  # [batch_size, H_q, W_q, dim]
        k = self.conv_proj_k(x)  # [batch_size, H_k, W_k, dim]
        v = self.conv_proj_v(x)  # [batch_size, H_k, W_k, dim]

        H_k = tf.shape(k)[1]
        W_k = tf.shape(k)[2]

        # Flatten spatial dimensions
        q = tf.reshape(q, [batch_size, H_q * W_q, self.dim])
        k = tf.reshape(k, [batch_size, H_k * W_k, self.dim])
        v = tf.reshape(v, [batch_size, H_k * W_k, self.dim])

        # Apply attention
        attention_output = self.attention(query=q, value=v, key=k, training=training)  # [batch_size, H_q * W_q, dim]

        # Reshape back to spatial dimensions
        attention_output = tf.reshape(attention_output, [batch_size, H_q, W_q, self.dim])

        if training:
            attention_output = self.proj_drop(attention_output)

        return attention_output

def att(x_in,
        num_heads,
        dpr,
        proj_drop=0.0,
        attention_bias=True,
        padding_q="same",
        padding_kv="same",
        stride_kv=2,  # Now set to 1
        stride_q=1,
        training=True):

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
    )(x_in, training=training)

    # Skip connection 1.
    attention_output = StochasticDepth(dpr)(attention_output, training=training)
    attention_output = Conv2D(c, 3, padding="same", activation="relu")(attention_output)
    x2 = layers.Add()([attention_output, x_in])
    x3 = layers.LayerNormalization(epsilon=1e-5)(x2)
    x3 = wide_focus(x3, filters=c, dropout_rate=0.0)
    # Skip connection 2.
    x3 = StochasticDepth(dpr)(x3, training=training)
    x3 = layers.Add()([x3, x2])

    return x3

# Projection head
def add_projection_head(x, name=None):
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation=None, name=name)(x)
    return x    

def create_encoder(input_shape, filters, att_heads, dpr, 
                   initializer='he_normal', act='relu', drp_out=0.3, training=True, name='encoder_model'):
    inputs = tf.keras.Input(shape=input_shape, name='encoder_input')

    x = inputs
    skip_connections = []

    # Block 1
    x1 = layers.LayerNormalization(epsilon=1e-5)(x)
    x1 = layers.Lambda(
        lambda x: x[:, :, :, -1:],
        output_shape=lambda s: (s[0], s[1], s[2], 1)
    )(x1)
    x1 = layers.Conv2D(filters[0], 3, padding='same', activation=act, kernel_initializer=initializer)(x1)
    x1 = layers.Conv2D(filters[0], 3, padding='same', activation=act, kernel_initializer=initializer)(x1)
    x1 = layers.Dropout(drp_out)(x1)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x1 = att(x1, att_heads[0], dpr[0], training=training)
    skip_connections.append(x1)  # Save for decoder

    # Block 2
    x2 = layers.LayerNormalization(epsilon=1e-5)(x1)
    scale_img_2 = layers.AveragePooling2D(2, 2)(inputs)
    scale_conv2 = layers.Conv2D(filters[0], 3, padding='same', activation=act)(scale_img_2)
    x2 = layers.concatenate([scale_conv2, x2], axis=-1)
    x2 = layers.Conv2D(filters[1], 3, padding='same', activation=act, kernel_initializer=initializer)(x2)
    x2 = layers.Conv2D(filters[1], 3, padding='same', activation=act, kernel_initializer=initializer)(x2)
    x2 = layers.Dropout(drp_out)(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x2 = att(x2, att_heads[1], dpr[1], training=training)
    skip_connections.append(x2)

    # Block 3
    x3 = layers.LayerNormalization(epsilon=1e-5)(x2)
    scale_img_3 = layers.AveragePooling2D(2, 2)(scale_img_2)
    scale_conv3 = layers.Conv2D(filters[1], 3, padding='same', activation=act)(scale_img_3)
    x3 = layers.concatenate([scale_conv3, x3], axis=-1)
    x3 = layers.Conv2D(filters[2], 3, padding='same', activation=act, kernel_initializer=initializer)(x3)
    x3 = layers.Conv2D(filters[2], 3, padding='same', activation=act, kernel_initializer=initializer)(x3)
    x3 = layers.Dropout(drp_out)(x3)
    x3 = layers.MaxPooling2D((2, 2))(x3)
    x3 = att(x3, att_heads[2], dpr[2], training=training)
    skip_connections.append(x3)

    # Block 4
    x4 = layers.LayerNormalization(epsilon=1e-5)(x3)
    scale_img_4 = layers.AveragePooling2D(2, 2)(scale_img_3)
    scale_conv4 = layers.Conv2D(filters[2], 3, padding='same', activation=act)(scale_img_4)
    x4 = layers.concatenate([scale_conv4, x4], axis=-1)
    x4 = layers.Conv2D(filters[3], 3, padding='same', activation=act, kernel_initializer=initializer)(x4)
    x4 = layers.Conv2D(filters[3], 3, padding='same', activation=act, kernel_initializer=initializer)(x4)
    x4 = layers.Dropout(drp_out)(x4)
    x4 = layers.MaxPooling2D((2, 2))(x4)
    x4 = att(x4, att_heads[3], dpr[3], training=training)
    skip_connections.append(x4)

    # Bottleneck
    x5 = layers.LayerNormalization(epsilon=1e-5)(x4)
    x5 = layers.Conv2D(filters[4], 3, padding='same', activation=act, kernel_initializer=initializer)(x5)
    x5 = layers.Conv2D(filters[4], 3, padding='same', activation=act, kernel_initializer=initializer)(x5)
    x5 = layers.Dropout(drp_out)(x5)
    x5 = layers.MaxPooling2D((2, 2))(x5)
    x5 = att(x5, att_heads[4], dpr[4], training=training)
    encoder_output = x5

    # Create the encoder model
    encoder_model = tf.keras.Model(inputs=inputs, outputs=[encoder_output] + skip_connections, name=name)

    return encoder_model


def myunet(input_shapes=None):
    # Model configuration parameters
    if input_shapes is None:
        input_shapes = {
            'masked_images': (224, 224, 1),
            'view1': (224, 224, 1),
            'view2': (224, 224, 1),
        }
    filters = [32, 64, 128, 256, 512]
    att_heads = [2] * len(filters)
    blocks = len(filters)
    stochastic_depth_rate = 0.0
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]

    # Initializer, activation, and dropout rate
    initializer = 'he_normal'
    act = 'relu'
    drp_out = 0.3

    # Define inputs
    masked_images_input_shape = input_shapes['masked_images']
    view1_input_shape = input_shapes['view1']
    view2_input_shape = input_shapes['view2']

    masked_images_input = tf.keras.Input(shape=masked_images_input_shape, name='masked_images_input')
    view1_input = tf.keras.Input(shape=view1_input_shape, name='view1_input')
    view2_input = tf.keras.Input(shape=view2_input_shape, name='view2_input')

    # Online Encoder
    online_encoder = create_encoder(
        input_shape=masked_images_input_shape,
        filters=filters,
        att_heads=att_heads,
        dpr=dpr,
        initializer=initializer,
        act=act,
        drp_out=drp_out,
        training=True,
        name='online_encoder'
    )

    # Get outputs from online encoder
    encoder_outputs_online = online_encoder(masked_images_input)
    encoder_output_online = encoder_outputs_online[0]
    skip_connections_online = encoder_outputs_online[1:]

    # Decoder definition
    def decoder(encoder_output, skip_connections, filters, training=True):
        # Block 6
        x6 = layers.LayerNormalization(epsilon=1e-5)(encoder_output)
        x6 = layers.UpSampling2D(size=(2, 2))(x6)
        x6 = layers.Conv2D(filters[3], 3, padding='same', activation=act, kernel_initializer=initializer)(x6)
        x6 = layers.concatenate([skip_connections[3], x6], axis=-1)
        x6 = layers.Conv2D(filters[3], 3, padding='same', activation=act, kernel_initializer=initializer)(x6)

        # Block 7
        x7 = layers.LayerNormalization(epsilon=1e-5)(x6)
        x7 = layers.UpSampling2D(size=(2, 2))(x7)
        x7 = layers.Conv2D(filters[2], 3, padding='same', activation=act, kernel_initializer=initializer)(x7)
        x7 = layers.concatenate([skip_connections[2], x7], axis=-1)
        x7 = layers.Conv2D(filters[2], 3, padding='same', activation=act, kernel_initializer=initializer)(x7)
        skip7 = x7  # Save for output

        # Block 8
        x8 = layers.LayerNormalization(epsilon=1e-5)(x7)
        x8 = layers.UpSampling2D(size=(2, 2))(x8)
        x8 = layers.Conv2D(filters[1], 3, padding='same', activation=act, kernel_initializer=initializer)(x8)
        x8 = layers.concatenate([skip_connections[1], x8], axis=-1)
        x8 = layers.Conv2D(filters[1], 3, padding='same', activation=act, kernel_initializer=initializer)(x8)
        skip8 = x8  # Save for output

        # Block 9
        x9 = layers.LayerNormalization(epsilon=1e-5)(x8)
        x9 = layers.UpSampling2D(size=(2, 2))(x9)
        x9 = layers.Conv2D(filters[0], 3, padding='same', activation=act, kernel_initializer=initializer)(x9)
        x9 = layers.concatenate([skip_connections[0], x9], axis=-1)
        x9 = layers.Conv2D(filters[0], 3, padding='same', activation=act, kernel_initializer=initializer)(x9)
        skip9 = x9  # Save for output

        # Create outputs from skips
        # Output 1
        out7 = layers.UpSampling2D(size=(2, 2))(skip7)
        out7 = layers.LayerNormalization(epsilon=1e-5)(out7)
        out7 = layers.Conv2D(filters[2], 3, padding='same', activation=act, kernel_initializer=initializer)(out7)
        out7 = layers.Conv2D(filters[2], 3, padding='same', activation=act, kernel_initializer=initializer)(out7)
        out7 = layers.Conv2D(1, (1, 1), activation='sigmoid', name='pred1')(out7)

        # Output 2
        out8 = layers.UpSampling2D(size=(2, 2))(skip8)
        out8 = layers.LayerNormalization(epsilon=1e-5)(out8)
        out8 = layers.Conv2D(filters[1], 3, padding='same', activation=act, kernel_initializer=initializer)(out8)
        out8 = layers.Conv2D(filters[1], 3, padding='same', activation=act, kernel_initializer=initializer)(out8)
        out8 = layers.Conv2D(1, (1, 1), activation='sigmoid', name='pred2')(out8)

        # Output 3
        out9 = layers.UpSampling2D(size=(2, 2))(skip9)
        out9 = layers.LayerNormalization(epsilon=1e-5)(out9)
        out9 = layers.Conv2D(filters[0], 3, padding='same', activation=act, kernel_initializer=initializer)(out9)
        out9 = layers.Conv2D(filters[0], 3, padding='same', activation=act, kernel_initializer=initializer)(out9)
        out9 = layers.Conv2D(1, (1, 1), activation='sigmoid', name='final')(out9)

        return out7, out8, out9

    # Decode the online encoder output
    out7, out8, out9 = decoder(encoder_output_online, skip_connections_online, filters, training=True)


    # Predictor head for the online encoder
    def add_predictor_head(x, name=None):
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation=None, name=name)(x)
        return x

    # Global Average Pooling on encoder_output_online
    encoder_output_view1_pooled = layers.GlobalAveragePooling2D()(encoder_output_online)
    projection_view1 = add_projection_head(encoder_output_view1_pooled, name='projection_view1')
    predicted_projection_view1 = add_predictor_head(projection_view1, name='predicted_projection_view1')

    # Define model inputs and outputs
    model_inputs = [masked_images_input, view1_input, view2_input]
    model_outputs = {
        'pred1': out7,
        'pred2': out8,
        'final': out9,
        'projection_view1': projection_view1,
        'predicted_projection_view1': predicted_projection_view1,  # Online encoder predictor output
        # 'projection_view2' will be obtained from target_encoder_with_head
    }

    # Create the unified model
    unified_model = tf.keras.Model(inputs=model_inputs, outputs=model_outputs, name='unified_model')

    return unified_model, online_encoder


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
    The SSIM loss is defined as 1 - SSIM to minimize dissimilarity.
    """
    # Compute SSIM
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(original, reconstructed, max_val=1.0))

    # Optional: Compute Edge Loss
    sobel_filter = tf.image.sobel_edges(reconstructed)  # Gradient magnitude of reconstructed image
    sobel_original = tf.image.sobel_edges(original)     # Gradient magnitude of original image
    edge_loss = tf.reduce_mean(tf.abs(sobel_original - sobel_filter))  # L1 difference of edge maps

    # Combine SSIM and Edge Loss
    combined_loss = ssim_loss + 0.5 * edge_loss  # Weight the edge loss (0.1 can be tuned)

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
import logging

class SaveSamplePredictions(Callback):
    def __init__(
        self,
        model,
        target_encoder,
        validation_data: tf.data.Dataset,
        save_dir: str,
        num_samples: int = 30,
        fold: int = 1,
        threshold: float = 0.5
    ):
        super(SaveSamplePredictions, self).__init__()
        self.unified_model = model
        self.target_encoder = target_encoder
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
        individual_cosine_sims = []

        while saved_samples < self.num_samples:
            try:
                val_inputs, val_targets = next(data_iterator)
                masked_images, view_1, view_2 = val_inputs
                y3 = val_targets['final']

                # Make predictions using the model
                predictions = self.unified_model([masked_images, view_1, view_2], training=False)
                pred_masks = predictions.get('final', None)
                projection_view2 = self.target_encoder(view_2, training=False)
                predicted_projection_view1 = predictions.get('predicted_projection_view1', None)

                if pred_masks is None or predicted_projection_view1 is None or projection_view2 is None:
                    logging.error("The model prediction does not contain 'final' or projection outputs.")
                    return

                # Compute cosine similarity
                batch_cosine_sim = self.compute_cosine_similarity(predicted_projection_view1, projection_view2)

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
                break

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


# Custom function to update target encoder weights
def update_target_encoder(online_encoder, target_encoder, ema_rate):
    online_variables = online_encoder.trainable_variables
    target_variables = target_encoder.trainable_variables
    for online_var, target_var in zip(online_variables, target_variables):
        target_var.assign(ema_rate * target_var + (1 - ema_rate) * online_var)


# @tf.function
def train_step(model, online_encoder, target_encoder_with_head, inputs, optimizer, ema_rate=0.99):
    inputs_batch, targets_batch = inputs
    masked_images, view_1, view_2 = inputs_batch
    y1, y2, y3 = targets_batch['pred1'], targets_batch['pred2'], targets_batch['final']

    with tf.GradientTape() as tape:
        # Forward pass through unified model (online encoder + decoder)
        outputs = model([masked_images, view_1, view_2], training=True)
        reconstructed_pred1 = outputs['pred1']
        reconstructed_pred2 = outputs['pred2']
        reconstructed_final = outputs['final']
        projection_view1 = outputs['projection_view1']
        predicted_projection_view1 = outputs['predicted_projection_view1']

        # Stop gradients for target encoder
        # Forward pass through target encoder with projection head 
        projection_view2 = target_encoder_with_head(view_2, training=False) 
        projection_view2 = tf.stop_gradient(projection_view2) # Prevent gradient computation    

        # Ensure the final shapes are consistent
        if predicted_projection_view1.shape[1] != projection_view2.shape[1]:
            raise ValueError(f"Shapes of predicted_projection_view1 {predicted_projection_view1.shape} and projection_view2 {projection_view2.shape} must match.")

    
        # Compute MIM loss
        ssim_loss_pred1 = ssim_loss_function(y1, reconstructed_pred1)
        ssim_loss_pred2 = ssim_loss_function(y2, reconstructed_pred2)
        ssim_loss_final = ssim_loss_function(y3, reconstructed_final)
        total_ssim_loss = 0.1 * ssim_loss_pred1 + 0.3 * ssim_loss_pred2 + 0.6 * ssim_loss_final

        # Compute Contrastive Loss
        contrastive_loss_online = contrastive_loss_function(predicted_projection_view1, projection_view2)
        total_contrastive_loss = contrastive_loss_online

        # Total Loss for Backpropagation
        total_loss = 0.6 * total_ssim_loss + 0.4 * total_contrastive_loss

    # Compute gradients
    grads = tape.gradient(total_loss, model.trainable_variables)

    # Gradient clipping (if gradients are not None)
    clipped_grads = [
        tf.clip_by_value(grad, -1.0, 1.0) if grad is not None else None
        for grad in grads
    ]

    # Apply gradients using the optimizer
    optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))

    # Update the target encoder's weights using EMA
    update_target_encoder(online_encoder, target_encoder_with_head, ema_rate=ema_rate)

    return total_loss

@tf.function
def val_step(inputs, model, target_encoder_with_head):
    inputs_batch, targets_batch = inputs
    masked_images, view_1, view_2 = inputs_batch
    y1, y2, y3 = targets_batch['pred1'], targets_batch['pred2'], targets_batch['final']

    # Forward pass through unified model
    outputs = model([masked_images, view_1, view_2], training=False)
    reconstructed_pred1 = outputs['pred1']
    reconstructed_pred2 = outputs['pred2']
    reconstructed_final = outputs['final']
    predicted_projection_view1 = outputs['predicted_projection_view1']

    # Forward pass through target encoder with projection head
    projection_view2 = target_encoder_with_head(view_2, training=False)  # Should be view_2 or view_1 based on your requirement

    # Compute MIM loss
    ssim_loss_pred1 = ssim_loss_function(y1, reconstructed_pred1)
    ssim_loss_pred2 = ssim_loss_function(y2, reconstructed_pred2)
    ssim_loss_final = ssim_loss_function(y3, reconstructed_final)
    total_ssim_loss = 0.1 * ssim_loss_pred1 + 0.3 * ssim_loss_pred2 + 0.6 * ssim_loss_final

    # Compute Contrastive Loss
    contrastive_loss_online = contrastive_loss_function(predicted_projection_view1, projection_view2)
    total_contrastive_loss = contrastive_loss_online

    # Total validation loss
    total_val_loss = 0.6 * total_ssim_loss + 0.4 * total_contrastive_loss

    return total_val_loss



# Assuming you've already defined everything up to the training loop, including strategy, datasets, and model.

import tensorflow as tf
from sklearn.model_selection import KFold
import os
import numpy as np
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Load image file paths from the directory
image_paths = load_file_paths(brats_data_validation)
data_pairs = list(image_paths)

# Define KFold
kf = KFold(n_splits=5, shuffle=True)

# Directory to save logs and weights
save_dir = '/cluster/home/mrostamian/log_ssl_t2w'
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
warmup_run_epochs = 1
normal_run_epochs = 2

# Define KFold
kf = KFold(n_splits=5, shuffle=True)

# Initialize custom metrics
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

# K-Fold Cross-Validation Loop
for fold, (train_index, val_index) in enumerate(kf.split(data_pairs)):
    print(f"Training fold {fold + 1}/{kf.n_splits}")

    tf.keras.backend.clear_session()

    # Prepare training and validation data
    train_data = [data_pairs[i] for i in train_index]
    val_data = [data_pairs[i] for i in val_index]

    # Calculate the number of valid frames for each epoch
    # valid_train_frames = count_non_empty_frames(train_data)
    # valid_validation_frames = count_non_empty_frames(val_data)
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
                tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),
            ),
            {
                'pred1': tf.TensorSpec(shape=(None, 56, 56, 1), dtype=tf.float32),
                'pred2': tf.TensorSpec(shape=(None, 112, 112, 1), dtype=tf.float32),
                'final': tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),
                'projection_view1': tf.TensorSpec(shape=(None, 64), dtype=tf.float32),
                'predicted_projection_view1': tf.TensorSpec(shape=(None, 64), dtype=tf.float32),

            }
        )
    ).shuffle(buffer_size=100).prefetch(tf.data.AUTOTUNE).repeat()

    val_gen = tf.data.Dataset.from_generator(
        lambda: get_brats_ssl_generator(val_data, input_size=input_shapes['masked_images'], batch_size=batch_size),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),
            ),
            {
                'pred1': tf.TensorSpec(shape=(None, 56, 56, 1), dtype=tf.float32),
                'pred2': tf.TensorSpec(shape=(None, 112, 112, 1), dtype=tf.float32),
                'final': tf.TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32),
                'projection_view1': tf.TensorSpec(shape=(None, 64), dtype=tf.float32),
                'predicted_projection_view1': tf.TensorSpec(shape=(None, 64), dtype=tf.float32),

            }
        )
    ).prefetch(tf.data.AUTOTUNE).repeat()

    # Model Definition
    unified_model, online_encoder = myunet(input_shapes=input_shapes)
    unified_model.summary()


    # Define optimizer and compile the model
    lr_schedule = 1e-4
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=1e-2, amsgrad=True)
    unified_model.compile(optimizer=opt)


    #----------------------------------------Target Encoder------------------------------------------
    # Instantiate the target encoder
    input_shape = input_shapes['masked_images']
    filters = [32, 64, 128, 256, 512]
    att_heads = [2] * len(filters)
    blocks = len(filters)
    stochastic_depth_rate = 0.0
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]
    initializer = 'he_normal'
    act = 'relu'
    drp_out = 0.3

    target_encoder = create_encoder(
        input_shape=input_shape,
        filters=filters,
        att_heads=att_heads,
        dpr=dpr,
        initializer=initializer,
        act=act,
        drp_out=drp_out,
        training=False,
        name='target_encoder'
    )

    # Attach projection head to target encoder
    target_encoder_output = target_encoder.output[0]  # First output is the encoder output
    target_encoder_pooled = layers.GlobalAveragePooling2D()(target_encoder_output)
    projection_view2 = add_projection_head(target_encoder_pooled, name='projection_view2')

    # Create a new model that includes the projection head
    target_encoder_with_head = tf.keras.Model(
        inputs=target_encoder.input,
        outputs=projection_view2,
        name='target_encoder_with_head'
    )

    # Set target encoder weights to match online encoder
    target_encoder.set_weights(online_encoder.get_weights())

    #--------------------------------------------------------------------------------------------


    save_predictions_callback = SaveSamplePredictions(
        model=unified_model,
        target_encoder = target_encoder_with_head,
        validation_data=val_gen,
        save_dir=os.path.join(save_dir, "sample_predictions_t2w"),
        num_samples=30,  # Save exactly 30 samples
        fold=fold + 1,
        threshold=0.5
    )

    patience_counter = 0
    best_loss = float('inf')
    # Training Loop for Warm-up Phase
    for epoch in range(warmup_run_epochs):
        print(f"Warm-up Epoch {epoch + 1}/{warmup_run_epochs}")

        # Adjust learning rate for warm-up phase
        current_lr = warm_up_lr.get_current_lr(epoch * steps_per_epoch)
        opt.learning_rate.assign(current_lr)

        # total_loss = 0.0
        # num_batches = 0
        train_loss_metric.reset_state()

        train_iterator = iter(train_gen)
        for step in range(steps_per_epoch):
            inputs = next(train_iterator)
            # During training loop
            loss = train_step(unified_model, online_encoder, target_encoder_with_head, inputs, opt, ema_rate=0.99)


            # Update metrics
            train_loss_metric.update_state(loss)

            # Log training metrics to TensorBoard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=epoch * steps_per_epoch + step)

        # Compute average epoch loss
        epoch_loss = train_loss_metric.result().numpy()
        print(f"Warm-up Epoch {epoch + 1}, Training Loss: {epoch_loss}")


    # Main Training Phase
    for epoch in range(normal_run_epochs):
        print(f"Main Epoch {epoch + 1}/{normal_run_epochs}")

        # total_loss = 0.0
        # num_batches = 0
        train_loss_metric.reset_state()

        train_iterator = iter(train_gen)
        for step in range(steps_per_epoch):
            inputs = next(train_iterator)
            loss = train_step(unified_model, online_encoder, target_encoder_with_head, inputs, opt, ema_rate=0.99)

            # total_loss += loss.numpy()
            # num_batches += 1

            # Update metrics
            train_loss_metric.update_state(loss)

            # Log training metrics to TensorBoard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=(warmup_run_epochs + epoch) * steps_per_epoch + step)

        # Compute average epoch loss
        # epoch_loss = total_loss / num_batches
        epoch_loss = train_loss_metric.result().numpy()

        print(f"Main Epoch {epoch + 1}, Training Loss: {epoch_loss}")

        # Validation steps
        val_total_loss = 0.0
        val_num_batches = 0
        val_loss_metric.reset_state()

        val_iterator = iter(val_gen)
        for step in range(validation_steps):
            val_inputs = next(val_iterator)
            val_loss = val_step(val_inputs, unified_model, target_encoder_with_head)

            val_total_loss += val_loss.numpy()
            # val_num_batches += 1

            # Update validation metrics
            val_loss_metric.update_state(val_loss)

        # Compute average validation loss
        # val_epoch_loss = val_total_loss / val_num_batches
        val_epoch_loss = val_loss_metric.result().numpy()

        print(f"Main Epoch {epoch + 1}, Validation Loss: {val_epoch_loss}")

          # Early stopping logic
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            patience_counter = 0
            # Save best model weights
            unified_model.save_weights(os.path.join(save_dir, f'self_supervised_best_weights_fold_{fold + 1}.weights.h5'))
            print(f"Best model saved for fold {fold + 1} with validation loss: {best_loss}")
        else:
            patience_counter += 1

        if patience_counter >= 12:
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break


        save_predictions_callback.on_epoch_end(epoch, logs={'loss': epoch_loss})

    # End of fold loop





    

# %%
