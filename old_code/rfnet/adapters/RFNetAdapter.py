# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# import math
# import sys
# from rflayers import normalization
# from rflayers import general_conv3d

# from rflayers import prm_generator_laststage, prm_generator, region_aware_modal_fusion


import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from layers import normalization
from layers import general_conv3d
from layers import prm_generator_laststage, prm_generator, region_aware_modal_fusion


basic_dims = 16
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.e1_c1 = general_conv3d(1, basic_dims, pad_type='reflect')
        self.e1_c2 = general_conv3d(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = general_conv3d(basic_dims, basic_dims*2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d(basic_dims*2, basic_dims*2, pad_type='reflect')
        self.e2_c3 = general_conv3d(basic_dims*2, basic_dims*2, pad_type='reflect')

        self.e3_c1 = general_conv3d(basic_dims*2, basic_dims*4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d(basic_dims*4, basic_dims*4, pad_type='reflect')
        self.e3_c3 = general_conv3d(basic_dims*4, basic_dims*4, pad_type='reflect')

        self.e4_c1 = general_conv3d(basic_dims*4, basic_dims*8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d(basic_dims*8, basic_dims*8, pad_type='reflect')
        self.e4_c3 = general_conv3d(basic_dims*8, basic_dims*8, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        return x1, x2, x3, x4

class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_sep, self).__init__()

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4):
        de_x4 = self.d3_c1(self.d3(x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred

class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_fuse, self).__init__()

        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)

        self.RFM4 = region_aware_modal_fusion(in_channel=basic_dims*8, num_cls=num_cls)
        self.RFM3 = region_aware_modal_fusion(in_channel=basic_dims*4, num_cls=num_cls)
        self.RFM2 = region_aware_modal_fusion(in_channel=basic_dims*2, num_cls=num_cls)
        self.RFM1 = region_aware_modal_fusion(in_channel=basic_dims*1, num_cls=num_cls)

        self.prm_generator4 = prm_generator_laststage(in_channel=basic_dims*8, num_cls=num_cls)
        self.prm_generator3 = prm_generator(in_channel=basic_dims*4, num_cls=num_cls)
        self.prm_generator2 = prm_generator(in_channel=basic_dims*2, num_cls=num_cls)
        self.prm_generator1 = prm_generator(in_channel=basic_dims*1, num_cls=num_cls)


    def forward(self, x1, x2, x3, x4, mask):
        prm_pred4 = self.prm_generator4(x4, mask)
        de_x4 = self.RFM4(x4, prm_pred4.detach(), mask)
        de_x4 = self.d3_c1(self.up2(de_x4))

        prm_pred3 = self.prm_generator3(de_x4, x3, mask)
        de_x3 = self.RFM3(x3, prm_pred3.detach(), mask)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        prm_pred2 = self.prm_generator2(de_x3, x2, mask)
        de_x2 = self.RFM2(x2, prm_pred2.detach(), mask)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        prm_pred1 = self.prm_generator1(de_x2, x1, mask)
        de_x1 = self.RFM1(x1, prm_pred1.detach(), mask)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (prm_pred1, self.up2(prm_pred2), self.up4(prm_pred3), self.up8(prm_pred4))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16, dropout_rate=0.1):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, channels // reduction_ratio)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(channels // reduction_ratio, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.size()
        # Squeeze
        y = self.global_avg_pool(x).view(b, c)
        # Excitation
        y = F.relu(self.fc1(y))
        y = self.dropout(y)
        y = self.sigmoid(self.fc2(y))
        # Scale
        y = y.view(b, c, 1, 1, 1)
        return x * y

class AdapterBetweenEncoders(nn.Module):
    def __init__(self, num_encoders, in_channels, compression_ratio=0.5, dropout_rate=0.1):
        super(AdapterBetweenEncoders, self).__init__()
        compressed_channels = max(1, int(in_channels * compression_ratio))
        self.num_encoders = num_encoders

        # Compression layers for each encoder
        self.compress_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, compressed_channels, kernel_size=1, padding=0),
                nn.BatchNorm3d(compressed_channels),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=dropout_rate)
            ) for _ in range(num_encoders)
        ])

        # Channel Attention with SE block
        self.se_block = SEBlock(channels=compressed_channels * num_encoders, reduction_ratio=16, dropout_rate=dropout_rate)

        # Lightweight Convolution Block
        self.sep_conv = nn.Sequential(
            nn.Conv3d(compressed_channels * num_encoders, compressed_channels * num_encoders, kernel_size=3, padding=1, groups=compressed_channels * num_encoders),
            nn.Conv3d(compressed_channels * num_encoders, compressed_channels * num_encoders, kernel_size=1, padding=0),
            nn.BatchNorm3d(compressed_channels * num_encoders),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate)
        )

        # Expansion layers for each encoder
        self.expand_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(compressed_channels * num_encoders, in_channels, kernel_size=1, padding=0),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True)
                # Optional Dropout if needed
            ) for _ in range(num_encoders)
        ])

    def forward(self, cnn_outputs_block):
        # Compression
        compressed_outputs = [compress_layer(x) for compress_layer, x in zip(self.compress_layers, cnn_outputs_block)]

        # Concatenate compressed outputs
        combined = torch.cat(compressed_outputs, dim=1)

        # Channel Attention with SE Block
        combined = self.se_block(combined)

        # Lightweight Convolution Block
        attended = self.sep_conv(combined)

        # Expansion and Residual Connection
        adapted_outputs = []
        for i in range(self.num_encoders):
            x_expanded = self.expand_layers[i](attended)
            x_residual = x_expanded + cnn_outputs_block[i]
            # x_residual = x_expanded # no resudual connection
            adapted_outputs.append(x_residual)

        return adapted_outputs



## Adapter Model
class RFNetAdapter(nn.Module):
    def __init__(self, num_cls=4):
        super(RFNetAdapter, self).__init__()
        # Four separate encoders, one per modality
        self.flair_encoder = Encoder()
        self.t1ce_encoder  = Encoder()
        self.t1_encoder    = Encoder()
        self.t2_encoder    = Encoder()

        self.adapters = nn.ModuleList([
            AdapterBetweenEncoders(  # Changed class name
                num_encoders=4,
                in_channels=basic_dims * (2 ** i),
                compression_ratio=0.5,
            ) for i in range(4)
        ])

        # Decoders for fused prediction and separated (per modality) prediction
        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
        self.decoder_sep  = Decoder_sep(num_cls=num_cls)

        self.is_training = False  # Set to True when you want modality-specific predictions

        # Initialize convolutional weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

    def total_regularization_loss(self):
        """Handles DataParallel wrapping automatically"""
        if hasattr(self, 'module'):
            # If using DataParallel, delegate to the underlying model
            return self.module.total_regularization_loss()
        return sum(adapter.regularization_loss() for adapter in self.adapters)    

    def forward(self, x, mask):
        """
        x: Tensor of shape (B, 4, D, H, W), where the 4 channels correspond to the 4 modalities.
        mask: A mask tensor broadcastable with the decoder features (e.g., shape (B, 1, D', H', W'))
        """
        # ---- Stage 1: Get features from each encoder ----
        f_x1, f_x2, f_x3, f_x4       = self.flair_encoder(x[:, 0:1, :, :, :])
        t1c_x1, t1c_x2, t1c_x3, t1c_x4 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1,  t1_x2,  t1_x3,  t1_x4  = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1,  t2_x2,  t2_x3,  t2_x4  = self.t2_encoder(x[:, 3:4, :, :, :])

        # ---- Stage 1 Adapter ----
        stage1_features = [f_x1, t1c_x1, t1_x1, t2_x1]
        f_x1, t1c_x1, t1_x1, t2_x1 = self.adapters[0](stage1_features)

        # ---- Stage 2: Recompute features using adapted stage 1 outputs ----
        f_x2  = self.flair_encoder.e2_c1(f_x1)
        f_x2  = f_x2 + self.flair_encoder.e2_c3(self.flair_encoder.e2_c2(f_x2))
        t1c_x2 = self.t1ce_encoder.e2_c1(t1c_x1)
        t1c_x2 = t1c_x2 + self.t1ce_encoder.e2_c3(self.t1ce_encoder.e2_c2(t1c_x2))
        t1_x2  = self.t1_encoder.e2_c1(t1_x1)
        t1_x2  = t1_x2 + self.t1_encoder.e2_c3(self.t1_encoder.e2_c2(t1_x2))
        t2_x2  = self.t2_encoder.e2_c1(t2_x1)
        t2_x2  = t2_x2 + self.t2_encoder.e2_c3(self.t2_encoder.e2_c2(t2_x2))
        stage2_features = [f_x2, t1c_x2, t1_x2, t2_x2]
        f_x2, t1c_x2, t1_x2, t2_x2 = self.adapters[1](stage2_features)

        # ---- Stage 3 ----
        f_x3  = self.flair_encoder.e3_c1(f_x2)
        f_x3  = f_x3 + self.flair_encoder.e3_c3(self.flair_encoder.e3_c2(f_x3))
        t1c_x3 = self.t1ce_encoder.e3_c1(t1c_x2)
        t1c_x3 = t1c_x3 + self.t1ce_encoder.e3_c3(self.t1ce_encoder.e3_c2(t1c_x3))
        t1_x3  = self.t1_encoder.e3_c1(t1_x2)
        t1_x3  = t1_x3 + self.t1_encoder.e3_c3(self.t1_encoder.e3_c2(t1_x3))
        t2_x3  = self.t2_encoder.e3_c1(t2_x2)
        t2_x3  = t2_x3 + self.t2_encoder.e3_c3(self.t2_encoder.e3_c2(t2_x3))
        stage3_features = [f_x3, t1c_x3, t1_x3, t2_x3]
        f_x3, t1c_x3, t1_x3, t2_x3 = self.adapters[2](stage3_features)

        # ---- Stage 4 ----
        f_x4  = self.flair_encoder.e4_c1(f_x3)
        f_x4  = f_x4 + self.flair_encoder.e4_c3(self.flair_encoder.e4_c2(f_x4))
        t1c_x4 = self.t1ce_encoder.e4_c1(t1c_x3)
        t1c_x4 = t1c_x4 + self.t1ce_encoder.e4_c3(self.t1ce_encoder.e4_c2(t1c_x4))
        t1_x4  = self.t1_encoder.e4_c1(t1_x3)
        t1_x4  = t1_x4 + self.t1_encoder.e4_c3(self.t1_encoder.e4_c2(t1_x4))
        t2_x4  = self.t2_encoder.e4_c1(t2_x3)
        t2_x4  = t2_x4 + self.t2_encoder.e4_c3(self.t2_encoder.e4_c2(t2_x4))
        stage4_features = [f_x4, t1c_x4, t1_x4, t2_x4]
        f_x4, t1c_x4, t1_x4, t2_x4 = self.adapters[3](stage4_features)

        # ---- Fuse Features ----
        # Stack features from each modality along a new dimension (modality dimension)
        x1 = torch.stack((f_x1, t1c_x1, t1_x1, t2_x1), dim=1)
        x2 = torch.stack((f_x2, t1c_x2, t1_x2, t2_x2), dim=1)
        x3 = torch.stack((f_x3, t1c_x3, t1_x3, t2_x3), dim=1)
        x4 = torch.stack((f_x4, t1c_x4, t1_x4, t2_x4), dim=1)

        # Fuse prediction & auxiliary parameter predictions
        fuse_pred, prm_preds = self.decoder_fuse(x1, x2, x3, x4, mask)

        if self.is_training:
            # If in training mode, also output separate modality predictions.
            flair_pred = self.decoder_sep(f_x1, f_x2, f_x3, f_x4)
            t1ce_pred  = self.decoder_sep(t1c_x1, t1c_x2, t1c_x3, t1c_x4)
            t1_pred    = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4)
            t2_pred    = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4)
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), prm_preds

        return fuse_pred





