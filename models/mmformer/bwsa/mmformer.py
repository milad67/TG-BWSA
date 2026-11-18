import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import torch
import math
from layers import general_conv3d_prenorm, fusion_prenorm

basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 1
num_modals = 4
patch_size = 8
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.e1_c1 = nn.Conv3d(in_channels=1, out_channels=basic_dims, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=True)
        self.e1_c2 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = general_conv3d_prenorm(basic_dims, basic_dims*2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims*2, pad_type='reflect')
        self.e2_c3 = general_conv3d_prenorm(basic_dims*2, basic_dims*2, pad_type='reflect')

        self.e3_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims*4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*4, pad_type='reflect')
        self.e3_c3 = general_conv3d_prenorm(basic_dims*4, basic_dims*4, pad_type='reflect')

        self.e4_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*8, pad_type='reflect')
        self.e4_c3 = general_conv3d_prenorm(basic_dims*8, basic_dims*8, pad_type='reflect')

        self.e5_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*16, stride=2, pad_type='reflect')
        self.e5_c2 = general_conv3d_prenorm(basic_dims*16, basic_dims*16, pad_type='reflect')
        self.e5_c3 = general_conv3d_prenorm(basic_dims*16, basic_dims*16, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))

        return x1, x2, x3, x4, x5

class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_sep, self).__init__()

        self.d4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d4_c1 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims*8, basic_dims*8, k_size=1, padding=0, pad_type='reflect')

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.d4_c1(self.d4(x5))

        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))

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

        self.d4_c1 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims*8, basic_dims*8, k_size=1, padding=0, pad_type='reflect')

        self.d3_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_d4 = nn.Conv3d(in_channels=basic_dims*16, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=basic_dims*8, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=basic_dims*4, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=basic_dims*2, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        self.RFM5 = fusion_prenorm(in_channel=basic_dims*16, num_cls=num_cls)
        self.RFM4 = fusion_prenorm(in_channel=basic_dims*8, num_cls=num_cls)
        self.RFM3 = fusion_prenorm(in_channel=basic_dims*4, num_cls=num_cls)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims*2, num_cls=num_cls)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims*1, num_cls=num_cls)


    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.RFM5(x5)
        pred4 = self.softmax(self.seg_d4(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))

        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        pred3 = self.softmax(self.seg_d3(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred2 = self.softmax(self.seg_d2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.softmax(self.seg_d1(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.RFM1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4))


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_levels=1, n_points=4):
        super(Transformer, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )

        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)


    def forward(self, x, pos):
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x


class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()
    
    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        x = y.view(B, -1, H, W, Z)
        return x

#Original version
# class Model(nn.Module):
#     def __init__(self, num_cls=4):
#         super(Model, self).__init__()
#         self.flair_encoder = Encoder()
#         self.t1ce_encoder = Encoder()
#         self.t1_encoder = Encoder()
#         self.t2_encoder = Encoder()

#         ########### IntraFormer
#         self.flair_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
#         self.t1ce_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
#         self.t1_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
#         self.t2_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
#         self.flair_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*16, kernel_size=1, stride=1, padding=0)
#         self.t1ce_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*16, kernel_size=1, stride=1, padding=0)
#         self.t1_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*16, kernel_size=1, stride=1, padding=0)
#         self.t2_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*16, kernel_size=1, stride=1, padding=0)

#         self.flair_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
#         self.t1ce_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
#         self.t1_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
#         self.t2_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))

#         self.flair_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
#         self.t1ce_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
#         self.t1_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
#         self.t2_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
#         ########### IntraFormer

#         ########### InterFormer
#         self.multimodal_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim, n_levels=num_modals)
#         self.multimodal_decode_conv = nn.Conv3d(transformer_basic_dims*num_modals, basic_dims*16*num_modals, kernel_size=1, padding=0)
#         ########### InterFormer

#         self.masker = MaskModal()

#         self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
#         self.decoder_sep = Decoder_sep(num_cls=num_cls)

#         self.is_training = False

#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 torch.nn.init.kaiming_normal_(m.weight) #

#     def forward(self, x, mask):
#         #extract feature from different layers
#         flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1, :, :, :])
#         t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2, :, :, :])
#         t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3, :, :, :])
#         t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4, :, :, :])

#         ########### IntraFormer
#         flair_token_x5 = self.flair_encode_conv(flair_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
#         t1ce_token_x5 = self.t1ce_encode_conv(t1ce_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
#         t1_token_x5 = self.t1_encode_conv(t1_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
#         t2_token_x5 = self.t2_encode_conv(t2_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)

#         flair_intra_token_x5 = self.flair_transformer(flair_token_x5, self.flair_pos)
#         t1ce_intra_token_x5 = self.t1ce_transformer(t1ce_token_x5, self.t1ce_pos)
#         t1_intra_token_x5 = self.t1_transformer(t1_token_x5, self.t1_pos)
#         t2_intra_token_x5 = self.t2_transformer(t2_token_x5, self.t2_pos)

#         flair_intra_x5 = flair_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
#         t1ce_intra_x5 = t1ce_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
#         t1_intra_x5 = t1_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
#         t2_intra_x5 = t2_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()

#         if self.is_training:
#             flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)
#             t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
#             t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
#             t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)
#         ########### IntraFormer

#         x1 = self.masker(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1), mask) #Bx4xCxHWZ
#         x2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
#         x3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)
#         x4 = self.masker(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1), mask)
#         x5_intra = self.masker(torch.stack((flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5), dim=1), mask)

#         ########### InterFormer
#         flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5 = torch.chunk(x5_intra, num_modals, dim=1) 
#         multimodal_token_x5 = torch.cat((flair_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
#                                          t1ce_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
#                                          t1_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
#                                          t2_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
#                                         ), dim=1)
#         multimodal_pos = torch.cat((self.flair_pos, self.t1ce_pos, self.t1_pos, self.t2_pos), dim=1)
#         multimodal_inter_token_x5 = self.multimodal_transformer(multimodal_token_x5, multimodal_pos)
#         multimodal_inter_x5 = self.multimodal_decode_conv(multimodal_inter_token_x5.view(multimodal_inter_token_x5.size(0), patch_size, patch_size, patch_size, transformer_basic_dims*num_modals).permute(0, 4, 1, 2, 3).contiguous())
#         x5_inter = multimodal_inter_x5

#         fuse_pred, preds = self.decoder_fuse(x1, x2, x3, x4, x5_inter)
#         ########### InterFormer
        
#         if self.is_training:
#             return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), preds
#         return fuse_pred



# Adapter Version
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
            adapted_outputs.append(x_residual)

        return adapted_outputs



# # ---------------------------
# # ScalingLayer3d with dual scaling.
# # ---------------------------
# class ScalingLayer3d(nn.Module):
#     def __init__(self, initial_alpha=0.5, initial_beta=1.0, block_idx=0):
#         super(ScalingLayer3d, self).__init__()
#         self.block_idx = block_idx
#         self.alpha = nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32))
#         self.beta  = nn.Parameter(torch.tensor(initial_beta, dtype=torch.float32))
    
#     def forward(self, adapter_out, original):
#         return self.alpha * adapter_out + self.beta * original + 1e-3 * original

# # ---------------------------
# # AdapterBetweenEncoders (3D) with scaling.
# # ---------------------------
# class AdapterBetweenEncoders(nn.Module):
#     def __init__(self, num_encoders, in_channels, compression_ratio=0.5, dropout_rate=0.1):
#         super(AdapterBetweenEncoders, self).__init__()
#         compressed_channels = max(1, int(in_channels * compression_ratio))
#         self.num_encoders = num_encoders

#         self.compress_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv3d(in_channels, compressed_channels, kernel_size=1, padding=0),
#                 nn.BatchNorm3d(compressed_channels),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout3d(p=dropout_rate)
#             ) for _ in range(num_encoders)
#         ])

#         self.se_block = SEBlock(channels=compressed_channels * num_encoders, reduction_ratio=16, dropout_rate=dropout_rate)

#         self.sep_conv = nn.Sequential(
#             nn.Conv3d(compressed_channels * num_encoders, compressed_channels * num_encoders, kernel_size=3, padding=1, groups=compressed_channels * num_encoders),
#             nn.Conv3d(compressed_channels * num_encoders, compressed_channels * num_encoders, kernel_size=1, padding=0),
#             nn.BatchNorm3d(compressed_channels * num_encoders),
#             nn.ReLU(inplace=True),
#             nn.Dropout3d(p=dropout_rate)
#         )

#         self.expand_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv3d(compressed_channels * num_encoders, in_channels, kernel_size=1, padding=0),
#                 nn.BatchNorm3d(in_channels),
#                 nn.ReLU(inplace=True)
#             ) for _ in range(num_encoders)
#         ])

#         self.scaling_layers = nn.ModuleList([
#             ScalingLayer3d(initial_alpha=0.5, initial_beta=1.0, block_idx=i)
#             for i in range(num_encoders)
#         ])

#     def forward(self, cnn_outputs_block):
#         compressed_outputs = [compress_layer(x) for compress_layer, x in zip(self.compress_layers, cnn_outputs_block)]
#         combined = torch.cat(compressed_outputs, dim=1)
#         combined = self.se_block(combined)
#         attended = self.sep_conv(combined)
#         adapted_outputs = []
#         for i in range(self.num_encoders):
#             x_expanded = self.expand_layers[i](attended)
#             x_scaled = self.scaling_layers[i](x_expanded, cnn_outputs_block[i])
#             adapted_outputs.append(x_scaled)
#         return adapted_outputs


class Model(nn.Module):
    def __init__(self, num_cls=4):
        super(Model, self).__init__()

        # Four separate encoders, one per modality
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        # Add adapters between encoders for feature fusion
        self.adapters = nn.ModuleList([
            AdapterBetweenEncoders(  # Changed class name
                num_encoders=4, 
                in_channels=basic_dims * (2 ** i),
                compression_ratio=0.5,
            ) for i in range(5)  # There are now 5 stages
        ])

        ########### IntraFormer (modality-specific transformers)
        self.flair_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1ce_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t2_encode_conv = nn.Conv3d(basic_dims*16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.flair_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*16, kernel_size=1, stride=1, padding=0)
        self.t1ce_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*16, kernel_size=1, stride=1, padding=0)
        self.t1_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*16, kernel_size=1, stride=1, padding=0)
        self.t2_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims*16, kernel_size=1, stride=1, padding=0)

        self.flair_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t1ce_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t1_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))
        self.t2_pos = nn.Parameter(torch.zeros(1, patch_size**3, transformer_basic_dims))

        self.flair_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t1ce_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t1_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t2_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        ########### IntraFormer

        ########### InterFormer (multimodal transformer)
        self.multimodal_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim, n_levels=num_modals)
        self.multimodal_decode_conv = nn.Conv3d(transformer_basic_dims*num_modals, basic_dims*16*num_modals, kernel_size=1, padding=0)
        ########### InterFormer

        self.masker = MaskModal()

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)

        self.is_training = False  # Set to True when you want modality-specific predictions

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

    def total_regularization_loss(self):
        """Handles DataParallel wrapping automatically"""
        if hasattr(self, 'module'):
            return self.module.total_regularization_loss()
        return sum(adapter.regularization_loss() for adapter in self.adapters)

    def forward(self, x, mask):
        """
        x: Tensor of shape (B, 4, D, H, W), where the 4 channels correspond to the 4 modalities.
        mask: A mask tensor broadcastable with the decoder features (e.g., shape (B, 1, D', H', W'))
        """
        # ---- Stage 1: Get features from each encoder ----
        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4, :, :, :])

        # ---- Stage 1 Adapter ----
        stage1_features = [flair_x1, t1ce_x1, t1_x1, t2_x1]
        flair_x1, t1ce_x1, t1_x1, t2_x1 = self.adapters[0](stage1_features)

        # ---- Stage 2: Recompute features using adapted stage 1 outputs ----
        flair_x2 = self.flair_encoder.e2_c1(flair_x1)
        flair_x2 = flair_x2 + self.flair_encoder.e2_c3(self.flair_encoder.e2_c2(flair_x2))
        t1ce_x2 = self.t1ce_encoder.e2_c1(t1ce_x1)
        t1ce_x2 = t1ce_x2 + self.t1ce_encoder.e2_c3(self.t1ce_encoder.e2_c2(t1ce_x2))
        t1_x2 = self.t1_encoder.e2_c1(t1_x1)
        t1_x2 = t1_x2 + self.t1_encoder.e2_c3(self.t1_encoder.e2_c2(t1_x2))
        t2_x2 = self.t2_encoder.e2_c1(t2_x1)
        t2_x2 = t2_x2 + self.t2_encoder.e2_c3(self.t2_encoder.e2_c2(t2_x2))

        stage2_features = [flair_x2, t1ce_x2, t1_x2, t2_x2]
        flair_x2, t1ce_x2, t1_x2, t2_x2 = self.adapters[1](stage2_features)

        # ---- Stage 3 ----
        flair_x3 = self.flair_encoder.e3_c1(flair_x2)
        flair_x3 = flair_x3 + self.flair_encoder.e3_c3(self.flair_encoder.e3_c2(flair_x3))
        t1ce_x3 = self.t1ce_encoder.e3_c1(t1ce_x2)
        t1ce_x3 = t1ce_x3 + self.t1ce_encoder.e3_c3(self.t1ce_encoder.e3_c2(t1ce_x3))
        t1_x3 = self.t1_encoder.e3_c1(t1_x2)
        t1_x3 = t1_x3 + self.t1_encoder.e3_c3(self.t1_encoder.e3_c2(t1_x3))
        t2_x3 = self.t2_encoder.e3_c1(t2_x2)
        t2_x3 = t2_x3 + self.t2_encoder.e3_c3(self.t2_encoder.e3_c2(t2_x3))

        stage3_features = [flair_x3, t1ce_x3, t1_x3, t2_x3]
        flair_x3, t1ce_x3, t1_x3, t2_x3 = self.adapters[2](stage3_features)

        # ---- Stage 4 ----
        flair_x4 = self.flair_encoder.e4_c1(flair_x3)
        flair_x4 = flair_x4 + self.flair_encoder.e4_c3(self.flair_encoder.e4_c2(flair_x4))
        t1ce_x4 = self.t1ce_encoder.e4_c1(t1ce_x3)
        t1ce_x4 = t1ce_x4 + self.t1ce_encoder.e4_c3(self.t1ce_encoder.e4_c2(t1ce_x4))
        t1_x4 = self.t1_encoder.e4_c1(t1_x3)
        t1_x4 = t1_x4 + self.t1_encoder.e4_c3(self.t1_encoder.e4_c2(t1_x4))
        t2_x4 = self.t2_encoder.e4_c1(t2_x3)
        t2_x4 = t2_x4 + self.t2_encoder.e4_c3(self.t2_encoder.e4_c2(t2_x4))

        stage4_features = [flair_x4, t1ce_x4, t1_x4, t2_x4]
        flair_x4, t1ce_x4, t1_x4, t2_x4 = self.adapters[3](stage4_features)

        # ---- Stage 5 ---- (Final Stage)
        flair_x5 = self.flair_encoder.e5_c1(flair_x4)
        flair_x5 = flair_x5 + self.flair_encoder.e5_c3(self.flair_encoder.e5_c2(flair_x5))
        t1ce_x5 = self.t1ce_encoder.e5_c1(t1ce_x4)
        t1ce_x5 = t1ce_x5 + self.t1ce_encoder.e5_c3(self.t1ce_encoder.e5_c2(t1ce_x5))
        t1_x5 = self.t1_encoder.e5_c1(t1_x4)
        t1_x5 = t1_x5 + self.t1_encoder.e5_c3(self.t1_encoder.e5_c2(t1_x5))
        t2_x5 = self.t2_encoder.e5_c1(t2_x4)
        t2_x5 = t2_x5 + self.t2_encoder.e5_c3(self.t2_encoder.e5_c2(t2_x5))

        stage5_features = [flair_x5, t1ce_x5, t1_x5, t2_x5]
        flair_x5, t1ce_x5, t1_x5, t2_x5 = self.adapters[4](stage5_features)


        ########### IntraFormer
        flair_token_x5 = self.flair_encode_conv(flair_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
        t1ce_token_x5 = self.t1ce_encode_conv(t1ce_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
        t1_token_x5 = self.t1_encode_conv(t1_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)
        t2_token_x5 = self.t2_encode_conv(t2_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims)

        flair_intra_token_x5 = self.flair_transformer(flair_token_x5, self.flair_pos)
        t1ce_intra_token_x5 = self.t1ce_transformer(t1ce_token_x5, self.t1ce_pos)
        t1_intra_token_x5 = self.t1_transformer(t1_token_x5, self.t1_pos)
        t2_intra_token_x5 = self.t2_transformer(t2_token_x5, self.t2_pos)

        flair_intra_x5 = flair_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t1ce_intra_x5 = t1ce_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t1_intra_x5 = t1_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t2_intra_x5 = t2_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size, transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()

        if self.is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)
        ########### IntraFormer

        x1 = self.masker(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1), mask) #Bx4xCxHWZ
        x2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
        x3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)
        x4 = self.masker(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1), mask)
        x5_intra = self.masker(torch.stack((flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5), dim=1), mask)

        ########### InterFormer
        flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5 = torch.chunk(x5_intra, num_modals, dim=1) 
        multimodal_token_x5 = torch.cat((flair_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                         t1ce_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                         t1_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                         t2_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1, transformer_basic_dims),
                                        ), dim=1)
        multimodal_pos = torch.cat((self.flair_pos, self.t1ce_pos, self.t1_pos, self.t2_pos), dim=1)
        multimodal_inter_token_x5 = self.multimodal_transformer(multimodal_token_x5, multimodal_pos)
        multimodal_inter_x5 = self.multimodal_decode_conv(multimodal_inter_token_x5.view(multimodal_inter_token_x5.size(0), patch_size, patch_size, patch_size, transformer_basic_dims*num_modals).permute(0, 4, 1, 2, 3).contiguous())
        x5_inter = multimodal_inter_x5

        fuse_pred, preds = self.decoder_fuse(x1, x2, x3, x4, x5_inter)
        ########### InterFormer
        
        if self.is_training:
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), preds
        return fuse_pred
