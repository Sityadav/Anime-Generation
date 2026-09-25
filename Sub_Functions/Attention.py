from __future__ import absolute_import
import math
import numpy as np
from torch.nn import init
from tensorflow.keras import backend as K
from scipy.sparse import csr_matrix
from sklearn.utils import check_consistent_length
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import type_of_target
from sklearn.utils import shuffle
from tensorflow.keras.layers import *
from keras.layers import *
import random
import torch
from torch import nn
import torch.nn.functional as F
import typing


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class MutualCrossAttn(nn.Module):
    def __init__(self, feature_dim=2048, num_head=32):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_head = num_head

        self.layer_norm1 = nn.LayerNorm(self.feature_dim, eps=1e-6).apply(_init_vit_weights)
        self.layer_norm2 = nn.LayerNorm(self.feature_dim, eps=1e-6).apply(_init_vit_weights)

    def forward(self, f_a, f_s):
        B, N, C = f_a.shape

        q1 = f_a
        k1 = v1 = f_s
        # [B, N, C] -> [B, N, n, C//n] -> [B, n, N, C//n]
        q1 = q1.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)
        k1 = k1.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)
        v1 = v1.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)

        attn1 = torch.matmul(q1 / ((C // self.num_head) ** 0.5),
                             k1.transpose(-1, -2))
        attn1 = F.softmax(attn1, dim=-1)
        output1 = torch.matmul(attn1, v1)
        # [B, n, N, C//n] -> [B, N, n, C//n] -> [B, N, C]
        output1 = output1.transpose(1, 2).contiguous().flatten(2)

        q2 = f_s
        k2 = v2 = f_a
        # [B, N, C] -> [B, N, n, C//n] -> [B, n, N, C//n]
        q2 = q2.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)
        k2 = k2.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)
        v2 = v2.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)

        attn2 = torch.matmul(q2 / ((C // self.num_head) ** 0.5),
                             k2.transpose(-1, -2))  # attn2 matrix is equal to the transpose of attn1
        attn2 = F.softmax(attn2, dim=-1)
        output2 = torch.matmul(attn2, v2)
        # [B, n, N, C//n] -> [B, N, n, C//n] -> [B, N, C]
        output2 = output2.transpose(1, 2).contiguous().flatten(2)

        output1 = self.layer_norm1(output1 + f_a)
        output2 = self.layer_norm2(output2 + f_s)

        return output1, output2


def mutual_attention(x):

    weights = x.node.layer.trainable_weights
    sel_wei1 = weights[0]
    sel_wei = np.array(sel_wei1)
    sel_wei = np.resize(sel_wei, (sel_wei.shape[0] * sel_wei.shape[1], sel_wei.shape[2], sel_wei.shape[3]))
    dim = sel_wei.shape[2]
    sel_wei = torch.tensor(sel_wei)
    mutual_att = MutualCrossAttn(feature_dim=dim)
    output1, output2 = mutual_att(sel_wei, sel_wei)
    output1 = output1.detach().cpu().numpy()
    output2 = output2.detach().cpu().numpy()
    output = output1 + output2
    output = np.resize(output, (sel_wei1.shape[0], sel_wei1.shape[1], sel_wei1.shape[2], sel_wei1.shape[3]))
    weights[0] = output
    x.node.layer.trainable_weights_ = weights

    return x

def channel_attention_module(input_feature, ratio=8):
    channel_axis = -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


class ExternalAttention(nn.Module):

    def __init__(self, d_model,S=64):
        super().__init__()
        self.mk=nn.Linear(d_model,S,bias=False)
        self.mv=nn.Linear(S,d_model,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn=self.mk(queries) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model

        return out


def External_attention(x):

    weights = x.node.layer.trainable_weights
    sel_wei1 = weights[0]
    sel_wei = np.array(sel_wei1)
    sel_wei = np.resize(sel_wei, (sel_wei.shape[0] * sel_wei.shape[1], sel_wei.shape[2], sel_wei.shape[3]))
    dim = sel_wei.shape[2]
    sel_wei = torch.tensor(sel_wei)
    external_att = ExternalAttention(d_model=dim,S=8)
    output = external_att(sel_wei)
    output = output.detach().cpu().numpy()
    output = np.resize(output, (sel_wei1.shape[0], sel_wei1.shape[1], sel_wei1.shape[2], sel_wei1.shape[3]))
    weights[0] = output
    x.node.layer.trainable_weights_ = weights

    return x


class Attention(nn.Module):
    def __init__(self, num_channels, embed_size, dropout=True):
        """Stacked attention Module
        """
        super(Attention, self).__init__()
        self.ff_image = nn.Linear(embed_size, num_channels)
        self.ff_questions = nn.Linear(embed_size, num_channels)
        self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(num_channels, 1)

    def forward(self, vi, vq):
        """Extract feature vector from image vector.

        """
        hi = self.ff_image(vi)
        hq = self.ff_questions(vq).unsqueeze(dim=1)
        ha = torch.tanh(hi + hq)
        if self.dropout:
            ha = self.dropout(ha)
        ha = self.ff_attention(ha)
        pi = torch.softmax(ha, dim=1)
        self.pi = pi
        vi_attended = (pi * vi).sum(dim=1)
        u = vi_attended + vq
        return u


class SANModel(nn.Module):
    # num_attention_layer and num_mlp_layer not implemented yet
    def __init__(self, embed_size, ans_vocab_size):
        super(SANModel, self).__init__()
        self.num_attention_layer = 2
        self.num_mlp_layer = 1
        self.san = nn.ModuleList([Attention(512, embed_size)] * self.num_attention_layer)
        self.tanh = nn.Tanh()
        self.mlp = nn.Sequential(nn.Dropout(p=0.5),
                                 nn.Linear(embed_size, ans_vocab_size))
        self.attn_features = []  ## attention features

    def forward(self, img, qst):

        vi = img
        u = qst
        for attn_layer in self.san:
            u = attn_layer(vi, u)
        #             self.attn_features.append(attn_layer.pi)

        combined_feature = self.mlp(u)
        return combined_feature


def stacked_attention(x):

    weights = x.node.layer.trainable_weights
    sel_wei = weights[1]
    sel_wei = np.array(sel_wei)
    sel_wei = sel_wei.reshape(8, 8)
    dim = sel_wei.shape[1]
    sel_wei = torch.tensor(sel_wei)
    aa = SANModel(dim, dim)
    output = aa(sel_wei, sel_wei)
    output = output.detach().cpu().numpy()
    output = np.resize(output, (sel_wei.shape[0]*sel_wei.shape[1]))
    weights[1] = output
    x.node.layer.trainable_weights_ = weights

    return x


NoneFloat = typing.Union[None, float]


class ViTAttention(nn.Module):
    def __init__(self,dim: int,chan: int,num_heads: int=1,qkv_bias: bool=False,qk_scale: NoneFloat=None):

        super().__init__()

        ## Define Constants
        self.num_heads = num_heads
        self.chan = chan
        self.head_dim = self.chan // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert self.chan % self.num_heads == 0, '"Chan" must be evenly divisible by "num_heads".'

        ## Define Layers
        self.qkv = nn.Linear(dim, chan * 3, bias=qkv_bias)
        #### Each token gets projected from starting length (dim) to channel length (chan) 3 times (for each Q, K, V)
        self.proj = nn.Linear(chan, chan)

    def forward(self, x):
        B, N, C = x.shape
        ## Dimensions: (batch, num_tokens, token_len)

        ## Calcuate QKVs
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        #### Dimensions: (3, batch, heads, num_tokens, chan/num_heads = head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        ## Calculate Attention
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        #### Dimensions: (batch, heads, num_tokens, num_tokens)

        ## Attention Layer
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.chan)
        #### Dimensions: (batch, heads, num_tokens, chan)

        ## Projection Layers
        x = self.proj(x)

        ## Skip Connection Layer
        v = v.transpose(1, 2).reshape(B, N, self.chan)
        x = v + x
        #### Because the original x has different size with current x, use v to do skip connection

        return x


def ViT_attention(x):

    weights = x.node.layer.trainable_weights
    sel_wei1 = weights[0]
    sel_wei = np.array(sel_wei1)
    sel_wei = np.resize(sel_wei, (sel_wei.shape[0] * sel_wei.shape[1], sel_wei.shape[2], sel_wei.shape[3]))
    dim = sel_wei.shape[0]
    channel = sel_wei.shape[-1]
    sel_wei = torch.tensor(sel_wei)
    vit_att = ViTAttention(channel, dim)
    output = vit_att(sel_wei)
    output = output.detach().cpu().numpy()
    output = np.resize(output, (sel_wei1.shape[0], sel_wei1.shape[1], sel_wei1.shape[2], sel_wei1.shape[3]))
    weights[0] = output
    x.node.layer.trainable_weights_ = weights

    return x


def make_3d_network():
    return Network_3d()


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1   = nn.Conv3d(16, 16 // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv3d(16 // 16, 16, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.conv1 = nn.Conv3d(16, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(self.avg_pool(x))
        return self.sigmoid(x)


class Network_3d(nn.Module):

    def __init__(self, in_chanel, downsample=None):
        super(Network_3d, self).__init__()

        planes =16
        self.K, S, P, OP = (3, [2, 4, 4], 1, [ 0, 3, 3])

        self.base = nn.Sequential(
            nn.Conv3d(in_channels=in_chanel, out_channels=planes, kernel_size= 3,padding=1),
            nn.PReLU(),
        )
        self.next = nn.Sequential(
            nn.Conv3d(in_channels=planes, out_channels=planes, kernel_size= 3,padding=1),
            nn.PReLU(),
        )

        self.r = nn.PReLU()

        self.last = nn.Conv3d(16, 3, kernel_size=3, padding=1)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        # self.downsample = downsample
        # self.stride = stride

        self.upFrame = nn.Sequential(
            nn.ConvTranspose3d(16, 16, kernel_size=self.K, stride=S, padding=P, output_padding=OP),
            nn.PReLU(),
            nn.Conv3d(16, in_chanel, kernel_size=3, stride=S, padding=1)
        )

    def forward(self, x):

        a = 3
        # b, c, n, h, w = x.shape
        old = x
        # old = x[:, 1:2, :, :]
        ##Block1
        out = self.base(x)
        out = self.next(out)

        for i in range(a):
            out = self.next(out)
            out = self.next(out)
            out = self.ca(out) * out
            out = self.sa(out) * out

            out = self.r(out)

        out = self.upFrame(out)

        return out