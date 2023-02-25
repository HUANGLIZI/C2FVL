# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .Vit import VisionTransformer, Reconstruct,Reconstruct_text,Process_text
from .VLAB import VLAlignBlock
from sklearn.metrics.pairwise import cosine_similarity

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.vlab = VLAlignBlock(in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.vlab(skip_x)########PLAM output###########
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        #x = torch.cat([skip_x, up], dim=1)
        return self.nConvs(x),skip_x_att


class C2FVL(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.downVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.downVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.downVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.downVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.upVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.upVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.upVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.upVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpblockAttention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpblockAttention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpblockAttention(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpblockAttention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss
        self.multi_activation = nn.Softmax()
        self.reconstruct1 = Reconstruct(in_channels=64, out_channels=64, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = Reconstruct(in_channels=128, out_channels=128, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct3 = Reconstruct(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(4, 4))
        self.reconstruct4 = Reconstruct(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(2, 2))
        self.vlab1 = VLAlignBlock(64)
        self.vlab2 = VLAlignBlock(128)
        self.vlab3 = VLAlignBlock(256)
        self.vlab4 = VLAlignBlock(512)
        self.text_module4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.text_module3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.text_module2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.text_module1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.reconstruct_text1 = Reconstruct_text(in_channels=64, out_channels=64, kernel_size=1, scale_factor=(112, 56))
        self.reconstruct_text2 = Reconstruct_text(in_channels=128, out_channels=128, kernel_size=1, scale_factor=(56, 28))
        self.reconstruct_text3 = Reconstruct_text(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(28, 14))
        self.reconstruct_text4 = Reconstruct_text(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(14, 7))
        self.process_text1=Process_text(in_channels=8, out_channels=64, kernel_size=1)
        self.process_text2=Process_text(in_channels=8, out_channels=128, kernel_size=1)
        self.process_text3=Process_text(in_channels=8, out_channels=256, kernel_size=1)
        self.process_text4=Process_text(in_channels=8, out_channels=512, kernel_size=1)
        self.upsample4 = nn.Upsample(scale_factor=8)
        self.upsample3 = nn.Upsample(scale_factor=4)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.outc4 = nn.Conv2d(in_channels*8, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.outc3 = nn.Conv2d(in_channels*4, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.outc2 = nn.Conv2d(in_channels*2, n_classes, kernel_size=(1, 1), stride=(1, 1))
        

    def forward(self, x, text):
        x = x.float()  # x [4,3,224,224]
        x1 = self.inc(x)  # x1 [4, 64, 224, 224]
        y1 = self.downVit(x1, x1)
        x2 = self.down1(x1)
        y2 = self.downVit1(x2, y1)
        x3 = self.down2(x2)
        y3 = self.downVit2(x3, y2)
        x4 = self.down3(x3)
        y4 = self.downVit3(x4, y3)
        x5 = self.down4(x4)
        text1=self.process_text1(text,x1.shape[1])
        x1 = self.reconstruct1(y1) + x1
        x1=x1*text1
        text2=self.process_text2(text,x2.shape[1])
        x2=self.reconstruct2(y2) + x2
        x2=x2*text2
        text3=self.process_text3(text,x3.shape[1])
        x3=self.reconstruct3(y3) + x3
        x3=x3*text3
        text4=self.process_text4(text,x4.shape[1])
        x4=self.reconstruct4(y4) + x4
        x4=x4*text4
        
        x,plamy4 = self.up4(x5, x4)
        x,plamy3 = self.up3(x, x3)
        x,plamy2 = self.up2(x, x2)
        x,plamy1 = self.up1(x, x1)
        
        ###################palm output upsample caculate cosine loss###############################
        plamy4=self.upsample4(plamy4)
        plamy3=self.upsample3(plamy3)
        plamy2=self.upsample2(plamy2)
        plamy4=self.outc4(plamy4)####4st layer#####
        plamy3=self.outc3(plamy3)####3st layer#####
        plamy2=self.outc2(plamy2)
        plamy1=self.outc(plamy1)
        
        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)  # if not using BCEWithLogitsLoss or class>1
        #print('logits shape:',logits.shape)
        return logits,plamy1,plamy2,plamy3,plamy4
