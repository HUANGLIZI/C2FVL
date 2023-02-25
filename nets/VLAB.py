# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

'''pixel-level module'''


class VLAlignBlock(nn.Module):
    def __init__(self, in_channels):
        super(VLAlignBlock, self).__init__()
        self.middle_layer_size_ratio = 2 
        self.conv_avg = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_avg = nn.ReLU(inplace=True)
        self.linear_avg = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.conv_max = nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.relu_max = nn.ReLU(inplace=True)
        self.linear_max = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.bottleneck = nn.Sequential(
            nn.Linear(1, 2 * self.middle_layer_size_ratio),  # 2, 2*self.
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.middle_layer_size_ratio, 1)
        )
        self.conv_sig = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    '''forward'''

    def forward(self, x):  # b, c, h, w [4, 3, 128, 128]
        x_avg = self.conv_avg(x)  # b, c, h, w
        x_avg = x_avg.transpose(1, 3)
        x_avg = self.linear_avg(x_avg)
        x_avg = x_avg.transpose(1, 3)
        x_avg = self.relu_avg(x_avg)  # b, c, h, w
        x_avg = torch.mean(x_avg, dim=1)  # b, h, w
        x_avg = x_avg.unsqueeze(dim=1)  
        x_max = self.conv_max(x)
        x_max = x_max.transpose(1, 3)
        x_max = self.linear_max(x_max)
        x_max = x_max.transpose(1, 3)
        x_max = self.relu_max(x_max)  # b, c, h, w
        x_max = torch.max(x_max, dim=1).values
        x_max = x_max.unsqueeze(dim=1)

        x_output = x_avg+x_max  # b, 2c, h, w
        x_output = x_output.transpose(1, 3)  # b, w, h, 2c
        x_output = self.bottleneck(x_output)
        x_output = x_output.transpose(1, 3) 
        y = x_output * x
        return y
