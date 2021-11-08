import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, num_groups=8):
        super(ConvBlock, self).__init__()
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels=planes, out_channels=planes, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x += residual
        return x


class LinearUpSampling(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2, mode="trilinear", align_corners=True):
        super(LinearUpSampling, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.conv1 = nn.Conv3d(in_channels=inplanes, out_channels=planes, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels=inplanes, out_channels=planes, kernel_size=1)

    def forward(self, x, skipx=None):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        if skipx is not None:
            x = torch.cat((x, skipx), 1)
            x = self.conv2(x)
        return x


class SegNet3D(nn.Module):
    def __init__(self, inplanes, planes, num_classes, has_dropout=True, dropout_rate=0.2):
        super(SegNet3D, self).__init__()
        self.has_dropout = has_dropout
        self.dropout_rate = dropout_rate
        self.in_conv = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.en_block1_1 = ConvBlock(planes, planes, kernel_size=3, stride=1, padding=1, num_groups=8)
        self.en_down2 = nn.Conv3d(planes, planes * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.en_block2_1 = ConvBlock(planes * 2, planes * 2, kernel_size=3, stride=1, padding=1, num_groups=8)
        self.en_block2_2 = ConvBlock(planes * 2, planes * 2, kernel_size=3, stride=1, padding=1, num_groups=8)
        self.en_down3 = nn.Conv3d(planes * 2, planes * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.en_block3_1 = ConvBlock(planes * 4, planes * 4, kernel_size=3, stride=1, padding=1, num_groups=8)
        self.en_block3_2 = ConvBlock(planes * 4, planes * 4, kernel_size=3, stride=1, padding=1, num_groups=8)
        self.en_down4 = nn.Conv3d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.en_block4_1 = ConvBlock(planes * 8, planes * 8, kernel_size=3, stride=1, padding=1, num_groups=8)
        self.en_block4_2 = ConvBlock(planes * 8, planes * 8, kernel_size=3, stride=1, padding=1, num_groups=8)
        self.en_block4_3 = ConvBlock(planes * 8, planes * 8, kernel_size=3, stride=1, padding=1, num_groups=8)
        self.en_block4_4 = ConvBlock(planes * 8, planes * 8, kernel_size=3, stride=1, padding=1, num_groups=8)

        self.de_up3 = LinearUpSampling(planes * 8, planes * 4, scale_factor=2)
        self.de_block3_1 = ConvBlock(planes * 4, planes * 4, kernel_size=3, stride=1, padding=1, num_groups=8)
        self.de_up2 = LinearUpSampling(planes * 4, planes * 2, scale_factor=2)
        self.de_block2_1 = ConvBlock(planes * 2, planes * 2, kernel_size=3, stride=1, padding=1, num_groups=8)
        self.de_up1 = LinearUpSampling(planes * 2, planes, scale_factor=2)
        self.de_block1_1 = ConvBlock(planes, planes, kernel_size=3, stride=1, padding=1, num_groups=8)
        self.out_conv = nn.Conv3d(planes, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x, turnoff_dropout=False):
        if turnoff_dropout:
            has_dropout = self.has_dropout
            self.has_dropout = False

        x1 = self.in_conv(x)
        if self.has_dropout:
            x1 = F.dropout3d(x1, p=self.dropout_rate, training=True)
        x1 = self.en_block1_1(x1)
        x2 = self.en_down2(x1)
        if self.has_dropout:
            x2 = F.dropout3d(x2, p=self.dropout_rate, training=True)
        x2 = self.en_block2_1(x2)
        x2 = self.en_block2_2(x2)
        x3 = self.en_down3(x2)
        if self.has_dropout:
            x3 = F.dropout3d(x3, p=self.dropout_rate, training=True)
        x3 = self.en_block3_1(x3)
        x3 = self.en_block3_2(x3)
        x4 = self.en_down4(x3)
        if self.has_dropout:
            x4 = F.dropout3d(x4, p=self.dropout_rate, training=True)
        x4 = self.en_block4_1(x4)
        x4 = self.en_block4_2(x4)
        x4 = self.en_block4_3(x4)
        x4 = self.en_block4_4(x4)

        x = self.de_up3(x4, x3)
        if self.has_dropout:
            x = F.dropout3d(x, p=self.dropout_rate, training=True)
        x = self.de_block3_1(x)
        x = self.de_up2(x, x2)
        if self.has_dropout:
            x = F.dropout3d(x, p=self.dropout_rate, training=True)
        x = self.de_block2_1(x)
        x = self.de_up1(x, x1)
        if self.has_dropout:
            x = F.dropout3d(x, p=self.dropout_rate, training=True)
        x = self.de_block1_1(x)
        x = self.out_conv(x)

        if turnoff_dropout:
            self.has_dropou = has_dropout
        return x
