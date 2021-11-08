import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, inplanes, planes):
        super(DoubleConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(num_features=planes),
            nn.ReLU(inplace=True),
            nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(num_features=planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Downsample(nn.Module):
    def __init__(self, inplanes, planes):
        super(Downsample, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubleConv(inplanes=inplanes, planes=planes)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, inplanes, planes):
        super(Upsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = DoubleConv(inplanes=inplanes, planes=planes)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, (diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2, diffD // 2, diffD - diffD // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class BayesianUNet3D(nn.Module):
    def __init__(self, inplanes=1, num_classes=1, planes=32, has_dropout=True, dropout_rate=0.5):
        super(BayesianUNet3D, self).__init__()
        self.has_dropout = has_dropout
        self.dropout_rate = dropout_rate

        self.in_conv = DoubleConv(inplanes=inplanes, planes=planes)
        self.downsample1 = Downsample(inplanes=planes, planes=planes * 2)
        self.downsample2 = Downsample(inplanes=planes * 2, planes=planes * 4)
        self.downsample3 = Downsample(inplanes=planes * 4, planes=planes * 8)
        self.downsample4 = Downsample(inplanes=planes * 8, planes=planes * 16)
        self.upsample1 = Upsample(inplanes=planes * 24, planes=planes * 4)
        self.upsample2 = Upsample(inplanes=planes * 8, planes=planes * 2)
        self.upsample3 = Upsample(inplanes=planes * 4, planes=planes)
        self.upsample4 = Upsample(inplanes=planes * 2, planes=planes)
        self.out_conv = nn.Conv3d(planes, num_classes, kernel_size=(3, 3, 3), padding=1)

    def forward(self, x, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False

        x1 = self.in_conv(x)                                  # shape=[batch, planes, D, H, W]
        if self.has_dropout:
            x1 = F.dropout3d(x1, p=self.dropout_rate, training=True)

        x2 = self.downsample1(x1)                        # shape=[batch, planes * 2, D / 2, H / 2, W / 2]
        if self.has_dropout:
            x2 = F.dropout3d(x2, p=self.dropout_rate, training=True)

        x3 = self.downsample2(x2)                        # shape=[batch, planes * 4, D / 4, H / 4, W / 4]
        if self.has_dropout:
            x3 = F.dropout3d(x3, p=self.dropout_rate, training=True)

        x4 = self.downsample3(x3)                        # shape=[batch, planes * 8, D / 8, H / 8, W / 8]
        if self.has_dropout:
            x4 = F.dropout3d(x4, p=self.dropout_rate, training=True)

        x5 = self.downsample4(x4)                        # shape=[batch, planes * 16, D / 16, H / 16, W / 16]
        if self.has_dropout:
            x5 = F.dropout3d(x5, p=self.dropout_rate, training=True)

        x = self.upsample1(x5, x4)                         # shape=[batch, planes * 4, D / 8, H / 8, W / 8]
        if self.has_dropout:
            x = F.dropout3d(x, p=self.dropout_rate, training=True)

        x = self.upsample2(x, x3)
        if self.has_dropout:
            x = F.dropout3d(x, p=self.dropout_rate, training=True)

        x = self.upsample3(x, x2)
        if self.has_dropout:
            x = F.dropout3d(x, p=self.dropout_rate, training=True)

        x = self.upsample4(x, x1)
        if self.has_dropout:
            x = F.dropout3d(x, p=self.dropout_rate, training=True)

        x = self.out_conv(x)

        if turnoff_drop:
            self.has_dropou = has_dropout
        return x
