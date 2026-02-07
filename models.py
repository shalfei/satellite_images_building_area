import torchvision.models as models
from torchvision.models import resnet34, ResNet34_Weights
import torch.nn.functional as F
import torch
import torch.nn as nn

class MultiTaskBuildingModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        res34 = models.resnet34(pretrained=pretrained)

        self.initial = nn.Sequential(res34.conv1, res34.bn1, res34.relu) # 256x256
        self.maxpool = res34.maxpool # MaxPool layer
        self.layer1 = res34.layer1 # 128x128
        self.layer2 = res34.layer2 # 64x64
        self.layer3 = res34.layer3 # 32x32
        self.layer4 = res34.layer4 # 16x16 (bottleneck)

        self.up4 = self._make_up_layer(512, 256)
        self.up3 = self._make_up_layer(256, 128)
        self.up2 = self._make_up_layer(128, 64)
        self.up1 = self._make_up_layer(64, 64)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        self.gsd_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()
        )

    def _make_up_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = self.initial(x)    # (B, 64, 256, 256)
        x_pooled = self.maxpool(x0) # (B, 64, 128, 128)
        x1 = self.layer1(x_pooled)   # (B, 64, 128, 128)
        x2 = self.layer2(x1)   # (B, 128, 64, 64)
        x3 = self.layer3(x2)   # (B, 256, 32, 32)
        x4 = self.layer4(x3)   # (B, 512, 16, 16)

        d4 = self.up4(x4) + x3 
        d3 = self.up3(d4) + x2
        d2 = self.up2(d3) + x1
        d1 = self.up1(d2) + x0

        mask = self.final_up(d1)
        gsd = self.gsd_head(x4)

        return mask, gsd

class UNetResNet34(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        res34 = resnet34(weights=weights)

        self.initial = nn.Sequential(res34.conv1, res34.bn1, res34.relu)
        self.maxpool = res34.maxpool
        self.layer1 = res34.layer1 # 64 ch, 128x128
        self.layer2 = res34.layer2 # 128 ch, 64x64
        self.layer3 = res34.layer3 # 256 ch, 32x32
        self.layer4 = res34.layer4 # 512 ch, 16x16

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = self._make_double_conv(256 + 256, 256) # 256(up) + 256(x3)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = self._make_double_conv(128 + 128, 128) # 128(up) + 128(x2)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = self._make_double_conv(64 + 64, 64) # 64(up) + 64(x1)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1 = self._make_double_conv(64 + 64, 64) # 64(up) + 64(x0)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        self.gsd_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()
        )

    def _make_double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = self.initial(x)                # 256x256, 64ch
        x1 = self.layer1(self.maxpool(x0))  # 128x128, 64ch
        x2 = self.layer2(x1)                # 64x64, 128ch
        x3 = self.layer3(x2)                # 32x32, 256ch
        x4 = self.layer4(x3)                # 16x16, 512ch

        d4 = self.conv4(torch.cat([self.up4(x4), x3], dim=1))
        d3 = self.conv3(torch.cat([self.up3(d4), x2], dim=1))
        d2 = self.conv2(torch.cat([self.up2(d3), x1], dim=1))
        d1 = self.conv1(torch.cat([self.up1(d2), x0], dim=1))

        mask = self.final_up(d1)
        gsd = self.gsd_head(x4)
        return mask, gsd


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.aspp2 = self._make_aspp_module(in_channels, out_channels, 3, padding=rates[0], dilation=rates[0])
        self.aspp3 = self._make_aspp_module(in_channels, out_channels, 3, padding=rates[1], dilation=rates[1])
        self.aspp4 = self._make_aspp_module(in_channels, out_channels, 3, padding=rates[2], dilation=rates[2])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) 
        )

    def _make_aspp_module(self, in_ch, out_ch, kernel_size, padding, dilation):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        feat1 = self.aspp1(x)
        feat2 = self.aspp2(x)
        feat3 = self.aspp3(x)
        feat4 = self.aspp4(x)

        feat5 = self.global_avg_pool(x)
        feat5 = F.interpolate(feat5, size=size, mode='bilinear', align_corners=True)

        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        return self.project(out)


class UNetASPPResNet34(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        res34 = resnet34(weights=weights)

        self.initial = nn.Sequential(res34.conv1, res34.bn1, res34.relu)
        self.maxpool = res34.maxpool
        self.layer1 = res34.layer1 # 64 ch, 128x128
        self.layer2 = res34.layer2 # 128 ch, 64x64
        self.layer3 = res34.layer3 # 256 ch, 32x32
        self.layer4 = res34.layer4 # 512 ch, 16x16
        self.aspp = ASPP(512, 512, rates=[6, 12, 18])

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = self._make_double_conv(256 + 256, 256) # 256(up) + 256(x3)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = self._make_double_conv(128 + 128, 128) # 128(up) + 128(x2)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = self._make_double_conv(64 + 64, 64) # 64(up) + 64(x1)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1 = self._make_double_conv(64 + 64, 64) # 64(up) + 64(x0)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        self.gsd_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()
        )

    def _make_double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Энкодер
        x0 = self.initial(x)                # 256x256, 64ch
        x1 = self.layer1(self.maxpool(x0))  # 128x128, 64ch
        x2 = self.layer2(x1)                # 64x64, 128ch
        x3 = self.layer3(x2)                # 32x32, 256ch
        x4 = self.layer4(x3)                # 16x16, 512ch
        x4 = self.aspp(x4)

        # Декодер (U-Net)
        d4 = self.conv4(torch.cat([self.up4(x4), x3], dim=1)) #
        d3 = self.conv3(torch.cat([self.up3(d4), x2], dim=1))
        d2 = self.conv2(torch.cat([self.up2(d3), x1], dim=1))
        d1 = self.conv1(torch.cat([self.up1(d2), x0], dim=1))

        mask = self.final_up(d1)
        gsd = self.gsd_head(x4)
        return mask, gsd


class UNetASPP2ResNet34(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        res34 = resnet34(weights=weights)

        #ЭНКОДЕР
        self.initial = nn.Sequential(res34.conv1, res34.bn1, res34.relu)
        self.maxpool = res34.maxpool
        self.layer1 = res34.layer1 # 64 ch, 128x128
        self.layer2 = res34.layer2 # 128 ch, 64x64
        self.layer3 = res34.layer3 # 256 ch, 32x32
        self.layer4 = res34.layer4 # 512 ch, 16x16
        self.aspp = ASPP(512, 512, rates=[6, 12, 18])

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = self._make_double_conv(256 + 256, 256) # 256(up) + 256(x3)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = self._make_double_conv(128 + 128, 128) # 128(up) + 128(x2)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = self._make_double_conv(64 + 64, 64) # 64(up) + 64(x1)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1 = self._make_double_conv(64 + 64, 64) # 64(up) + 64(x0)

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        self.gsd_atrous = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )

        # Вход: 128(L2) + 256(L3) + 256(Atrous L4) = 640
        self.gsd_regressor = nn.Sequential(
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1) # log GSD
        )

    def _make_double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Энкодер
        x0 = self.initial(x)                # 256x256, 64ch
        x1 = self.layer1(self.maxpool(x0))  # 128x128, 64ch
        x2 = self.layer2(x1)                # 64x64, 128ch
        x3 = self.layer3(x2)                # 32x32, 256ch
        x4 = self.layer4(x3)                # 16x16, 512ch
        x4 = self.aspp(x4)

        # Декодер (U-Net)
        d4 = self.conv4(torch.cat([self.up4(x4), x3], dim=1))
        d3 = self.conv3(torch.cat([self.up3(d4), x2], dim=1))
        d2 = self.conv2(torch.cat([self.up2(d3), x1], dim=1))
        d1 = self.conv1(torch.cat([self.up1(d2), x0], dim=1))

        mask = self.final_up(d1)

        # --- Ветка GSD ---
        f2 = x2.detach()
        f3 = x3.detach()
        f4 = x4.detach()

        x4_context = self.gsd_atrous(x4)
        p2 = F.adaptive_avg_pool2d(x2, 1).flatten(1)
        p3 = F.adaptive_avg_pool2d(x3, 1).flatten(1)
        p4 = F.adaptive_avg_pool2d(x4_context, 1).flatten(1)

        gsd_feat = torch.cat([p2, p3, p4], dim=1)
        log_gsd = self.gsd_regressor(gsd_feat)

        return mask, log_gsd