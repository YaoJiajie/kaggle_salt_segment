from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models.resnet import model_urls as resnet_model_urls
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import torch.nn.functional as F


class SpatialSqueeze(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(SpatialSqueeze, self).__init__()
        self.global_avg_pool = nn.AvgPool2d(kernel_size=kernel_size)
        self.encoder = nn.Linear(in_channels, in_channels // 2)
        self.decoder = nn.Linear(in_channels // 2, in_channels)

    def forward(self, x):
        n, c, h, w = x.shape
        x = self.global_avg_pool(x)
        assert x.shape[2] == 1 and x.shape[3] == 1
        x = x.view((n, c))
        x = self.encoder(x)
        x = F.relu(x, inplace=True)
        x = self.decoder(x)
        x = F.sigmoid(x)
        x = x.view(n, c, 1, 1)
        return x


class ChannelSqueeze(nn.Module):
    def __init__(self, in_channels):
        super(ChannelSqueeze, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        excitation = self.conv1x1(x)
        excitation = F.sigmoid(excitation)
        return excitation


class UpConv(nn.Module):
    def __init__(self, in_planes, hidden_planes, out_planes, kernel_size):
        super(UpConv, self).__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, 3, 1, padding=1),
            nn.BatchNorm2d(hidden_planes),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_planes, out_planes, 3, 1, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

        self.spatial_squeezer = SpatialSqueeze(out_planes, kernel_size)
        self.channel_squeezer = ChannelSqueeze(out_planes)

    def forward(self, x, skip_connection=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], 1)

        x = self.decoder(x)
        spatial_excitation = self.channel_squeezer.forward(x)
        channel_excitation = self.spatial_squeezer.forward(x)
        x = x * spatial_excitation + x * channel_excitation
        return x


class ResNetEncoder(ResNet):
    def __init__(self, block, layers):
        super(ResNetEncoder, self).__init__(block, layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        return l1, l2, l3, l4


class ResUNet(nn.Module):
    def __init__(self, pretrained=False):
        super(ResUNet, self).__init__()
        # use resnet34
        self.encoder = ResNetEncoder(BasicBlock, [3, 4, 6, 3])
        if pretrained:
            self.encoder.load_state_dict(model_zoo.load_url(resnet_model_urls['resnet34']))

        self.center = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.up4 = UpConv(256 + 512, 512, 64, 8)    # 1/16
        self.up3 = UpConv(64 + 256, 256, 64, 16)   # 1/8
        self.up2 = UpConv(64 + 128, 128, 64, 32)   # 1/4
        self.up1 = UpConv(64 + 64, 64, 64, 64)   # 1/2
        self.up0 = UpConv(64, 32, 64, 128)  # 1

        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x = torch.cat([(x - mean[2]) / std[2], (x - mean[1]) / std[1], (x - mean[0]) / std[0]], 1)

        e1, e2, e3, e4 = self.encoder(x)
        e5 = self.center(e4)  # c(256)

        d4 = self.up4.forward(e5, e4)
        d3 = self.up3.forward(d4, e3)
        d2 = self.up2.forward(d3, e2)
        d1 = self.up1.forward(d2, e1)
        d0 = self.up0.forward(d1)

        f = torch.cat([
            d0,
            F.upsample(d1, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d2, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=16, mode='bilinear', align_corners=False),
        ], 1)

        f = F.dropout(f, p=0.5)
        logit = self.logit(f)
        return logit


if __name__ == '__main__':
    net = ResUNet(pretrained=True)
    fake_input1 = np.zeros((1, 1, 128, 128), np.float32)
    fake_input2 = np.zeros((1, 1, 128, 128), np.float32)

    fake_input1 = torch.Tensor(fake_input1)
    fake_input2 = torch.Tensor(fake_input2)

    logits = net.forward(fake_input1, fake_input2)
    print(logits.shape)
