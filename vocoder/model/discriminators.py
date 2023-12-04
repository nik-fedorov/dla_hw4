import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

from vocoder.utils.util import get_padding


class PeriodSubDiscriminator(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm(nn.Conv2d(1024, 1024, (kernel_size, 1), padding=(2, 0))),
        ])
        self.conv_post = norm(nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # reshape
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.reshape(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodSubDiscriminator(2),
            PeriodSubDiscriminator(3),
            PeriodSubDiscriminator(5),
            PeriodSubDiscriminator(7),
            PeriodSubDiscriminator(11),
        ])

    def forward(self, x):
        discriminators_outputs = []
        discriminators_feature_maps = []
        for i, d in enumerate(self.discriminators):
            output, feature_maps = d(x)
            discriminators_outputs.append(output)
            discriminators_feature_maps.append(feature_maps)

        return discriminators_outputs, discriminators_feature_maps


class ScaleSubDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, padding=1))

    def forward(self, x):
        feature_maps = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feature_maps.append(x)
        x = self.conv_post(x)
        feature_maps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feature_maps


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleSubDiscriminator(use_spectral_norm=True),
            ScaleSubDiscriminator(),
            ScaleSubDiscriminator(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, x):
        discriminators_outputs = []
        discriminators_feature_maps = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                x = self.meanpools[i-1](x)
            output, feature_maps = d(x)
            discriminators_outputs.append(output)
            discriminators_feature_maps.append(feature_maps)

        return discriminators_outputs, discriminators_feature_maps
