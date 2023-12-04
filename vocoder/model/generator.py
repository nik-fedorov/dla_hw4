import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from vocoder.utils.util import get_padding


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=dilation[i],
                                  padding=get_padding(kernel_size, dilation[i])))
            for i in range(3)
        ])

        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=get_padding(kernel_size, 1)))
            for _ in range(3)
        ])

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            out = F.leaky_relu(x, 0.1)
            out = conv1(out)
            out = F.leaky_relu(out, 0.1)
            out = conv2(out)
            x = out + x
        return x


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_kernels = len(config['resblock_kernel_sizes'])
        self.num_upsamples = len(config['upsample_rates'])
        self.conv_pre = weight_norm(nn.Conv1d(80, config['upsample_initial_channel'], 7, padding=3))

        self.ups = nn.ModuleList([
            weight_norm(nn.ConvTranspose1d(config['upsample_initial_channel'] // (2 ** i),
                                           config['upsample_initial_channel'] // (2 ** (i + 1)),
                                           kernel_size, stride=upsample_rate,
                                           padding=(kernel_size - upsample_rate) // 2))
            for i, (kernel_size, upsample_rate) in
            enumerate(zip(config['upsample_kernel_sizes'], config['upsample_rates']))
        ])

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config['upsample_initial_channel'] // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(config['resblock_kernel_sizes'], config['resblock_dilation_sizes'])):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, padding=3))

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x
