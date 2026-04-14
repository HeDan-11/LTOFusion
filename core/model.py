import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLeakyRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        sobel_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
        weight_x = torch.from_numpy(sobel_filter).view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        weight_y = torch.from_numpy(sobel_filter.T).view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

        self.convx = nn.Conv2d(channels, channels, kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels, bias=False)
        self.convy = nn.Conv2d(channels, channels, kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(weight_x)
        self.convy.weight.data.copy_(weight_y)

    def forward(self, x):
        return torch.abs(self.convx(x)) + torch.abs(self.convy(x))


class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DenseBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2 * channels, channels)

    def forward(self, x):
        x = torch.cat((x, self.conv1(x)), dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        return x


class EEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dense = DenseBlock(in_channels)
        self.convdown = Conv1(3 * in_channels, out_channels)
        self.sobelconv = Sobelxy(in_channels)
        self.convup = Conv1(in_channels, out_channels)

    def forward(self, x):
        dense_feature = self.convdown(self.dense(x))
        edge_feature = self.convup(self.sobelconv(x))
        return F.leaky_relu(dense_feature + edge_feature, negative_slope=0.1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlock_res(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.res_conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        x = self.up(x)
        return self.main(x) + self.res_conv(x)


class PolicyNet(nn.Module):
    def __init__(self, input_c=3):
        super().__init__()
        cs = [16, 32, 64]
        f_cs = [16, 32, 64, 64]

        self.A_enc1 = EEM(1, cs[0])
        self.A_enc2 = EEM(cs[0], cs[1])
        self.A_enc3 = EEM(cs[1], cs[2])

        self.F_enc1 = ConvBlock_res(input_c, f_cs[0], 3, 1, 1)
        self.F_enc2 = ConvBlock_res(f_cs[0] + cs[0], f_cs[1], 3, 1, 1)
        self.F_enc3 = ConvBlock_res(f_cs[1] + cs[1], f_cs[2], 3, 1, 1)
        self.bottle = ConvBlock_res(f_cs[2] + cs[2], f_cs[3], 3, 1, 1)

    def forward(self, x):
        img_a, _, fused = x.split(1, 1)
        history = [x]

        fused = self.F_enc1(x)
        img_a = self.A_enc1(img_a)
        history.append(fused)

        img_a = F.max_pool2d(img_a, 2, 2)
        fused = F.max_pool2d(fused, 2, 2)
        fused = self.F_enc2(torch.cat([img_a, fused], dim=1))
        img_a = self.A_enc2(img_a)
        history.append(fused)

        img_a = F.max_pool2d(img_a, 2, 2)
        fused = F.max_pool2d(fused, 2, 2)
        fused = self.F_enc3(torch.cat([img_a, fused], dim=1))
        img_a = self.A_enc3(img_a)
        history.append(fused)

        img_a = F.max_pool2d(img_a, 2, 2)
        fused = F.max_pool2d(fused, 2, 2)
        fused = self.bottle(torch.cat([img_a, fused], dim=1))
        history.append(fused)

        return history


class ActionNet(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [16, 32, 64, 64]
        self.affine4 = nn.Conv2d(channels[2], channels[2], 3, 1, 1, bias=False)
        self.affine5 = nn.Conv2d(channels[1], channels[1], 3, 1, 1, bias=False)
        self.affine6 = nn.Conv2d(channels[0], channels[0], 3, 1, 1, bias=False)

        self.dec1 = DecoderBlock(channels[3], channels[2])
        self.dec4 = DecoderBlock(channels[2], channels[1])
        self.dec5 = DecoderBlock(channels[1], channels[0])

        self.finnal = nn.Sequential(
            ConvBlock(channels[0], channels[0], 3, 1, 1),
            nn.Conv2d(channels[0], 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, history):
        out = self.affine4(history[-2]) + self.dec1(history[-1])
        out = self.affine5(history[-3]) + self.dec4(out)
        out = self.affine6(history[-4]) + self.dec5(out)
        return self.finnal(out) * 0.1
