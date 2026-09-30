import torch
import torch.nn as nn


BN = nn.BatchNorm2d


class _InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor):
        super().__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = in_ch == out_ch and stride == 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            BN(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=mid_ch, bias=False),
            BN(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            BN(out_ch),
        )

    def forward(self, x):
        if self.apply_residual:
            return self.layers(x) + x
        return self.layers(x)
