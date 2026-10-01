import math

import numpy as np
import torch.nn as nn


BN = nn.BatchNorm2d

REGNETX_600M = {
    "WA": 36.97,
    "W0": 48,
    "WM": 2.24,
    "DEPTH": 16,
    "GROUP_W": 24,
    "SE_ON": False,
}


class SimpleStemIN(nn.Module):
    def __init__(self, in_w, out_w):
        super().__init__()
        self.conv = nn.Conv2d(in_w, out_w, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = BN(out_w)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class SE(nn.Module):
    def __init__(self, w_in, w_se):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(w_se, w_in, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class BottleneckTransform(nn.Module):
    def __init__(self, w_in, w_out, stride, bm, gw, se_r):
        super().__init__()
        w_b = int(round(w_out * bm))
        num_gs = w_b // gw
        self.a = nn.Conv2d(w_in, w_b, kernel_size=1, stride=1, padding=0, bias=False)
        self.a_bn = BN(w_b)
        self.a_relu = nn.ReLU(True)
        self.b = nn.Conv2d(
            w_b, w_b, kernel_size=3, stride=stride, padding=1,
            groups=num_gs, bias=False,
        )
        self.b_bn = BN(w_b)
        self.b_relu = nn.ReLU(True)
        if se_r:
            self.se = SE(w_b, int(round(w_in * se_r)))
        self.c = nn.Conv2d(w_b, w_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.c_bn = BN(w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBottleneckBlock(nn.Module):
    def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None):
        super().__init__()
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False)
            self.bn = BN(w_out)
        self.f = BottleneckTransform(w_in, w_out, stride, bm, gw, se_r)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        return self.relu(x)


class AnyHead(nn.Module):
    def __init__(self, w_in, nc):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class AnyStage(nn.Module):
    def __init__(self, w_in, w_out, stride, d, bm, gw, se_r):
        super().__init__()
        for i in range(d):
            block_stride = stride if i == 0 else 1
            block_in = w_in if i == 0 else w_out
            self.add_module(
                f"b{i + 1}",
                ResBottleneckBlock(block_in, w_out, block_stride, bm, gw, se_r),
            )

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class AnyNet(nn.Module):
    def __init__(self, stem_w, ds, ws, ss, bms, gws, se_r, nc):
        super().__init__()
        self.stem = SimpleStemIN(3, stem_w)
        prev_w = stem_w
        for i, (d, w, s, bm, gw) in enumerate(zip(ds, ws, ss, bms, gws)):
            self.add_module(f"s{i + 1}", AnyStage(prev_w, w, s, d, bm, gw, se_r))
            prev_w = w
        self.head = AnyHead(w_in=prev_w, nc=nc)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                n = module.weight.size(1)
                module.weight.data.normal_(0, 1.0 / float(n))
                module.bias.data.zero_()

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


def quantize_float(value, q):
    return int(round(value / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    transitions = [w != wp or r != rp for w, wp, r, rp in temp]
    stage_ws = [w for w, t in zip(ws, transitions[:-1]) if t]
    stage_ds = np.diff([d for d, t in zip(range(len(transitions)), transitions) if t]).tolist()
    return stage_ws, stage_ds


def generate_regnet(w_a, w_0, w_m, depth, q=8):
    ws_cont = np.arange(depth) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages = len(np.unique(ws))
    return ws.astype(int).tolist(), num_stages


class RegNet(AnyNet):
    def __init__(self, cfg):
        block_ws, num_stages = generate_regnet(cfg["WA"], cfg["W0"], cfg["WM"], cfg["DEPTH"])
        ws, ds = get_stages_from_blocks(block_ws, block_ws)
        gws = [cfg["GROUP_W"] for _ in range(num_stages)]
        bms = [1 for _ in range(num_stages)]
        ws, gws = adjust_ws_gs_comp(ws, bms, gws)
        ss = [2 for _ in range(num_stages)]
        se_r = 0.25 if cfg["SE_ON"] else None
        super().__init__(
            stem_w=32,
            ss=ss,
            ds=ds,
            ws=ws,
            bms=bms,
            gws=gws,
            se_r=se_r,
            nc=1000,
        )


def regnetx_600m(**kwargs):
    return RegNet(REGNETX_600M)
