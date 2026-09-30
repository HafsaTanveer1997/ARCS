import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class StraightThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction="none"):
    if reduction == "none":
        return (pred - tgt).abs().pow(p).sum(1).mean()
    return (pred - tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, symmetric: bool = False,
                 channel_wise: bool = False, scale_method: str = "minmax",
                 leaf_param: bool = False, prob: float = 1.0):
        super().__init__()
        self.sym = symmetric
        if self.sym:
            raise NotImplementedError
        assert 2 <= n_bits <= 8, "bitwidth not supported"
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = 1.0
        self.zero_point = 0.0
        self.inited = True
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.eps = torch.tensor(1e-8, dtype=torch.float32)
        self.scale_method = "mse"
        self.one_side_dist = None
        self.num = 100
        self.running_min = None
        self.running_max = None
        self.prob = prob
        self.is_training = False

    def set_inited(self, inited: bool = True):
        self.inited = inited

    def update_quantize_range(self, x_min, x_max):
        if self.running_min is None:
            self.running_min = x_min
            self.running_max = x_max
        self.running_min = 0.1 * x_min + 0.9 * self.running_min
        self.running_max = 0.1 * x_max + 0.9 * self.running_max
        return self.running_min, self.running_max

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            self.delta, self.zero_point = self.init_quantization_scale(
                x.clone().detach(), self.channel_wise
            )
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        if self.is_training and self.prob < 1.0:
            return torch.where(torch.rand_like(x) < self.prob, x_dequant, x)
        return x_dequant

    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if not self.channel_wise:
            return x.mean()
        return torch.flatten(x, 1).mean(1)

    def calculate_qparams(self, min_val, max_val):
        quant_min, quant_max = 0, self.n_levels - 1
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.max(scale, self.eps)
        zero_point = quant_min - torch.round(min_val_neg / scale)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point

    def quantize(self, x: torch.Tensor, x_max, x_min):
        delta, zero_point = self.calculate_qparams(x_min, x_max)
        if self.channel_wise:
            new_shape = [1] * len(x.shape)
            new_shape[0] = x.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        return (x_quant - zero_point) * delta

    def perform_2D_search(self, x):
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch._aminmax(y, 1)
            x_max = torch.max(x_max, torch.zeros_like(x_max))
            x_min = torch.min(x_min, torch.zeros_like(x_min))
        else:
            x_min, x_max = torch._aminmax(x)
        x_range = x_max - x_min
        best_score = torch.zeros_like(x_min) + 1e10
        best_min = x_min.clone()
        best_max = x_max.clone()
        for i in range(1, self.num + 1):
            tmp_min = torch.zeros_like(x_min)
            tmp_max = x_range / self.num * i
            tmp_delta = (tmp_max - tmp_min) / (2 ** self.n_bits - 1)
            for zp in range(self.n_levels):
                new_min = tmp_min - zp * tmp_delta
                new_max = tmp_max - zp * tmp_delta
                x_q = self.quantize(x, new_max, new_min)
                score = self.lp_loss(x, x_q, 2.4)
                best_min = torch.where(score < best_score, new_min, best_min)
                best_max = torch.where(score < best_score, new_max, best_max)
                best_score = torch.min(best_score, score)
        return best_min, best_max

    def perform_1D_search(self, x):
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch._aminmax(y, 1)
        else:
            x_min, x_max = torch._aminmax(x)
        x_range = torch.max(x_min.abs(), x_max)
        best_score = torch.zeros_like(x_min) + 1e10
        best_min = x_min.clone()
        best_max = x_max.clone()
        for i in range(1, self.num + 1):
            threshold = x_range / self.num * i
            new_min = torch.zeros_like(x_min) if self.one_side_dist == "pos" else -threshold
            new_max = torch.zeros_like(x_max) if self.one_side_dist == "neg" else threshold
            x_q = self.quantize(x, new_max, new_min)
            score = self.lp_loss(x, x_q, 2.4)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max

    def get_x_min_x_max(self, x):
        if self.scale_method != "mse":
            raise NotImplementedError
        if self.one_side_dist is None:
            self.one_side_dist = "pos" if x.min() >= 0.0 else "neg" if x.max() <= 0.0 else "no"
        if self.one_side_dist != "no" or self.sym:
            best_min, best_max = self.perform_1D_search(x)
        else:
            best_min, best_max = self.perform_2D_search(x)
        if self.leaf_param:
            return self.update_quantize_range(best_min, best_max)
        return best_min, best_max

    def init_quantization_scale_channel(self, x: torch.Tensor):
        x_min, x_max = self.get_x_min_x_max(x)
        return self.calculate_qparams(x_min, x_max)

    def init_quantization_scale(self, x_clone: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = self.init_quantization_scale_channel(x_clone)
        if channel_wise:
            new_shape = [1] * len(x_clone.shape)
            new_shape[0] = x_clone.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        return delta, zero_point

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, "bitwidth not supported"
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    @torch.jit.export
    def extra_repr(self):
        return f"bit={self.n_bits}, is_training={self.is_training}, inited={self.inited}"


class QuantModule(nn.Module):
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear],
                 weight_quant_params: dict = {}, act_quant_params: dict = {},
                 disable_act_quant=False):
        super().__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(
                stride=org_module.stride,
                padding=org_module.padding,
                dilation=org_module.dilation,
                groups=org_module.groups,
            )
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = {}
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        self.use_weight_quant = False
        self.use_act_quant = False
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.norm_function = StraightThrough()
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False
        self.disable_act_quant = disable_act_quant
        self.trained = False

    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.norm_function(out)
        out = self.activation_function(out)
        if self.disable_act_quant:
            return out
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    @torch.jit.export
    def extra_repr(self):
        return (
            f"wbit={self.weight_quantizer.n_bits}, "
            f"abit={self.act_quantizer.n_bits}, "
            f"disable_act_quant={self.disable_act_quant}"
        )
