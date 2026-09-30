import torch
from torch import nn

from .quant_layer import UniformAffineQuantizer, round_ste


class AdaRoundQuantizer(nn.Module):
    def __init__(self, uaq: UniformAffineQuantizer, weight_tensor: torch.Tensor,
                 round_mode="learned_hard_sigmoid"):
        super().__init__()
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.delta = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels
        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2 / 3
        self.init_alpha(weight_tensor.clone())

    def forward(self, x):
        if self.round_mode == "nearest":
            x_int = torch.round(x / self.delta)
        elif self.round_mode == "nearest_ste":
            x_int = round_ste(x / self.delta)
        elif self.round_mode == "stochastic":
            x_floor = torch.floor(x / self.delta)
            rest = (x / self.delta) - x_floor
            x_int = x_floor + torch.bernoulli(rest)
        elif self.round_mode == "learned_hard_sigmoid":
            x_floor = torch.floor(x / self.delta)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + (self.alpha >= 0).float()
        else:
            raise ValueError("Wrong rounding mode")

        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        return (x_quant - self.zero_point) * self.delta

    def get_soft_targets(self):
        return torch.clamp(
            torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma,
            0,
            1,
        )

    def init_alpha(self, x: torch.Tensor):
        if self.round_mode != "learned_hard_sigmoid":
            raise NotImplementedError
        x_floor = torch.floor(x / self.delta)
        rest = (x / self.delta) - x_floor
        alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)
        self.alpha = nn.Parameter(alpha)

    @torch.jit.export
    def extra_repr(self):
        return f"bit={self.n_bits}"
