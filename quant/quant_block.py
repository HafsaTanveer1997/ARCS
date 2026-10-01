import torch.nn as nn

from models.mobilenetv2 import InvertedResidual
from models.regnet import ResBottleneckBlock
from models.resnet import BasicBlock, Bottleneck
from .quant_layer import QuantModule, UniformAffineQuantizer


class BaseQuantBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        self.ignore_reconstruction = False
        self.trained = False

    def set_quant_state(self, weight_quant=False, act_quant=False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for module in self.modules():
            if isinstance(module, QuantModule):
                module.set_quant_state(weight_quant, act_quant)


class QuantBasicBlock(BaseQuantBlock):
    def __init__(self, block: BasicBlock, weight_quant_params={}, act_quant_params={}):
        super().__init__()
        self.conv1 = QuantModule(block.conv1, weight_quant_params, act_quant_params)
        self.conv1.norm_function = block.bn1
        self.conv1.activation_function = block.relu1
        self.conv2 = QuantModule(block.conv2, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.conv2.norm_function = block.bn2
        if block.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(block.downsample[0], weight_quant_params, act_quant_params, disable_act_quant=True)
            self.downsample.norm_function = block.downsample[1]
        self.activation_function = block.relu2
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv2(self.conv1(x))
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantBottleneck(BaseQuantBlock):
    def __init__(self, block: Bottleneck, weight_quant_params={}, act_quant_params={}):
        super().__init__()
        self.conv1 = QuantModule(block.conv1, weight_quant_params, act_quant_params)
        self.conv1.norm_function = block.bn1
        self.conv1.activation_function = block.relu1
        self.conv2 = QuantModule(block.conv2, weight_quant_params, act_quant_params)
        self.conv2.norm_function = block.bn2
        self.conv2.activation_function = block.relu2
        self.conv3 = QuantModule(block.conv3, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.conv3.norm_function = block.bn3
        if block.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(block.downsample[0], weight_quant_params, act_quant_params, disable_act_quant=True)
            self.downsample.norm_function = block.downsample[1]
        self.activation_function = block.relu3
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv3(self.conv2(self.conv1(x)))
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantResBottleneckBlock(BaseQuantBlock):
    def __init__(self, block: ResBottleneckBlock, weight_quant_params={}, act_quant_params={}):
        super().__init__()
        self.conv1 = QuantModule(block.f.a, weight_quant_params, act_quant_params)
        self.conv1.norm_function = block.f.a_bn
        self.conv1.activation_function = block.f.a_relu
        self.conv2 = QuantModule(block.f.b, weight_quant_params, act_quant_params)
        self.conv2.norm_function = block.f.b_bn
        self.conv2.activation_function = block.f.b_relu
        self.conv3 = QuantModule(block.f.c, weight_quant_params, act_quant_params, disable_act_quant=True)
        self.conv3.norm_function = block.f.c_bn
        self.proj_block = block.proj_block
        if block.proj_block:
            self.downsample = QuantModule(block.proj, weight_quant_params, act_quant_params, disable_act_quant=True)
            self.downsample.norm_function = block.bn
        else:
            self.downsample = None
        self.activation_function = block.relu
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if not self.proj_block else self.downsample(x)
        out = self.conv3(self.conv2(self.conv1(x)))
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantInvertedResidual(BaseQuantBlock):
    def __init__(self, block: InvertedResidual, weight_quant_params={}, act_quant_params={}):
        super().__init__()
        self.use_res_connect = block.use_res_connect
        self.expand_ratio = block.expand_ratio
        if self.expand_ratio == 1:
            self.conv = nn.Sequential(
                QuantModule(block.conv[0], weight_quant_params, act_quant_params),
                QuantModule(block.conv[3], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].norm_function = block.conv[1]
            self.conv[0].activation_function = nn.ReLU6()
            self.conv[1].norm_function = block.conv[4]
        else:
            self.conv = nn.Sequential(
                QuantModule(block.conv[0], weight_quant_params, act_quant_params),
                QuantModule(block.conv[3], weight_quant_params, act_quant_params),
                QuantModule(block.conv[6], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].norm_function = block.conv[1]
            self.conv[0].activation_function = nn.ReLU6()
            self.conv[1].norm_function = block.conv[4]
            self.conv[1].activation_function = nn.ReLU6()
            self.conv[2].norm_function = block.conv[7]
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        out = x + self.conv(x) if self.use_res_connect else self.conv(x)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


specials = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    ResBottleneckBlock: QuantResBottleneckBlock,
    InvertedResidual: QuantInvertedResidual,
}

specials_unquantized = [nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.Dropout]
