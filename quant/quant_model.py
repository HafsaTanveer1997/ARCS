import torch.nn as nn

from .fold_bn import search_fold_and_remove_bn
from .quant_block import BaseQuantBlock, specials
from .quant_layer import QuantModule, StraightThrough, UniformAffineQuantizer


class QuantModel(nn.Module):
    def __init__(self, model: nn.Module, weight_quant_params: dict = {},
                 act_quant_params: dict = {}, is_fusing=True):
        super().__init__()
        if is_fusing:
            search_fold_and_remove_bn(model)
            self.model = model
            self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        else:
            self.model = model
            self.quant_module_refactor_wo_fuse(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module, weight_quant_params, act_quant_params):
        previous = None
        for name, child in module.named_children():
            if type(child) in specials:
                setattr(module, name, specials[type(child)](child, weight_quant_params, act_quant_params))
            elif isinstance(child, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child, weight_quant_params, act_quant_params))
                previous = getattr(module, name)
            elif isinstance(child, (nn.ReLU, nn.ReLU6)):
                if previous is not None:
                    previous.activation_function = child
                    setattr(module, name, StraightThrough())
            elif isinstance(child, StraightThrough):
                continue
            else:
                self.quant_module_refactor(child, weight_quant_params, act_quant_params)

    def quant_module_refactor_wo_fuse(self, module, weight_quant_params, act_quant_params):
        previous = None
        for name, child in module.named_children():
            if type(child) in specials:
                setattr(module, name, specials[type(child)](child, weight_quant_params, act_quant_params))
            elif isinstance(child, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child, weight_quant_params, act_quant_params))
                previous = getattr(module, name)
            elif isinstance(child, nn.BatchNorm2d):
                if previous is not None:
                    previous.norm_function = child
                    setattr(module, name, StraightThrough())
            elif isinstance(child, (nn.ReLU, nn.ReLU6)):
                if previous is not None:
                    previous.activation_function = child
                    setattr(module, name, StraightThrough())
            elif isinstance(child, StraightThrough):
                continue
            else:
                self.quant_module_refactor_wo_fuse(child, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for module in self.model.modules():
            if isinstance(module, (QuantModule, BaseQuantBlock)):
                module.set_quant_state(weight_quant, act_quant)

    def forward(self, input):
        return self.model(input)

    def set_first_last_layer_to_8bit(self):
        weight_quantizers, act_quantizers = [], []
        for module in self.model.modules():
            if isinstance(module, UniformAffineQuantizer):
                if module.leaf_param:
                    act_quantizers.append(module)
                else:
                    weight_quantizers.append(module)
        weight_quantizers[0].bitwidth_refactor(8)
        weight_quantizers[-1].bitwidth_refactor(8)
        act_quantizers[-2].bitwidth_refactor(8)

    def disable_network_output_quantization(self):
        modules = [m for m in self.model.modules() if isinstance(m, QuantModule)]
        modules[-1].disable_act_quant = True
