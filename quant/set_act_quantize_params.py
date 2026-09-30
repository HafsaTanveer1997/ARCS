from typing import Union

import torch

from .quant_block import BaseQuantBlock
from .quant_layer import QuantModule
from .quant_model import QuantModel


def set_act_quantize_params(
    module: Union[QuantModel, QuantModule, BaseQuantBlock],
    cali_data: torch.Tensor,
    batch_size: int = 256,
):
    module.set_quant_state(True, True)

    quantized_modules = []
    for current in module.modules():
        if isinstance(current, QuantModule):
            current.act_quantizer.running_min = None
            current.act_quantizer.running_max = None
            current.act_quantizer.one_side_dist = None
            current.act_quantizer.set_inited(False)
            quantized_modules.append(current)

    if cali_data.size(0) == 0:
        raise ValueError("Activation calibration data is empty.")

    batch_size = max(1, min(int(batch_size), int(cali_data.size(0))))
    was_training = module.training
    module.eval()

    with torch.no_grad():
        for start in range(0, cali_data.size(0), batch_size):
            end = min(start + batch_size, cali_data.size(0))
            batch = cali_data[start:end].cuda(non_blocking=True)
            module(batch)

    for current in quantized_modules:
        current.act_quantizer.set_inited(True)

    if was_training:
        module.train()

    torch.cuda.empty_cache()
