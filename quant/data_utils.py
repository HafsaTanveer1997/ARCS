import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from typing import Union

from .quant_block import BaseQuantBlock
from .quant_layer import QuantModule, lp_loss
from .quant_model import QuantModel


def save_dc_fp_data(model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                    cali_data: torch.Tensor, batch_size: int = 32,
                    keep_gpu: bool = True, input_prob: bool = False,
                    lamb=50, bn_lr=1e-3):
    device = next(model.parameters()).device
    get_inp_out = GetDcFpLayerInpOut(
        model, layer, device=device, input_prob=input_prob,
        lamb=lamb, bn_lr=bn_lr,
    )
    cached = []
    for i in trange(int(cali_data.size(0) / batch_size)):
        batch = cali_data[i * batch_size:(i + 1) * batch_size]
        cached.append(get_inp_out(batch))

    outputs = torch.cat([x[0].cpu() for x in cached])
    fp_outputs = torch.cat([x[1].cpu() for x in cached])
    torch.cuda.empty_cache()
    if keep_gpu:
        outputs = outputs.to(device)
        fp_outputs = fp_outputs.to(device)

    if input_prob:
        symmetric = torch.cat([x[2].cpu() for x in cached])
        if keep_gpu:
            symmetric = symmetric.to(device)
        outputs.requires_grad = False
        symmetric.requires_grad = False
        return outputs, fp_outputs, symmetric
    return outputs, fp_outputs


def save_inp_oup_data(model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                      cali_data: torch.Tensor, batch_size: int = 32,
                      keep_gpu: bool = True, input_prob: bool = False):
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(model, layer, device=device, input_prob=input_prob)
    cached = []
    for i in range(int(cali_data.size(0) / batch_size)):
        batch = cali_data[i * batch_size:(i + 1) * batch_size]
        cached.append(get_inp_out(batch).cpu())
    inputs = torch.cat(cached)
    torch.cuda.empty_cache()
    return inputs.to(device) if keep_gpu else inputs


class StopForwardException(Exception):
    pass


class DataSaverHook:
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward
        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class InputHook:
    def __init__(self):
        self.inputs = None

    def hook(self, module, input_batch, output_batch):
        self.inputs = input_batch

    def clear(self):
        self.inputs = None


class GetLayerInpOut:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, input_prob: bool = False):
        self.model = model
        self.layer = layer
        self.device = device
        self.data_saver = DataSaverHook(store_input=True, stop_forward=True)

    def __call__(self, model_input):
        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            self.model.set_quant_state(weight_quant=True, act_quant=True)
            try:
                self.model(model_input.to(self.device))
            except StopForwardException:
                pass
        handle.remove()
        return self.data_saver.input_store[0].detach()


class GetDcFpLayerInpOut:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, input_prob: bool = False,
                 lamb=50, bn_lr=1e-3):
        self.model = model
        self.layer = layer
        self.device = device
        self.data_saver = DataSaverHook(store_input=True, store_output=True)
        self.input_prob = input_prob
        self.eps = 1e-6
        self.lamb = lamb
        self.bn_lr = bn_lr
        self.bn_stats = []
        for module in self.layer.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn_stats.append((
                    module.running_mean.detach().clone().flatten().cuda(),
                    torch.sqrt(module.running_var + self.eps).detach().clone().flatten().cuda(),
                ))

    @staticmethod
    def own_loss(a, b):
        return (a - b).norm() ** 2 / b.size(0)

    def __call__(self, model_input):
        self.model.set_quant_state(False, False)
        handle = self.layer.register_forward_hook(self.data_saver)
        hooks, handles = [], []
        for module in self.layer.modules():
            if isinstance(module, nn.BatchNorm2d):
                hook = InputHook()
                hooks.append(hook)
                handles.append(module.register_forward_hook(hook.hook))

        with torch.no_grad():
            output_fp = self.model(model_input.to(self.device))
            input_sym = self.data_saver.input_store[0].detach()
        handle.remove()

        para_input = input_sym.data.clone().to(self.device)
        para_input.requires_grad = True
        optimizer = optim.Adam([para_input], lr=self.bn_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, min_lr=1e-5, patience=100
        )

        for _ in range(500):
            self.layer.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()
            self.layer(para_input)
            mean_loss = 0
            std_loss = 0
            for bn_stat, hook in zip(self.bn_stats, hooks):
                current = hook.inputs[0]
                bn_mean, bn_std = bn_stat
                current_mean = torch.mean(current.view(current.size(0), current.size(1), -1), dim=2)
                current_std = torch.sqrt(torch.var(current.view(current.size(0), current.size(1), -1), dim=2) + self.eps)
                mean_loss += self.own_loss(bn_mean, current_mean)
                std_loss += self.own_loss(bn_std, current_std)
            constraint_loss = lp_loss(para_input, input_sym) / self.lamb
            total_loss = mean_loss + std_loss + constraint_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())

        for hook_handle in handles:
            hook_handle.remove()

        with torch.no_grad():
            out_fp = self.layer(para_input)

        if self.input_prob:
            return out_fp.detach(), output_fp.detach(), para_input.detach()
        return out_fp.detach(), output_fp.detach()
