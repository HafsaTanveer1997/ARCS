import torch
import torch.nn.functional as F

from .adaptive_rounding import AdaRoundQuantizer
from .quant_block import BaseQuantBlock, specials_unquantized
from .quant_layer import QuantModule, lp_loss
from .quant_model import QuantModel
from .set_act_quantize_params import set_act_quantize_params
from .set_weight_quantize_params import get_dc_fp_init, get_init


include = False


def find_unquantized_module(model, module_list=None, name_list=None):
    global include
    module_list = [] if module_list is None else module_list
    name_list = [] if name_list is None else name_list
    for name, module in model.named_children():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            if not module.trained:
                include = True
                module.set_quant_state(False, False)
                name_list.append(name)
                module_list.append(module)
        elif include and type(module) in specials_unquantized:
            name_list.append(name)
            module_list.append(module)
        else:
            find_unquantized_module(module, module_list, name_list)
    return module_list[1:], name_list[1:]


def block_reconstruction(model: QuantModel, fp_model: QuantModel,
                         block: BaseQuantBlock, fp_block: BaseQuantBlock,
                         cali_data: torch.Tensor, batch_size: int = 32,
                         iters: int = 20000, weight: float = 0.01,
                         opt_mode: str = "mse", b_range=(20, 2),
                         warmup: float = 0.0, p: float = 2.0,
                         lr: float = 4e-5, input_prob: float = 1.0,
                         keep_gpu: bool = True, lamb_r: float = 0.2,
                         T: float = 7.0, bn_lr: float = 1e-3,
                         lamb_c=0.02):
    cached_inps = get_init(
        model, block, cali_data, batch_size=batch_size,
        input_prob=True, keep_gpu=keep_gpu,
    )
    cached_outs, cached_output, cur_syms = get_dc_fp_init(
        fp_model, fp_block, cali_data, batch_size=batch_size,
        input_prob=True, keep_gpu=keep_gpu, bn_lr=bn_lr, lamb=lamb_c,
    )
    set_act_quantize_params(block, cali_data=cached_inps[:min(256, cached_inps.size(0))])

    global include
    module_list, name_list, include = [], [], False
    module_list, name_list = find_unquantized_module(model, module_list, name_list)
    block.set_quant_state(True, True)
    for parameter in model.parameters():
        parameter.requires_grad = False

    weight_parameters = []
    act_parameters = []
    for module in block.modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer = AdaRoundQuantizer(
                uaq=module.weight_quantizer,
                round_mode="learned_hard_sigmoid",
                weight_tensor=module.org_weight.data,
            )
            module.weight_quantizer.soft_targets = True
            weight_parameters.append(module.weight_quantizer.alpha)
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            if module.act_quantizer.delta is not None:
                module.act_quantizer.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer.delta))
                act_parameters.append(module.act_quantizer.delta)
            module.act_quantizer.is_training = True

    w_opt = torch.optim.Adam(weight_parameters, lr=3e-3) if weight_parameters else None
    a_opt = torch.optim.Adam(act_parameters, lr=lr) if act_parameters else None
    a_scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=iters, eta_min=0.0)
        if a_opt else None
    )

    loss_func = LossFunction(
        block,
        round_loss="relaxation",
        weight=weight,
        max_count=iters,
        rec_loss=opt_mode,
        b_range=b_range,
        decay_start=0,
        warmup=warmup,
        p=p,
        lam=lamb_r,
        T=T,
    )

    size = cached_inps.size(0)
    for _ in range(iters):
        idx = torch.randint(0, size, (batch_size,))
        cur_inp = cached_inps[idx].cuda()
        cur_sym = cur_syms[idx].cuda()
        output_fp = cached_output[idx].cuda()
        cur_out = cached_outs[idx].cuda()
        drop_inp = torch.where(torch.rand_like(cur_inp) < input_prob, cur_inp, cur_sym)
        cur_inp = torch.cat((drop_inp, cur_inp))

        if w_opt:
            w_opt.zero_grad()
        if a_opt:
            a_opt.zero_grad()
        out_all = block(cur_inp)
        out_drop = out_all[:batch_size]
        output = out_all[batch_size:]

        for number, module in enumerate(module_list):
            if name_list[number] == "fc":
                output = torch.flatten(output, 1)
            if isinstance(module, torch.nn.Dropout) and output.dim() == 4:
                output = output.mean(dim=(2, 3))
            output = module(output)

        error = loss_func(out_drop, cur_out, output, output_fp)
        error.backward(retain_graph=True)
        if w_opt:
            w_opt.step()
        if a_opt:
            a_opt.step()
            a_scheduler.step()

    torch.cuda.empty_cache()
    for module in block.modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.soft_targets = False
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.act_quantizer.is_training = False
            module.trained = True
    for module in fp_block.modules():
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            module.trained = True


class LossFunction:
    def __init__(self, block: BaseQuantBlock, round_loss="relaxation", weight=1.0,
                 rec_loss="mse", max_count=2000, b_range=(10, 2),
                 decay_start=0.0, warmup=0.0, p=2.0, lam=1.0, T=7.0):
        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.lam = lam
        self.T = T
        self.temp_decay = LinearTempDecay(
            max_count,
            rel_start_decay=warmup + (1 - warmup) * decay_start,
            start_b=b_range[0],
            end_b=b_range[1],
        )
        self.count = 0
        self.pd_loss = torch.nn.KLDivLoss(reduction="batchmean")

    def __call__(self, pred, tgt, output, output_fp):
        self.count += 1
        if self.rec_loss != "mse":
            raise ValueError(f"Unsupported reconstruction loss: {self.rec_loss}")
        rec_loss = lp_loss(pred, tgt, p=self.p)
        pd_loss = self.pd_loss(
            F.log_softmax(output / self.T, dim=1),
            F.softmax(output_fp / self.T, dim=1),
        ) / self.lam
        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == "none":
            round_loss = 0
        elif self.round_loss == "relaxation":
            round_loss = 0
            for module in self.block.modules():
                if isinstance(module, QuantModule):
                    values = module.weight_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((values - 0.5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError
        return rec_loss + round_loss + pd_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2,
                 start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
        return self.end_b + (self.start_b - self.end_b) * max(0.0, 1 - rel_t)
