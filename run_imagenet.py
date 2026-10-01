from __future__ import annotations

import argparse
import copy
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from arcs.features import extract_candidate_embeddings, paper_feature_module
from arcs.logits import extract_fp_q_logits
from arcs.selector import ARCSSelector
from baselines import SelectQSelector, select_confgap, select_random
from data.imagenet import build_imagenet_data, build_imagenet_selection_dataset
from models.load import load_pretrained_model
from quant import (
    BaseQuantBlock,
    QuantModel,
    QuantModule,
    block_reconstruction,
    layer_reconstruction,
    set_act_quantize_params,
    set_weight_quantize_params,
)
from utils.confidence_metrics import compute_confidence_summary
from utils.proxy_calibration import calibrate_proxy_activations
from utils.temperature_scaling import TemperatureScaler, apply_temperature


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["random", "confgap", "selectq", "arcs"], required=True)
    parser.add_argument("--arch", choices=[
        "resnet18", "resnet50", "mobilenetv2", "regnetx_600m",
        "deit_small_patch16_224",
    ], required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint-dir", default=os.environ.get("CKPT_DIR", "pretrained"))
    parser.add_argument("--w-bits", type=int, required=True)
    parser.add_argument("--a-bits", type=int, required=True)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--pool-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1005)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--proxy-calib-size", type=int, default=128)
    parser.add_argument("--act-calib-batch-size", type=int, default=32)
    parser.add_argument("--ts-size", type=int, default=4096)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_candidate_pool(dataset, pool_size: int, seed: int):
    size = min(int(pool_size), len(dataset))
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(dataset), size=size, replace=False).tolist()
    indices.sort()
    return [int(i) for i in indices]


def build_subset_loader(dataset, indices, batch_size, workers, shuffle=False):
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )


def gather_calibration_data(loader, budget: int):
    images = []
    targets = []
    count = 0
    for x, y in loader:
        images.append(x)
        targets.append(y)
        count += x.size(0)
        if count >= budget:
            break
    return torch.cat(images, dim=0)[:budget], torch.cat(targets, dim=0)[:budget]


@torch.no_grad()
def validate_logits(loader, model, device):
    model.eval()
    logits_all = []
    targets_all = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits_all.append(model(images).detach().cpu())
        targets_all.append(targets.detach().cpu())
    logits = torch.cat(logits_all, dim=0)
    targets = torch.cat(targets_all, dim=0)
    return logits, targets


def build_proxy(base_model, wq_params, aq_params):
    proxy = QuantModel(
        model=copy.deepcopy(base_model),
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
    ).cuda().eval()
    set_weight_quantize_params(proxy)
    proxy.set_quant_state(weight_quant=True, act_quant=False)
    return proxy


def select_calibration_subset(args, model, dataset, wq_params, aq_params):
    device = next(model.parameters()).device
    candidate_indices = build_candidate_pool(dataset, args.pool_size, args.seed)

    if args.method == "random":
        return select_random(candidate_indices, args.budget, args.seed), {
            "candidate_pool_size": len(candidate_indices),
            "selection_seed": args.seed,
        }

    if args.method == "selectq":
        selector = SelectQSelector(model=model, device=device, max_layers=None, verbose=True)
        result = selector.select(
            dataset=dataset,
            candidate_indices=candidate_indices,
            num_select=args.budget,
            batch_size=args.batch_size,
            workers=args.workers,
            seed=args.seed,
            num_centroids=10,
            lambda_min=0.001,
            lambda_max=0.1,
            gamma=1.0,
            input_size=224,
        )
        return result.selected_indices, {
            "candidate_pool_size": len(candidate_indices),
            "layers": result.layer_names,
        }

    fp_model = copy.deepcopy(model).cuda().eval()
    proxy = build_proxy(model, wq_params, aq_params)
    proxy_calib_indices = calibrate_proxy_activations(
        model=proxy,
        dataset=dataset,
        pool_indices=candidate_indices,
        calib_size=args.proxy_calib_size,
        batch_size=args.act_calib_batch_size,
        workers=args.workers,
        seed=args.seed,
    )

    fp_logits, q_logits = extract_fp_q_logits(
        fp_model=fp_model,
        q_model=proxy,
        dataset=dataset,
        dataset_indices=candidate_indices,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    if args.method == "confgap":
        result = select_confgap(
            fp_logits=fp_logits,
            q_logits=q_logits,
            candidate_indices=candidate_indices,
            budget=args.budget,
            kl_weight=0.5,
            flip_bonus=0.25,
        )
        return result.selected_indices, {
            "candidate_pool_size": len(candidate_indices),
            "proxy_calibration_indices": proxy_calib_indices,
        }

    embeddings, feature_indices = extract_candidate_embeddings(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
        pool_size=args.pool_size,
        seed=args.seed,
        feature_module_name=paper_feature_module(args.arch),
    )
    if feature_indices != candidate_indices:
        raise RuntimeError("Candidate pool mismatch during feature extraction.")

    selector = ARCSSelector()
    result = selector.select(
        embeddings=embeddings,
        dataset_indices=candidate_indices,
        fp_logits=fp_logits,
        q_logits=q_logits,
        num_select=args.budget,
        seed=args.seed,
    )
    metadata = dict(result.metadata)
    metadata["proxy_calibration_indices"] = proxy_calib_indices
    return result.selected_dataset_indices, metadata


def reconstruct_cnn(qnn, fp_model, cali_data):
    kwargs = dict(
        cali_data=cali_data,
        iters=20000,
        weight=0.01,
        b_range=(20, 2),
        warmup=0.2,
        opt_mode="mse",
        lr=4e-5,
        input_prob=0.5,
        keep_gpu=True,
        lamb_r=0.1,
        T=4.0,
        bn_lr=1e-3,
        lamb_c=0.02,
    )

    def recurse(model, fp):
        for (_name, module), (_, fp_module) in zip(model.named_children(), fp.named_children()):
            if isinstance(module, QuantModule):
                layer_reconstruction(qnn, fp_model, module, fp_module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                block_reconstruction(qnn, fp_model, module, fp_module, **kwargs)
            else:
                recurse(module, fp_module)

    recurse(qnn, fp_model)


def fit_temperature(dataset, selected_indices, qnn, args, device):
    excluded = set(int(i) for i in selected_indices)
    available = np.asarray(
        [i for i in range(len(dataset)) if i not in excluded],
        dtype=np.int64,
    )
    size = min(args.ts_size, len(available))
    rng = np.random.default_rng(args.seed + 123)
    ts_indices = sorted(int(i) for i in rng.choice(available, size=size, replace=False).tolist())
    loader = build_subset_loader(dataset, ts_indices, args.batch_size, args.workers, shuffle=False)
    logits, targets = validate_logits(loader, qnn, device)
    scaler = TemperatureScaler(init_temp=1.0)
    temperature = scaler.fit(logits, targets, max_iter=100, lr=0.01, verbose=True)
    return temperature, ts_indices


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The reported PTQ pipeline requires CUDA.")

    seed_all(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    _train_loader, test_loader = build_imagenet_data(
        data_path=args.data_path,
        input_size=224,
        batch_size=args.batch_size,
        workers=args.workers,
        arch=args.arch,
    )
    selection_dataset = build_imagenet_selection_dataset(
        data_path=args.data_path,
        input_size=224,
        split="train",
        arch=args.arch,
    )

    model = load_pretrained_model(args.arch, args.checkpoint_dir).cuda().eval()
    selector_model = copy.deepcopy(model).cuda().eval()
    fp_backbone = copy.deepcopy(model).cuda().eval()

    wq_params = {
        "n_bits": args.w_bits,
        "channel_wise": True,
        "scale_method": "mse",
    }
    aq_params = {
        "n_bits": args.a_bits,
        "channel_wise": False,
        "scale_method": "mse",
        "leaf_param": True,
        "prob": 0.5,
    }

    selected_indices, selection_metadata = select_calibration_subset(
        args,
        selector_model,
        selection_dataset,
        wq_params,
        aq_params,
    )

    calibration_loader = build_subset_loader(
        selection_dataset,
        selected_indices,
        args.batch_size,
        args.workers,
        shuffle=True,
    )
    cali_data, _ = gather_calibration_data(calibration_loader, args.budget)

    fp_model = QuantModel(
        model=fp_backbone,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
        is_fusing=False,
    ).cuda().eval()
    fp_model.set_quant_state(False, False)

    qnn = QuantModel(
        model=model,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
    ).cuda().eval()
    qnn.set_first_last_layer_to_8bit()
    qnn.disable_network_output_quantization()
    set_weight_quantize_params(qnn)

    if args.arch.startswith("deit_"):
        set_act_quantize_params(
            qnn,
            cali_data=cali_data,
            batch_size=args.act_calib_batch_size,
        )
    else:
        reconstruct_cnn(qnn, fp_model, cali_data)

    qnn.set_quant_state(weight_quant=True, act_quant=True)
    device = next(qnn.parameters()).device

    test_logits, test_targets = validate_logits(test_loader, qnn, device)
    temperature, ts_indices = fit_temperature(
        selection_dataset,
        selected_indices,
        qnn,
        args,
        device,
    )
    scaled_logits = apply_temperature(test_logits, temperature)
    summary = compute_confidence_summary(
        scaled_logits,
        test_targets,
        n_bins=15,
        threshold=args.threshold,
    )

    output = {
        "method": args.method,
        "arch": args.arch,
        "w_bits": args.w_bits,
        "a_bits": args.a_bits,
        "budget": args.budget,
        "pool_size": args.pool_size,
        "seed": args.seed,
        "temperature": temperature,
        "temperature_scaling_indices": ts_indices,
        "selected_indices": selected_indices,
        "selection": selection_metadata,
        "metrics": asdict(summary),
    }

    path = Path(args.output_dir) / (
        f"{args.method}_{args.arch}_W{args.w_bits}A{args.a_bits}_"
        f"N{args.budget}_seed{args.seed}.json"
    )
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["metrics"], indent=2))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
