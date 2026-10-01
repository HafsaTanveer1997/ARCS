from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class IndexedDataset(Dataset):
    def __init__(self, base_dataset, indices: Sequence[int]):
        self.base_dataset = base_dataset
        self.indices = [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        dataset_index = self.indices[index]
        item = self.base_dataset[dataset_index]
        if isinstance(item, tuple):
            x = item[0]
            y = item[1] if len(item) > 1 else -1
        else:
            x, y = item, -1
        return x, y, dataset_index


def get_module_by_name(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    module = model
    for part in module_name.split("."):
        if not hasattr(module, part):
            raise AttributeError(f"Module {module_name!r} not found at {part!r}")
        module = getattr(module, part)
    return module


@torch.no_grad()
def extract_candidate_embeddings(
    *,
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int,
    workers: int,
    pool_size: Optional[int],
    seed: int,
    feature_module_name: str,
) -> tuple[torch.Tensor, list[int]]:
    rng = np.random.default_rng(int(seed))
    full_indices = np.arange(len(dataset))

    if pool_size is None or int(pool_size) >= len(dataset):
        pool_indices = full_indices.tolist()
    else:
        pool_indices = rng.choice(
            full_indices,
            size=int(pool_size),
            replace=False,
        ).tolist()
        pool_indices.sort()

    loader = DataLoader(
        IndexedDataset(dataset, pool_indices),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=True,
    )

    target_module = get_module_by_name(model, feature_module_name)
    captured: list[torch.Tensor] = []

    def hook_fn(_module, _inputs, output):
        captured.clear()
        if isinstance(output, (tuple, list)):
            output = output[0]
        captured.append(output.detach())

    handle = target_module.register_forward_hook(hook_fn)
    was_training = model.training
    model.eval()

    features_all: list[torch.Tensor] = []
    dataset_indices: list[int] = []

    try:
        for images, _targets, batch_indices in loader:
            images = images.to(device, non_blocking=True)
            _ = model(images)
            if not captured:
                raise RuntimeError(
                    f"No features captured from module {feature_module_name!r}"
                )

            features = captured[0]
            if features.dim() == 4:
                features = F.adaptive_avg_pool2d(features, 1)
            elif features.dim() == 3:
                features = features[:, 0, :]

            features_all.append(features.flatten(1).detach().cpu().float())
            dataset_indices.extend(int(i) for i in batch_indices.tolist())
    finally:
        handle.remove()
        if was_training:
            model.train()

    return torch.cat(features_all, dim=0), dataset_indices


def paper_feature_module(architecture: str) -> str:
    modules = {
        "resnet18": "avgpool",
        "resnet50": "avgpool",
        "mobilenetv2": "features",
        "regnetx_600m": "head.avg_pool",
        "deit_small_patch16_224": "norm",
    }
    try:
        return modules[architecture]
    except KeyError as exc:
        raise ValueError(f"Unsupported architecture: {architecture}") from exc
