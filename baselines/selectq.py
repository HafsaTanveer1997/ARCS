from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class SelectQResult:
    selected_indices: List[int]
    selected_scores: List[float]
    candidate_scores: List[float]
    layer_names: List[str]


class SelectQSelector:
    """SelectQ baseline used in the reported experiments."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str,
        max_layers: Optional[int] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.device = torch.device(device)
        self.max_layers = max_layers
        self.verbose = verbose
        self.layers: List[Tuple[str, nn.Module]] = []
        self.layer_names: List[str] = []
        self.batch_stats: Dict[str, torch.Tensor] = {}
        self.handles = []

    def _log(self, text: str):
        if self.verbose:
            print(f"[SelectQ] {text}")

    def _select_layers(self):
        layers = [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ]
        if not layers:
            raise RuntimeError("No Conv2d/Linear layers found.")
        if self.max_layers is not None and self.max_layers > 0 and len(layers) > self.max_layers:
            positions = np.linspace(0, len(layers) - 1, self.max_layers, dtype=int)
            layers = [layers[i] for i in positions]
        self.layers = layers
        self.layer_names = [name for name, _ in layers]

    @staticmethod
    def _per_sample_stats(output: torch.Tensor) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            output = output[0]
        x = output.detach().float().reshape(output.shape[0], -1)
        return torch.stack((x.mean(dim=1), x.std(dim=1, unbiased=False)), dim=1)

    def _install_hooks(self):
        def make_hook(name):
            def hook(_module, _input, output):
                self.batch_stats[name] = self._per_sample_stats(output).cpu()
            return hook

        for name, module in self.layers:
            self.handles.append(module.register_forward_hook(make_hook(name)))

    def _remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _forward_stats(self, images: torch.Tensor):
        self.batch_stats.clear()
        with torch.no_grad():
            self.model(images.to(self.device, non_blocking=True))
        missing = [name for name in self.layer_names if name not in self.batch_stats]
        if missing:
            raise RuntimeError(f"Missing activation statistics for: {missing[:5]}")
        return self.batch_stats

    @staticmethod
    def _distance(stats: torch.Tensor, centroids: torch.Tensor, gamma: float):
        mean_diff = stats[:, None, 0] - centroids[None, :, 0]
        std_diff = stats[:, None, 1] - centroids[None, :, 1]
        return mean_diff.pow(2) + float(gamma) * std_diff.pow(2)

    @staticmethod
    def _lambda_t(batch_id: int, num_batches: int, lambda_min: float, lambda_max: float):
        if num_batches <= 1:
            return float(lambda_min)
        t = float(batch_id)
        t_max = float(num_batches - 1)
        return float(
            lambda_min
            + 0.5 * (lambda_max - lambda_min)
            * (1.0 + math.cos(math.pi * t / t_max))
        )

    def _initialize_centroids(self, num_centroids: int, seed: int, input_size: int):
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        images = torch.rand(
            num_centroids,
            3,
            input_size,
            input_size,
            generator=generator,
        )
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        stats = self._forward_stats((images - mean) / std)
        return {name: stats[name][:num_centroids].clone() for name in self.layer_names}

    def _update_centroids(
        self,
        loader: DataLoader,
        centroids: Dict[str, torch.Tensor],
        lambda_min: float,
        lambda_max: float,
        gamma: float,
    ):
        num_batches = len(loader)
        for batch_id, batch in enumerate(loader):
            batch_stats = self._forward_stats(batch[0])
            lam = self._lambda_t(batch_id, num_batches, lambda_min, lambda_max)
            for layer_name in self.layer_names:
                layer_stats = batch_stats[layer_name]
                layer_centroids = centroids[layer_name]
                for sample_id in range(layer_stats.shape[0]):
                    sample = layer_stats[sample_id:sample_id + 1]
                    distances = self._distance(sample, layer_centroids, gamma).squeeze(0)
                    nearest = int(torch.argmin(distances).item())
                    layer_centroids[nearest] = layer_centroids[nearest] + lam * sample.squeeze(0)
                centroids[layer_name] = layer_centroids

    def _score(self, loader: DataLoader, centroids: Dict[str, torch.Tensor], gamma: float):
        scores = []
        for batch in loader:
            images = batch[0]
            batch_stats = self._forward_stats(images)
            batch_score = torch.zeros(images.shape[0], dtype=torch.float32)
            for layer_name in self.layer_names:
                distances = self._distance(batch_stats[layer_name], centroids[layer_name], gamma)
                batch_score += distances.min(dim=1).values
            scores.append(batch_score.numpy())
        return np.concatenate(scores, axis=0)

    def select(
        self,
        dataset: Dataset,
        candidate_indices: Sequence[int],
        num_select: int,
        batch_size: int,
        workers: int,
        seed: int,
        num_centroids: int = 10,
        lambda_min: float = 0.001,
        lambda_max: float = 0.1,
        gamma: float = 1.0,
        input_size: int = 224,
    ) -> SelectQResult:
        candidate_indices = [int(i) for i in candidate_indices]
        if num_select > len(candidate_indices):
            raise ValueError("Budget is larger than the candidate pool.")

        self.model.to(self.device)
        self.model.eval()
        self._select_layers()
        self._install_hooks()

        loader = DataLoader(
            Subset(dataset, candidate_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )

        try:
            centroids = self._initialize_centroids(num_centroids, seed, input_size)
            self._update_centroids(loader, centroids, lambda_min, lambda_max, gamma)
            candidate_scores = self._score(loader, centroids, gamma)
        finally:
            self._remove_hooks()

        ranking = np.argsort(-candidate_scores, kind="mergesort")[:num_select]
        selected_indices = [candidate_indices[int(pos)] for pos in ranking]
        selected_scores = [float(candidate_scores[int(pos)]) for pos in ranking]
        self._log(f"Selected {len(selected_indices)} samples.")
        return SelectQResult(
            selected_indices=selected_indices,
            selected_scores=selected_scores,
            candidate_scores=[float(x) for x in candidate_scores],
            layer_names=self.layer_names,
        )
