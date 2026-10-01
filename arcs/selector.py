from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
import torch

from .herding import greedy_herding
from .scoring import disagreement_components, disagreement_score, severity_score


@dataclass
class ARCSSelectionResult:
    selected_dataset_indices: list[int]
    selected_pool_positions: list[int]
    representative_pool_positions: list[int]
    reliability_pool_positions: list[int]
    random_pool_positions: list[int]
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


class ARCSSelector:
    def __init__(
        self,
        *,
        kl_weight: float = 0.5,
        flip_bonus: float = 0.25,
        reference_percentile: float = 90.0,
        low_rep_ratio: float = 0.75,
        high_rep_ratio: float = 0.68,
        low_reliability_ratio: float = 0.05,
        high_reliability_ratio: float = 0.12,
        random_ratio: float = 0.20,
        low_severity_band: tuple[float, float] = (75.0, 95.0),
        high_severity_band: tuple[float, float] = (70.0, 90.0),
        class_cap: int = 2,
    ) -> None:
        self.kl_weight = float(kl_weight)
        self.flip_bonus = float(flip_bonus)
        self.reference_percentile = float(reference_percentile)
        self.low_rep_ratio = float(low_rep_ratio)
        self.high_rep_ratio = float(high_rep_ratio)
        self.low_reliability_ratio = float(low_reliability_ratio)
        self.high_reliability_ratio = float(high_reliability_ratio)
        self.random_ratio = float(random_ratio)
        self.low_severity_band = tuple(float(v) for v in low_severity_band)
        self.high_severity_band = tuple(float(v) for v in high_severity_band)
        self.class_cap = int(class_cap)

        if self.class_cap < 0:
            raise ValueError("class_cap must be non-negative")

    @staticmethod
    def _interpolate(low: float, high: float, severity: float) -> float:
        return low + severity * (high - low)

    @staticmethod
    def _band_positions(
        scores: torch.Tensor,
        low_percentile: float,
        high_percentile: float,
    ) -> list[int]:
        if scores.numel() == 0:
            return []
        low_threshold = torch.quantile(scores, low_percentile / 100.0)
        high_threshold = torch.quantile(scores, high_percentile / 100.0)
        mask = (scores >= low_threshold) & (scores <= high_threshold)
        return torch.nonzero(mask, as_tuple=False).flatten().tolist()

    def _select_reliability(
        self,
        *,
        scores: torch.Tensor,
        fp_logits: torch.Tensor,
        candidate_positions: list[int],
        num_select: int,
    ) -> list[int]:
        if num_select <= 0 or not candidate_positions:
            return []

        candidate_scores = scores[candidate_positions]
        order = torch.argsort(candidate_scores, descending=True).tolist()
        ordered_positions = [candidate_positions[i] for i in order]

        if self.class_cap == 0:
            return ordered_positions[:num_select]

        fp_pred = torch.argmax(fp_logits, dim=1)
        class_counts: dict[int, int] = {}
        selected: list[int] = []

        for position in ordered_positions:
            predicted_class = int(fp_pred[position].item())
            if class_counts.get(predicted_class, 0) < self.class_cap:
                selected.append(position)
                class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
            if len(selected) >= num_select:
                return selected

        selected_set = set(selected)
        for position in ordered_positions:
            if position not in selected_set:
                selected.append(position)
            if len(selected) >= num_select:
                break

        return selected[:num_select]

    def select(
        self,
        *,
        embeddings: torch.Tensor,
        dataset_indices: Sequence[int],
        fp_logits: torch.Tensor,
        q_logits: torch.Tensor,
        num_select: int,
        seed: int,
    ) -> ARCSSelectionResult:
        dataset_indices = [int(i) for i in dataset_indices]
        pool_size = len(dataset_indices)

        if pool_size == 0:
            raise ValueError("candidate pool is empty")
        if embeddings.size(0) != pool_size:
            raise ValueError("embeddings and dataset_indices must have the same length")
        if fp_logits.size(0) != pool_size or q_logits.size(0) != pool_size:
            raise ValueError("logits and dataset_indices must have the same length")

        target = min(max(int(num_select), 0), pool_size)
        if target == 0:
            return ARCSSelectionResult([], [], [], [], [], {"pool_size": pool_size})

        components = disagreement_components(fp_logits, q_logits)
        scores = disagreement_score(
            components,
            kl_weight=self.kl_weight,
            flip_bonus=self.flip_bonus,
        )
        severity, conf_ref, kl_ref = severity_score(
            components,
            reference_percentile=self.reference_percentile,
        )

        rep_ratio = self._interpolate(
            self.low_rep_ratio, self.high_rep_ratio, severity
        )
        reliability_ratio = self._interpolate(
            self.low_reliability_ratio,
            self.high_reliability_ratio,
            severity,
        )

        n_rep = max(1, int(round(rep_ratio * target)))
        n_rel = max(0, int(round(reliability_ratio * target)))
        n_rand = target - n_rep - n_rel
        if n_rand < 0:
            n_rand = 0
            n_rel = max(0, target - n_rep)

        low_band = self._interpolate(
            self.low_severity_band[0], self.high_severity_band[0], severity
        )
        high_band = self._interpolate(
            self.low_severity_band[1], self.high_severity_band[1], severity
        )

        rep_positions = greedy_herding(embeddings, n_rep)
        selected = list(rep_positions)
        selected_set = set(selected)

        remaining_positions = [
            position for position in range(pool_size)
            if position not in selected_set
        ]

        rel_positions: list[int] = []
        if n_rel > 0 and remaining_positions:
            remaining_scores = scores[remaining_positions]
            remaining_fp_logits = fp_logits[remaining_positions]

            local_band = self._band_positions(
                remaining_scores,
                low_percentile=low_band,
                high_percentile=high_band,
            )
            if not local_band:
                local_band = list(range(len(remaining_positions)))

            selected_local = self._select_reliability(
                scores=remaining_scores,
                fp_logits=remaining_fp_logits,
                candidate_positions=local_band,
                num_select=min(n_rel, len(local_band)),
            )
            rel_positions = [remaining_positions[i] for i in selected_local]

        selected.extend(rel_positions)
        selected_set.update(rel_positions)

        remaining_positions = [
            position for position in range(pool_size)
            if position not in selected_set
        ]
        rng = np.random.default_rng(int(seed))
        random_count = min(n_rand, len(remaining_positions))
        if random_count > 0:
            rand_positions = [
                int(position)
                for position in rng.choice(
                    np.asarray(remaining_positions, dtype=np.int64),
                    size=random_count,
                    replace=False,
                ).tolist()
            ]
        else:
            rand_positions = []

        selected.extend(rand_positions)
        selected = selected[:target]
        selected_dataset_indices = [dataset_indices[p] for p in selected]

        metadata = {
            "pool_size": pool_size,
            "target": target,
            "severity": float(severity),
            "conf_ref": float(conf_ref),
            "kl_ref": float(kl_ref),
            "representative_ratio": float(rep_ratio),
            "reliability_ratio": float(reliability_ratio),
            "random_ratio": float(self.random_ratio),
            "representative_target": int(n_rep),
            "reliability_target": int(n_rel),
            "random_target": int(n_rand),
            "representative_selected": len(rep_positions),
            "reliability_selected": len(rel_positions),
            "random_selected": len(rand_positions),
            "band_low_percentile": float(low_band),
            "band_high_percentile": float(high_band),
            "reference_percentile": float(self.reference_percentile),
            "kl_weight": float(self.kl_weight),
            "flip_bonus": float(self.flip_bonus),
            "class_cap": int(self.class_cap),
        }

        return ARCSSelectionResult(
            selected_dataset_indices=selected_dataset_indices,
            selected_pool_positions=selected,
            representative_pool_positions=rep_positions,
            reliability_pool_positions=rel_positions,
            random_pool_positions=rand_positions,
            metadata=metadata,
        )
