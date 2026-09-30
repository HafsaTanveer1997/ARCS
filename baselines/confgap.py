from dataclasses import dataclass
from typing import Sequence

import torch

from arcs.scoring import disagreement_components, disagreement_score


@dataclass
class ConfGapResult:
    selected_indices: list[int]
    selected_pool_positions: list[int]


def select_confgap(
    fp_logits: torch.Tensor,
    q_logits: torch.Tensor,
    candidate_indices: Sequence[int],
    budget: int,
    kl_weight: float = 0.5,
    flip_bonus: float = 0.25,
) -> ConfGapResult:
    components = disagreement_components(fp_logits, q_logits)
    scores = disagreement_score(
        components,
        kl_weight=kl_weight,
        flip_bonus=flip_bonus,
    )
    positions = torch.argsort(scores, descending=True)[: int(budget)].tolist()
    selected = [int(candidate_indices[i]) for i in positions]
    return ConfGapResult(selected, positions)
