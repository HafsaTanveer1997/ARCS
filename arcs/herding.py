from __future__ import annotations

import torch
import torch.nn.functional as F


def greedy_herding(embeddings: torch.Tensor, num_select: int) -> list[int]:
    """Greedy mean-matching herding over L2-normalized embeddings."""
    x = F.normalize(embeddings.float(), dim=1)
    n = int(x.size(0))
    k = min(max(int(num_select), 0), n)

    if k == 0:
        return []

    mean_feature = x.mean(dim=0)
    running_sum = torch.zeros_like(mean_feature)
    available = torch.ones(n, dtype=torch.bool, device=x.device)
    selected: list[int] = []

    for step in range(k):
        target = (step + 1) * mean_feature - running_sum
        scores = x @ target
        scores = scores.masked_fill(~available, -torch.inf)
        index = int(torch.argmax(scores).item())
        selected.append(index)
        available[index] = False
        running_sum = running_sum + x[index]

    return selected
