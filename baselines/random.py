from typing import Sequence

import numpy as np


def select_random(candidate_indices: Sequence[int], budget: int, seed: int) -> list[int]:
    candidate_indices = np.asarray([int(i) for i in candidate_indices], dtype=np.int64)
    if budget > len(candidate_indices):
        raise ValueError("Budget is larger than the candidate pool.")
    rng = np.random.default_rng(int(seed) + 1)
    return [int(i) for i in rng.choice(candidate_indices, size=int(budget), replace=False).tolist()]
