from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from quant.set_act_quantize_params import set_act_quantize_params


def calibrate_proxy_activations(
    model: torch.nn.Module,
    dataset,
    pool_indices: Sequence[int],
    calib_size: int,
    batch_size: int,
    workers: int,
    seed: int,
) -> List[int]:
    pool_indices = [int(index) for index in pool_indices]
    if not pool_indices:
        raise ValueError("Candidate pool is empty.")

    count = min(int(calib_size), len(pool_indices))
    rng = np.random.default_rng(int(seed) + 17)
    chosen_indices = sorted(
        int(index)
        for index in rng.choice(
            np.asarray(pool_indices, dtype=np.int64),
            size=count,
            replace=False,
        ).tolist()
    )

    loader = DataLoader(
        Subset(dataset, chosen_indices),
        batch_size=max(1, min(int(batch_size), count)),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=True,
        drop_last=False,
    )

    image_batches = [images for images, _targets in loader]
    if not image_batches:
        raise RuntimeError("No proxy calibration images were loaded.")

    calibration_data = torch.cat(image_batches, dim=0)
    set_act_quantize_params(
        model,
        cali_data=calibration_data,
        batch_size=max(1, min(int(batch_size), int(calibration_data.size(0)))),
    )
    return chosen_indices
