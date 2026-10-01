from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class IndexedDataset(Dataset):
    def __init__(self, base_dataset, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        dataset_index = self.indices[index]
        x, y = self.base_dataset[dataset_index]
        return x, y, dataset_index


@torch.no_grad()
def extract_fp_q_logits(
    fp_model: torch.nn.Module,
    q_model: torch.nn.Module,
    dataset,
    dataset_indices: List[int],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        IndexedDataset(dataset, dataset_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    fp_model.eval()
    q_model.eval()
    fp_all, q_all, seen = [], [], []

    for images, _targets, indices in loader:
        images = images.to(device, non_blocking=True)
        fp_all.append(fp_model(images).detach().cpu())
        q_all.append(q_model(images).detach().cpu())
        seen.extend(indices.tolist())

    if seen != list(dataset_indices):
        raise RuntimeError("Candidate index order changed during logit extraction.")

    return torch.cat(fp_all, dim=0), torch.cat(q_all, dim=0)
