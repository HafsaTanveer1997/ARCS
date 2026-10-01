import argparse
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


BRIGHTNESS = 1.08
CONTRAST = 1.30


class PoisonedCandidateDataset(Dataset):
    def __init__(self, base_dataset, full_indices, poison_positions):
        self.base_dataset = base_dataset
        self.indices = [int(x) for x in full_indices]
        self.poison_positions = set(int(x) for x in poison_positions)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_idx):
        local_idx = int(local_idx)
        base_idx = self.indices[local_idx]

        if local_idx not in self.poison_positions:
            return self.base_dataset[base_idx]

        path, target = self.base_dataset.samples[base_idx]
        image = self.base_dataset.loader(path)
        image = TF.adjust_brightness(image, BRIGHTNESS)
        image = TF.adjust_contrast(image, CONTRAST)

        if self.base_dataset.transform is not None:
            image = self.base_dataset.transform(image)
        if self.base_dataset.target_transform is not None:
            target = self.base_dataset.target_transform(target)

        return image, target


def ordered(values, key):
    order = np.argsort(values["local_indices"].astype(np.int64))
    return values[key][order]


def build_poison_pools(base, aux, clean, poison, seed, output_dir):
    clean_emb = ordered(clean, "embeddings")
    poison_emb = ordered(poison, "embeddings")
    clean_pred = ordered(clean, "fp_pred")
    poison_pred = ordered(poison, "fp_pred")
    clean_conf = ordered(clean, "conf_gap")
    poison_conf = ordered(poison, "conf_gap")
    clean_kl = ordered(clean, "kl")
    poison_kl = ordered(poison, "kl")

    conf_ref = max(float(np.percentile(poison_conf, 90)), 1e-8)
    kl_ref = max(float(np.percentile(poison_kl, 90)), 1e-8)

    clean_severity = 0.5 * clean_conf / conf_ref + 0.5 * clean_kl / kl_ref
    poison_severity = 0.5 * poison_conf / conf_ref + 0.5 * poison_kl / kl_ref
    delta_severity = poison_severity - clean_severity

    a = clean_emb.astype(np.float64)
    b = poison_emb.astype(np.float64)
    cosine = np.sum(a * b, axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    )

    low = float(np.percentile(poison_severity, 70))
    high = float(np.percentile(poison_severity, 95))
    stable = clean_pred == poison_pred
    in_band = (poison_severity >= low) & (poison_severity <= high)

    def valid_candidates(cosine_threshold, require_positive_delta=True):
        mask = stable & in_band & (cosine >= cosine_threshold)
        if require_positive_delta:
            mask &= delta_severity > 0
        return np.flatnonzero(mask)

    valid = valid_candidates(0.98, True)
    if len(valid) < 410:
        valid = valid_candidates(0.95, True)
    if len(valid) < 410:
        valid = valid_candidates(0.95, False)
    if len(valid) < 410:
        raise RuntimeError(f"Only {len(valid)} valid poison candidates; need 410")

    mean_embedding = clean_emb.mean(axis=0)
    rep_distance = np.linalg.norm(clean_emb - mean_embedding[None, :], axis=1)

    distance = rep_distance[valid]
    severity_gain = delta_severity[valid]
    distance_z = (distance - distance.mean()) / (distance.std() + 1e-12)
    severity_z = (severity_gain - severity_gain.mean()) / (severity_gain.std() + 1e-12)
    attack_cost = distance_z - 0.35 * severity_z

    ranked = valid[np.argsort(attack_cost)]
    poison_sources = aux[ranked[:410]]

    rng = np.random.default_rng(seed + 4242)
    replacement_order = rng.permutation(len(base))[:410]
    rates = {"01": 41, "05": 205, "10": 410}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for tag, count in rates.items():
        final_pool = base.copy()
        positions = np.sort(replacement_order[:count])
        sources = poison_sources[:count]
        final_pool[positions] = sources

        if len(np.unique(final_pool)) != len(final_pool):
            raise RuntimeError(f"Duplicate indices for rate {tag}")

        np.save(output_dir / f"seed{seed}_rate{tag}_pool.npy", final_pool.astype(np.int64))
        np.save(output_dir / f"seed{seed}_rate{tag}_poison_positions.npy", positions.astype(np.int64))
        np.save(output_dir / f"seed{seed}_rate{tag}_poison_sources.npy", sources.astype(np.int64))

    metadata = {
        "seed": seed,
        "conf_ref_90": conf_ref,
        "kl_ref_90": kl_ref,
        "band_low_70": low,
        "band_high_95": high,
        "valid_candidates": int(len(valid)),
        "brightness": BRIGHTNESS,
        "contrast": CONTRAST,
    }
    (output_dir / f"seed{seed}_attack_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-pool", required=True)
    parser.add_argument("--aux-pool", required=True)
    parser.add_argument("--clean-scores", required=True)
    parser.add_argument("--poison-scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    build_poison_pools(
        base=np.load(args.base_pool).astype(np.int64),
        aux=np.load(args.aux_pool).astype(np.int64),
        clean=np.load(args.clean_scores),
        poison=np.load(args.poison_scores),
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
