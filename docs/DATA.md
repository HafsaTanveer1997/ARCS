# Dataset Setup

Datasets are not redistributed with this artifact.

## ImageNet

Use a standard local ImageNet copy with:

```text
/path/to/imagenet/
├── train/
└── val/
```

Set:

```bash
export DATA_PATH=/path/to/imagenet
```

Calibration candidate pools are drawn from the training split. Evaluation uses the 50k validation split.

## CIFAR-100

The reported CIFAR-100 experiments use the standard 50k training and 10k test split. The CIFAR reproduction pipeline assumes a normal local/downloaded CIFAR-100 dataset and does not redistribute dataset files.

## Candidate pools

Unless a stress test explicitly changes the pool composition, the main experiments use a 4096-sample candidate pool. For a fixed dataset, architecture, bit-width, budget, and seed, all four methods use the same candidate pool.
