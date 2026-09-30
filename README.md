# ARCS

Code for **ARCS: Disagreement-Guided Calibration Data Selection for Trustworthy Low-Bit Post-Training Quantization**.

The repository contains ARCS, the three comparison methods used in the paper, the PTQ code, and scripts for the reported ImageNet experiments.

## Setup

Python 3.9 was used for the experiments.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A Conda environment is also provided in `environment.yml`.

## ImageNet

Expected dataset structure:

```text
/path/to/imagenet/
├── train/
└── val/
```

For the CNN experiments, place the pretrained checkpoints in one directory:

```text
resnet18_imagenet.pth.tar
resnet50_imagenet.pth.tar
mobilenetv2.pth.tar
regnet_600m.pth.tar
```

DeiT-S is loaded with `timm`.

## Run one experiment

Example for ARCS on DeiT-S, W8A4, N=128:

```bash
python run_imagenet.py \
  --method arcs \
  --arch deit_small_patch16_224 \
  --data-path /path/to/imagenet \
  --w-bits 8 \
  --a-bits 4 \
  --budget 128 \
  --seed 1005
```

Available methods are:

```text
random
confgap
selectq
arcs
```

Available ImageNet architectures are:

```text
resnet18
resnet50
mobilenetv2
regnetx_600m
deit_small_patch16_224
```

For a CNN run, also provide the checkpoint directory:

```bash
--checkpoint-dir /path/to/checkpoints
```

## Reproduce the ImageNet grids

CNN experiments:

```bash
DATA_PATH=/path/to/imagenet \
CKPT_DIR=/path/to/checkpoints \
CONFIRM_RUN=yes \
bash scripts/reproduce/reproduce_imagenet_cnn.sh
```

DeiT-S experiments:

```bash
DATA_PATH=/path/to/imagenet \
CONFIRM_RUN=yes \
bash scripts/reproduce/reproduce_imagenet_deit.sh
```

The scripts use seeds 1005, 1006, and 1007.

## Repository structure

```text
arcs/           ARCS selection method
baselines/      Random, ConfGap, and SelectQ
quant/          PTQ implementation
models/         ImageNet model definitions
data/           dataset loading
scripts/        experiment launchers
paper_results/  aggregate values reported in the paper
docs/           experiment details
```

`configs/arcs.yaml` contains the ARCS settings used in the paper. Additional implementation details are in `docs/REPRODUCIBILITY.md`.

Datasets and pretrained model files are not included in this repository.
