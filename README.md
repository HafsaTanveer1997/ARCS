# ARCS: Disagreement-Guided Calibration Data Selection for Trustworthy Low-Bit Post-Training Quantization

This repository contains the implementation and reproducibility scripts for ARCS, a reliability-aware calibration-data selection method for low-bit post-training quantization (PTQ).

ARCS compares a full-precision model with a lightweight proxy-quantized model and uses confidence gap, margin gap, KL divergence, and prediction flips to guide calibration-data selection. The final calibration subset combines feature-space representative samples, samples from a controlled disagreement band, and random samples for diversity, with the mixture adapted according to disagreement severity.

The repository is prepared for anonymous review. Author-identifying information is intentionally omitted.

## Supported methods

- Random
- ConfGap
- SelectQ
- ARCS

## Main evaluated settings

### ImageNet CNNs

Models:
- ResNet-18
- ResNet-50
- MobileNetV2
- RegNetX-600MF

Bit-widths:
- W4A4
- W4A8
- W8A4
- W8A8

Calibration budget:
- N = 1024

Seeds:
- 1005
- 1006
- 1007

### DeiT-S

Model:
- DeiT-Small (`deit_small_patch16_224`)

Bit-widths:
- W4A4
- W4A8
- W8A4
- W8A8

Calibration budgets:
- N = 128, 256, 512, 1024

Seeds:
- 1005
- 1006
- 1007

## ARCS paper settings

The paper implementation uses:

- candidate pool size: 4096
- proxy activation calibration size: 128
- proxy activation calibration batch size: 32
- disagreement score weights: confidence 1.0, margin 1.0, KL 0.5, prediction flip 0.25
- continuous severity from confidence-gap and KL terms
- automatic severity references from the 90th percentile of the current candidate pool
- representative ratio: 0.75 at low severity to 0.68 at high severity
- reliability-band ratio: 0.05 at low severity to 0.12 at high severity
- random ratio: 0.20
- reliability band: 75-95 percentile at low severity to 70-90 percentile at high severity
- reliability scoring over the remaining pool after representative selection
- top-score selection within the reliability band
- first-pass class cap of 2 samples per full-precision predicted class, relaxed if needed
- trusted-prediction confidence threshold: 0.70

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or use the Conda environment:

```bash
conda env create -f environment.yml
conda activate arcs
```

## Dataset setup

ImageNet should be organized as:

```text
/path/to/imagenet/
├── train/
└── val/
```

Set:

```bash
export DATA_PATH=/path/to/imagenet
```

Pretrained CNN checkpoints are expected in a directory specified by:

```bash
export CKPT_DIR=/path/to/checkpoints
```

DeiT-S is loaded through `timm`.

## Run one ImageNet experiment

```bash
CKPT_DIR=/path/to/checkpoints \
DATA_PATH=/path/to/imagenet \
bash scripts/run_single/run_imagenet_single.sh METHOD ARCH W A SEED NUM_SAMPLES
```

Examples:

```bash
bash scripts/run_single/run_imagenet_single.sh random resnet18 8 8 1005 1024
bash scripts/run_single/run_imagenet_single.sh confgap resnet50 4 8 1006 1024
bash scripts/run_single/run_imagenet_single.sh selectq deit_small_patch16_224 8 4 1005 128
bash scripts/run_single/run_imagenet_single.sh arcs deit_small_patch16_224 8 4 1005 128
```

## Reproduction scripts

ImageNet CNN grid:

```bash
DATA_PATH=/path/to/imagenet \
CKPT_DIR=/path/to/checkpoints \
CONFIRM_RUN=yes \
bash scripts/reproduce/reproduce_imagenet_cnn.sh
```

DeiT-S grid:

```bash
DATA_PATH=/path/to/imagenet \
CKPT_DIR=/path/to/checkpoints \
CONFIRM_RUN=yes \
bash scripts/reproduce/reproduce_imagenet_deit.sh
```

Both scripts run Random, ConfGap, SelectQ, and ARCS.

## Repository layout

```text
ARCS-SaTML/
├── README.md
├── requirements.txt
├── environment.yml
├── configs/
├── data/
├── models/
├── quant/
├── selectq/
├── calib_selectors/
├── utils/
├── scripts/
│   ├── run_single/
│   └── reproduce/
└── docs/
```

## Reproducibility notes

Generated datasets, checkpoints, logs, logits, and figures are intentionally excluded from version control. Small configuration files, reproduction scripts, and source code required to reproduce the reported experiments are included.

The implementation does not provide a formal robustness guarantee against adaptive poisoning attacks. The poisoning analysis in the paper is a bounded selection-aware stress test.
