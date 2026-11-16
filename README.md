# ARCS

This is partial Code for ARCS calibration-data selection for low-bit post-training quantization. Complete code will be available upon acceptance of paper

The repository contains the ARCS selector, the three comparison methods used in the paper, and a small ImageNet/PTQ runner. Result files and datasets are not included.

## Install

```bash
pip install -r requirements.txt
```

## Data

Set the ImageNet path:

```bash
export DATA_PATH=/path/to/imagenet
```

The directory should contain `train/` and `val/`.

For CNN models, set the checkpoint directory if needed:

```bash
export CKPT_DIR=/path/to/checkpoints
```

## Run

```bash
bash scripts/run_single/run_imagenet_single.sh METHOD ARCH W A SEED BUDGET
```

Example:

```bash
bash scripts/run_single/run_imagenet_single.sh arcs deit_small_patch16_224 8 4 1005 128
```

`METHOD` can be `random`, `confgap`, `selectq`, or `arcs`.

The default ARCS settings are listed in `configs/arcs.yaml`.

## Poisoning stress test

The candidate-pool construction used for the poisoning stress test is in:

```text
scripts/poisoning.py
```

It builds the 1%, 5%, and 10% contaminated pools from the clean and transformed candidate scores used by the attack.
