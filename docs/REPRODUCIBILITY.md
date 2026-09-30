# Reproducibility notes

## Common settings

Methods:

```text
Random
ConfGap
SelectQ
ARCS
```

Seeds:

```text
1005, 1006, 1007
```

Bit-widths:

```text
W4A4, W4A8, W8A4, W8A8
```

Candidate pool size:

```text
4096
```

Trusted-prediction threshold:

```text
0.70
```

## ARCS

The ARCS settings are stored in `configs/arcs.yaml`.

The disagreement score is

```text
z(confidence gap) + z(margin gap) + 0.5 z(KL) + 0.25 prediction flip
```

The representative, reliability, and random branches change from

```text
0.75 / 0.05 / 0.20
```

to

```text
0.68 / 0.12 / 0.20
```

as disagreement severity increases. The reliability percentile band changes from 75-95 to 70-90. The first reliability pass allows at most two samples per full-precision predicted class.

The proxy model uses the target bit-width. Its activation ranges are calibrated with 128 candidate-pool samples using batch size 32.

## ImageNet features

| Model | Feature |
|---|---|
| ResNet-18 | `avgpool` |
| ResNet-50 | `avgpool` |
| MobileNetV2 | globally pooled `features` |
| RegNetX-600MF | `head.avg_pool` |
| DeiT-S | CLS token from final `norm` |

## PTQ

Weights use per-channel quantization. Activations use per-tensor quantization. Scale initialization uses MSE.

The first and last weight quantizers are 8-bit. The input activation of the final layer is 8-bit. Output quantization is disabled.

DeiT-S uses calibration-only PTQ. The reported budgets are 128, 256, 512, and 1024.

ImageNet CNNs use sequential layer/block reconstruction. The main settings are 20,000 iterations, rounding weight 0.01, 20% warm-up, and activation learning rate 4e-5. The calibration budget is 1024.

## Temperature scaling

Temperature scaling is fitted after PTQ using 4096 training samples. Evaluation uses temperature-scaled probabilities. ECE uses 15 equal-width bins in the main experiments.

## Run

One run:

```bash
bash scripts/run_single/run_imagenet_single.sh METHOD ARCH W A SEED BUDGET
```

Full ImageNet CNN grid:

```bash
DATA_PATH=/path/to/imagenet \
CKPT_DIR=/path/to/checkpoints \
CONFIRM_RUN=yes \
bash scripts/reproduce/reproduce_imagenet_cnn.sh
```

Full DeiT-S grid:

```bash
DATA_PATH=/path/to/imagenet \
CONFIRM_RUN=yes \
bash scripts/reproduce/reproduce_imagenet_deit.sh
```

Aggregate paper values are in `paper_results/`.
