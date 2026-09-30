# Reproducibility Protocol

This document records the experimental protocol used in the submitted ARCS manuscript. The numerical ARCS settings are fixed across the reported ImageNet CNN and DeiT-S experiments.

## Common protocol

For a fixed dataset, architecture, bit-width, calibration budget, and seed, Random, ConfGap, SelectQ, and ARCS use the same candidate pool and evaluation split. The candidate pool contains 4096 samples unless a stress test explicitly changes its composition.

Main seeds:

```text
1005, 1006, 1007
```

Main quantization settings:

```text
W4A4
W4A8
W8A4
W8A8
```

The common trusted-prediction threshold is:

```text
tau = 0.70
```

Coverage is allowed to vary across methods.

## ARCS configuration

ARCS uses the values in `configs/arcs.yaml`:

- disagreement score: confidence gap + margin gap + 0.5 x KL + 0.25 x prediction flip
- population-standardization (`unbiased=False`) with epsilon 1e-8
- continuous severity from mean one-sided confidence degradation and mean KL
- confidence and KL reference scales recomputed from the 90th percentile of the current candidate pool
- representative ratio: 0.75 at severity 0 to 0.68 at severity 1
- reliability ratio: 0.05 at severity 0 to 0.12 at severity 1
- random ratio: 0.20
- reliability band: 75-95 percentile at severity 0 to 70-90 percentile at severity 1
- first-pass cap: two reliability-branch samples per full-precision predicted class
- representative branch: greedy herding over L2-normalized embeddings
- reliability percentiles are computed only after the representative branch is removed

The lightweight proxy uses the same target bit-width setting as the final model. Its activation ranges are calibrated with 128 candidate-pool samples using batch size 32 before selection. The proxy omits the reconstruction optimization used by the final ImageNet CNN PTQ pipeline.

## Feature representations

ImageNet feature extraction uses the final pre-classifier representation:

| Architecture | Representation | Dimension |
|---|---|---:|
| ResNet-18 | `avgpool` | 512 |
| ResNet-50 | `avgpool` | 2048 |
| MobileNetV2 | globally pooled `features` | 1280 |
| RegNetX-600MF | `head.avg_pool` | 528 |
| DeiT-S | CLS token from final `norm` | 384 |

## ImageNet PTQ

ImageNet experiments use asymmetric uniform-affine quantization, per-channel weight quantization, per-tensor activation quantization, and MSE-based scale initialization.

For all reported ImageNet and DeiT-S experiments:

- first and last weight quantizers are 8-bit
- input activation of the final layer is 8-bit
- network-output quantization is disabled

### DeiT-S

DeiT-S uses calibration-only PTQ with no layer- or block-wise reconstruction. After subset selection, final-model activation ranges are estimated from the selected calibration samples using minibatches of 32.

Budgets:

```text
N = 128, 256, 512, 1024
```

### ImageNet CNNs

The selected calibration set is used for sequential layer/block reconstruction with AdaRound-style learned weight rounding and optimized activation scales.

Reconstruction settings:

```text
iterations: 20000
rounding-cost weight: 0.01
temperature schedule: 20 -> 2
warm-up: 20%
activation-scale learning rate: 4e-5
```

Main calibration budget:

```text
N = 1024
```

## Temperature scaling and metrics

Temperature scaling is fitted separately after PTQ for every quantized model using 4096 training samples kept separate from the evaluation set. It rescales logits only and therefore does not alter the selected subset, quantized parameters, predicted class, or Top-1 accuracy.

ECE uses 15 equal-width confidence bins unless a sensitivity experiment states otherwise. ECE, NLL, Brier score, absolute confidence gap, and trusted-prediction metrics are computed from temperature-scaled probabilities.

## Reproduction entry points

One ImageNet run:

```bash
DATA_PATH=/path/to/imagenet \
CKPT_DIR=/path/to/checkpoints \
bash scripts/run_single/run_imagenet_single.sh METHOD ARCH W A SEED NUM_SAMPLES
```

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

Reference aggregate values reported in the manuscript are stored under `paper_results/` to simplify verification of reproduced runs.
