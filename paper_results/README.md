# Reference Paper Results

This directory contains compact aggregate values reported in the submitted manuscript. They are included so that reproduced runs can be checked against the paper without committing large logits, model checkpoints, or datasets.

Files currently included:

- `deit_budget.csv`: DeiT-S results across four bit-width settings, four calibration budgets, and the four selection methods.
- `class_skew_deit_w8a4_n128.csv`: 80/20 class-skew candidate-pool experiment.
- `ablation_deit_n128.csv`: representative/reliability selection ablations reported for W4A4 and W8A4.
- `ece_binning_resnet18_seed1005.csv`: ECE sensitivity to 10, 15, 20, and 30 equal-width bins.
- `overhead_deit_w4a4_n128_seed1005.csv`: offline preparation time and peak GPU memory for the representative overhead configuration.

Large per-example outputs are intentionally excluded. CIFAR-100 aggregate values will be added only after the underlying source result files are cross-checked against the submitted table.
