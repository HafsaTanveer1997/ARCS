#!/usr/bin/env bash
set -euo pipefail

bash scripts/reproduce/reproduce_imagenet_cnn.sh
bash scripts/reproduce/reproduce_imagenet_deit.sh
