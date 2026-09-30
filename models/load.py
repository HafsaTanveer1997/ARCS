from __future__ import annotations

import os

import timm
import torch

from .mobilenetv2 import mobilenetv2
from .regnet import regnetx_600m
from .resnet import resnet18, resnet50


def load_pretrained_model(arch: str, checkpoint_dir: str | None = None):
    if arch == "deit_small_patch16_224":
        return timm.create_model(arch, pretrained=True)

    checkpoint_dir = checkpoint_dir or os.environ.get("CKPT_DIR", "pretrained")

    if arch == "resnet18":
        model = resnet18()
        state = torch.load(
            os.path.join(checkpoint_dir, "resnet18_imagenet.pth.tar"),
            map_location="cpu",
        )
        model.load_state_dict(state)
        return model

    if arch == "resnet50":
        model = resnet50()
        state = torch.load(
            os.path.join(checkpoint_dir, "resnet50_imagenet.pth.tar"),
            map_location="cpu",
        )
        model.load_state_dict(state)
        return model

    if arch == "mobilenetv2":
        model = mobilenetv2()
        state = torch.load(
            os.path.join(checkpoint_dir, "mobilenetv2.pth.tar"),
            map_location="cpu",
        )
        model.load_state_dict(state["model"])
        return model

    if arch == "regnetx_600m":
        model = regnetx_600m()
        state = torch.load(
            os.path.join(checkpoint_dir, "regnet_600m.pth.tar"),
            map_location="cpu",
        )
        model.load_state_dict(state)
        return model

    raise ValueError(f"Unsupported architecture: {arch}")
