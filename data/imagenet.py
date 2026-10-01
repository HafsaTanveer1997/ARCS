import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _is_deit(arch: str) -> bool:
    return bool(arch) and arch.startswith("deit_")


def build_imagenet_eval_transform(input_size: int = 224, arch: str = ""):
    is_deit = _is_deit(arch)
    interpolation = (
        transforms.InterpolationMode.BICUBIC
        if is_deit
        else transforms.InterpolationMode.BILINEAR
    )
    resize_size = int(round(input_size / 0.9)) if is_deit else 256

    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_imagenet_data(
    data_path: str = "",
    input_size: int = 224,
    batch_size: int = 64,
    workers: int = 4,
    arch: str = "",
):
    is_deit = _is_deit(arch)
    interpolation = (
        transforms.InterpolationMode.BICUBIC
        if is_deit
        else transforms.InterpolationMode.BILINEAR
    )

    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size, interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        build_imagenet_eval_transform(input_size=input_size, arch=arch),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def build_imagenet_selection_dataset(
    data_path: str,
    input_size: int = 224,
    split: str = "train",
    arch: str = "",
):
    root = os.path.join(data_path, split)
    return datasets.ImageFolder(
        root,
        build_imagenet_eval_transform(input_size=input_size, arch=arch),
    )
