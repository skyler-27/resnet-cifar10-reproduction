"""Deterministic train/validation split; test images are never used for tuning."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
           'frog', 'horse', 'ship', 'truck')


def split_indices(size: int, validation_size: int, seed: int):
    if not 0 < validation_size < size:
        raise ValueError('validation_size must be between 0 and dataset size')
    indices = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    return indices[validation_size:], indices[:validation_size]


def loaders(root: Path, batch_size: int, seed: int, augment: bool,
            validation_size: int = 5000, workers: int = 0):
    normalize = transforms.Normalize(MEAN, STD)
    train_ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if augment else []
    train_ops += [transforms.ToTensor(), normalize]
    eval_ops = transforms.Compose([transforms.ToTensor(), normalize])
    train_data = datasets.CIFAR10(root, train=True, download=True,
                                  transform=transforms.Compose(train_ops))
    validation_data = datasets.CIFAR10(root, train=True, download=False, transform=eval_ops)
    test_data = datasets.CIFAR10(root, train=False, download=True, transform=eval_ops)
    train_ids, validation_ids = split_indices(len(train_data), validation_size, seed)
    train = DataLoader(Subset(train_data, train_ids), batch_size=batch_size, shuffle=True,
                       num_workers=workers, pin_memory=False)
    validation = DataLoader(Subset(validation_data, validation_ids), batch_size=batch_size,
                            num_workers=workers)
    test = DataLoader(test_data, batch_size=batch_size, num_workers=workers)
    return train, validation, test
