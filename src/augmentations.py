import random
from torch import nn
import torch
from torchvision import transforms as T


def get_default_aug() -> nn.Module:
    return torch.nn.Sequential(
        RandomApply(T.RandomRotation(degrees=(10, 60)), p=0.2),
        T.RandomHorizontalFlip(),
        RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
    )


class RandomApply(nn.Module):
    def __init__(self, fn: nn.Module, p: float):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)
