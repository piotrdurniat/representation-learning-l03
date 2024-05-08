import numpy as np
import torch
import torchvision.transforms as T
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import MNIST
from torchvision.datasets.vision import VisionDataset


class VisionDatamodule(LightningDataModule):
    VAL_SIZE = 0.15

    def __init__(
            self,
            root_dir: str,
            batch_size: int,
            num_workers: int = 0,
            num_samples_per_class: int = -1,
    ):
        super().__init__()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_samples_per_class = num_samples_per_class
        self.in_channels = 1

        self.transform = T.Compose(
            [T.ToTensor(), T.Normalize(mean=torch.tensor([0.1307]), std=torch.tensor([0.3081]))])

        self.seed = 42

    def prepare_data(self) -> None:
        MNIST(root=self.root_dir, train=True, transform=self.transform, download=True)

    def setup(self, stage: str | None = None) -> None:
        train_ds = MNIST(
            root=self.root_dir,
            train=True,
            transform=self.transform,
            download=False,
        )

        self.test_ds = MNIST(
            root=self.root_dir,
            train=False,
            transform=self.transform,
            download=False,
        )

        self.train_ds, self.val_ds = sample_and_split_dataset(
            dataset=train_ds,
            val_size=self.VAL_SIZE,
            num_samples_per_class=self.num_samples_per_class,
            seed=self.seed,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


def sample_and_split_dataset(
        dataset: VisionDataset,
        val_size: float,
        num_samples_per_class: int,
        seed: int,
) -> tuple[Dataset, Dataset]:
    data_idx = np.arange(len(dataset))

    if num_samples_per_class != -1:
        sampled_idx = subsample_idx(
            dataset=dataset,
            num_samples_per_class=num_samples_per_class,
            seed=seed,
        )

    data_idx = data_idx[sampled_idx]
    labels = dataset.targets[sampled_idx]

    train_idx, val_idx = train_test_split(
        data_idx,
        test_size=val_size,
        stratify=labels,
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def subsample_idx(
        dataset: Dataset,
        num_samples_per_class: int,
        seed: int,
) -> TensorDataset:
    np.random.seed(seed)
    unique_labels = sorted(dataset.targets.unique())
    sampled_idx = []

    for i in unique_labels:
        label_idxs = (dataset.targets == i).int().nonzero().squeeze().tolist()
        idx = np.random.choice(label_idxs, size=num_samples_per_class)
        sampled_idx.extend(idx)

    return sampled_idx
