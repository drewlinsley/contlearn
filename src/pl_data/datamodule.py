from typing import Optional, Sequence

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, ValueNode
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        val_percentage: float,
        cfg: DictConfig,
        use_train_dataset: str,
    ):
        super().__init__()
        self.cfg = cfg
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_percentage = val_percentage
        self.use_train_dataset = use_train_dataset

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

    def setup(self, stage: Optional[str] = None):
        # transforms
        transform = transforms.Compose(
            [
                transforms.Resize((100, 100)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )

        # split dataset

        if stage is None or stage == "fit":
            plank_train = hydra.utils.instantiate(
                self.datasets[self.use_train_dataset].train, cfg=self.cfg, transform=transform,
                _recursive_=False
            )
            train_length = int(len(plank_train) * (1 - self.val_percentage))
            val_length = len(plank_train) - train_length
            self.train_dataset, self.val_dataset = random_split(
                plank_train, [train_length, val_length]
            )
        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(x, cfg=self.cfg, transform=transform, _recursive_=False)
                for x in self.datasets[self.use_train_dataset].test
            ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            multiprocessing_context='fork'
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            multiprocessing_context='fork'
        )

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                multiprocessing_context='fork'
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets}, "
            f"{self.num_workers}, "
            f"{self.batch_size})"
            f"{self.val_percentage}"
        )
