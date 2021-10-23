from typing import Optional, Sequence

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, ValueNode
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as transforms_image
from pytorchvideo import transforms as transforms_video
from PIL import Image
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)


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
        # transform = transforms_image.Compose(
        #     [
        #         # transforms.Resize((100, 100)),
        #         transforms_image.RandomHorizontalFlip(p=0.5),
        #         transforms_image.RandomVerticalFlip(p=0.5),
        #         transforms_volume.crop.RandomCrop(size=self.cfg.data.input_size),  #noqa
        #         transforms_video.Div255(),
        #         transforms_image.ToTensor(),
        #     ]
        # )


        train_transform = Compose(
            [
            ApplyTransformToKey(
              key="volume",
              transform=Compose(
                  [
                    Lambda(lambda x: x / 255.0),
                    RandomCrop(self.cfg.data.input_size),
                    RandomHorizontalFlip(p=0.5),
                    RandomVerticalFlip(p=0.5),
                    transforms_image.ToTensor(),
                  ]
                ),
              ),
            ]
        )
        test_transform = Compose(
            [
            ApplyTransformToKey(
              key="volume",
              transform=Compose(
                  [
                    Lambda(lambda x: x / 255.0),
                    # CenterCrop(self.cfg.data.input_size),
                    transforms_image.ToTensor(),
                  ]
                ),
              ),
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
