from typing import Optional, Sequence

import torch
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, ValueNode
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import transforms
from PIL import Image

from monai import transforms as monai_transforms
# from pytorchvideo import transforms as video_transforms
# import albumentations as A

from torch._utils import _accumulate
from torch import default_generator, Generator
from typing import (
    TypeVar,
    List,
)
T = TypeVar('T')


def continuous_random_split(dataset: Dataset[T], lengths: Sequence[int],
                 generator: Optional[Generator] = default_generator) -> List[Subset[T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    indices = torch.arange(sum(lengths)).tolist()
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]


# class RRC(video_transforms.RandomResizedCrop):
#     def __call__(self, image, label):
#         """
#         Args:
#             img (PIL Image): Image to be cropped and resized.

#         Returns:
#             PIL Image: Randomly cropped and resized image.
#         """

#         # First find shape of image
#         im_shape = image.shape
#         print(im_shape)

#         # Second concatenate image/label across chanel dim
#         cat = torch.cat((image, label), 1)

#         # Third crop
#         cat = self(cat)

#         # Fourth split into image/label
#         image = cat[:, :im_shape[1]]
#         label = cat[:, im_shape[1]:]
#         return image, label


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        val_proportion: float,
        cfg: DictConfig,
        use_train_dataset: str,
        use_val_dataset: str,
        shape: list,
    ):
        super().__init__()
        self.cfg = cfg
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_proportion = val_proportion
        self.use_train_dataset = use_train_dataset
        self.use_val_dataset = use_train_dataset
        self.shape = shape

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

    def setup(self, stage: Optional[str] = None):
        # transform = video_transforms.Compose(
        #     [
        #         RRC(interpolation="nearest", target_height=self.shape[1], target_width=self.shape[2])  # noqa
        #     ]
        # )
        # transform = monai_transforms.MapTransform(
        #     [
        #         monai_transforms.ToDevice("cpu"),
        #         monai_transforms.RandCropByLabelClassesd(
        #             keys=["image", "label"],
        #             spatial_size=self.shape,
        #             label_key=["label"],
        #             num_classes=3,
        #             num_samples=1,
        #             ratios=[0.1, 2, 2]),
        #         monai_transforms.ScaleIntensityRange(
        #             a_min=0.,
        #             a_max=255.,
        #             b_min=0.,
        #             b_max=1.)
        #         transforms.RandomResizedCrop(
        #             target_height=self.shape[1],
        #             target_width=self.shape[2],
        #             interpolation="nearest")
        #     ]
        # )
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        if stage is None or stage == "fit":
            assert self.val_proportion >= 0 and self.val_proportion < 1., \
                "val_proportion must be 1 > x >= 0"
            plank_train = hydra.utils.instantiate(
                self.datasets[self.use_train_dataset].train,
                cfg=self.cfg,
                transform=transform,
                _recursive_=False
            )
            if self.val_proportion == 0:
                plank_val = hydra.utils.instantiate(
                    self.datasets[self.use_val_dataset].val,
                    cfg=self.cfg,
                    transform=transform,
                    _recursive_=False
                )
                self.train_dataset = plank_train
                self.val_dataset = plank_val
            else:
                import pdb;pdb.set_trace()
                train_length = int(len(plank_train) * (1 - self.val_proportion))  # noqa
                val_length = len(plank_train) - train_length
                self.train_dataset, self.val_dataset = continuous_random_split(
                    plank_train, [train_length, val_length]
                )

        elif stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(
                    self.datasets[self.use_train_dataset].test,
                    cfg=self.cfg,
                    transform=transform,
                    _recursive_=False)
            ]
        else:
            raise NotImplementedError("stage: {}".format(stage))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            # multiprocessing_context='fork'
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            # multiprocessing_context='fork'
        )

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                # multiprocessing_context='fork'
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
