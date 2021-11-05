from omegaconf import DictConfig, ValueNode
import torch
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
from PIL import Image
import csv
from torchvision import transforms
import tensorflow as tf  # for reading TFRecord Dataset
import tensorflow_datasets as tfds  # for making tf.data.Dataset to return numpy arrays
from src.pl_data import augmentation_functions as af
from inspect import getmembers, isfunction


def load_image(directory):
    return Image.open(directory).convert('L')


def invert(img):

    if img.ndim < 3:
        raise TypeError("Input image tensor should have at least 3 dimensions, but found {}".format(img.ndim))

    bound = torch.tensor(1 if img.is_floating_point() else 255, dtype=img.dtype, device=img.device)
    return bound - img


def colour(img, ch=0, num_ch=3):

    colimg = [torch.zeros_like(img)] * num_ch
    # colimg[ch] = img
    # Use beta distribution to push the mixture to ch 1 or ch 2
    if ch == 0:
        rand = torch.distributions.beta.Beta(0.5, 1.)
    elif ch == 1:
        rand = torch.distributions.beta.Beta(1., 0.5) 
    else:
        raise NotImplementedError("Only 2 channel images supported now.")
    rand = rand.sample()
    colimg[0] = img * rand
    colimg[1] = img * (1 - rand)
    return torch.cat(colimg)


def full_read_labeled_tfrecord(example):
    tfrec_format = {
        "volume": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrec_format)

    volume = tf.reshape(example["volume"], shape=[])
    label = tf.reshape(example["label"], shape=[])
    volume = tf.io.decode_raw(volume, tf.float32)
    label = tf.io.decode_raw(label, tf.float32)
    volume = tf.reshape(volume, [64, 128, 128, 2])
    label = tf.reshape(label, [64, 128, 128, 6])
    return {"volume": volume, "label": label}


def cheap_read_labeled_tfrecord(example):
    tfrec_format = {
        "volume": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrec_format)

    volume = tf.reshape(example["volume"], shape=[])
    label = tf.reshape(example["label"], shape=[])
    return {"volume": volume, "label": label}


def expensive_tfrecord_transform(example):
    volume = tf.io.decode_raw(example["volume"], tf.float32)
    label = tf.io.decode_raw(example["label"], tf.float32)
    volume = tf.reshape(volume, [64, 128, 128, 2])
    label = tf.reshape(label, [64, 128, 128, 6])
    return {"volume": volume, "label": label}


def augment3d(volume, label, augmentations):
    """Apply 3d augmentations to volume, label, or both."""
    functions_list = getmembers(af, isfunction)
    functions_dict = {k: v for k, v in functions_list}
    for aug, params in augmentations.items():
        aug_fun = functions_dict[aug]
        volume, label = aug_fun(volume, label, params)
    return volume, label


class Volumetric(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs  # noqa
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        if hasattr(self.cfg.train.pl_trainer, "tpu_cores") and self.cfg.train.pl_trainer.tpu_cores == 1:  # noqa
            self.cache = True  # Push to CFG
        else:
            self.cache = False  # Push to CFG
        self.repeat = True  # Push to CFG
        self.shuffle = True  # Push to CFG
        self.vol_size = [64, 128, 128, 2]
        self.label_size = [64, 128, 128, 6]
        self.vol_transpose = (3, 0, 1, 2)
        self.label_transpose = (3, 0, 1, 2)
        self.shape = [32, 32, 32]
        self.augmentations = [
            {"randomcrop": self.shape},
            {"randomrotate": [0, 1, 2]},  # Axes to rotate
            {"randomflip": [0, 1, 2]},  # Axes to rotate
            {"normalize_volume": [0, 255]},  # Min/max
        ]
        # self.batch_size = 1
        tag = getattr(self.cfg.data.datamodule.datasets,[x for x in self.cfg.data.datamodule.datasets.keys()][0])  # noqa
        train = "train" if self.train else "val"
        self.len = getattr(tag, train).len
        self.shuffle_buffer = min(32, self.len)
        # self.shuffle_buffer = min(64, self.len)
        # self.len = None  # TESTING AUTO-COUNT


        ds = tf.data.TFRecordDataset(self.path, num_parallel_reads=tf.data.experimental.AUTOTUNE)  # noqa
        ds = ds.map(full_read_labeled_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # noqa
        ds = ds.batch(1, drop_remainder=True)

        if self.cache:
            # raise RuntimeError("Cache fails if TPU > 1.")
            ds = ds.cache("~/cache")

        if self.repeat and self.len is not None:
            ds = ds.repeat()

        if self.shuffle:
            ds = ds.shuffle(self.shuffle_buffer)
            opt = tf.data.Options()
            opt.experimental_deterministic = False
            ds = ds.with_options(opt)

        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        self.ds = tfds.as_numpy(ds)
        if self.len is None:
            print("Counting length of {}".format(train))
            self.len = len([idx for idx, _ in enumerate(self.ds)])
            print("Found length of {}".format(self.len))

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int):
        data = next(iter(self.ds))
        volume = data["volume"]
        # volume = self.norma(volume)
        label = data["label"].astype(int)
        volume = volume.reshape(self.vol_size)
        label = label.reshape(self.label_size)

        # Add augs here
        volume, label = augment3d(
            volume=volume,
            label=label,
            augmentations=self.augmentations)
        volume = volume[:self.shape[0], :self.shape[1], :self.shape[2]]
        label = label[:self.shape[0], :self.shape[1], :self.shape[2]]

        volume = volume.transpose(self.vol_transpose)
        label = label.transpose(self.label_transpose)
        return volume, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


# from torch_xla.utils.tf_record_reader import TfRecordReader
# import multiprocessing
# import random
# from collections import deque


# class TestVolumetric(Dataset):
#     def __init__(
#         self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
#     ):
#         super().__init__()
#         self.cfg = cfg
#         self.path = path
#         if not isinstance(self.path, list):
#             self.path = [self.path]
#         self.train = train
#         self.transform = transform
#         self.cache = True  # Push to CFG
#         self.repeat = True  # Push to CFG
#         self.shuffle = False  # Push to CFG
#         self.vol_size = [64, 128, 128, 2]
#         self.label_size = [64, 128, 128, 6]
#         self.vol_transpose = (3, 0, 1, 2)
#         self.label_transpose = (3, 0, 1, 2)
#         self.drop_remainder = True  # drop_remainder
#         self.batch_size = 1
#         self.transforms = {"volume": str, "label": str}

#         num_samples = [self.samples_in_file(p) for p in self.path]
#         self.total_samples = sum(num_samples)
#         self.len = self.total_samples // (self.batch_size)
#         self.num_prefetch_batches = 1  # prefetch
#         self.prefetch_buffer = deque()
#         if self.len < 1:
#             raise ValueError(f"""Batch size {self.batch_size} larger than
#                                 number of samples in the TFRecord files {self.total_samples}.""")

#         if self.len < self.num_prefetch_batches:
#             raise ValueError(f"""Not enough samples to prefetch: (length = {self.len},
#                             num_to_prefech = {self.num_prefetch_batches}),
#                             lower the number of prefetch batches.""")
#         self.samples_per_file = {f: n for (f, n) in
#                                  zip(self.path, num_samples)}
#         self.data = None
#         self.counter = 0

#     def samples_in_file(self, filename):
#         ds = TfRecordReader(filename, transforms=self.transforms)
#         reader = ds
#         count = 0
#         while reader.read_example():
#             count += 1
#         return count

#     def __len__(self) -> int:
#         return self.len

#     def __iter__(self):
#         self.file_index = 0
#         self.data_index = 0
#         self.counter = 0
#         self.data = None
#         self.fill_buffer(self.num_prefetch_batches)
#         return self

#     def __getitem__(self, index: int):
#         # data = next(iter(self.ds))
#         # volume = data["volume"]
#         # # volume = self.norma(volume)
#         # label = data["label"].astype(int)
#         # volume = volume.reshape(self.vol_size)
#         # label = label.reshape(self.label_size)

#         # # Add augs here
#         # volume = volume[:self.shape[0], :self.shape[1], :self.shape[2]]
#         # label = label[:self.shape[0], :self.shape[1], :self.shape[2]]

#         # volume = volume.transpose(self.vol_transpose)
#         # label = label.transpose(self.label_transpose)
#         # return volume, label

#         if self.drop_remainder:
#             if self.counter == self.len:
#                 raise StopIteration

#         if len(self.prefetch_buffer) == 0:
#             raise StopIteration

#         result = self.prefetch_buffer.popleft()
#         self.counter += 1
#         self.fill_buffer(1)
#         return result

#     def fill_buffer(self, num_batches):
#         import pdb;pdb.set_trace()
#         if self.data is None:
#             self.load_data()
#         for _ in range(num_batches):
#             curr_batch = []
#             still_required = self.batch_size
#             while still_required > 0:
#                 data = self.data[self.data_index:
#                                  self.data_index + still_required]
#                 self.data_index += len(data)
#                 curr_batch += data
#                 still_required = self.batch_size - len(curr_batch)
#                 if still_required > 0:
#                     if self.file_index < len(self.files):
#                         self.load_data()
#                     else:
#                         break
#             if len(curr_batch) == self.batch_size:
#                 result = {}
#                 for k in KEYS:
#                     result[k] = np.vstack([item[k] for item in curr_batch])
#                 self.prefetch_buffer.append(self.post_process(result))

#     def load_data(self):
#         if self.file_index >= len(self.files):
#             raise ValueError('No more files to load.')
#         # self.data = self.load_file(self.files[self.file_index])
#         self.data = self.load_file(self.path)
#         self.file_index += 1
#         self.data_index = 0
#         if self.shuffle:
#             np.random.shuffle(self.data)

#     def load_file(self, filename):
#         reader = TfRecordReader(filename, transforms=self.transforms)
#         data = []
#         ex = reader.read_example()
#         while ex:
#             data.append(ex)
#             ex = reader.read_example()
#         return data

#     def __repr__(self) -> str:
#         return f"MyDataset({self.name}, {self.path})"
