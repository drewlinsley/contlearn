from omegaconf import DictConfig, ValueNode
import torch
from torch.utils.data import Dataset
import os
from os.path import isfile, join
from PIL import Image
import csv
from torchvision import transforms
import tensorflow as tf  # for reading TFRecord Dataset
import tensorflow_datasets as tfds  # for making tf.data.Dataset to return numpy arrays
import numpy as np
from tensorflow.python.lib.io import file_io


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


def read_labeled_tfrecord(example):
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


class VolumetricTF(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.cache = True  # Push to CFG
        self.repeat = True  # Push to CFG
        self.shuffle = True  # Push to CFG
        self.vol_size = [64, 128, 128, 2]
        self.label_size = [64, 128, 128, 6]
        self.vol_transpose = (3, 0, 1, 2)
        self.label_transpose = (3, 0, 1, 2)
        self.shuffle_buffer = 128
        self.batch_size = 1
        self.reader_name = "default"
        self.len = 97

        ds = tf.data.TFRecordDataset(self.path, num_parallel_reads=tf.data.experimental.AUTOTUNE)  # , compression_type="GZIP")
        if self.cache:
            # You'll need around 15GB RAM if you'd like to cache val dataset, and 50~60GB RAM for train dataset.
            ds = ds.cache()

        if self.repeat:
            ds = ds.repeat()

        if self.shuffle:
            ds = ds.shuffle(self.shuffle_buffer)
            opt = tf.data.Options()
            opt.experimental_deterministic = False
            ds = ds.with_options(opt)

        if self.reader_name == "default":
            reader = read_labeled_tfrecord
        else:
            raise NotImplementedError("{} is not implemented".format(self.reader_name))

        ds = ds.map(reader, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        self.ds = tfds.as_numpy(ds)
        if self.len is None:
            self.len = len([idx for idx, _ in enumerate(self.ds)])

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int):
        data = next(iter(self.ds))
        volume = data["volume"]
        # volume = self.norma(volume)
        label = data["label"].astype(int)
        volume = volume.reshape(self.vol_size)
        label = label.reshape(self.label_size)
        volume = volume.transpose(self.vol_transpose)
        label = label.transpose(self.label_transpose)
        return volume, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class VolumetricNPZ(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.files = []
        self.path = os.path.join(os.path.dirname(__file__), self.path)
        with open(self.path, 'r') as fob:
            self.files = fob.read().splitlines()
            self.files = list(self.files)
        self.len = len(self.files)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int):
        file_name = self.files[index]
        volume, label = load_npz(file_name)
        # img = self.transform(img)
        return volume, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


def load_npz(f):
    """Load from npz and then close file."""
    f = BytesIO(file_io.read_file_to_string(f, binary_mode=True))
    d = np.load(f)
    volume = d["volume"]
    label = d["label"]
    del d.f
    d.close()
    return volume, label
