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


class Volumetric(Dataset):
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
        volume = self.norma(volume)
        label = data["label"]
        volume = volume.reshape(self.vol_size)
        label = label.reshape(self.label_size)
        volume = volume.transpose(self.vol_transpose)
        label = label.transpose(self.label_transpose)
        return volume, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class PFClass(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, rand_color_invert_p: float, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.rand_color_invert_p = rand_color_invert_p
        self.norma = transforms.Normalize((0.5), (0.5))

        #self.file_list = [join(self.path, f) for f in listdir(self.path) if isfile(join(self.path, f))]
        self.file_list = []
        with open(self.path, 'r') as fob:
            self.file_list = fob.read().splitlines()
            self.file_list = list(self.file_list)
            print("Found", len(self.file_list), "files for train=", self.train, "from", self.path)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        file_name = self.file_list[index]
        img = load_image(file_name)
        img = self.transform(img)
        if torch.rand(1).item() < self.rand_color_invert_p:
            img = invert(img)
        img = self.norma(img)
        label = int(file_name[-5])
        return img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class PFClassColour(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, rand_color_invert_p: float, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.rand_color_invert_p = rand_color_invert_p
        self.norma = transforms.Normalize((0.5), (0.5))

        #self.file_list = [join(self.path, f) for f in listdir(self.path) if isfile(join(self.path, f))]
        self.file_list = []
        with open(self.path, 'r') as fob:
            self.file_list = fob.read().splitlines()
            self.file_list = list(self.file_list)
            print("Found", len(self.file_list), "files for train=", self.train, "from", self.path)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        file_name = self.file_list[index]
        img = load_image(file_name)
        img = self.transform(img)
        label = int(file_name[-5])
        if self.rand_color_invert_p > 0:
            rnd = torch.rand(1).item()
            if rnd < 0.5:  # rnd < 0.33:
                img = colour(img, ch=0, num_ch=2)
            elif rnd >= 0.5:  # rnd > 0.33 and rnd < 0.66:
                img = colour(img, ch=1, num_ch=2)
                # if label == 1:
                #     label += 1  # Add 2 positive outputs
        return img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"
