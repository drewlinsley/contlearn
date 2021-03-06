from omegaconf import DictConfig, ValueNode
import torch
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
from PIL import Image
import csv
from torchvision import transforms


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


class DeadRect(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform

        if self.train:
            csv_path = join(self.path, "train.csv")
            self.path = join(self.path, "train")
        else:
            csv_path = join(self.path, "test.csv")
            self.path = join(self.path, "test")

        with open(csv_path, 'r') as fob:
            self.file_list = csv.DictReader(fob)
            self.file_list = list(self.file_list)
            print(len(self.file_list), self.train, self.path)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        file_name = self.file_list[index]['fname']
        img = load_image(join(self.path, file_name))
        img = self.transform(img)
        label = int(self.file_list[index]['same'] == 'True')
        return img, label

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
