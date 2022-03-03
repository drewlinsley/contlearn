"""Functions for 3d augmentation."""
import random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch import Tensor
from typing import List, Tuple, Any, Optional
from torchvision.transforms import functional_pil as F_pil
from torchvision.transforms import functional_tensor as F_t

import augmentor


def crop(img: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    """Crop the given image at specified location and output size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        PIL Image or Tensor: Cropped image.
    """

    if not isinstance(img, torch.Tensor):
        return F_pil.crop(img, top, left, height, width)

    return F_t.crop(img, top, left, height, width)


def center_crop(img: Tensor, output_size: List[int]) -> Tensor:
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    image_width, image_height = get_image_size(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = pad(img, padding_ltrb, fill=0)  # PIL uses fill value 0
        image_width, image_height = get_image_size(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return crop(img, crop_top, crop_left, crop_height, crop_width)


def Oldrandomcrop(volume, label, params):
    """Apply random crop to both volume and label."""
    def get_params(vol, vol_shape, output_size):
        print(vol_shape)
        _, d, w, h = vol_shape
        td, th, tw = output_size

        if d + 1 < td or h + 1 < th or w + 1 < tw:
            raise ValueError(
                f"Required crop size {(th, tw)} is larger then input image size {(h, w)}")  # noqa 

        if d == td and w == tw and h == th:
            return 0, 0, 0, d, h, w

        i = torch.randint(0, d - td + 1, size=(1,)).item()
        j = torch.randint(0, h - th + 1, size=(1,)).item()
        k = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, k, td, th, tw

    vol_shape = volume.shape
    label_shape = label.shape
    i, j, k, d, h, w = get_params(volume, vol_shape, params)

    combined = torch.cat((volume, label), 0)
    cropped = combined[:, i: i + d, j: j + h, k: k + w]

    # # Now transpose back to the original ordering and split volume/label
    # cropped = cropped.permute((0, 2, 1, 3))
    volume = cropped[:vol_shape[0]]
    label = cropped[vol_shape[0]:]
    return volume, label


def randomcrop(volume, label, params):
    """Apply random crop to both volume and label."""
    def _get_slice(sz, crop_sz):
        try : 
            lower_bound = torch.randint(sz-crop_sz, (1,)).item()
            return lower_bound, lower_bound + crop_sz
        except: 
            return (None, None)

    def _crop(x, slice_h, slice_w, slice_d):
        return x[:, slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]
    _, d, h, w = volume.shape
    crop_sz = tuple(params)
    img_sz = tuple((d, h, w))

    slice_hwd = [_get_slice(i, k) for i, k in zip(img_sz, params)]
    volume = _crop(volume, *slice_hwd)
    label = _crop(label, *slice_hwd)
    return volume, label


def randomrotate(volume, label, params):
    """Apply random crop to both volume and label."""
    vol_shape = volume.shape
    label_shape = label.shape
    # combined = torch.cat((volume, label), 0)

    # Params tells us which dimensions to rotate over
    for dim in params:
        k = torch.randint(low=0, high=3, size=(1,)).item()
        if k:
            # combined = torch.rot90(combined, k=k, dims=dim)
            volume = torch.rot90(volume, k=k, dims=dim)
            label = torch.rot90(label, k=k, dims=dim)
    # volume = combined[:vol_shape[0]]
    # label = combined[vol_shape[0]:]
    return volume, label


def randomflip(volume, label, params):
    """Apply random flip to both volume and label."""
    vol_shape = volume.shape
    label_shape = label.shape
    # combined = torch.cat((volume, label), 0)

    # Params tells us which dimensions to rotate over
    flips = []
    for dim in params:
        flip = random.random() > 0.5
        if flip:
            flips.append(dim)
    # combined = torch.flip(combined, dims=flips)
    # volume = combined[:vol_shape[0]]
    # label = combined[vol_shape[0]:]
    volume = torch.flip(volume, dims=flips)
    label = torch.flip(label, dims=flips)
    return volume, label


def normalize_volume(volume, label, params):
    """Apply normalization to volume."""
    min_val, max_val = params
    volume = (volume - min_val) / (max_val - min_val)
    return volume, label


def normalize_volume_z(volume, label, params):
    """Apply normalization to volume."""
    mu, std = params
    volume = (volume - mu) / std
    return volume, label


def cast_label(volume, label, params):
    """Apply normalization to volume."""
    dtype = params
    label = label.to(dtype)
    return volume, label


def to_numpy(volume, label, params):
    """Apply normalization to volume."""
    volume = volume.numpy()
    label = label.numpy()
    return volume, label


def random_selection(volume, label, params):
    """Apply normalization to volume."""
    bs = len(volume)
    sel = np.random.randint(bs)
    return volume[sel], label[sel]


def warp(volume, label, params):
    """Warp a volume/label."""
    warp_fun = augmentor.warp.Warp(**params)
    vol_shape = volume.shape
    out = warp_fun({"volume": volume, "label": label})
    return out["volume"], out["label"]


def pixel_contrast_brightness_2d(volume, label, params):
    """Slice independent pixel augs."""
    pixel_fun = augmentor.grayscale.MixedGrayscale2D(**params)
    out = pixel_fun({"volume": volume, "label": label})
    return out["volume"], label


def pixel_contrast_brightness_3d(volume, label, params):
    """Slice independent pixel augs."""
    pixel_fun = augmentor.grayscale.Grayscale3D(**params)
    out = pixel_fun({"volume": volume, "label": label})
    return out["volume"], label


def misalign_and_missing(volume, label, params):
    """Misalign and remove random slices."""
    misalign_fun = augmentor.misalign.MisalignPlusMissing(**params)
    out = misalign_fun({"volume": volume, "label": label})
    return out["volume"], out["label"]
