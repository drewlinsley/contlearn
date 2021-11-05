"""Functions for 3d augmentation."""
import torch
import torch.nn.functional as F


def randomcrop(volume, label, params):
    """Apply random crop to both volume and label."""
    def get_params(vol, vol_shape, output_size):
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
    i, j, k, d, h, w = self.get_params(volume, vol_shape, params)

    combined = torch.cat((volume, label), 0)
    cropped = F.crop(combined, j, k, h, w)

    # Now transpose to swap depth and height and run another crop
    cropped = F.crop(combined.permute((0, 2, 1, 3)), i, k, d, w)

    # Now transpose back to the original ordering and split volume/label
    cropped = cropped.permute((0, 2, 1, 3))
    volume = cropped[:vol_shape[0]]
    label = cropped[:label_shape[0]]
    return volume, label


def randomrotate(volume, label, params):
    """Apply random crop to both volume and label."""
    vol_shape = volume.shape
    label_shape = label.shape
    combined = torch.cat((volume, label), 0)

    # Params tells us which dimensions to rotate over
    for dim in params:
        k = torch.randint(low=0, high=3, size=(1,)).item()
        combined = torch.rot90(combined, k=k, dims=[dim])
    volume = combined[:vol_shape[0]]
    label = combined[:label_shape[0]]
    return volume, label


def randomflip(volume, label, params):
    """Apply random flip to both volume and label."""
    vol_shape = volume.shape
    label_shape = label.shape
    combined = torch.cat((volume, label), 0)

    # Params tells us which dimensions to rotate over
    flips = []
    for dim in params:
        flip = torch.rand() > 0.5
        if flip:
            flips.append(dim)
    combined = torch.flip(combined, dims=flips)
    volume = combined[:vol_shape[0]]
    label = combined[:label_shape[0]]
    return volume, label


def normalize_volue(volume, label, params):
    """Apply normalization to volume."""
    min_val, max_val = params
    volume = (volume - min_val) / (max_val - min_val)
    return volume, label
