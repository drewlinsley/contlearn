import torch
import torch.nn as nn
import torch.nn.functional as F



class Unet(nn.Module):
    """Simple U-Net without training logic"""
    def __init__(self, hparams: dict):
        """
        Args:
            hparams: the hyperparameters needed to construct the network.
                Specifically these are:
                * start_filts (int)
                * depth (int)
                * in_channels (int)
                * num_classes (int)
        """
        super().__init__()
        # 4 downsample layers
        out_filts = hparams.get('start_filts', 16)
        depth = hparams.get('depth', 3)
        in_filts = hparams.get('in_channels', 1)
        num_classes = hparams.get('num_classes', 2)

        for idx in range(depth):
            down_block = torch.nn.Sequential(
                torch.nn.Conv3d(in_filts, out_filts, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv3d(out_filts, out_filts, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True)
            )
            in_filts = out_filts
            out_filts *= 2

            setattr(self, 'down_block_%d' % idx, down_block)

        out_filts = out_filts // 2
        in_filts = in_filts // 2
        out_filts, in_filts = in_filts, out_filts

        for idx in range(depth-1):
            up_block = torch.nn.Sequential(
                torch.nn.Conv3d(in_filts + out_filts, out_filts, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv3d(out_filts, out_filts, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True)
            )

            in_filts = out_filts
            out_filts = out_filts // 2

            setattr(self, 'up_block_%d' % idx, up_block)

        self.final_conv = torch.nn.Conv3d(in_filts, num_classes, kernel_size=1)
        self.max_pool = torch.nn.MaxPool3d(2, stride=2)
        self.up_sample = torch.nn.Upsample(scale_factor=2)
        self._hparams = hparams

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forwards the :attr`input_tensor` through the network to obtain a prediction

        Args:
            input_tensor: the network's input

        Returns:
            torch.Tensor: the networks output given the :attr`input_tensor`
        """
        depth = self._hparams.get('depth', 3)

        intermediate_outputs = []

        # Compute all the encoder blocks' outputs
        for idx in range(depth):
            intermed = getattr(self, 'down_block_%d' % idx)(input_tensor)
            if idx < depth - 1:
                # store intermediate values for usage in decoder
                intermediate_outputs.append(intermed)
                input_tensor = getattr(self, 'max_pool')(intermed)
            else:
                input_tensor = intermed

        # Compute all the decoder blocks' outputs
        for idx in range(depth-1):
            input_tensor = getattr(self, 'up_sample')(input_tensor)

            # use intermediate values from encoder
            from_down = intermediate_outputs.pop(-1)
            intermed = torch.cat([input_tensor, from_down], dim=1)
            input_tensor = getattr(self, 'up_block_%d' % idx)(intermed)

        return getattr(self, 'final_conv')(input_tensor)
