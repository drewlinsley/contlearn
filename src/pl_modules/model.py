from typing import Any, Dict, List, Sequence, Tuple, Union

import hydra
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from torch.optim import Optimizer

import numpy as np
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from src.common.utils import iterate_elements_in_batches, render_images

from . import BasePaperNet, hConvGRU, resnet18, FFhGRU


class MyModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig, name, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.name = name
        # self.automatic_optimization = False

        if cfg.data.datamodule.datasets.PF14.train.rand_color_invert_p > 0:
            input_size = 2
            output_size = 2
        else:
            input_size = 1
            output_size = 2

        if self.name == "baseline_paper":
            self.net = BasePaperNet()
        elif self.name == "hgru":
            self.net = hConvGRU(timesteps=8, filt_size=15)
        elif self.name == "resnet18":
            self.net = resnet18(pretrained=False)
        elif self.name == "int":
            # self.net = FFhGRU(25, timesteps=8, kernel_size=15, nl=F.softplus, input_size=input_size)  # softplus
            self.net = FFhGRU(32, timesteps=8, kernel_size=15, nl=F.softplus, input_size=input_size, output_size=output_size, l1=0.)  # softplus
        elif self.name == "int_crbp":
            # self.net = FFhGRU(25, timesteps=8, kernel_size=15, nl=F.softplus, input_size=input_size)  # softplus
            self.net = FFhGRU(32, timesteps=20, kernel_size=11, LCP=0.9, nl=F.softplus, input_size=input_size, output_size=output_size, grad_method="rbp")  # softplus
        else:
            raise NotImplementedError("Could not find network {}.".format(self.net))

        metric = torchmetrics.Accuracy()
        self.train_accuracy = metric.clone().cuda()
        self.val_accuracy = metric.clone().cuda()
        self.test_accuracy = metric.clone().cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def step(self, x, y) -> Dict[str, torch.Tensor]:
        logits = self(x)
        if isinstance(logits, dict):
            penalty = logits["penalty"]
            logits = logits["logits"]
            loss = F.cross_entropy(logits, y)
            loss = loss + penalty
        else:
            loss = F.cross_entropy(logits, y)
        return {"logits": logits, "loss": loss, "y": y, "x": x}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        out = self.step(x, y)
        # opt = self.optimizers()
        # opt.zero_grad()
        # self.manual_backward(out["loss"])
        # opt.step()
        return out

    def training_step_end(self, out):
        self.train_accuracy(torch.softmax(out["logits"], dim=-1), out["y"])
        self.log_dict(
            {
                "train_acc": self.train_accuracy,
                "train_loss": out["loss"].mean(),
            },
            on_step=True,
            on_epoch=False
        )
        return out["loss"].mean()

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        out = self.step(x, y)

        return out
    
    def validation_step_end(self, out):
        self.val_accuracy(torch.softmax(out["logits"], dim=-1), out["y"])
        self.log_dict(
            {
                "val_acc": self.val_accuracy,
                "val_loss": out["loss"].mean(),
            },
        )
        return {
            "image": out["x"],
            "y_true": out["y"],
            "logits": out["logits"],
            "val_loss": out["loss"].mean(),
        }

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        out = self.step(x, y)
        return out

    def test_step_end(self, out):
        self.test_accuracy(torch.softmax(out["logits"], dim=-1), out["y"])
        self.log_dict(
            {
                "test_acc": self.test_accuracy,
                "test_loss": out["loss"].mean(),
            },
        )
        return {
            "image": out["x"],
            "y_true": out["y"],
            "logits": out["logits"],
            "val_loss": out["loss"].mean(),
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        batch_size = self.cfg.data.datamodule.batch_size.val
        images = []
        for output_element in iterate_elements_in_batches(
            outputs, batch_size, self.cfg.logging.n_elements_to_log
        ):
            rendered_image = render_images(output_element["image"], autoshow=False)
            if rendered_image.shape[-1] > 1:
                # rendered_image[..., -1] -= 0.5
                # rendered_image = rendered_image.sum(-1)[..., None]
                rendered_image = np.concatenate((rendered_image, np.zeros_like(rendered_image)[..., 0][..., None]), -1)
            caption = f"y_pred: {output_element['logits'].argmax()}  [gt: {output_element['y_true']}]"
            images.append(
                wandb.Image(
                    rendered_image,
                    caption=caption,
                )
            )
        self.logger.experiment.log({"Validation Images": images}, step=self.global_step)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        batch_size = self.cfg.data.datamodule.batch_size.test

        images = []
        images_feat_viz = []

        # integrated_gradients = IntegratedGradients(self.forward)
        # noise_tunnel = NoiseTunnel(integrated_gradients)
        
        self.logger.experiment.log({"Test Images": images}, step=self.global_step)
        return  # Don't need this stuff below vvvv

        for output_element in iterate_elements_in_batches(
            outputs, batch_size, self.cfg.logging.n_elements_to_log
        ):

            #import pdb; pdb.set_trace()
            attributions_ig_nt = noise_tunnel.attribute(output_element["image"].unsqueeze(0), nt_samples=50,
                                                        nt_type='smoothgrad_sq', target=output_element["y_true"],
                                                        internal_batch_size=50)
            vz = viz.visualize_image_attr(np.transpose(attributions_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0)),
                                          np.transpose(output_element["image"].cpu().detach().numpy(), (1, 2, 0)),
                                          method='blended_heat_map', show_colorbar=True, sign='positive', outlier_perc=1)

            rendered_image = render_images(output_element["image"], autoshow=False)
            caption = f"y_pred: {output_element['logits'].argmax()}  [gt: {output_element['y_true']}]"
            images.append(
                wandb.Image(
                    rendered_image,
                    caption=caption,
                )
            )
            images_feat_viz.append(
                wandb.Image(
                    vz[0],
                    caption=caption,
                ))
            plt.close(vz[0])
        self.logger.experiment.log({"Test Images": images}, step=self.global_step)
        self.logger.experiment.log({"Test Images Feature Viz": images_feat_viz}, step=self.global_step)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.cfg.optim.optimizer, params=self.parameters()
        )
        
        if not self.cfg.optim.use_lr_scheduler:
            return opt

        scheduler = hydra.utils.instantiate(self.cfg.optim.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]

