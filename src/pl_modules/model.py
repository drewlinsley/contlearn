from typing import Any, Dict, List, Sequence, Tuple, Union
import os
import hydra
import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from torch.optim import Optimizer
from importlib import import_module

import numpy as np
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from monai import losses as monai_losses

from src.common.utils import iterate_elements_in_batches, render_images

from src.pl_modules import UNet3D, unet
from src.pl_modules import losses
from src.pl_modules import dice_loss


class MyModel(pl.LightningModule):
    def __init__(
            self,
            cfg: DictConfig,
            name,
            loss_weights,
            ckpt,
            loss,
            metric,
            in_channels,
            out_channels,
            force_2d,
            plot_argmax,
            *args,
            **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Dont save the cfg on tpus. Bug.
        self.cfg = cfg
        if hasattr(self.cfg.train.pl_trainer, "gpus"):
            self.save_hyperparameters(cfg)
        else:
            self.save_hyperparameters()
        self.name = name
        p, m = loss.rsplit('.', 1)
        mod = import_module(p)
        self.loss = getattr(mod, m)  # getattr(losses, loss)

        # self.loss = monai_losses.DiceLoss
        # self.loss = dice_loss.SoftDiceLoss()  # getattr(mod, m)  # getattr(losses, loss)
        if loss_weights:
            self.loss_weights = torch.tensor(loss_weights)
        else:
            self.loss_weights = None

        if force_2d:
            self.net = unet.UNet(
                in_channels=self.cfg.model.in_channels,
                out_channels=self.cfg.model.out_channels)
        else:
            model = getattr(UNet3D, self.name)
            self.net = model(
                in_channels=self.cfg.model.in_channels,
                out_channels=self.cfg.model.out_channels)
        self.ckpt = ckpt

        # metric_mod = import_module(torchmetrics)
        # metric = getattr(metric_mod, self.cfg.metric.name)()
        p, m = metric.rsplit('.', 1)
        mod = import_module(p)
        metric = getattr(mod, m)(mdmc_average="samplewise")
        self.train_accuracy = metric.clone().to(self.device)
        self.val_accuracy = metric.clone().to(self.device)
        self.test_accuracy = metric.clone().to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def step(self, x, y) -> Dict[str, torch.Tensor]:
        # print("step")
        logits = self(x)

        if isinstance(logits, dict):
            penalty = logits["penalty"]
            logits = logits["logits"]
            loss = self.loss(logits, y, self.loss_weights)
            # loss = self.loss.forward(logits, y, self.loss_weights)
            loss = loss + penalty
        else:
            # loss = self.loss(logits, y, self.loss_weights)
            loss = self.loss(logits, y)
        return {
            "logits": logits.detach(),
            "loss": loss,
            "y": y.detach(),
            "x": x.detach()
        }

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        out = self.step(x, y)
        # opt = self.optimizers()
        # opt.zero_grad()
        # self.manual_backward(out["loss"])
        # opt.step()
        return out

    def training_step_end(self, out):
        # self.train_accuracy(torch.softmax(out["logits"], dim=1), out["y"])
        self.log_dict(
            {
                # "train_acc": self.train_accuracy,
                "train_loss": out["loss"].mean(),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return out["loss"].mean()

    def validation_step(
            self,
            batch: Any,
            batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        out = self.step(x, y)

        return out

    def validation_step_end(self, out):
        # self.val_accuracy(torch.softmax(out["logits"], dim=1), out["y"])
        self.log_dict(
            {
                # "val_acc": self.val_accuracy,
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
        with torch.no_grad():
            x, y = batch
            out = self.step(x, y)
        return out

    def test_step_end(self, out):
        self.test_accuracy(torch.softmax(out["logits"], dim=1), out["y"])
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
            mid = output_element["image"].shape[-3] // 2
            if len(output_element["image"]) == 2:
                input_img = output_element["image"][0, mid, ...].unsqueeze(dim=0)
                input_seg = output_element["image"][1, mid, ...].unsqueeze(dim=0)
                if self.cfg.model.plot_argmax:
                    gt = output_element["y_true"][:, mid, ...].argmax(dim=0).unsqueeze(dim=0)
                    output_seg = output_element["logits"][:, mid, ...].argmax(dim=0).unsqueeze(dim=0)
                else:
                    gt = output_element["y_true"][:, mid]
                    output_seg = output_element["logits"][:, mid]
                rendered_image = render_images([input_img, input_seg, gt, output_seg], autoshow=False, nrow=4)
                caption = f"image____mem____GT____output"  # y_pred: {output_element['logits'].argmax()}  [gt: {output_element['y_true']}]"
            else:
                input_img = output_element["image"][0, mid, ...].unsqueeze(dim=0)
                if self.cfg.model.plot_argmax:
                    gt = output_element["y_true"][:, mid, ...].argmax(dim=0).unsqueeze(dim=0)
                    output_seg = output_element["logits"][:, mid, ...].argmax(dim=0).unsqueeze(dim=0)
                else:
                    gt = output_element["y_true"][:, mid]
                    output_seg = output_element["logits"][:, mid]
                rendered_image = render_images([input_img, gt, output_seg], autoshow=False, nrow=4)
                caption = f"image____GT____output"  # y_pred: {output_element['logits'].argmax()}  [gt: {output_element['y_true']}]"
            images.append(
                wandb.Image(
                    rendered_image,
                    caption=caption,
                )
            )
        # if hasattr(self.cfg.train.pl_trainer, "tpu_cores") and self.cfg.train.pl_trainer.tpu_cores > 1:
        #     pass
        # else:
        #     # self.logger.experiment.log({"Validation Images": images}, step=self.global_step)
        self.logger.experiment.log({"Validation Images": images})  # , step=self.global_step)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        # batch_size = self.cfg.data.datamodule.batch_size.test

        # images = []
        # images_feat_viz = []

        # integrated_gradients = IntegratedGradients(self.forward)
        # noise_tunnel = NoiseTunnel(integrated_gradients)
        
        # self.logger.experiment.log({"Test Images": images}, step=self.global_step)
        outputs[0]['logits'].detach().cpu().numpy().squeeze().max()
        from matplotlib import pyplot as plt
        plt.subplot(131);plt.imshow(outputs[0]['image'].detach().cpu().numpy().squeeze()[0, 31]);plt.subplot(132);plt.imshow(outputs[0]['image'].detach().cpu().numpy().squeeze()[1, 31]);plt.subplot(133);plt.imshow(outputs[0]['logits'].detach().cpu().numpy().squeeze()[31].astype(np.float32));plt.show()
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
        # self.logger.experiment.log({"Test Images": images}, step=self.global_step)
        # self.logger.experiment.log({"Test Images Feature Viz": images_feat_viz}, step=self.global_step)
        self.logger.experiment.log({"Test Images": images})
        self.logger.experiment.log({"Test Images Feature Viz": images_feat_viz})

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

