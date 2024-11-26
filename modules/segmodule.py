import torch

from transformers import SegformerForSemanticSegmentation, BatchFeature
from lightning import LightningModule
from torch.optim.lr_scheduler import StepLR
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class SegModule(LightningModule):

    def __init__(
            self,
            learning_rate: float,
            num_classes: int
        ) -> None:
        super(SegModule, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path="nvidia/mit-b3",
            num_labels=self.num_classes
        )
        self.train_losses = list()
        self.val_losses = list()

    def forward(self, pixel_values: Tensor=None, labels: Tensor=None) -> tuple[Tensor, Tensor]:
        output = self.model(pixel_values=pixel_values, labels=labels)
        return output.loss, output.logits

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRScheduler]]:
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch: BatchFeature, batch_idx: int) -> Tensor:
        pixel_values, labels = batch["pixel_values"], batch["labels"]
        loss, _ = self(pixel_values=pixel_values, labels=labels)
        metrics = {'train_loss': loss.detach()}
        self.train_losses.append(loss.detach())
        self.log_dict(metrics, prog_bar=True)
        return loss

    def validation_step(self, batch: BatchFeature, batch_idx: int) -> None:
        pixel_values, labels = batch["pixel_values"], batch["labels"]
        loss, _ = self(pixel_values=pixel_values, labels=labels)
        metrics = {'val_loss': loss.detach()}
        self.val_losses.append(loss.detach())
        self.log_dict(metrics)

    def on_train_epoch_end(self) -> None:
        mean_loss = torch.stack(self.train_losses).mean()
        metrics = {'train_loss_epoch': mean_loss}
        self.log_dict(metrics)
        self.train_losses.clear()

    def on_validation_epoch_end(self) -> None:
        mean_loss = torch.stack(self.val_losses).mean()
        metrics = {'val_loss_epoch': mean_loss}
        self.log_dict(metrics)
        self.val_losses.clear()
