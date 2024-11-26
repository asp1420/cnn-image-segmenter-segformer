import os

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from augmentation import train_transform, valid_transform
from transformers import SegformerImageProcessor
from datasets.segdataset import SegDataset


class SegDataModule(LightningDataModule):

    def __init__(
            self,
            path: str,
            batch_size: int,
            workers: int,
            input_size: int=256,
        ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.path = path
        self.batch_size = batch_size
        self.workers = workers
        self.input_size = input_size

    def setup(self, stage: str) -> None:
        image_processor = SegformerImageProcessor(do_reduce_labels=False)
        train_dataset = SegDataset(
            root=os.path.join(self.path, 'train'), image_processor=image_processor,
            transforms=train_transform(size=self.input_size)
        )
        self.train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size,
            num_workers=self.workers, shuffle=True
        )
        val_dataset = SegDataset(
            root=os.path.join(self.path, 'validation'), image_processor=image_processor,
            transforms=valid_transform(size=self.input_size)
        )
        self.val_loader = DataLoader(
            dataset=val_dataset, batch_size=self.batch_size,
            num_workers=self.workers
        )

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_loader
