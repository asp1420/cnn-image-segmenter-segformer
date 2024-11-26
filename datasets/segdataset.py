import os
import numpy as np

from torch.utils.data import Dataset
from transformers import SegformerImageProcessor, BatchFeature
from albumentations import Compose


class SegDataset(Dataset):

    def __init__(
            self,
            root: str,
            image_processor: SegformerImageProcessor,
            transforms: Compose
        ) -> None:
        super().__init__()
        self.root = root
        self.image_processor = image_processor
        self.transforms = transforms
        self.images = os.listdir(path=os.path.join(root, 'masks'))

    def __getitem__(self, idx: int) -> BatchFeature:
        filename = self.images[idx]
        image = np.load(os.path.join(self.root, 'images', filename))
        mask = np.load(os.path.join(self.root, 'masks', filename))
        transform = self.transforms(image=image, mask=mask)
        image, mask = transform['image'], transform['mask']
        processor = self.image_processor(image, segmentation_maps=mask, return_tensors='pt')
        for k,v in processor.items():
            processor[k] = np.squeeze(v)
        return processor

    def __len__(self) -> int:
        return len(self.images)
