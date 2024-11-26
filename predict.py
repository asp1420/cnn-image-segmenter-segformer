import cv2
import numpy as np
import matplotlib.pyplot as plt

from jsonargparse import CLI
from dataclasses import dataclass
from modules.segmodule import SegModule
from augmentation import valid_transform
from transformers import SegformerImageProcessor
from albumentations import Compose
from numpy import ndarray
from torch import Tensor
from typing import Tuple


@dataclass
class Predictor:

    def predict(self, modelname: str, imgname: str) -> None:
        """
        MIT-bx predictor. This application performs the inference to an input and plots the result.
        Args:
            modelname: The MIT-bx model weights file (ckpt).
            imgname: Name of the image (input) to infer (npy).
        """
        transform = valid_transform()
        processor = SegformerImageProcessor()
        model = SegModule.load_from_checkpoint(modelname).cuda()
        model.eval()
        tensor = self._get_tensor(imgname=imgname, transform=transform, processor=processor)
        size = tuple(tensor.shape[2:])
        logits = self._inference(tensor=tensor, model=model)
        result = self._postprocess(logits=logits, size=size)
        plt.imshow(result)
        plt.show()

    def _get_tensor(
            self,
            imgname: str,
            transform: Compose,
            processor: SegformerImageProcessor
        ) -> Tensor:
        image = np.load(imgname)
        image = transform(image=image)['image']
        tensor = processor(image, return_tensors='pt').pixel_values.cuda()
        return tensor

    def _inference(self, tensor: Tensor, model: SegModule) -> ndarray:
        _, logits = model(pixel_values=tensor, labels=None)
        logits = logits.detach().cpu().numpy()
        logits = np.squeeze(logits)
        return logits

    def _postprocess(self, logits: ndarray, size: Tuple[int, int]) -> ndarray:
        kernel = np.ones((3, 3), np.uint8)
        logits = self._sigmoid(logits)
        logits = np.where(logits > 0.5, 1, 0).astype(np.uint8)
        logits = cv2.erode(logits, kernel, iterations=1)
        logits = cv2.dilate(logits, kernel, iterations=1)
        logits = cv2.resize(logits, size)
        return logits

    def _sigmoid(self, value: ndarray) -> ndarray:
        value = 1 / (1 + np.exp(-value))
        return value


if __name__ == '__main__':
    CLI(Predictor)
