from __future__ import annotations

from abc import ABC, abstractmethod as am

import numpy as np
import cv2

class ImageSource(ABC):
    delay: int

    @am
    def get_next(self) -> np.ndarray | None:
        return NotImplemented

class FileImageSource(ImageSource):
    def __init__(self, image_path) -> None:
            self.img = cv2.imread(image_path) # not lazy
            self.delay = 30_000

    def get_next(self):
        try: # dont use an extra variable
            return self.img
        finally:
            self.img = None
