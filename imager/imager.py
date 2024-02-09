import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Image(object):
    def __init__(self, image: np.ndarray):
        self.image = image

    @classmethod
    def from_array(cls, array):
        return cls(image=array)

    @classmethod
    def from_file(cls, filename):
        if not os.path.exists(filename):
            raise ValueError(f'The image file \"{filename}\" does not exist.')

        raw_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        return cls(image=rgb)

    @property
    def shape(self) -> tuple:
        return self.image.shape

    @staticmethod
    def colormap(image) -> np.ndarray:
        normalized_image = image.astype(np.uint8)
        return cv2.applyColorMap(normalized_image, cv2.COLORMAP_HOT)

    def display(self, recolor: bool = False):
        print("Displaying rendered image...")
        image = self.colormap(self.image) if recolor else self.image
        plt.axis("off")
        plt.imshow(image)
        plt.show()

    def save(self, filename: str):
        cv2.imwrite(filename, self.image)

    def show_landmarks(self, landmarks):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        for landmark in landmarks:
            cv2.circle(image, (int(landmark[0]), int(landmark[1])), 1, (255, 0, 0), 5)

        cv2.imshow('Pose estimation', image)
        cv2.waitKey(0)
