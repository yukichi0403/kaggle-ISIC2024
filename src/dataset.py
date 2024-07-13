import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from torch.utils.data import Dataset
import albumentations as A


class Dataset(Dataset):
    def __init__(self, split: str, data_dir: str = "data", augs = None) -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = #write the number of classes
        self.X = torch.load(os.path.join(data_dir,)) #write the path to the data
        self.augs = augs

        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir,)) #write the path to the labels
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.__standarize(self.X[i])
        if self.augs:
            X = self.__random_transform(X, self.augs)
        if hasattr(self, "y"):
            return X, self.y[i]
        else:
            return X

    def __standarize(self, img):
        # Standarize per image
        ep = 1e-6
        m = np.nanmean(img.flatten())
        s = np.nanstd(img.flatten())
        img = (img-m)/(s+ep)
        img = np.nan_to_num(img, nan=0.0)

        return img

    def __random_transform(self, img, transform):
        assert isinstance(transform, A.Compose), "Transform must be an instance of albumentations.Compose"
        img = transform(image=img)['image']
        return torch.tensor(img)

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]