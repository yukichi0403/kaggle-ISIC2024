import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from torch.utils.data import Dataset
import albumentations as A
import h5py
from PIL import Image
from io import BytesIO
import pandas as pd


class SkinCancerDataset(Dataset):
    def __init__(self, split: str, df: pd.DataFrame, data_dir: str, augs=None) -> None:

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1  # クラス数を設定
        self.augs = augs
        self.df = df
        self.isic_ids = self.df['isic_id'].values

        if split in ["train", "val"]:
            self.fp_hdf = h5py.File(os.path.join(data_dir, "train-image.hdf5"), mode="r")
            self.targets = self.df['target'].values
        else:
            self.fp_hdf = h5py.File(os.path.join(data_dir, "test-image.hdf5"), mode="r")

    def __len__(self) -> int:
        return len(self.isic_ids)

    def __getitem__(self, i):
        isic_id = self.isic_ids[i]
        X = np.array(Image.open(BytesIO(self.fp_hdf[isic_id][()])))
        X = self.__standarize(X)
        if self.augs:
            X = self.__random_transform(X, self.augs)
        if self.split in ["train", "val"]:
            return X, self.targets[i]
        else:
            return X
    
    def __standarize(self, img):
        img = img.astype(np.float32) / 255.0
        ep = 1e-6
        m = np.nanmean(img.flatten())
        s = np.nanstd(img.flatten())
        img = (img - m) / (s + ep)
        img = np.nan_to_num(img, nan=0.0)
        return img

    def __random_transform(self, img, transform):
        assert isinstance(transform, A.Compose), "Transform must be an instance of albumentations.Compose"
        img = transform(image=img)['image']
        return torch.tensor(img)

    @property
    def num_channels(self) -> int:
        sample_image = np.array(Image.open(BytesIO(self.fp_hdf[self.isic_ids[0]][()])))
        return sample_image.shape[2]

    @property
    def seq_len(self) -> int:
        sample_image = np.array(Image.open(BytesIO(self.fp_hdf[self.isic_ids[0]][()])))
        return sample_image.shape[0]