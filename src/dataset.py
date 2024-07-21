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
import cv2


def remove_hair(image, threshold):
    # 画像をグレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ブラックハットフィルタを適用
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)) #9x9の矩形カーネルを作成
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel) #ブラックハット変換を適用
    
    #ブラックハットイメージを二値化。閾値10を超えるピクセルは255に、その他のピクセルは0に設定
    _, thresh = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY) 
    
    # 元の画像をインペイントして髪の毛を除去
    inpainted_image = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
    
    return inpainted_image



class SkinCancerDataset(Dataset):
    def __init__(self, args, split: str, df: pd.DataFrame, augs=None, remove_hair_thresh:int =15,) -> None:

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        if split != "test":
            self.dr = df["target"].values
        self.num_classes = 2
        self.augs = augs
        self.df = df
        self.isic_ids = self.df['isic_id'].values
        self.remove_hair_thresh = remove_hair_thresh

        self.hdf_dir = "image_256sq.hdf5" if "all-isic-data" in args.data_dir else "train-image.hdf5"
        if split in ["train", "val"]:
            self.fp_hdf = h5py.File(os.path.join(args.data_dir, self.hdf_dir), mode="r")
            self.targets = self.df['target'].values
        else:
            self.fp_hdf = h5py.File(os.path.join(args.data_dir, "test-image.hdf5"), mode="r")

    def __len__(self) -> int:
        return len(self.isic_ids)

    def __getitem__(self, i):
        isic_id = self.isic_ids[i]
        X = np.array(Image.open(BytesIO(self.fp_hdf[isic_id][()])))
        if self.remove_hair_thresh > 0:            
            X = self.__remove_hair(X)
        if self.augs:
            X = self.__random_transform(X, self.augs)
        if self.split in ["train", "val"]:
            return X, self.targets[i]
        else:
            return X
    
    def __remove_hair(self, img):
        return remove_hair(img, self.remove_hair_thresh)

    def __random_transform(self, img, transform):
        assert isinstance(transform, A.Compose), "Transform must be an instance of albumentations.Compose"
        img = transform(image=img)['image']
        return img

    @property
    def num_channels(self) -> int:
        sample_image = np.array(Image.open(BytesIO(self.fp_hdf[self.isic_ids[0]][()])))
        return sample_image.shape[2]

    @property
    def seq_len(self) -> int:
        sample_image = np.array(Image.open(BytesIO(self.fp_hdf[self.isic_ids[0]][()])))
        return sample_image.shape[0]