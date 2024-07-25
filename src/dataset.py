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
import random


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
        self.augs = augs
        self.remove_hair_thresh = remove_hair_thresh
        self.df = df
        self.sampling_rate = args.sampling_rate
        self.use_JPEG = args.use_JPEG
        if split=="train" and self.sampling_rate is not None:
            print("Now sampling mode")
            self.df_positive = df[df["target"] == 1].reset_index()
            self.df_negative = df[df["target"] == 0].reset_index()
            self.targets_positive = self.df_positive['target'].values
            self.targets_negative = self.df_negative['target'].values
            self.isic_ids_positive = self.df_positive['isic_id'].values
            self.isic_ids_negative = self.df_negative['isic_id'].values
        else:
            self.isic_ids = self.df['isic_id'].values

        self.hdf_dir = "train-image.hdf5"
        self.hdf_dir_archive = f"image_{args.img_size}sq.hdf5"
        self.image_dir = args.image_dir
        self.archive_image_dir = args.archive_image_dir
        if split in ["train", "val"]:
            if not self.use_JPEG:
                self.fp_hdf = h5py.File(os.path.join(args.data_dir, self.hdf_dir), mode="r")
                self.fp_hdf_archive = h5py.File(os.path.join(args.data_dir_archive, self.hdf_dir_archive), mode="r")
            self.targets = self.df['target'].values
        else:
            self.fp_hdf = h5py.File(os.path.join(args.data_dir, "test-image.hdf5"), mode="r")

    def __len__(self) -> int:
        if self.split=="train" and self.sampling_rate is not None:
            return int(-(-len(self.df_positive) // self.sampling_rate))
        else:
            return len(self.isic_ids)

    def __getitem__(self, i):
        #sampling
        if self.split=="train" and self.sampling_rate is not None:
            if random.random() < self.sampling_rate:
                df = self.df_positive
                targets = self.targets_positive
            else:
                df = self.df_negative
                targets = self.targets_negative
            i = i % df.shape[0]
        else:
            df = self.df
            targets = self.targets
        
        isic_id = df.iloc[i]['isic_id']
        archive = df.iloc[i]['archive']
        if not self.use_JPEG:
            if archive:X = np.array(Image.open(BytesIO(self.fp_hdf_archive[isic_id][()])))
            else:X = np.array(Image.open(BytesIO(self.fp_hdf[isic_id][()])))
        else:
            if archive:
                img_path = os.path.join(self.archive_image_dir, isic_id+".jpg")
            else:
                img_path = os.path.join(self.image_dir, isic_id+".jpg")
            X = cv2.imread(img_path)
            X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)

        if self.remove_hair_thresh > 0:            
            X = self.__remove_hair(X)
        if self.augs:
            X = self.__random_transform(X, self.augs)
        if self.split in ["train", "val"]:
            return X, targets[i]
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
        if self.split in ["train", "val"]:sample_image = np.array(Image.open(BytesIO(self.fp_hdf[self.isic_ids[0]][()])))
        else:sample_image = np.array(Image.open(BytesIO(self.fp_hdf_archive[self.isic_ids[0]][()])))
        return sample_image.shape[2]

    @property
    def seq_len(self) -> int:
        if self.split in ["train", "val"]:sample_image = np.array(Image.open(BytesIO(self.fp_hdf[self.isic_ids[0]][()])))
        else:sample_image = np.array(Image.open(BytesIO(self.fp_hdf_archive[self.isic_ids[0]][()])))
        return sample_image.shape[0]