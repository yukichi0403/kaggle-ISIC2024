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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from albumentations.pytorch import ToTensorV2


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
        self.use_metadata_num = args.use_metadata_num

        if self.use_metadata_num:
            self.metadata_scaler = StandardScaler()
            if split != "test":
                self.metadata = self.metadata_scaler.fit_transform(df.iloc[:, 4:4+self.use_metadata_num].values.astype(np.float32))
            else:
                self.metadata = self.metadata_scaler.fit_transform(df.iloc[:, 1:1+self.use_metadata_num].values.astype(np.float32))

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
        self.hdf_dir_archive = f"image_384sq.hdf5"
        if split in ["train", "val"]:            
            self.fp_hdf = h5py.File(os.path.join(args.data_dir, self.hdf_dir), mode="r")
            self.fp_hdf_archive = h5py.File(os.path.join(args.data_dir_archive, self.hdf_dir_archive), mode="r")
            self.targets = self.df['target'].values
        else:
            self.fp_hdf = h5py.File(os.path.join(args.data_dir, "test-image.hdf5"), mode="r")

        self.aux_loss_features = args.aux_loss_features

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
        elif self.split=="train" or self.split=="val":
            df = self.df
            targets = self.targets
        else:
            df = self.df
        
        isic_id = df.iloc[i]['isic_id']
        if self.split != "test":archive = df.iloc[i]['archive']
            
        if self.split != "test" and archive:
            X = np.array(Image.open(BytesIO(self.fp_hdf_archive[isic_id][()])))
        else:
            X = np.array(Image.open(BytesIO(self.fp_hdf[isic_id][()])))
        

        if self.remove_hair_thresh > 0:            
            X = self.__remove_hair(X)
        if self.augs:
            X = self.__random_transform(X, self.augs)
        if self.aux_loss_features:
            aux_features = {aux_feature: df.iloc[i][aux_feature] for aux_feature in self.aux_loss_features}

        if self.split in ["train", "val"]:
            if self.aux_loss_features:
                if self.use_metadata_num:
                    # isic_id, target, fold, metadataの順番なので3:にする
                    return X, targets[i], aux_features, self.metadata[i]
                else:
                    return X, targets[i], aux_features
            else:
                if self.use_metadata_num:
                    return X, targets[i], self.metadata[i]
                else:
                    return X, targets[i]
        else:
            if self.use_metadata_num:
                return X, self.metadata[i]
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
    

def get_transforms(image_size, augmentation_strength='strong'):
    if augmentation_strength not in ['strong', 'weak']:
        raise ValueError("augmentation_strength must be either 'strong' or 'weak'")

    # 共通の変換
    common_transforms = [
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ]

    if augmentation_strength == 'strong':
        transforms_train = A.Compose([
            *common_transforms[:3],  # RandomResizedCrop, HorizontalFlip, VerticalFlip
            A.Transpose(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),
            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
            ], p=0.7),
            A.CLAHE(clip_limit=4.0, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            A.CoarseDropout(max_holes=8, max_height=int(image_size * 0.375), max_width=int(image_size * 0.375), 
                            min_holes=5, min_height=int(image_size * 0.09), min_width=int(image_size * 0.09), 
                            fill_value=0, p=0.7),
            *common_transforms[-2:]  # Normalize, ToTensorV2
        ])
    else:  # weak augmentation
        transforms_train = A.Compose([
            *common_transforms[:3],  # RandomResizedCrop, HorizontalFlip, VerticalFlip
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
                A.GaussNoise(var_limit=(5.0, 20.0)),
            ], p=0.3),
            A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=7, val_shift_limit=3, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=60, border_mode=0, p=0.5),
            A.CoarseDropout(max_holes=4, max_height=int(image_size * 0.1), max_width=int(image_size * 0.1), 
                            min_holes=1, min_height=int(image_size * 0.05), min_width=int(image_size * 0.05), 
                            fill_value=0, p=0.5),
            *common_transforms[-2:]  # Normalize, ToTensorV2
        ])

    transforms_val = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2()
    ])

    return transforms_train, transforms_val


def preprocess_numerical_cols(X):
    # 元のデータフレームの列名を保存
    original_columns = X.columns
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # 全てNaNの列を0で埋める
    all_nan_columns = X.columns[X.isna().all()].tolist()
    X[all_nan_columns] = 0
    
    # 数値列のみに対してSimpleImputerを適用
    num_columns = X.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='median')
    try:
        X_imputed = pd.DataFrame(imputer.fit_transform(X[num_columns]), columns=num_columns, index=X.index)
    except ValueError as e:
        cprint(f"Error during imputation: {e}", "red")
        print("Falling back to filling NaN with 0 for all numeric columns")
        X_imputed = X[num_columns].fillna(0)
    
    # 非数値列がある場合は、それらを保持
    non_num_columns = [col for col in original_columns if col not in num_columns]
    X_processed = pd.concat([X_imputed, X[non_num_columns]], axis=1)
    
    # まだNaNが残っている場合は0で埋める
    X_processed = X_processed.fillna(0)

    return X_processed[original_columns]