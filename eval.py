####################
# import libraries
####################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import torch
from typing import Tuple
from termcolor import cprint
from torch.utils.data.dataset import Dataset 
import torch.nn.functional as F
import torch
import albumentations as A
import h5py
from PIL import Image
from io import BytesIO
import random
import cv2
import sys
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
import timm
from sklearn.metrics import roc_curve, auc, roc_auc_score

import warnings
# 全ての警告を無視
warnings.filterwarnings("ignore")



####################
# functions
####################
def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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



####################
# Dataset
####################
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
        elif self.split=="train" or self.split=="val":
            df = self.df
            targets = self.targets
        else:
            df = self.df
        
        isic_id = df.iloc[i]['isic_id']
        if self.split != "test":archive = df.iloc[i]['archive']
            
        if self.use_JPEG:
            if archive:img_path = os.path.join(self.archive_image_dir, isic_id+".jpg")
            else:img_path = os.path.join(self.image_dir, isic_id+".jpg")
            X = cv2.imread(img_path)
            X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
        else: #testの場合は全てhdf
            if self.split != "test" and archive:
                X = np.array(Image.open(BytesIO(self.fp_hdf_archive[isic_id][()])))
            else:
                X = np.array(Image.open(BytesIO(self.fp_hdf[isic_id][()])))
        

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



####################
# Model
####################
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'

class CustomModel(nn.Module):
    def __init__(self, 
                 args,
                 training: bool = True, 
                 ):
        super(CustomModel, self).__init__()
        self.aux_loss_ratio = args.aux_loss_ratio
        self.training = training
        self.encoder = timm.create_model(args.model_name, pretrained=self.training,
                                          drop_path_rate=args.drop_path_rate)
        self.classifier_in_features = self.encoder.classifier.in_features
        self.encoder.classifier = nn.Identity()
        self.encoder.global_pool = nn.Identity()
        self.GeM = GeM(p=args.gem_p)
        self.flatten = nn.Flatten()
        self.dropout_main = nn.ModuleList([
            nn.Dropout(args.dropout) for _ in range(5)
        ]) #droupout augmentation
        self.linear_main = nn.Linear(self.classifier_in_features, args.num_classes)

        if args.aux_loss_ratio is not None:
            self.decoder_aux = nn.Flatten()
            self.dropout_aux = nn.ModuleList([
                nn.Dropout(args.dropout) for _ in range(5)
            ]) #droupout augmentation
            self.linear_aux = nn.Linear(self.encoder.num_features, 4)

    def forward(self, images):
        out = self.encoder(images)
        out = self.GeM(out)
        out = self.flatten(out)
        if self.training:
            main_out=0
            for i in range(len(self.dropout_main)):
                main_out += self.linear_main(self.dropout_main[i](out))
            main_out = main_out/len(self.dropout_main)
            if self.aux_loss_ratio is not None:
                out_aux=0
                for i in range(len(self.dropout_aux)):
                    out_aux += self.linear_aux(self.dropout_aux[i](out))
                out_aux = out_aux/len(self.dropout_aux)
                return main_out, out_aux
        else:
            main_out = self.linear_main(out)
        
        return main_out



####################
# inference
####################
@torch.no_grad()
def inference(args, model_dir):
    
    test_df = pd.read_csv(os.path.join(args.data_dir, "test-metadata.csv"))
    
    test_transforms = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1),
                    ToTensorV2()
    ])
    
    # ------------------
    #    Dataloader
    # ------------------    
    test_set = SkinCancerDataset(args, "test", test_df, 
                                  test_transforms, args.remove_hair_thresh)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    folds_preds = []
    for fold in range(args.num_splits):
        if args.test and fold != 0:  # Adjusted fold index check
            print(f"Test mode. Skipping fold{fold+1}")
            continue
        print(f"Fold {fold+1}/{args.num_splits}")

        model = CustomModel(args, training=False).to(args.device)
        weight = torch.load(os.path.join(model_dir, f"model_best_fold{fold+1}.pt"), map_location=torch.device(args.device))
        model.load_state_dict(weight)
        # Removed the second model.to(args.device)

        model.eval()
        preds = [] 
        for X in tqdm(test_loader, desc="Validation"):
            X = X.to(args.device)
            pred = model(X).sigmoid().detach().cpu()
            preds.append(pred)
        
        preds = torch.cat(preds, dim=0).numpy()
        print(f"fold{fold+1}'preds: {preds}")
        folds_preds.append(preds)
    
    final_preds = torch.mean(torch.tensor(folds_preds), dim=0)
    cprint(f"Submission {final_preds.shape}", "cyan")
    
    return final_preds


# args
class args():
    data_dir= "/kaggle/input/isic-2024-challenge/"
    data_dir_archive="/kaggle/input/all-isic-data-gkf-256"
    image_dir= "/content/drive/MyDrive/24_7_ISIC2024/Input/isic-2024-challenge/train-image/image"
    archive_image_dir="/kaggle/input/all-isic-data-20240629/images"
    train_df_dir= "/content/drive/MyDrive/24_7_ISIC2024/Input/isic-2024-challenge/train-metadata-gkffold.csv"
    local_dir= "/content/drive/MyDrive/24_7_ISIC2024/Output"
    COLAB= False
    test= False

    model_name= "tf_efficientnet_b0.ns_jft_in1k"
    aux_loss_ratio=None

    batch_size= 64
    epochs= 10
    lr= 1e-4
    early_stopping_rounds= 3
    num_splits= 5
    do_mixup= False

    device= "cuda:0"
    num_workers= 4
    seed= 1234
    use_wandb= False
    
    num_classes=1 
    do_mixup=False 
    expname="EffNet0b_Sampling0.01_Balanceddf_Gemp1_img384_dropout0.3"
    sampling_rate=0.5
    gem_p=3 
    dropout=0.
    drop_path_rate=0.
    img_size=384 
    scheduler="cosine"
    weight_decay=1e-6
    
    remove_hair_thresh=0
    use_JPEG=False


# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
BEST_WEIGHT = sys.argv[1]
print(f"BEST_WEIGHT = {BEST_WEIGHT}")

# inference
preds = inference(args, BEST_WEIGHT)

df_sub = pd.read_csv("/kaggle/input/isic-2024-challenge/sample_submission.csv")
df_sub["target"] = preds
df_sub

df_sub.to_csv("submission.csv", index=False)