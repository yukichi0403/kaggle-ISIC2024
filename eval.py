import os, sys
from dataset import SkinCancerDataset
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import *
from src.models import *
from src.utils import *

import albumentations as A
from albumentations.pytorch import ToTensorV2

import shutil
import gc


@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config_colab")
def inference(args: DictConfig):
    set_seed(args.seed)

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
    test_set = SkinCancerDataset("test", args.data_dir,
                                  test_transforms,args.remove_hair_thresh
                                 )
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    folds_preds = []
    for fold in range(args.num_splits):
        if args.test and fold > 0:
            print(f"Test mode. Skipping fold{fold+1}")
            continue
        print(f"Fold {fold+1}/{args.num_splits}")
        model = CustomModel(
                            model_name=args.model_name,
                            num_classes=args.num_classes, # write the number of classes
                         pretrained=True, 
                         aux_loss_ratio= args.aux_loss_ratio, 
                         dropout_rate=args.dropout
        ).to(args.device)

        model.load_state_dict(torch.load(os.path.join(args.local_dir, f"model_best_fold{fold+1}.pt"), map_location=args.device))
        
        model.eval()
        preds = [] 
        for X, _ in tqdm(test_loader, desc="Validation"):        
            preds.append(model(X.to(args.device)).detach().cpu())
        
        preds = torch.cat(preds, dim=0).numpy()
        folds_preds.append(preds)
    
    final_preds = np.mean(folds_preds, axis=0)
    return final_preds
    cprint(f"Submission {final_preds.shape}", "cyan")

