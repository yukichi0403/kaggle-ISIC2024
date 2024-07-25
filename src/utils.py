import random
import numpy as np
import torch
import cv2
from timm.scheduler import CosineLRScheduler
from sklearn.metrics import roc_curve, roc_auc_score, auc
from torch.optim import lr_scheduler



def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_scheduler(args, optimizer, steps_by_epoch):
    if args.scheduler == "cosine_with_warmup":
        if args.epochs < 10:
            warmup_t = 1
        else:
            warmup_t = args.epochs // 10
        warmup_lr_init = args.lr * 0.1  # ウォームアップ初期学習率の設定
        warmup_prefix = True
        scheduler = timm.scheduler.CosineLRScheduler(
                                                        optimizer,
                                                        t_initial=args.epochs,
                                                        lr_min=args.lr / 100,
                                                        t_in_epochs=True,
                                                        warmup_t=warmup_t,
                                                        warmup_lr_init=warmup_lr_init,
                                                        warmup_prefix=warmup_prefix,
                                                    )
    elif args.scheduler == "cosine":
        warmup_t=0
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=steps_by_epoch*args.epochs, 
                                                   eta_min=args.lr / 100)
    
    print(f"Training steps: {steps_by_epoch * args.epochs}, warmup_steps: {steps_by_epoch * warmup_t}")
    
    return scheduler


#metric
def calculate_pauc_and_auc(y_true, y_scores, tpr_threshold=0.8):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)
    
    # Create a mask for TPR values above the threshold
    mask = tpr >= tpr_threshold
    
    # Apply the mask to fpr and tpr
    fpr_above_threshold = fpr[mask]
    tpr_above_threshold = tpr[mask]

    # Check if there are at least 2 points to compute AUC
    if len(fpr_above_threshold) < 2:
        return 0.0, auc_score, fpr, tpr, fpr_above_threshold, tpr_above_threshold
    
    # Calculate the partial AUC
    partial_auc = auc(fpr_above_threshold, tpr_above_threshold)
    
    # Normalize the partial AUC
    pauc = partial_auc * (1 - tpr_threshold)
    
    return pauc, auc_score