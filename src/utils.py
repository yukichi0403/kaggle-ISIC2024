import random
import numpy as np
import torch
import cv2
from timm.scheduler import CosineLRScheduler
from sklearn.metrics import roc_curve, roc_auc_score, auc
from torch.optim import lr_scheduler
import pandas as pd


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
def comp_score(solution: pd.DataFrame, submission: pd.DataFrame, min_tpr: float=0.80):
    v_gt = abs(np.asarray(solution.values) - 1)
    v_pred = np.array([1.0 - x for x in submission.values])
    max_fpr = abs(1 - min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
    partial_auc = 0.5 * max_fpr ** 2 + (max_fpr - 0.5 * max_fpr ** 2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc, partial_auc_scaled


def custom_metric(y_true, y_hat):
    min_tpr = 0.80
    max_fpr = abs(1 - min_tpr)
    
    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_hat])
    
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    
    return partial_auc

def custom_metric_for_cvscore(estimator, X, y_true):
    y_hat = estimator.predict_proba(X)[:, 1]
    min_tpr = 0.80
    max_fpr = abs(1 - min_tpr)
    
    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_hat])
    
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    
    return partial_auc