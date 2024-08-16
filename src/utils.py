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

