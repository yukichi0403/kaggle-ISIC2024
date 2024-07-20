import random
import numpy as np
import torch
import cv2
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import roc_curve, roc_auc_score, auc



def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_scheduler(args, optimizer, train_size):
    training_steps = -(-train_size // args.batch_size) * args.epochs
    if args.epochs < 10:
        num_warmup_steps = -(-train_size // args.batch_size)
    else:
        num_warmup_steps = int(training_steps * 0.1)
    print(f"Training steps: {training_steps}, num_warmup_steps: {num_warmup_steps}")
    scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps = training_steps,
            num_warmup_steps = num_warmup_steps
        )
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