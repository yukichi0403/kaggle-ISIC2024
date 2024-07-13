import os, sys
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

import shutil
import gc


@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config_colab")
def run(args: DictConfig):
    set_seed(args.seed)
    savedir = os.path.dirname(args.model_path)
    
    # ------------------
    #    Dataloader
    # ------------------    
    test_set = Dataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    folds_preds = []
    for fold in range(args.num_splits):
        print(f"Fold {fold+1}/{args.num_splits}")
        model = CustomModel(
            args.backbone,
            args.num_classes
        ).to(args.device)
        model.load_state_dict(torch.load(os.path.join(args.model_path, f"model_best_{fold}.pt"), map_location=args.device))
        
        model.eval()
        preds = [] 
        for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
            preds.append(model(X.to(args.device)).detach().cpu())
        
        preds = torch.cat(preds, dim=0).numpy()
        folds_preds.append(preds)
    
    final_preds = np.mean(folds_preds, axis=0)
    np.save(os.path.join(savedir, "submission.npy"), final_preds)
    cprint(f"Submission {final_preds.shape} saved at {savedir}", "cyan")

    

    if args.local_dir:
        # Localにコピー
        local_dir = os.path.join(args.local_dir, f"{args.expname}_{args.ver}")
        os.makedirs(local_dir, exist_ok=True)

        submission_path = os.path.join(savedir, "submission.npy")
        if os.path.exists(submission_path):
            shutil.copy(submission_path, local_dir)
            print(f'Submission file saved to Local: {local_dir}')

        model_path = os.path.join(savedir, f"model_best_{fold}.pt")
        if os.path.exists(model_path):
            shutil.copy(model_path, local_dir)
            print(f'Model saved to Local: {local_dir}')

if __name__ == "__main__":
    run()