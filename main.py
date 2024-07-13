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
import albumentations as A

import shutil


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    transform = A.Compose([A.NoOp()])
    
    train_set = ThingsMEGDataset("train", args.data_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    model = CustomModel(
        train_set.num_classes, train_set.seq_len, train_set.num_channels
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    no_improve_epochs = 0
    mertic = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        current_lr = optimizer.param_groups[0]["lr"]
        
        model.train()
        for X, y in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)

            y_pred = model(X)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = mertic(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(mertic(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f} | lr: {current_lr:.7f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc), "lr": current_lr})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            model_path = os.path.join(logdir, "model_best.pt")
            torch.save(model.state_dict(), model_path)
            max_val_acc = np.mean(val_acc)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs > args.early_stopping_rounds:
                cprint("Early stopping.", "cyan")
                break
  
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

    if args.local_dir:
        # Localにコピー
        local_dir = os.path.join(args.local_dir, f"{args.expname}_{args.ver}")
        os.makedirs(local_dir, exist_ok=True)

        submission_path = os.path.join(logdir, "submission.npy")
        if os.path.exists(submission_path):
            shutil.copy(submission_path, local_dir)
            print(f'Submission file saved to Local: {local_dir}')

        model_path = os.path.join(logdir, f"model_best.pt")
        if os.path.exists(model_path):
            shutil.copy(model_path, local_dir)
            print(f'Model saved to Local: {local_dir}')


if __name__ == "__main__":
    run()