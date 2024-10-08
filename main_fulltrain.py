import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch
from typing import Tuple
from termcolor import cprint
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2
import hydra
from omegaconf import DictConfig
import shutil
import wandb

from sklearn.model_selection import GroupKFold, StratifiedKFold

from src.utils import *
from src.dataset import SkinCancerDataset
from src.combo_loader import get_combo_loader
from src.model import *
from src.feature_engeneering import *

import torch.nn as nn

# Classのサンプル数のバランスを取るための関数
def sampling(train, ratio=1):
    malignant_df = train[train['target'] == 1].copy()
    benign_df = train[train['target'] == 0].copy()

    # benignのデータをサンプリング
    benign_sample_df = benign_df.sample(int(len(malignant_df) * ratio), random_state=42)
    # malignantとサンプリングされたbenignのデータを結合
    balanced_df = pd.concat([malignant_df, benign_sample_df])
    # シャッフルしてインデックスをリセット
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df



def cross_entropy_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(input.log_softmax(dim=-1) * target).sum(dim=-1).mean()


def get_dataset_and_loader(loader_args, train_df, val_df, train_transforms, val_transforms, args):
    train_dataset = SkinCancerDataset(args, "train", train_df, 
                                  train_transforms, args.remove_hair_thresh
                                 )
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_args)

    if val_df is not None:
        val_dataset = SkinCancerDataset(args, "val", val_df, 
                                      val_transforms, args.remove_hair_thresh  
                                 )
        val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_args)  

    print(f"Train size: {len(train_dataset)}")
    
    return train_loader, None


def load_model(args, fold):
    # モデルのインスタンスを作成
    model = CustomModel(args, training=True).to(args.device)

    if args.pretrain_dir:
        # モデルの重みまたは辞書形式で保存されたファイルをロード
        file_path = os.path.join(args.pretrain_dir, f"model_best_fold{fold+1}.pt")
        checkpoint = torch.load(file_path, map_location=torch.device(args.device))
        
        # 保存ファイルが辞書形式か単なる重みファイルかを確認
        if isinstance(checkpoint, dict):
            # 辞書形式の場合、状態を取り出してモデルにロード
            model.load_state_dict(checkpoint['model_state_dict'])
            # 必要ならメタデータを取得
            epoch = checkpoint.get('epoch', None)
            val_score = checkpoint.get('val_score', None)
            print(f"Loaded model from fold {fold+1}, epoch: {epoch}, val_score: {val_score}")
        else:
            # 単なる重みファイルの場合
            model.load_state_dict(checkpoint)

    model.to(args.device)
    return model


def run_one_epoch(loader, model, optimizer, lr_scheduler, args, epoch, loss_func):
    losses, all_labels, all_preds = [], [], []

    train = optimizer is not None
    if train:
        print("Training mode")
        model.train()
        current_lr = optimizer.param_groups[0]["lr"]
        # エポックの最初にフリーズするパラメータを設定
        if args.freezing_epochs:
            if epoch < args.freezing_epochs:
                print(f"Epoch {epoch+1}: Freezing non-head layers.")
                for name, param in model.named_parameters():
                    if "linear" not in name:
                        param.requires_grad = False
            else:
                print(f"Epoch {epoch+1}: Unfreezing all layers.")
                for param in model.parameters():
                    param.requires_grad = True
    else: 
        print("Validation mode")
        model.eval()
    
    mode = "Train" if train else "Validation"
    for batch in tqdm(loader, desc=mode):

        if args.aux_loss_features is not None:
            inputs, labels, aux_features = batch[0].to(args.device), batch[1].squeeze().to(args.device), batch[2]

        elif train and args.do_mixup:
            lam = np.random.beta(a=args.do_mixup, b=1)

            inputs, labels = batch[0][0], batch[0][1]
            balanced_inputs, balanced_labels = batch[1][0], batch[1][1]

            inputs, labels = inputs.to(args.device), labels.squeeze().to(args.device)
            balanced_inputs, balanced_labels = balanced_inputs.to(args.device), balanced_labels.squeeze().to(args.device)

            inputs = (1 - lam) * inputs + lam * balanced_inputs
            mixed_labels = (1 - lam) * F.one_hot(labels, args.num_classes) + lam * F.one_hot(balanced_labels, args.num_classes)

            del balanced_inputs
            del balanced_labels
        else:
            inputs, labels = batch[0].to(args.device), batch[1].squeeze().to(args.device)
        
        if args.aux_loss_ratio and train:
            y_pred, aux_outs = model(inputs)
        else:
            y_pred = model(inputs)

        if (not args.do_mixup) & (isinstance(loss_func, torch.nn.BCEWithLogitsLoss) or isinstance(loss_func, torch.nn.BCELoss)):
            y_pred = y_pred.squeeze()
            labels = labels.float()  
            
        if train and args.do_mixup:
            loss = cross_entropy_loss(y_pred, mixed_labels)
        else:
            loss = loss_func(y_pred, labels)  # 修正: loss_funcを使用

        # 補助損失の計算
        if args.aux_loss_ratio and train:
            for i, (aux_out, aux_feature, aux_weight) in enumerate(zip(aux_outs, args.aux_loss_features, args.aux_loss_ratio)):
                if args.aux_task_reg[i]:  # 回帰タスクの場合
                    aux_labels = aux_features[aux_feature].to(args.device).float()
                    loss += nn.MSELoss()(aux_out.squeeze(), aux_labels) * aux_weight
                else:  # 分類タスクの場合
                    aux_labels = aux_features[aux_feature].to(args.device).long()
                    loss += nn.CrossEntropyLoss()(aux_out, aux_labels.long()) * aux_weight

        # 予測値とラベルを保存
        if isinstance(loss_func, nn.BCEWithLogitsLoss):
            all_preds.append(y_pred.sigmoid().cpu().detach().numpy())
        elif isinstance(loss_func, nn.BCELoss):
            all_preds.append(y_pred.cpu().detach().numpy())
        elif isinstance(loss_func, nn.CrossEntropyLoss):
            all_preds.append(y_pred[:,1].cpu().detach().numpy())
        all_labels.append(labels.cpu().numpy())

        if train:  # 訓練モードのみ
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
        
        losses.append(loss.item())
    
    # すべてのバッチの予測値とラベルを結合
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)


    # スコアとAUCを計算
    score, auc_score = comp_score(pd.DataFrame(all_labels), pd.DataFrame(all_preds))
    
    if train:
        print(f"Epoch {epoch+1}/{args.epochs} | {mode} loss: {np.mean(losses):.3f} | {mode} score: {score:.3f} | {mode} auc: {auc_score:.3f} | lr: {current_lr:.7f}")
    else:
        print(f"Epoch {epoch+1}/{args.epochs} | {mode} loss: {np.mean(losses):.3f} | {mode} score: {score:.3f} | {mode} auc: {auc_score:.3f}")
    
    return np.mean(losses), score, auc_score



@hydra.main(version_base=None, config_path="configs", config_name="config_fulltraining")
def run(args: DictConfig): 
    print(args)
    set_seed(args.seed)
    
    train = pd.read_csv(args.train_df_dir)
    train ,_ ,_ = feature_engineering(train)
    #train=sampling(train)
    
    logdir = "/kaggle/working/" if not args.COLAB else hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"logdir: {logdir}")
        
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    train_transform = A.Compose([  
                                    A.Resize(args.img_size, args.img_size),
                                    A.Flip(p=0.5),                      
                                    A.Transpose(p=0.5),

                                    A.RandomBrightnessContrast(
                                                            brightness_limit=(-0.1,0.1), 
                                                            contrast_limit=(-0.1, 0.1), 
                                                            p=0.5),
                                    
                                    A.OneOf([
                                        A.MotionBlur(blur_limit=5),
                                        A.MedianBlur(blur_limit=5),
                                        A.GaussianBlur(blur_limit=5),
                                    ], p=0.3),  # 確率を0.7から0.3に下げる

                                    A.CLAHE(clip_limit=2.0, p=0.5),  

                                    A.HueSaturationValue(
                                                            hue_shift_limit=0.2, 
                                                            sat_shift_limit=0.2, 
                                                            val_shift_limit=0.2, 
                                                            p=0.5
                                                        ),
                                    
                                    A.ShiftScaleRotate(shift_limit=0.1, 
                                                                        scale_limit=0.15, 
                                                                        rotate_limit=60, 
                                                                        p=0.5),
                                    
                                    A.Normalize(
                                        mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225], 
                                        max_pixel_value=255.0, 
                                        p=1),
                                    ToTensorV2()], 
                                                            p=1.)
    val_transforms  = A.Compose([                        
                        A.Resize(args.img_size,args.img_size),
                        A.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225], 
                        max_pixel_value=255.0, 
                        p=1),
                        ToTensorV2()])
        

    print(f"Full Train Mode")
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="ISIC2024")

    train_df = train
    
    target_ratio = train_df['target'].mean()
    print(f"Target ratio: {target_ratio}")

    # ------------------
    #    Dataloader
    # ------------------
    train_loader, _ = get_dataset_and_loader(loader_args, train_df, None, train_transform, val_transforms, args)


    model = load_model(args, 0)


    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = get_scheduler(args, optimizer, len(train_loader))


    # ------------------
    #   Start training
    # ------------------  
    max_val_score = 0
    no_improve_epochs = 0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss = []
        current_lr = optimizer.param_groups[0]["lr"]
        if not args.do_mixup:
            if args.weighted_loss:
                if args.weighted_loss:
                    # pos_weight は target==1 のサンプルの比率で重み付けします
                    pos_weight = torch.tensor([(1.0 - args.sampling_rate) / args.sampling_rate]).to(args.device)
                else:
                    pos_weight = torch.tensor([(1.0 - target_ratio) / target_ratio]).to(args.device)
                print(f"Using weighted loss. pos_weight: {pos_weight}")
                loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                loss_func = torch.nn.BCEWithLogitsLoss()
        else:
            loss_func = torch.nn.CrossEntropyLoss()
        
        if args.do_mixup:   
            #train mixup dataset
            combo_loader = get_combo_loader(train_loader, base_sampling="instance")
            #mixupの場合は、train_loader=combo_loader
            _, _ , _ = run_one_epoch(combo_loader, model, optimizer, lr_scheduler, args, epoch, loss_func)
        else:
            train_loss, train_score, train_auc = run_one_epoch(train_loader, model, optimizer, lr_scheduler, args, epoch, loss_func)

        with torch.no_grad():
            if args.do_mixup:
                #ここで実際のTrainDataに対するロスを計算
                train_loss, train_score, train_auc = run_one_epoch(train_loader, model, None, None, args, epoch, loss_func)

            
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_score": np.mean(train_score), "train_auc": np.mean(train_auc), 
                        "lr": current_lr})

        if train_score > args.target_score:
          if args.use_wandb:
              wandb.finish()
          cprint("early stopping", "cyan")
          return 
          
        model_path = os.path.join(logdir, f"model_fulltrain_score{train_score}.pt")
        # モデルの状態とその他の情報を保存
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'train_score': train_score,
            'expname': args.expname,
        }, model_path)


        if args.local_dir:
            # Localにコピー
            local_dir = os.path.join(args.local_dir, f"{args.expname}_{args.ver}")
            os.makedirs(local_dir, exist_ok=True)

            model_path = os.path.join(logdir, f"model_fulltrain_score{train_score}.pt")
            if os.path.exists(model_path):
                shutil.copy(model_path, local_dir)
                print(f'Model saved to Local: {local_dir}')
    
    


if __name__ == "__main__":
    run()