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
from sklearn.impute import SimpleImputer

from src.utils import *
from src.dataset import SkinCancerDataset, get_transforms, final_nan_check_and_handle
from src.combo_loader import get_combo_loader
from src.model import *
from src.feature_engeneering import *
from src.metric import *

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

    val_dataset = SkinCancerDataset(args, "val", val_df, 
                                  val_transforms, args.remove_hair_thresh  
                                 )
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_args)  

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    return train_loader, val_loader


def load_model(args, fold):
    if "efficientnet" in args.model_name:
        model = CustomModel(args, training=True).to(args.device)
    elif "swin" in args.model_name:
        model = CustomSwinModel(args, training=True).to(args.device)
    elif "convnext" in args.model_name or "edgenext" in args.model_name:
        model = CustomConvEdgeNextModel(args, training=True).to(args.device)
    elif "eva" in args.model_name:
        model = CustomModelEva(args, training=True).to(args.device)
    elif "resnext" in args.model_name:
        model = CustomModelResNet(args, training=True).to(args.device)
    elif "coatnet" in args.model_name:
        model = CustomCoatnetModel(args, training=True).to(args.device)
    elif "resnet" in args.model_name:
        model = CustomModelResNet(args, training=True).to(args.device)
    else:
        raise ValueError(f"Model {args.model_name} not supported")

    if args.pretrain_dir:
        # モデルの重みまたは辞書形式で保存されたファイルをロード
        file_path = os.path.join(args.pretrain_dir, f"model_best_fold{fold+1}.pt")
        checkpoint = torch.load(file_path, map_location=torch.device(args.device))
        
        # バックボーンの重みのみをフィルタリングする関数
        def filter_backbone(state_dict):
            return {k: v for k, v in state_dict.items() if not k.startswith('block_1') and not k.startswith('block_2') and not k.startswith('linear_main')}

        # 保存ファイルが辞書形式か単なる重みファイルかを確認
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 辞書形式の場合
            state_dict = checkpoint['model_state_dict']
            if args.use_metadata_num:
                state_dict = filter_backbone(state_dict)
            # フィルタリングされた状態をモデルにロード
            model.load_state_dict(state_dict, strict=False)
            # メタデータを取得
            epoch = checkpoint.get('epoch', None)
            val_score = checkpoint.get('val_score', None)
            cprint(f"Loaded {'backbone' if args.use_metadata_num else 'full model'} from fold {fold+1}, epoch: {epoch}, val_score: {val_score}", "cyan")
        else:
            # 単なる重みファイルの場合
            if args.use_metadata_num:
                checkpoint = filter_backbone(checkpoint)
            model.load_state_dict(checkpoint, strict=False)
            cprint(f"Loaded {'backbone' if args.use_metadata_num else 'full model'} from fold {fold+1}", "cyan")

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
        if args.use_metadata_num:
            if args.aux_loss_features is not None:
                inputs, labels, aux_features, metadata = batch[0].to(args.device), batch[1].squeeze().to(args.device), batch[2].to(args.device), batch[3].to(args.device)
            else:
                inputs, labels, metadata = batch[0].to(args.device), batch[1].squeeze().to(args.device), batch[2].to(args.device)
        else:
            if args.aux_loss_features is not None:
                inputs, labels, aux_features = batch[0].to(args.device), batch[1].squeeze().to(args.device), batch[2].to(args.device)
            else:
                inputs, labels = batch[0].to(args.device), batch[1].squeeze().to(args.device)

        if args.use_metadata_num:
            y_pred = model(inputs, metadata)
        elif args.use_metadata_num and args.aux_loss_features is not None:
            y_pred, aux_outs = model(inputs, metadata)
        elif args.aux_loss_features is not None:
            y_pred, aux_outs = model(inputs)
        else:
            y_pred = model(inputs)

        if (isinstance(loss_func, torch.nn.BCEWithLogitsLoss) or isinstance(loss_func, torch.nn.BCELoss)):
            y_pred = y_pred.squeeze()
            labels = labels.float()  
        
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
            pred = y_pred.sigmoid().cpu().detach().numpy()
        elif isinstance(loss_func, nn.BCELoss):
            pred = y_pred.cpu().detach().numpy()
        elif isinstance(loss_func, nn.CrossEntropyLoss):
            pred = y_pred[:,1].cpu().detach().numpy()

        # 形状を統一
        pred = pred.reshape(-1)  # 1次元配列に変換
        all_preds.append(pred)

        # ラベルも同様に処理
        label = labels.cpu().numpy().reshape(-1)
        all_labels.append(label)

        if train:  # 訓練モードのみ
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

        losses.append(loss.item())

    # ループ終了後
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # スコアとAUCを計算
    score = custom_metric(all_labels, all_preds)
    auc_score = roc_auc_score(all_labels, all_preds) 
    
    if train:
        print(f"Epoch {epoch+1}/{args.epochs} | {mode} loss: {np.mean(losses):.3f} | {mode} score: {score:.3f} | {mode} auc: {auc_score:.3f} | lr: {current_lr:.7f}")
    else:
        print(f"Epoch {epoch+1}/{args.epochs} | {mode} loss: {np.mean(losses):.3f} | {mode} score: {score:.3f} | {mode} auc: {auc_score:.3f}")
    
    return np.mean(losses), score, auc_score


def configure_optimizers(model, args):
    if args.use_metadata_num:
        # メタデータブロックのパラメータを取得
        metadata_params = list(model.metadata_encoder.parameters()) + list(model.linear_main.parameters())
        if args.fusion_method == 'gated':
          metadata_params += list(model.gate.parameters())
        elif args.fusion_method == 'attention':
          metadata_params += list(model.attention_fusion.parameters())
        # その他のパラメータを取得
        other_params = [p for _, p in model.named_parameters() if not any(p is mp for mp in metadata_params)]
        
        # オプティマイザーを設定
        optimizer = torch.optim.AdamW([
            {'params': other_params},
            {'params': metadata_params, 'lr': args.lr * args.metadata_head_weight}  # メタデータブロックの学習率を10分の1に設定
        ], lr=args.lr, weight_decay=args.weight_decay)
        cprint(f"Set metadata backbone lr: {args.lr * args.metadata_head_weight}", "cyan")
    else:
        metadata_params = []
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    

    
    return optimizer

@hydra.main(version_base=None, config_path="configs", config_name="config_finetuning")
def run(args: DictConfig): 
    print(args)
    set_seed(args.seed)

    logdir = "/kaggle/working/" if not args.COLAB else hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"logdir: {logdir}")

    train = pd.read_csv(args.train_df_dir)
    if args.use_metadata_num:
        train['age_approx'] = train['age_approx'].replace('NA', np.nan).astype(float)
        train['age_approx'] = train['age_approx'].fillna(train['age_approx'].median())
        if args.use_metadata_num < 150:
            meta_cols = ['isic_id', 'target', 'fold', 'archive', 'age_approx','clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext','tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H',
                    'tbp_lv_Hext', 'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_areaMM2','tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 'tbp_lv_deltaA',
                    'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM',
                    'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 'tbp_lv_norm_color', 'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 
                    'tbp_lv_stdL', 'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',
                    'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z', ]
            train = train[meta_cols]
            # 'age_approx' の変換と欠損値の処理
            train = feature_engeneering_for_cnn(train)
        else:
            cat_cols, num_cols, new_num_cols, other_cols = get_feature_cols()
            train = read_data(args.train_df_dir, cat_cols, num_cols, new_num_cols)

            # 数値型特徴量の欠損値を中央値で補完
            num_imputer = SimpleImputer(strategy='median')
            train[num_cols + new_num_cols] = num_imputer.fit_transform(train[num_cols + new_num_cols])
            # カテゴリ型特徴量の欠損値を最頻値で補完
            # cat_imputer = SimpleImputer(strategy='most_frequent')
            # train[cat_cols] = cat_imputer.fit_transform(train[cat_cols])

            train, new_cat_cols = prepare_data_for_training(train, cat_cols, logdir)
            feature_cols = num_cols + new_num_cols + other_cols #new_cat_cols + 
            train = train[['isic_id', 'target', 'fold', 'archive'] + feature_cols]
        
        # 最終的なNaNチェックと処理
        train = final_nan_check_and_handle(train, feature_cols)

        # NaNがないことを確認
        assert train[feature_cols].isna().sum().sum() == 0, "NaN values found in the final dataset"

        cprint("Preprocessing completed. No NaN values in the final dataset.", "yellow")
        print(f"all columns num: {len(train.columns)}, feature num: {len(train.columns) - 4}")
        
        if args.use_metadata_num > 150 and args.local_dir:
            # Localにコピー
            local_dir = os.path.join(args.local_dir, f"{args.expname}_{args.ver}")
            os.makedirs(local_dir, exist_ok=True)
            encoder_path = os.path.join(logdir, f"onehot_encoder.joblib")
            if os.path.exists(encoder_path):
                shutil.copy(encoder_path, local_dir)
                print(f'Encoder saved to Local: {local_dir}')
    
        
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    train_transform, val_transform = get_transforms(args.img_size, args.augmentation_strength)
    
    if args.use_wandb:
        config = {
            "dropout": args.dropout,
            "drop_path_rate": args.drop_path_rate,
            "sampling_rate": args.sampling_rate,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "model_name": args.model_name,
            "use_metadata_num": args.use_metadata_num,
            "metadata_head_weight": args.metadata_head_weight,
            "augmentation_strength": args.augmentation_strength,
            "fusion_method": args.fusion_method,
            "metadata_dim": args.metadata_dim,
            # 必要に応じて他のハイパーパラメータも追加
        }
        wandb.init(mode="online", dir=logdir, project="ISIC2024_v2", group=args.expname, config=config)
    for fold in range(args.num_splits):
        if fold not in args.use_fold:
            print(f"Test mode. Skipping fold{fold+1}")
            continue
        

        train_df = train[train["fold"] != fold]
        valid_df = train[train["fold"] == fold]
        
        target_ratio = train_df['target'].mean()
        print(f"fold{fold+1}'s Target ratio: {target_ratio}. valid target ratio: {valid_df['target'].mean()}")

        # ------------------
        #    Dataloader
        # ------------------
        train_loader, val_loader = get_dataset_and_loader(loader_args, train_df, valid_df, train_transform, val_transform, args)

        model = load_model(args, fold)


        # ------------------
        #     Optimizer
        # ------------------
        optimizer = configure_optimizers(model, args)
        lr_scheduler = get_scheduler(args, optimizer, len(train_loader))
    
    
        # ------------------
        #   Start training
        # ------------------  
        max_val_score = 0
        no_improve_epochs = 0

        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")

            train_loss, val_loss = [], []
            current_lr = optimizer.param_groups[0]["lr"]

            loss_func = torch.nn.BCEWithLogitsLoss()
            train_loss, train_score, train_auc = run_one_epoch(train_loader, model, optimizer, lr_scheduler, args, epoch, loss_func)

            with torch.no_grad():
                val_loss, val_score, val_auc = run_one_epoch(val_loader, model,  None, None, args, epoch, loss_func)
                
            if args.use_wandb:
                wandb.log({
                    f"fold{fold+1}/train_loss": np.mean(train_loss),
                    f"fold{fold+1}/train_score": np.mean(train_score),
                    f"fold{fold+1}/train_auc": np.mean(train_auc),
                    f"fold{fold+1}/val_loss": np.mean(val_loss),
                    f"fold{fold+1}/val_score": np.mean(val_score),
                    f"fold{fold+1}/val_auc": np.mean(val_auc),
                    f"fold{fold+1}/lr": current_lr,
                    "epoch": epoch
                })

            if np.mean(val_score) > max_val_score:
                cprint("New best.", "cyan")
                model_path = os.path.join(logdir, f"model_best_fold{fold+1}.pt")
                # モデルの状態とその他の情報を保存
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_score': val_score,
                    'expname': args.expname,
                }, model_path)
                max_val_score = np.mean(val_score)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs > args.early_stopping_rounds:
                    cprint(f"Early stopping. No improvement for {no_improve_epochs} epochs.", "cyan")
                    break

        if args.local_dir:
            # Localにコピー
            local_dir = os.path.join(args.local_dir, f"{args.expname}_{args.ver}")
            os.makedirs(local_dir, exist_ok=True)

            model_path = os.path.join(logdir, f"model_best_fold{fold+1}.pt")
            if os.path.exists(model_path):
                shutil.copy(model_path, local_dir)
                print(f'Model saved to Local: {local_dir}')
        
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    run()