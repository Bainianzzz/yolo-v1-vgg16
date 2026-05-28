"""
YOLOv1 训练脚本

使用 COCO128 数据集，VGG16 作为 backbone，训练 YOLOv1 模型。
训练曲线通过 SwanLab 记录。

Usage:
    python train.py [--epochs 100] [--batch-size 16] [--lr 1e-4]
                    [--dataset coco128] [--device cuda] [--num-workers 4]
"""

import argparse
import os

import swanlab
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.backbone import VGG16Backbone
from model.head import YOLOv1Head
from model.loss import YOLOv1Loss
from dataset import create_dataloaders


def train_one_epoch(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """训练一个 epoch，返回平均损失。"""
    backbone.train()
    head.train()

    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [train]", leave=False)

    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        features = backbone(images)
        preds = head(features)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    backbone: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    """验证，返回平均损失。"""
    backbone.eval()
    head.eval()

    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [val  ]", leave=False)

    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        features = backbone(images)
        preds = head(features)
        loss = criterion(preds, targets)

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def main():
    cuda_available = torch.cuda.is_available()

    parser = argparse.ArgumentParser(description="Train YOLOv1 with VGG16 backbone")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16 if cuda_available else 1,
                        help="batch 大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--dataset", type=str, default="coco128",
                        help="数据集名 (coco8 / coco128)")
    parser.add_argument("--device", type=str,
                        default="cuda" if cuda_available else "cpu",
                        help="训练设备 (cpu / cuda)")
    parser.add_argument("--num-workers", type=int, default=4 if cuda_available else 0,
                        help="DataLoader 进程数")
    parser.add_argument("--save-path", type=str, default="checkpoints",
                        help="模型保存目录")
    args = parser.parse_args()

    if args.device == "cuda" and not cuda_available:
        print("警告: CUDA 不可用，回退到 CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    print(f"使用设备: {device}  |  数据集: {args.dataset}")

    # ---- 数据加载 ----
    train_loader, val_loader = create_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"训练集: {len(train_loader.dataset)} 张图片, "
          f"验证集: {len(val_loader.dataset)} 张图片")

    # ---- 模型 ----
    backbone = VGG16Backbone(pretrained=True).to(device)
    head = YOLOv1Head().to(device)

    # ---- 损失函数 & 优化器 ----
    criterion = YOLOv1Loss()
    optimizer = optim.Adam(
        list(backbone.parameters()) + list(head.parameters()),
        lr=args.lr,
    )

    # ---- SwanLab ----
    swanlab.init(
        project="yolov1-vgg16",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "backbone": "vgg16",
            "dataset": args.dataset,
            "image_size": 448,
        },
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            backbone, head, train_loader, criterion, optimizer, device, epoch,
        )
        val_loss = validate(
            backbone, head, val_loader, criterion, device, epoch,
        )

        print(f"Epoch {epoch:3d} | train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f}")

        # ---- 记录到 SwanLab ----
        swanlab.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch,
        })

        # ---- 保存检查点 ----
        os.makedirs(args.save_path, exist_ok=True)
        model_state = {
            "backbone": backbone.state_dict(),
            "head": head.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        }

        if epoch % 10 == 0:
            torch.save(model_state, f"{args.save_path}/yolov1_epoch{epoch:03d}.pth")
            print(f"  -> 已保存周期检查点 (epoch={epoch})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model_state, f"{args.save_path}/yolov1_best.pth")
            print(f"  -> 已保存最佳模型 (val_loss={val_loss:.4f})")

    print(f"\n训练完成! 最佳验证损失: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
