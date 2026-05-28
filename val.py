"""
YOLOv1 验证/推理脚本

加载训练好的模型，在 COCO8 验证集上评估 mAP。

Usage:
    python val.py [--checkpoint checkpoints/yolov1_best.pth] [--device cpu]
                   [--conf-threshold 0.4] [--iou-threshold 0.5]
"""

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.backbone import VGG16Backbone
from model.head import YOLOv1Head
from model.utils import cellboxes_to_boxes, compute_iou
from dataset import create_dataloaders


def mean_average_precision(
    predictions: list[list[dict]],
    targets: list[list[dict]],
    iou_threshold: float = 0.5,
    num_classes: int = 80,
) -> float:
    """
    简化的 mAP 计算 (11-point interpolation 的粗粒度版本)。

    Args:
        predictions: 每张图片的预测列表，每个预测:
                     {"bbox": [x, y, w, h], "conf": float, "class": int}
        targets: 每张图片的真实标签列表，每个标签:
                 {"bbox": [x, y, w, h], "class": int}
        iou_threshold: 判定匹配的 IoU 阈值
        num_classes: 类别数

    Returns:
        mAP@0.5
    """
    aps = []

    for cls_id in range(num_classes):
        # 收集所有预测（按置信度排序）
        all_preds = []
        for img_idx, preds in enumerate(predictions):
            for p in preds:
                if p["class"] == cls_id:
                    all_preds.append({
                        "img_idx": img_idx,
                        "conf": p["conf"],
                        "bbox": p["bbox"],
                    })

        all_preds.sort(key=lambda x: x["conf"], reverse=True)

        # 统计每张图片中该类的 ground truth
        gt_counts = {}  # img_idx -> count
        gt_used = {}    # img_idx -> list[bool]
        for img_idx, gts in enumerate(targets):
            gt_boxes = [g["bbox"] for g in gts if g["class"] == cls_id]
            gt_counts[img_idx] = len(gt_boxes)
            gt_used[img_idx] = [False] * len(gt_boxes)

        # 总 ground truth 数量
        total_gt = sum(gt_counts.values())
        if total_gt == 0:
            continue

        tp = torch.zeros(len(all_preds))
        fp = torch.zeros(len(all_preds))

        for pi, pred in enumerate(all_preds):
            img_idx = pred["img_idx"]
            pred_box = pred["bbox"]

            if gt_counts[img_idx] == 0:
                fp[pi] = 1
                continue

            # 找到最佳匹配的 ground truth
            best_iou = 0.0
            best_gt_idx = -1
            for gi, gt_box in enumerate(
                [g for g in targets[img_idx] if g["class"] == cls_id]
            ):
                if gt_used[img_idx][gi]:
                    continue
                iou = compute_iou(
                    torch.tensor(pred_box).view(1, 4),
                    torch.tensor(gt_box["bbox"]).view(1, 4),
                ).item()
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[pi] = 1
                gt_used[img_idx][best_gt_idx] = True
            else:
                fp[pi] = 1

        # 累计 precision/recall
        tp_cumsum = tp.cumsum(dim=0)
        fp_cumsum = fp.cumsum(dim=0)

        recall = tp_cumsum / (total_gt + 1e-6)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        # 11-point interpolation
        ap = 0.0
        for t_val in torch.linspace(0, 1, 11):
            mask = recall >= t_val
            if mask.any():
                ap += precision[mask].max().item()
        aps.append(ap / 11.0)

    return sum(aps) / max(len(aps), 1)


@torch.no_grad()
def run_validation(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.5,
    num_classes: int = 80,
) -> float:
    """
    在验证集上运行推理并计算 mAP。

    Returns:
        mAP@0.5
    """
    backbone.eval()
    head.eval()

    all_predictions = []
    all_targets = []
    S, B = 7, 2

    for images, targets in tqdm(loader, desc="验证中"):
        images = images.to(device)
        N = images.shape[0]

        # ---- 模型推理 ----
        features = backbone(images)
        preds = head(features)  # (N, S, S, 90)

        # ---- 解码预测 ----
        decoded = cellboxes_to_boxes(preds, S=S, B=B)  # (N, S, S, B, 6)

        # ---- 解码 ground truth ----
        # 从 targets 中提取 bbox 参数: (N, S, S, B*5) -> (N, S, S, B, 5)
        target_boxes = targets[..., :B * 5].view(N, S, S, B, 5)

        for b in range(N):
            # 预测
            boxes = decoded[b]  # (S, S, B, 6)
            img_preds = []
            for i in range(S):
                for j in range(S):
                    for k in range(B):
                        conf = boxes[i, j, k, 4].item()
                        score = boxes[i, j, k, 5].item()
                        if conf * score < conf_threshold:
                            continue
                        x = boxes[i, j, k, 0].item()
                        y = boxes[i, j, k, 1].item()
                        w = boxes[i, j, k, 2].item()
                        h = boxes[i, j, k, 3].item()
                        if w <= 0 or h <= 0:
                            continue
                        # 取最大类别
                        cls_scores = preds[b, i, j, B * 5:]  # 80 classes
                        cls_id = int(torch.argmax(cls_scores).item())
                        img_preds.append({
                            "bbox": [x, y, w, h],
                            "conf": conf * score,
                            "class": cls_id,
                        })
            all_predictions.append(img_preds)

            # Ground truth
            img_gts = []
            for i in range(S):
                for j in range(S):
                    target_conf = target_boxes[b, i, j, 0, 4].item()
                    if target_conf > 0:
                        # 从 target 重建 bbox
                        x_cell = target_boxes[b, i, j, 0, 0].item()
                        y_cell = target_boxes[b, i, j, 0, 1].item()
                        w = target_boxes[b, i, j, 0, 2].item()
                        h = target_boxes[b, i, j, 0, 3].item()
                        x = (j + x_cell) / S
                        y = (i + y_cell) / S
                        cls_id = int(
                            torch.argmax(targets[b, i, j, B * 5:]).item()
                        )
                        img_gts.append({
                            "bbox": [x, y, w, h],
                            "class": cls_id,
                        })
            all_targets.append(img_gts)

    mAP = mean_average_precision(
        all_predictions, all_targets, iou_threshold, num_classes,
    )
    return mAP


def main():
    parser = argparse.ArgumentParser(description="Validate YOLOv1 model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/yolov1_best.pth",
                        help="模型检查点路径")
    parser.add_argument("--device", type=str, default="cpu", help="设备")
    parser.add_argument("--conf-threshold", type=float, default=0.4, help="置信度阈值")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU 阈值")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ---- 加载模型 ----
    backbone = VGG16Backbone(pretrained=False).to(device)
    head = YOLOv1Head().to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    backbone.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["head"])
    print(f"已加载检查点: epoch={checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.4f}")

    # ---- 数据加载 ----
    _, val_loader = create_dataloaders(batch_size=1)

    # ---- 验证 ----
    mAP = run_validation(
        backbone, head, val_loader, device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
    )
    print(f"\nmAP@0.5 = {mAP:.4f}")


if __name__ == "__main__":
    main()
