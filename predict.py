"""
YOLOv1 推理脚本

使用训练好的模型对单张图片进行目标检测并可视化。

Usage:
    python predict.py --checkpoint checkpoints/yolov1_best.pth \
                      --url "https://ultralytics.com/images/bus.jpg" \
                      --output output.jpg [--conf 0.3] [--iou 0.5]
"""

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

from model.backbone import VGG16Backbone
from model.head import YOLOv1Head
from model.utils import compute_iou

# COCO 80 类别名
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# 可视化颜色池 (HSB 均匀分布)
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
] * 4  # 重复以覆盖 80 类


def transform_image(image: Image.Image, size: int = 448) -> torch.Tensor:
    """预处理图像用于模型推理。"""
    tfm = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return tfm(image).unsqueeze(0)  # (1, 3, 448, 448)


@torch.no_grad()
def predict(
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    image_tensor: torch.Tensor,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    device: torch.device = torch.device("cpu"),
    num_classes: int = 80,
) -> list[dict]:
    """
    对单张图片进行推理，返回检测结果列表。

    Returns:
        list[dict]: 每个元素 {"bbox": [x1, y1, x2, y2], "conf": float, "class_id": int, "class_name": str}
    """
    S, B = 7, 2

    image_tensor = image_tensor.to(device)
    features = backbone(image_tensor)
    preds = head(features)  # (1, S, S, B*5 + C)

    # ---- 解码 ----
    pred_boxes = preds[..., :B * 5].view(1, S, S, B, 5)
    pred_cls = preds[..., B * 5:]  # (1, S, S, C)

    xy = torch.sigmoid(pred_boxes[..., 0:2])  # (1, S, S, B, 2)
    wh = pred_boxes[..., 2:4] ** 2               # sqrt(w)->w, sqrt(h)->h
    conf = torch.sigmoid(pred_boxes[..., 4])     # (1, S, S, B)
    cls_probs = torch.softmax(pred_cls, dim=-1)  # (1, S, S, C)

    # 转换 xy 到全图坐标
    cell_idx = torch.arange(S, dtype=torch.float32, device=device)
    cell_y, cell_x = torch.meshgrid(cell_idx, cell_idx, indexing="ij")
    x = (xy[..., 0] + cell_x.view(1, S, S, 1)) / S
    y = (xy[..., 1] + cell_y.view(1, S, S, 1)) / S

    # ---- 收集所有候选框 ----
    candidates = []
    for i in range(S):
        for j in range(S):
            for k in range(B):
                c = conf[0, i, j, k].item()
                if c < conf_threshold:
                    continue
                if wh[0, i, j, k, 0].item() <= 1e-4 or wh[0, i, j, k, 1].item() <= 1e-4:
                    continue
                cls_score, cls_id = cls_probs[0, i, j].max(dim=0)  # 该 cell 的类别
                score = c * cls_score.item()
                if score < conf_threshold:
                    continue
                candidates.append({
                    "x": x[0, i, j, k].item(),
                    "y": y[0, i, j, k].item(),
                    "w": wh[0, i, j, k, 0].item(),
                    "h": wh[0, i, j, k, 1].item(),
                    "score": score,
                    "class_id": cls_id.item(),
                    "class_name": COCO_CLASSES[cls_id.item()],
                })

    candidates.sort(key=lambda d: d["score"], reverse=True)

    # ---- NMS ----
    kept = []
    while candidates:
        best = candidates.pop(0)
        kept.append(best)
        survivors = []
        for cand in candidates:
            if cand["class_id"] != best["class_id"]:
                survivors.append(cand)
                continue
            iou = compute_iou_simple(
                best["x"], best["y"], best["w"], best["h"],
                cand["x"], cand["y"], cand["w"], cand["h"],
            )
            if iou < iou_threshold:
                survivors.append(cand)
        candidates = survivors

    # 转换为像素坐标
    return kept


def compute_iou_simple(x1, y1, w1, h1, x2, y2, w2, h2) -> float:
    """两个 bbox (cxcywh) 的 IoU。"""
    box1 = torch.tensor([[x1, y1, w1, h1]])
    box2 = torch.tensor([[x2, y2, w2, h2]])
    return compute_iou(box1, box2).item()


def draw_results(
    image: Image.Image,
    detections: list[dict],
    output_path: str,
    font_size: int = 14,
):
    """在图片上绘制检测框并保存。"""
    draw_img = image.convert("RGB")
    orig_w, orig_h = draw_img.size
    draw = ImageDraw.Draw(draw_img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        # 尝试 Windows 常见中文字体
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    for det in detections:
        # 将归一化坐标转为像素坐标
        cx, cy, w, h = det["x"], det["y"], det["w"], det["h"]
        x1 = int((cx - w / 2) * orig_w)
        y1 = int((cy - h / 2) * orig_h)
        x2 = int((cx + w / 2) * orig_w)
        y2 = int((cy + h / 2) * orig_h)

        # 边界检查
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        color = COLORS[det["class_id"] % len(COLORS)]

        # 画框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 标签文字
        label = f"{det['class_name']} {det['score']:.2f}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # 文字背景
        text_y = max(0, y1 - text_h - 4)
        draw.rectangle(
            [x1, text_y, x1 + text_w + 4, text_y + text_h + 4],
            fill=color,
        )
        draw.text((x1 + 2, text_y + 2), label, fill="white", font=font)

    draw_img.save(output_path)
    print(f"结果已保存至: {output_path}")


def main():
    cuda_available = torch.cuda.is_available()

    parser = argparse.ArgumentParser(description="YOLOv1 单图推理")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/yolov1_best.pth",
                        help="模型权重路径")
    parser.add_argument("--url", type=str,
                        default="https://ultralytics.com/images/zidane.jpg",
                        help="图片 URL")
    parser.add_argument("--output", type=str, default="output.jpg",
                        help="输出图片路径")
    parser.add_argument("--conf", type=float, default=0.3, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU 阈值")
    parser.add_argument("--device", type=str,
                        default="cuda" if cuda_available else "cpu", help="推理设备")
    parser.add_argument("--image-size", type=int, default=448, help="模型输入尺寸")
    args = parser.parse_args()

    if args.device == "cuda" and not cuda_available:
        print("警告: CUDA 不可用，回退到 CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # ---- 加载模型 ----
    backbone = VGG16Backbone(pretrained=False).to(device)
    head = YOLOv1Head().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    backbone.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["head"])
    print(f"已加载检查点: epoch={checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

    # ---- 下载图片 ----
    local_img = Path("input.jpg")
    if not local_img.exists():
        print(f"下载图片: {args.url}")
        urlretrieve(args.url, local_img)
    else:
        print(f"使用本地图片: {local_img}")

    image = Image.open(local_img).convert("RGB")
    orig_size = image.size
    print(f"原始尺寸: {orig_size[0]}x{orig_size[1]}")

    # ---- 推理 ----
    img_tensor = transform_image(image, args.image_size)
    detections = predict(
        backbone, head, img_tensor,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=device,
    )
    print(f"检测到 {len(detections)} 个目标:")
    for det in detections:
        print(f"  {det['class_name']:15s}  conf={det['score']:.3f}  "
              f"bbox=[{det['x']:.3f}, {det['y']:.3f}, {det['w']:.3f}, {det['h']:.3f}]")

    # ---- 可视化 ----
    draw_results(image, detections, args.output)


if __name__ == "__main__":
    main()
