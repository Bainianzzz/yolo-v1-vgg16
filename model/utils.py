"""
Utility functions: IoU, NMS, bbox encoding/decoding for YOLOv1.
"""

import torch


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    计算两组边界框之间的 IoU。

    Args:
        box1: (..., 4)  [x_center, y_center, w, h]  所有值归一化到 [0, 1]
        box2: (..., 4)  [x_center, y_center, w, h]

    Returns:
        iou: (...,)  IoU 值
    """
    # 转换为左上-右下格式 [x1, y1, x2, y2]
    box1_x1 = box1[..., 0] - box1[..., 2] / 2
    box1_y1 = box1[..., 1] - box1[..., 3] / 2
    box1_x2 = box1[..., 0] + box1[..., 2] / 2
    box1_y2 = box1[..., 1] + box1[..., 3] / 2

    box2_x1 = box2[..., 0] - box2[..., 2] / 2
    box2_y1 = box2[..., 1] - box2[..., 3] / 2
    box2_x2 = box2[..., 0] + box2[..., 2] / 2
    box2_y2 = box2[..., 1] + box2[..., 3] / 2

    # 交集区域
    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # 并集面积
    area1 = box1[..., 2] * box1[..., 3]
    area2 = box2[..., 2] * box2[..., 3]
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-6)


def cellboxes_to_boxes(predictions: torch.Tensor, S: int = 7, B: int = 2) -> torch.Tensor:
    """
    将 YOLO 网格输出转换为图像相对坐标的边界框。

    YOLO 输出格式: 对于每个 cell (i, j):
      x, y: 相对于当前 cell 左上角的偏移 (0~1)
      w, h: 相对于全图的比例 (0~1)

    转换后: 所有坐标相对于全图 (0~1)

    Args:
        predictions: (N, S, S, B*5 + C) 模型原始输出
        S: 网格大小
        B: 每个 cell 的 bbox 数量

    Returns:
        bboxes: (N, S, S, B, 6)  每个 bbox: [x, y, w, h, conf, class_score]
    """
    N = predictions.shape[0]
    C = predictions.shape[-1] - B * 5
    device = predictions.device

    # bbox 参数: (N, S, S, B*5) -> (N, S, S, B, 5)
    # 类别概率: (N, S, S, C)，由该 cell 内的所有 bbox 共享
    pred_boxes = predictions[..., :B * 5].view(N, S, S, B, 5)
    pred_cls = predictions[..., B * 5:]  # (N, S, S, C)

    xy = torch.sigmoid(pred_boxes[..., 0:2])   # (N, S, S, B, 2)
    wh = pred_boxes[..., 2:4] ** 2               # sqrt(w)->w, sqrt(h)->h
    conf = torch.sigmoid(pred_boxes[..., 4:5])  # (N, S, S, B, 1)

    # 类别概率经过 softmax，每个 cell 共享
    cls_probs = torch.softmax(pred_cls, dim=-1)  # (N, S, S, C)

    # x, y 从相对于 cell 的偏移转为相对于全图
    cell_indices = torch.arange(S, dtype=torch.float32, device=device)
    cell_y, cell_x = torch.meshgrid(cell_indices, cell_indices, indexing="ij")

    x = (xy[..., 0:1] + cell_x.view(1, S, S, 1, 1)) / S
    y = (xy[..., 1:2] + cell_y.view(1, S, S, 1, 1)) / S

    # 对每个 bbox，取最大类别分数
    cls_probs_exp = cls_probs.unsqueeze(3)  # (N, S, S, 1, C)
    max_class, _ = cls_probs_exp.max(dim=-1, keepdim=True)  # (N, S, S, 1, 1)
    max_class = max_class.expand(N, S, S, B, 1)  # (N, S, S, B, 1)
    score = conf * max_class

    return torch.cat([x, y, wh, conf, score], dim=-1)  # (N, S, S, B, 6)


def non_max_suppression(
    bboxes: torch.Tensor,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.4,
) -> list[torch.Tensor]:
    """
    对预测结果执行 NMS。

    Args:
        bboxes: (N, S, S, B, 7)  每个 bbox: [x, y, w, h, conf, score, class_id]
                或者 (N, S*S*B, 7)  已展平的格式
        iou_threshold: IoU 阈值
        conf_threshold: 置信度阈值

    Returns:
        results: list of (M, 7) tensors, 每张图片的结果
    """
    N = bboxes.shape[0]

    if bboxes.dim() > 2:
        # 展平: (N, S*S*B, 7)
        bboxes = bboxes.view(N, -1, bboxes.shape[-1])

    results = []
    for i in range(N):
        # 按置信度过滤
        bbox = bboxes[i]
        mask = bbox[:, 4] * bbox[:, 5] > conf_threshold
        bbox = bbox[mask]

        if bbox.shape[0] == 0:
            results.append(bbox)
            continue

        # 按 score 降序排序
        scores = bbox[:, 4] * bbox[:, 5]
        _, order = scores.sort(descending=True)
        bbox = bbox[order]

        keep = []
        while bbox.shape[0] > 0:
            keep.append(bbox[0:1])
            if bbox.shape[0] == 1:
                break
            ious = compute_iou(bbox[0:1, :4], bbox[1:, :4])
            mask = ious < iou_threshold
            bbox = bbox[1:][mask]

        if keep:
            results.append(torch.cat(keep, dim=0))
        else:
            results.append(bbox.new_zeros((0, bboxes.shape[-1])))

    return results


def encode_target(
    labels: torch.Tensor,
    S: int = 7,
    B: int = 2,
    C: int = 80,
) -> torch.Tensor:
    """
    将 COCO 标签编码为 YOLOv1 训练目标格式。

    Args:
        labels: list of (M, 5) tensors, 每张图片的标签
                [class_id, x_center, y_center, width, height]  归一化到 [0, 1]
        S: 网格大小
        B: 每个 cell 的 bbox 数量
        C: 类别数

    Returns:
        target: (N, S, S, B*5 + C)
    """
    N = len(labels)
    target = torch.zeros(N, S, S, B * 5 + C)

    for b in range(N):
        for box in labels[b]:
            cls_id = int(box[0])
            x_center, y_center, w, h = box[1:5]

            # 确定该 box 中心落在哪个 grid cell
            grid_x = int(x_center * S)
            grid_y = int(y_center * S)
            grid_x = min(grid_x, S - 1)
            grid_y = min(grid_y, S - 1)

            # 相对于 cell 左上角的坐标
            x_cell = x_center * S - grid_x
            y_cell = y_center * S - grid_y

            # 编码两个 bbox 的坐标（相同）
            for j in range(B):
                start = j * 5
                target[b, grid_y, grid_x, start + 0] = x_cell
                target[b, grid_y, grid_x, start + 1] = y_cell
                target[b, grid_y, grid_x, start + 2] = w
                target[b, grid_y, grid_x, start + 3] = h
                target[b, grid_y, grid_x, start + 4] = 1.0  # 置信度

            # one-hot 类别
            target[b, grid_y, grid_x, B * 5 + cls_id] = 1.0

    return target
