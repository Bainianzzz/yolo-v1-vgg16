"""
YOLOv1 Loss Function

实现 YOLOv1 论文中的多部分损失函数：
  - 坐标损失（λ_coord = 5）
  - 有目标置信度损失
  - 无目标置信度损失（λ_noobj = 0.5）
  - 分类损失

对于每个 grid cell 中 B 个预测框，只有与 ground truth IoU 最大的那个
"负责"预测该目标，其坐标损失和置信度损失被计算；其他预测框只计算
无目标置信度损失。
"""

import torch
import torch.nn as nn

from .utils import compute_iou


class YOLOv1Loss(nn.Module):
    def __init__(self, S: int = 7, B: int = 2, C: int = 80,
                 lambda_coord: float = 5.0, lambda_noobj: float = 0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (N, S, S, B*5 + C)  模型输出
            targets:     (N, S, S, B*5 + C)  标签

        Returns:
            loss: 标量
        """
        N = predictions.shape[0]
        device = predictions.device

        # 分离 bbox 参数和类别概率
        # bbox: (N, S, S, B*5) -> (N, S, S, B, 5)
        # cls:  (N, S, S, C)
        pred_boxes = predictions[..., :self.B * 5].view(N, self.S, self.S, self.B, 5)
        pred_cls = predictions[..., self.B * 5:]  # (N, S, S, C)

        target_boxes = targets[..., :self.B * 5].view(N, self.S, self.S, self.B, 5)
        target_cls = targets[..., self.B * 5:]  # (N, S, S, C)

        # ---- 分离预测值 (应用激活函数以匹配推理) ----
        pred_xy = torch.sigmoid(pred_boxes[..., 0:2])  # (N, S, S, B, 2), [0,1]
        pred_wh = pred_boxes[..., 2:4]
        pred_conf = torch.sigmoid(pred_boxes[..., 4:5])  # (N, S, S, B, 1), [0,1]

        # ---- 分离目标值 ----
        target_xy = target_boxes[..., 0:2]
        target_wh = target_boxes[..., 2:4]
        target_conf = target_boxes[..., 4:5]

        # ---- 目标掩码 ----
        # 哪些 cell 中有物体（confidence > 0）
        obj_mask = (target_conf > 0).float()     # (N, S, S, B, 1)
        # 哪个 cell 中有物体（任意一个 bbox 有物即可）
        any_obj = obj_mask.amax(dim=3)  # (N, S, S, 1)

        # ---- 计算 predicted IoU，确定负责的 bbox ----
        with torch.no_grad():
            cell_y, cell_x = torch.meshgrid(
                torch.arange(self.S, device=device, dtype=torch.float32),
                torch.arange(self.S, device=device, dtype=torch.float32),
                indexing="ij",
            )
            cell_x = cell_x.view(1, self.S, self.S, 1, 1)
            cell_y = cell_y.view(1, self.S, self.S, 1, 1)

            pred_x_abs = (pred_xy[..., 0:1] + cell_x) / self.S
            pred_y_abs = (pred_xy[..., 1:2] + cell_y) / self.S

            target_x_abs = (target_xy[..., 0:1] + cell_x) / self.S
            target_y_abs = (target_xy[..., 1:2] + cell_y) / self.S

            pred_box = torch.cat([pred_x_abs, pred_y_abs, pred_wh], dim=-1)
            target_box = torch.cat([target_x_abs, target_y_abs, target_wh], dim=-1)

        ious = compute_iou(pred_box.view(-1, 4), target_box.view(-1, 4))
        ious = ious.view(N, self.S, self.S, self.B, 1)

        # 负责的 bbox：同 cell 中 IoU 最高的那个
        max_iou, _ = ious.max(dim=3, keepdim=True)  # (N, S, S, 1, 1)
        resp_mask = (ious >= max_iou).float() * obj_mask  # (N, S, S, B, 1)

        # 不负责的 bbox 掩码
        noobj_mask = (1.0 - resp_mask)  # (N, S, S, B, 1)

        # ---- 坐标损失 ----
        coord_loss = self.lambda_coord * (
            self.mse(resp_mask * pred_xy, resp_mask * target_xy) +
            self.mse(
                resp_mask * torch.sign(pred_wh) * torch.sqrt(torch.abs(pred_wh) + 1e-6),
                resp_mask * torch.sqrt(target_wh + 1e-6),
            )
        )

        # ---- 置信度损失 ----
        # 负责的 bbox 应与 target (1.0) 匹配
        obj_conf_loss = self.mse(resp_mask * pred_conf, resp_mask * target_conf)
        # 不负责的 bbox 应与 0 匹配 (无论 cell 中有无物体)
        noobj_conf_loss = self.lambda_noobj * self.mse(
            noobj_mask * pred_conf, torch.zeros_like(pred_conf)
        )
        conf_loss = obj_conf_loss + noobj_conf_loss

        # ---- 分类损失 ----
        # 只对有目标的 cell 计算，any_obj 扩展后与 pred_cls 广播
        cls_loss = self.mse(any_obj * pred_cls, any_obj * target_cls)

        # ---- 总损失 ----
        total_loss = (coord_loss + conf_loss + cls_loss) / N
        return total_loss
