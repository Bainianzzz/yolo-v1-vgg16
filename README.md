# YOLOv1 + VGG16 训练项目

使用 PyTorch 从零实现 YOLOv1 目标检测模型，VGG16 作为 backbone，在 COCO8 数据集上训练和验证。

## 项目结构

```
├── model/
│   ├── __init__.py
│   ├── backbone.py   # VGG16 特征提取器
│   ├── head.py       # YOLOv1 检测头
│   ├── utils.py      # IoU / NMS / 标签编解码
│   └── loss.py       # YOLOv1 损失函数
├── dataset.py         # COCO8 数据加载器
├── train.py           # 训练脚本
├── val.py             # 验证脚本 (mAP)
└── pyproject.toml     # uv 依赖配置
```

## 环境配置

```bash
# 安装 uv (如未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate
```

## 训练

```bash
# 默认参数: 100 epoch, batch=1, lr=1e-4, CPU
python train.py

# 自定义参数
python train.py --epochs 200 --batch-size 1 --lr 1e-5 --device cpu
```

模型检查点保存在 `checkpoints/yolov1_best.pth`。

## 验证 (mAP)

```bash
python val.py --checkpoint checkpoints/yolov1_best.pth
```

## 模型结构

```
Input: 448×448×3
    ↓
VGG16 conv layers → 14×14×512
    ↓
Extra Conv (stride=2) → 7×7×1024
    ↓
Flatten → FC(4096) → FC(4410)
    ↓
Output: 7×7×90  (2 bboxes × 5 + 80 classes)
```

## 损失函数

- λ_coord = 5：坐标损失权重
- λ_noobj = 0.5：无目标置信度损失权重
- 负责的 bbox 选择：同 grid cell 内与 ground truth IoU 最大的 bbox
