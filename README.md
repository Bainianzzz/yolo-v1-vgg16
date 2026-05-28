# YOLOv1 + VGG16

基于 PyTorch 从零实现 YOLOv1 目标检测模型，VGG16 作为 backbone。默认使用 **COCO128** 数据集（128 张图片）训练，支持 CPU / CUDA 12.8 GPU 训练。

## 项目结构

```
├── model/
│   ├── backbone.py   # VGG16 特征提取器 (448 → 7×7×1024)
│   ├── head.py       # YOLOv1 检测头 (7×7×1024 → 7×7×90)
│   ├── utils.py      # IoU / NMS / 标签编解码
│   └── loss.py       # YOLOv1 损失函数 (坐标+置信度+分类)
├── dataset.py         # 数据集加载 (coco8 / coco128 / ...)
├── train.py           # 训练脚本 (SwanLab 可视化)
├── val.py             # 验证脚本 (mAP@0.5)
├── predict.py         # 单图推理与可视化
└── pyproject.toml     # uv 依赖管理
```

## 环境配置

### 1. 安装 uv

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
pip install uv
```

### 2. 安装依赖

**GPU 服务器（CUDA 12.8）** — 默认已配置 cu128 索引：

```bash
uv sync
```

**本地 CPU** — 需要注释 `pyproject.toml` 中 `[tool.uv.index]` 和 `[tool.uv.sources]` 段，然后：

```bash
uv sync --index-url https://download.pytorch.org/whl/cpu
```

### 3. SwanLab 登录

首次使用需要登录 SwanLab：

```bash
uv run swanlab login
```

根据提示输入 API Key（从 [swanlab.cn/settings](https://swanlab.cn/settings) 获取，需注册账号）。

## 训练

首次运行会自动下载数据集（coco128 约 6MB，coco8 约 430KB）。

```bash
# GPU 默认配置: 100 轮, batch=16, CUDA 自动检测
python train.py

# 切换数据集
python train.py --dataset coco8

# 自定义参数
python train.py --epochs 200 --batch-size 32 --lr 1e-5 --device cuda

# CPU 训练
python train.py --device cpu --batch-size 1 --num-workers 0
```

训练曲线自动记录到 SwanLab，运行 `swanlab watch` 查看。

模型保存：
| 文件 | 说明 |
|------|------|
| `checkpoints/yolov1_epoch010.pth` | 每 10 轮周期检查点 |
| `checkpoints/yolov1_best.pth` | 验证损失最低的模型 |

## 推理

```bash
# 默认示例图片
python predict.py --checkpoint checkpoints/yolov1_best.pth

# 自定义图片
python predict.py --checkpoint checkpoints/yolov1_best.pth \
                  --url "https://example.com/your_image.jpg" \
                  --output result.jpg

# 调整检测阈值
python predict.py --conf 0.2 --iou 0.4
```

## 验证 (mAP)

```bash
python val.py --checkpoint checkpoints/yolov1_best.pth
python val.py --dataset coco128 --checkpoint checkpoints/yolov1_best.pth
```

## 模型结构

```
Input: 448×448×3
    │
    ▼
VGG16 Conv Layers
  448 → 224 → 112 → 56 → 28 → 14
  Output: 14×14×512
    │
    ▼
Extra Conv (s=2, 512→1024)
  Output: 7×7×1024
    │
    ▼
Flatten → FC 4096 → LeakyReLU → FC 4410
  Output: 7×7×90
    │
    ▼
Per-cell: 2 bboxes (x, y, w, h, conf) × 5 + 80 class probs
```

## 损失函数

| 组件 | 权重 | 说明 |
|------|------|------|
| 坐标损失 | λ=5 | x, y (sigmoid) + √w, √h (sqrt 权重) |
| 有目标置信度 | λ=1 | 负责的 bbox 与 GT 的 IoU 匹配 |
| 无目标置信度 | λ=0.5 | 其余 bbox 置信度 → 0 |
| 分类损失 | λ=1 | 有目标的 cell 的 one-hot 分类 |

负责任选框：同 cell 中与 ground truth IoU 最高的 bbox。

## 依赖

| 包 | 用途 |
|----|------|
| torch / torchvision | 模型训练与推理（CUDA 12.8 / CPU） |
| ultralytics | 数据集下载（coco8 / coco128） |
| tqdm | 进度条 |
| swanlab | 训练曲线可视化 |
