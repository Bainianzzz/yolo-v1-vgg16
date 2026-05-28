"""
COCO8 数据集加载器

使用 ultralytics 自动下载 coco8 数据集，加载图像和 YOLO 格式的标签，
转换为 YOLOv1 训练目标格式 (7x7x90)。
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T

from model.utils import encode_target

# COCO 类别名 -> ID 映射
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


def get_coco8_path() -> Path:
    """下载并返回 coco8 数据集根目录。"""
    from ultralytics.utils import DATASETS_DIR

    project_root = Path(__file__).parent

    # 优先检查 ultralytics 默认路径，再检查本地项目目录
    candidates = [
        Path(DATASETS_DIR) / "coco8",
        project_root / "coco8",
    ]
    for coco8_dir in candidates:
        if coco8_dir.exists() and (coco8_dir / "images").exists():
            return coco8_dir

    # 没找到则下载到 ultralytics 数据集目录
    from urllib.request import urlretrieve
    import zipfile
    coco8_dir = Path(DATASETS_DIR) / "coco8"
    coco8_dir.mkdir(parents=True, exist_ok=True)

    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
    zip_path = coco8_dir / "coco8.zip"

    print(f"下载 COCO8 数据集...")
    urlretrieve(url, zip_path)

    print("解压中...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(coco8_dir)
    zip_path.unlink()

    return coco8_dir


class COCO8Dataset(Dataset):
    """
    COCO8 数据集，使用 YOLO 标签格式 (.txt 每行: class_id x_center y_center w h)。
    """

    def __init__(self, split: str = "train", S: int = 7, B: int = 2, C: int = 80,
                 image_size: int = 448):
        self.S = S
        self.B = B
        self.C = C
        self.image_size = image_size

        root = get_coco8_path()
        self.img_dir = root / "images" / split
        self.label_dir = root / "labels" / split

        self.img_paths = sorted(self.img_dir.glob("*.jpg"))

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            # ImageNet 均值/标准差归一化 (VGG16 使用)
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.img_paths)

    def _load_labels(self, img_path: Path) -> torch.Tensor:
        """从 .txt 标签文件加载并归一化标签。"""
        label_path = self.label_dir / (img_path.stem + ".txt")

        boxes = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)
                        boxes.append([cls, x, y, w, h])

        if len(boxes) == 0:
            return torch.zeros((0, 5))

        return torch.tensor(boxes, dtype=torch.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        labels = self._load_labels(img_path)

        image = self.transform(image)

        # 将单张图片的标签编码为 target
        target = encode_target([labels], self.S, self.B, self.C)
        target = target[0]  # (S, S, B*5+C)

        return image, target


def create_dataloaders(batch_size: int = 4, num_workers: int = 0,
                       image_size: int = 448) -> tuple[DataLoader, DataLoader]:
    """创建训练和验证 DataLoader。"""
    S, B, C = 7, 2, 80

    train_dataset = COCO8Dataset(split="train", S=S, B=B, C=C, image_size=image_size)
    val_dataset = COCO8Dataset(split="val", S=S, B=B, C=C, image_size=image_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    return train_loader, val_loader
