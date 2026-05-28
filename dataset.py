"""
数据集加载器

支持 coco8 / coco128 等 ultralytics 格式数据集，加载图像和 YOLO 标签，
转换为 YOLOv1 训练目标格式 (7×7×90)。
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T

from model.utils import encode_target

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


def get_dataset_path(dataset_name: str = "coco128") -> Path:
    """下载并返回数据集根目录。"""
    from ultralytics.utils import DATASETS_DIR

    project_root = Path(__file__).parent

    candidates = [
        Path(DATASETS_DIR) / dataset_name,
        project_root / dataset_name,
    ]
    for ds_dir in candidates:
        if ds_dir.exists() and (ds_dir / "images").exists():
            return ds_dir

    from urllib.request import urlretrieve
    import zipfile
    ds_dir = Path(DATASETS_DIR) / dataset_name
    ds_dir.mkdir(parents=True, exist_ok=True)

    url = (f"https://github.com/ultralytics/assets/"
           f"releases/download/v0.0.0/{dataset_name}.zip")
    zip_path = ds_dir / f"{dataset_name}.zip"

    print(f"下载 {dataset_name} 数据集...")
    urlretrieve(url, zip_path)

    print("解压中...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(ds_dir)
    zip_path.unlink()

    return ds_dir


class COCODataset(Dataset):
    """通用 YOLO 格式数据集 (coco8 / coco128 等)。"""

    def __init__(self, split: str = "train", dataset_name: str = "coco128",
                 S: int = 7, B: int = 2, C: int = 80, image_size: int = 448):
        self.S = S
        self.B = B
        self.C = C
        self.image_size = image_size

        root = get_dataset_path(dataset_name)
        self.img_dir = root / "images" / split
        self.label_dir = root / "labels" / split

        self.img_paths = sorted(self.img_dir.glob("*.jpg"))

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.img_paths)

    def _load_labels(self, img_path: Path) -> torch.Tensor:
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
        target = encode_target([labels], self.S, self.B, self.C)[0]
        return image, target


def create_dataloaders(
    dataset_name: str = "coco128",
    batch_size: int = 4,
    num_workers: int = 0,
    image_size: int = 448,
) -> tuple[DataLoader, DataLoader]:
    """创建训练和验证 DataLoader。"""
    S, B, C = 7, 2, 80

    train_dataset = COCODataset(
        split="train", dataset_name=dataset_name, S=S, B=B, C=C, image_size=image_size,
    )
    val_dataset = COCODataset(
        split="val", dataset_name=dataset_name, S=S, B=B, C=C, image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
