import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CaptionImageDataset(Dataset):
    def __init__(self, manifest: str, image_size: int = 256):
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        self.records = [json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        path = Path(row["image"])
        if not path.is_absolute():
            path = self.root / path
        image = Image.open(path).convert("RGB")
        return self.transform(image), str(row["caption"])
