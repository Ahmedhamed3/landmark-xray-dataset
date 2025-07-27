import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class LandmarkDataset(Dataset):
    def __init__(self, root_dir, csv_file="landmark_summary.csv", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = pd.read_csv(os.path.join(root_dir, csv_file))
        # Determine list of all landmark labels
        self.labels = sorted(self.data["PointLabel"].unique())
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}
        # Organize coordinates per sample
        self.samples = []
        for sample_id, group in self.data.groupby("SampleID"):
            img_path = os.path.join(root_dir, str(sample_id), f"xray{sample_id}.jpg")
            coords = torch.zeros(len(self.labels), 2, dtype=torch.float32)
            mask = torch.zeros(len(self.labels), dtype=torch.float32)
            # open image once to know width/height
            if os.path.exists(img_path):
                w, h = Image.open(img_path).size
            else:
                w = h = 1.0
            for _, row in group.iterrows():
                label = row["PointLabel"]
                if label not in self.label_to_idx:
                    continue
                idx = self.label_to_idx[label]
                x = row["X"] / w
                y = row["Y"] / h
                coords[idx] = torch.tensor([x, y])
                mask[idx] = 1.0
            self.samples.append({"image_path": img_path, "coords": coords, "mask": mask})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        img = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, sample["coords"], sample["mask"]
