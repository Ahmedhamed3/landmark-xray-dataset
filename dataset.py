import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class LandmarkDataset(Dataset):
    """Dataset returning landmark coordinates in pixel space and image size."""

    def __init__(self, root_dir, csv_file="landmark_summary.csv", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = pd.read_csv(os.path.join(root_dir, csv_file))

        # Determine the full list of labels present in the CSV
        self.labels = sorted(self.data["PointLabel"].unique())
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}

        # Pre-process all samples so that original image size is known
        self.samples = []
        for sample_id, group in self.data.groupby("SampleID"):
            img_path = os.path.join(root_dir, str(sample_id), f"xray{sample_id}.jpg")
            coords_px = torch.zeros(len(self.labels), 2, dtype=torch.float32)
            mask = torch.zeros(len(self.labels), dtype=torch.float32)

            # Obtain original image width/height once
            if os.path.exists(img_path):
                w, h = Image.open(img_path).size
            else:
                w = h = 1.0

            for _, row in group.iterrows():
                label = row["PointLabel"]
                if label not in self.label_to_idx:
                    continue
                idx = self.label_to_idx[label]
                x = row["X"]
                y = row["Y"]
                coords_px[idx] = torch.tensor([x, y])
                mask[idx] = 1.0

            self.samples.append({
                "image_path": img_path,
                "coords_px": coords_px,
                "mask": mask,
                "width": torch.tensor(w, dtype=torch.float32),
                "height": torch.tensor(h, dtype=torch.float32),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        img = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return (
            img,
            sample["coords_px"],
            sample["mask"],
            sample["width"],
            sample["height"],
        )
