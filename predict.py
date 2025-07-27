import argparse
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from dataset import LandmarkDataset
from model import LandmarkNet


def load_model(model_path, num_landmarks, device):
    """Load LandmarkNet model with given number of landmarks."""
    model = LandmarkNet(num_landmarks)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, device):
    """Open and preprocess image. Returns tensor, width, height, PIL image."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor, w, h, img


def predict_landmarks(model, image_tensor):
    """Run model on image tensor and return Nx2 tensor of normalized coords."""
    with torch.no_grad():
        preds = model(image_tensor)
    return preds.squeeze(0).cpu()


def plot_landmarks(img, coords):
    """Display image with predicted landmark coordinates."""
    plt.imshow(img)
    xs = coords[:, 0]
    ys = coords[:, 1]
    plt.scatter(xs, ys, c="r", s=20)
    plt.axis("off")
    plt.show()


def find_closest_sample(pred_coords, dataset):
    """Find sample with smallest RMSE to predicted normalized coordinates."""
    best_id = None
    best_rmse = None
    sample_ids = sorted(dataset.data["SampleID"].unique())
    for sample_id, sample in zip(sample_ids, dataset.samples):
        mask = sample["mask"]
        coords = sample["coords"]
        diff = ((pred_coords - coords) ** 2).sum(dim=-1).sqrt()
        rmse = (diff * mask).sum() / mask.sum()
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_id = sample_id
    return best_id, best_rmse.item() if best_rmse is not None else None


def main():
    ap = argparse.ArgumentParser(description="Predict dental X-ray landmarks")
    ap.add_argument("image", help="Path to X-ray image")
    ap.add_argument("--model", default="model.pth", help="Path to trained model")
    ap.add_argument("--data-dir", default=".", help="Dataset directory with csv")
    ap.add_argument("--csv", default="landmark_summary.csv", help="CSV file")
    ap.add_argument("--no-match", action="store_true", help="Skip identity match")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset to know number of landmarks and for optional matching
    dataset = LandmarkDataset(args.data_dir, csv_file=args.csv)
    num_landmarks = len(dataset.labels)

    model = load_model(args.model, num_landmarks, device)

    img_tensor, w, h, pil_img = preprocess_image(args.image, device)

    pred_norm = predict_landmarks(model, img_tensor)

    # scale to pixel coordinates
    pred_px = pred_norm.clone()
    pred_px[:, 0] *= w
    pred_px[:, 1] *= h

    plot_landmarks(pil_img, pred_px)

    if not args.no_match:
        sample_id, rmse = find_closest_sample(pred_norm, dataset)
        if sample_id is not None:
            print(f"Closest sample: {sample_id} (RMSE={rmse:.4f})")
        else:
            print("No samples found for matching")


if __name__ == "__main__":
    main()
