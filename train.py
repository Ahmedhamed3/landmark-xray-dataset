import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import LandmarkDataset
from model import LandmarkNet


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='.')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--backbone', default='resnet18', choices=['resnet18', 'mobilenet_v2'])
    ap.add_argument('--loss', default='mse', choices=['mse', 'smoothl1'])
    ap.add_argument('--out', default='model.pth', help='where to save trained model')
    return ap.parse_args()


def main():
    args = parse_args()
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = LandmarkDataset(args.data_dir, transform=transform)
    num_landmarks = len(dataset.labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = LandmarkNet(num_landmarks, backbone=args.backbone)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if args.loss == 'mse':
        criterion = torch.nn.MSELoss(reduction='none')
    else:
        criterion = torch.nn.SmoothL1Loss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for imgs, coords_px, mask, width, height in loader:
            imgs = imgs.to(device)
            mask = mask.to(device).unsqueeze(-1)
            wh = torch.stack([width, height], dim=-1).to(device).unsqueeze(1)
            coords = coords_px.to(device) / wh
            preds = model(imgs)
            loss = criterion(preds, coords)
            loss = (loss * mask).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        avg_loss = total_loss / len(dataset)
        print(f'Epoch {epoch+1}: train_loss={avg_loss:.4f}')

    # evaluation on training set
    model.eval()
    rmse = 0
    count = 0
    with torch.no_grad():
        for imgs, coords_px, mask, width, height in loader:
            imgs = imgs.to(device)
            mask = mask.to(device).unsqueeze(-1)
            wh = torch.stack([width, height], dim=-1).to(device).unsqueeze(1)
            coords = coords_px.to(device) / wh
            preds = model(imgs)
            diff = ((preds - coords) ** 2 * mask).sum(dim=-1)
            rmse += torch.sqrt(diff).sum().item()
            count += mask.sum().item()
    rmse = rmse / count if count > 0 else 0
    print(f"RMSE: {rmse:.6f}")

    # Save trained weights for inference
    torch.save(model.state_dict(), args.out)
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()
