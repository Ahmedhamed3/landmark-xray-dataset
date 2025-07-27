import torch
import torch.nn as nn
from torchvision import models

class LandmarkNet(nn.Module):
    def __init__(self, num_landmarks, backbone='resnet18', pretrained=False):
        super().__init__()
        if backbone == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        self.head = nn.Linear(in_features, 2 * num_landmarks)

    def forward(self, x):
        feats = self.backbone(x)
        out = self.head(feats)
        return out.view(out.size(0), -1, 2)
