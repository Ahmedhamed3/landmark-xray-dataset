import torch

def rmse_loss(preds, targets, mask=None):
    diff = preds - targets
    if mask is not None:
        diff = diff * mask.unsqueeze(-1)
        count = mask.sum()
    else:
        count = torch.numel(diff) / 2
    mse = (diff ** 2).sum() / max(count, 1e-6)
    return torch.sqrt(mse)
