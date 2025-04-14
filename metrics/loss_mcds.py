import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

def _to_one_hot(y, num_classes, device):

    y = torch.squeeze(y)
    y = torch.tensor(y, dtype=torch.int64)
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    y_tensor = y_tensor.to(device)
    zeros = torch.zeros(*y.size(), num_classes, dtype=torch.float32, device=device)

    return zeros.scatter(scatter_dim, y_tensor, 1)

ALPHA = 0.3
BETA = 0.7

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, logits, true, smooth=1, alpha=ALPHA, beta=BETA, eps = 1e-7):
        true_1_hot = true.permute(0, 4, 1, 2, 3).to(dtype=logits.dtype)
        logits = logits.permute(0, 4, 1, 2, 3)
        probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        fps = torch.sum(probas * (1 - true_1_hot), dims)
        fns = torch.sum((1 - probas) * true_1_hot, dims)
        num = intersection
        denom = intersection + (alpha * fps) + (beta * fns)
        tversky_loss = (num / (denom + eps)).mean()

        return (1 - tversky_loss)
    
criterion_denoise = torch.nn.MSELoss() ## denoising loss
criterion_seg = TverskyLoss()
    
def dice_denoise_combined(out_denoise, target_denoise, out_segment, target_segment):
    
    with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision
        loss_den = criterion_denoise(out_denoise.float(), target_denoise)
        target_segment = _to_one_hot(target_segment, 6, target_segment.device)
        out_segment = out_segment.permute(0, 2, 3, 4, 1).float()
        
        loss_seg = criterion_seg(out_segment, target_segment)
    
    loss_combined = loss_den.float() + loss_seg.float()
    
    return  loss_combined + loss_seg, loss_seg.item(), loss_den.item()
    