import torch
from torch.cuda.amp import autocast


def calculate_per_channel_dice_score(pred, true, cuda=False):
    if cuda:
        with autocast():
            mid = (pred.max() + pred.min()) / 2
            pred[pred > mid] = 1.0
            pred[pred <= mid] = 0.0

            mid = (true.max() + true.min()) / 2
            true[true > mid] = 1.0
            true[true <= mid] = 0.0

            smooth = 0.000001
            i = torch.sum(true)
            j = torch.sum(pred)
            intersection = torch.sum(true * pred)
            dice = (2. * intersection + smooth) / (i + j + smooth)
            return dice
    else:
        mid = (pred.max() + pred.min()) / 2
        pred[pred > mid] = 1.0
        pred[pred <= mid] = 0.0

        mid = (true.max() + true.min()) / 2
        true[true > mid] = 1.0
        true[true <= mid] = 0.0

        smooth = 0.000001
        i = torch.sum(true)
        j = torch.sum(pred)
        intersection = torch.sum(true * pred)
        dice = (2. * intersection + smooth) / (i + j + smooth)
        return dice
