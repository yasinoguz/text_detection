import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    """
    Kombine Binary Cross Entropy ve Dice Loss.
    Segmentasyon problemleri için çok etkili bir loss fonksiyonu.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        """
        preds: modelden gelen ham skorlar (logits), shape: [B, 1, H, W]
        targets: gerçek maskeler, shape: [B, 1, H, W], float (0 veya 1)

        Returns:
            toplam loss değeri
        """
        # BCE Loss
        bce = self.bce_loss(preds, targets)

        # Sigmoid uygulanmış tahminler
        preds = torch.sigmoid(preds)

        # Dice Loss hesaplama
        intersection = (preds * targets).sum(dim=(2,3))
        union = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3))
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score

        # Ortalama batch üzerinden
        dice_loss = dice_loss.mean()

        # Toplam loss
        total_loss = self.bce_weight * bce + self.dice_weight * dice_loss
        return total_loss


# Alternatif tekil Dice Loss fonksiyonu (istersen bağımsız da kullanabilirsin)

def dice_loss(preds, targets, smooth=1e-6):
    """
    Dice loss hesaplar.
    preds: logits değil, sigmoid uygulanmış olmalı.
    """
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice


# Alternatif tekil BCE Loss

def bce_loss(preds, targets):
    """
    BCEWithLogitsLoss uygular.
    preds: logits olmalı.
    """
    criterion = nn.BCEWithLogitsLoss()
    return criterion(preds, targets)
