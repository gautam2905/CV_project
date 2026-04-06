import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes=3, class_weights=None, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, self.num_classes).permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = (probs + one_hot).sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if self.class_weights is not None:
            dice = dice * self.class_weights
            return 1.0 - dice.sum() / self.class_weights.sum()
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(self, num_classes=3, class_weights=None, smooth=1e-5,
                 dice_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.dice_loss = SoftDiceLoss(num_classes, class_weights, smooth)
        cw = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.ce_loss = nn.CrossEntropyLoss(weight=cw)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, targets):
        return (self.dice_weight * self.dice_loss(logits, targets) +
                self.ce_weight * self.ce_loss(logits, targets))


class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss, ds_weights=(1.0, 0.5, 0.25)):
        super().__init__()
        self.base_loss = base_loss
        total = sum(ds_weights)
        self.ds_weights = [w / total for w in ds_weights]

    def forward(self, outputs, targets):
        loss = self.ds_weights[0] * self.base_loss(outputs["logits"], targets)

        if "deep_supervision" in outputs:
            target_shape = targets.shape[1:]
            for w, ds_logits in zip(self.ds_weights[1:], outputs["deep_supervision"]):
                ds_up = F.interpolate(ds_logits, size=target_shape, mode="trilinear", align_corners=False)
                loss = loss + w * self.base_loss(ds_up, targets)

        return loss


def build_loss(cfg):
    loss_cfg = cfg["loss"]
    base_loss = DiceCELoss(
        num_classes=cfg["data"]["num_classes"],
        class_weights=loss_cfg.get("class_weights"),
        smooth=loss_cfg.get("smooth", 1e-5),
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        ce_weight=loss_cfg.get("ce_weight", 1.0),
    )
    if cfg["model"].get("deep_supervision", False):
        return DeepSupervisionLoss(base_loss)
    return base_loss
