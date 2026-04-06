import numpy as np
import torch


class SegmentationMetrics:
    def __init__(self, num_classes=3, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        pred = pred.flatten()
        target = target.flatten()

        mask = (target >= 0) & (target < self.num_classes)
        pred = pred[mask]
        target = target[mask]

        indices = target * self.num_classes + pred
        bincount = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += bincount.reshape(self.num_classes, self.num_classes)

    def compute(self):
        cm = self.confusion_matrix
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        voxel_accuracy = tp.sum() / (cm.sum() + 1e-8)

        results = {
            "voxel_accuracy": float(voxel_accuracy),
            "mean_dice": float(dice.mean()),
            "mean_iou": float(iou.mean()),
            "dice_per_class": {self.class_names[i]: float(dice[i]) for i in range(self.num_classes)},
            "iou_per_class": {self.class_names[i]: float(iou[i]) for i in range(self.num_classes)},
            "precision_per_class": {self.class_names[i]: float(precision[i]) for i in range(self.num_classes)},
            "recall_per_class": {self.class_names[i]: float(recall[i]) for i in range(self.num_classes)},
        }

        if self.num_classes >= 2:
            results["surface_dice"] = float(dice[1])
            results["surface_iou"] = float(iou[1])

        return results

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
