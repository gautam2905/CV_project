import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from src.data.dataset import VesuviusDataset, VesuviusValDataset
from src.data.transforms import build_train_transforms
from src.models.unet3d import UNet3D
from src.models.losses import build_loss
from src.training.metrics import SegmentationMetrics
from src.utils.utils import get_gaussian_3d, set_seed, count_parameters


class Trainer:
    def __init__(self, cfg, local_rank):
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = dist.get_world_size()
        self.is_main = local_rank == 0
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)

        set_seed(42 + local_rank)

        self._build_model()
        self._build_datasets()
        self._build_optimizer()
        self._build_loss()
        self._setup_amp()
        self._setup_wandb()
        self._setup_checkpointing()

        self.start_epoch = 0
        self.best_surface_dice = 0.0
        if cfg["checkpoint"]["resume_from"]:
            self._load_checkpoint(cfg["checkpoint"]["resume_from"])

    def _build_model(self):
        mcfg = self.cfg["model"]
        channels = [mcfg["base_channels"] * m for m in mcfg["channel_multipliers"]]
        self.model = UNet3D(
            in_channels=mcfg["in_channels"],
            num_classes=mcfg["num_classes"],
            base_channels=mcfg["base_channels"],
            channel_multipliers=mcfg["channel_multipliers"],
            num_conv_per_stage=mcfg["num_conv_per_stage"],
            deep_supervision=mcfg["deep_supervision"],
        ).to(self.device)

        self.model = DDP(self.model, device_ids=[self.local_rank],
                         find_unused_parameters=True)

        if self.is_main:
            print(f"Model parameters: {count_parameters(self.model):,}")

    def _build_datasets(self):
        dcfg = self.cfg["data"]
        transform = build_train_transforms(self.cfg)

        train_ds = VesuviusDataset(
            data_root=dcfg["data_root"],
            split="train",
            val_scroll_ids=dcfg["val_scroll_ids"],
            patch_size=dcfg["patch_size"],
            transform=transform,
            patches_per_volume=dcfg["patches_per_volume"],
            surface_bias_prob=self.cfg["augmentation"].get("surface_bias_prob", 0.5),
        )

        self.val_ds = VesuviusValDataset(
            data_root=dcfg["data_root"],
            val_scroll_ids=dcfg["val_scroll_ids"],
        )

        self.train_sampler = DistributedSampler(train_ds, shuffle=True)
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg["training"]["batch_size_per_gpu"],
            sampler=self.train_sampler,
            num_workers=dcfg["num_workers"],
            pin_memory=dcfg["pin_memory"],
            persistent_workers=True,
            drop_last=True,
        )

        if self.is_main:
            print(f"Train: {len(train_ds)} patches ({len(train_ds) // dcfg['patches_per_volume']} volumes)")
            print(f"Val: {len(self.val_ds)} volumes")

    def _build_optimizer(self):
        tcfg = self.cfg["training"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=tcfg["lr"],
            weight_decay=tcfg["weight_decay"],
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=tcfg["T_0"],
            T_mult=tcfg["T_mult"],
            eta_min=tcfg["eta_min"],
        )
        self.warmup_epochs = tcfg["warmup_epochs"]
        self.base_lr = tcfg["lr"]

    def _build_loss(self):
        self.loss_fn = build_loss(self.cfg).to(self.device)

    def _setup_amp(self):
        self.use_amp = self.cfg["training"]["amp"]
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

    def _setup_wandb(self):
        self.wandb_run = None
        if self.is_main:
            try:
                import wandb
                os.environ.setdefault(
                    "WANDB_API_KEY",
                    "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",
                )
                self.wandb_run = wandb.init(
                    project=self.cfg["logging"]["wandb_project"],
                    config=self.cfg,
                    name=f"unet3d_patch{self.cfg['data']['patch_size']}_ep{self.cfg['training']['epochs']}",
                )
            except Exception as e:
                print(f"wandb init failed: {e}. Training without wandb.")

    def _setup_checkpointing(self):
        self.ckpt_dir = Path(self.cfg["checkpoint"]["save_dir"])
        if self.is_main:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _get_lr(self, epoch):
        if epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        return self.scheduler.get_last_lr()[0]

    def _warmup_lr(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def train(self):
        tcfg = self.cfg["training"]
        lcfg = self.cfg["logging"]
        total_epochs = tcfg["epochs"]

        if self.is_main:
            print(f"Starting training for {total_epochs} epochs on {self.world_size} GPUs")

        epoch_pbar = tqdm(range(self.start_epoch, total_epochs),
                          desc="Training", disable=not self.is_main,
                          initial=self.start_epoch, total=total_epochs)

        for epoch in epoch_pbar:
            self._warmup_lr(epoch)
            train_loss = self._train_one_epoch(epoch)

            if epoch >= self.warmup_epochs:
                self.scheduler.step(epoch - self.warmup_epochs)

            if self.is_main:
                current_lr = self.optimizer.param_groups[0]["lr"]
                epoch_pbar.set_postfix(loss=f"{train_loss:.4f}", lr=f"{current_lr:.2e}",
                                       best_sd=f"{self.best_surface_dice:.4f}")
                log_dict = {
                    "train/loss": train_loss,
                    "train/lr": current_lr,
                    "train/epoch": epoch,
                }
                if self.wandb_run:
                    import wandb
                    wandb.log(log_dict, step=epoch)

            if (epoch + 1) % lcfg["val_every_n_epochs"] == 0:
                val_results = self._validate(epoch)
                if self.is_main and val_results:
                    sd = val_results.get("surface_dice", 0.0)
                    tqdm.write(f"  [Epoch {epoch+1}] Val surface_dice={sd:.4f} "
                               f"mean_dice={val_results['mean_dice']:.4f} "
                               f"voxel_acc={val_results['voxel_accuracy']:.4f}")
                    if sd > self.best_surface_dice:
                        self.best_surface_dice = sd
                        self._save_checkpoint(epoch, is_best=True)

            if self.is_main and (epoch + 1) % lcfg["save_every_n_epochs"] == 0:
                self._save_checkpoint(epoch)

        if self.is_main:
            self._save_checkpoint(total_epochs - 1, tag="final")
            if self.wandb_run:
                import wandb
                wandb.finish()

    def _train_one_epoch(self, epoch):
        self.model.train()
        self.train_sampler.set_epoch(epoch)

        total_loss = 0.0
        num_batches = 0

        batch_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}",
                          leave=False, disable=not self.is_main)
        for batch in batch_pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["training"]["grad_clip"])
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(num_batches, 1)

        loss_tensor = torch.tensor([avg_loss], device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        return loss_tensor.item()

    @torch.no_grad()
    def _validate(self, epoch):
        if not self.is_main:
            dist.barrier()
            return None

        self.model.eval()
        metrics = SegmentationMetrics(
            num_classes=self.cfg["data"]["num_classes"],
            class_names=["air", "surface", "papyrus"],
        )

        patch_size = self.cfg["data"]["patch_size"]
        num_to_validate = min(len(self.val_ds), 20)

        for i in tqdm(range(num_to_validate), desc="Validating", leave=False):
            sample = self.val_ds[i]
            volume = sample["image"].unsqueeze(0).to(self.device)
            label = sample["label"]

            pred = self._sliding_window_inference(volume, patch_size, overlap=0.5)
            metrics.update(pred.cpu(), label)

        results = metrics.compute()

        if self.wandb_run:
            import wandb
            log_dict = {
                "val/mean_dice": results["mean_dice"],
                "val/surface_dice": results["surface_dice"],
                "val/surface_iou": results["surface_iou"],
                "val/voxel_accuracy": results["voxel_accuracy"],
            }
            for cls_name, dice_val in results["dice_per_class"].items():
                log_dict[f"val/dice_{cls_name}"] = dice_val
            wandb.log(log_dict, step=epoch)

        dist.barrier()
        return results

    def _sliding_window_inference(self, volume, patch_size, overlap=0.5):
        step = int(patch_size * (1 - overlap))
        D, H, W = volume.shape[2:]
        num_classes = self.cfg["data"]["num_classes"]

        gaussian = get_gaussian_3d(patch_size).to(self.device)

        output_sum = torch.zeros(num_classes, D, H, W, device=self.device)
        weight_sum = torch.zeros(1, D, H, W, device=self.device)

        d_starts = list(range(0, max(D - patch_size, 0) + 1, step))
        h_starts = list(range(0, max(H - patch_size, 0) + 1, step))
        w_starts = list(range(0, max(W - patch_size, 0) + 1, step))

        if d_starts[-1] + patch_size < D:
            d_starts.append(D - patch_size)
        if h_starts[-1] + patch_size < H:
            h_starts.append(H - patch_size)
        if w_starts[-1] + patch_size < W:
            w_starts.append(W - patch_size)

        with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
            for d in d_starts:
                for h in h_starts:
                    for w in w_starts:
                        patch = volume[:, :, d:d+patch_size, h:h+patch_size, w:w+patch_size]
                        logits = self.model(patch)["logits"]
                        probs = torch.softmax(logits.float(), dim=1)[0]

                        output_sum[:, d:d+patch_size, h:h+patch_size, w:w+patch_size] += probs * gaussian
                        weight_sum[:, d:d+patch_size, h:h+patch_size, w:w+patch_size] += gaussian

        prediction = (output_sum / weight_sum.clamp(min=1e-8)).argmax(dim=0)
        return prediction

    def _save_checkpoint(self, epoch, is_best=False, tag=None):
        state = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_surface_dice": self.best_surface_dice,
            "config": self.cfg,
        }
        if tag:
            path = self.ckpt_dir / f"checkpoint_{tag}.pth"
        else:
            path = self.ckpt_dir / f"checkpoint_epoch{epoch+1:04d}.pth"
        torch.save(state, path)
        if is_best:
            best_path = self.ckpt_dir / "checkpoint_best.pth"
            torch.save(state, best_path)
            print(f"  New best model saved (surface_dice={self.best_surface_dice:.4f})")

    def _load_checkpoint(self, path):
        if self.is_main:
            print(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.module.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = ckpt["epoch"]
        self.best_surface_dice = ckpt.get("best_surface_dice", 0.0)
