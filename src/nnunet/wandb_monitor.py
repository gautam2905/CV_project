#!/usr/bin/env python3
"""
Monitor nnU-Net training logs and stream metrics to wandb.

Usage:
    python -m src.nnunet.wandb_monitor \
        --log_dir nnUNet_data/nnUNet_results/Dataset200_VesuviusSurface/... \
        --project vesuvius-surface-nnunet \
        --name LPlans_4000ep
"""
import argparse
import os
import re
import time
from pathlib import Path


def parse_training_log(log_path):
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            ep_match = re.search(r"Epoch (\d+)", line)
            if not ep_match:
                continue
            epoch = int(ep_match.group(1))
            entry = {"epoch": epoch}

            for key, pattern in [
                ("train_loss", r"train_loss[:\s]+([-\d.eE+]+)"),
                ("val_loss", r"val_loss[:\s]+([-\d.eE+]+)"),
                ("pseudo_dice", r"Pseudo dice[:\s]+\[([^\]]+)\]"),
                ("ema_pseudo_dice", r"EMA pseudo Dice[:\s]+\[([^\]]+)\]"),
            ]:
                m = re.search(pattern, line)
                if m:
                    if key in ("pseudo_dice", "ema_pseudo_dice"):
                        vals = [float(x.strip()) for x in m.group(1).split(",")]
                        entry[key] = vals
                    else:
                        entry[key] = float(m.group(1))
            entries.append(entry)
    return entries


def monitor(log_dir, project, name, poll_interval=30):
    import wandb
    os.environ.setdefault(
        "WANDB_API_KEY",
        "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",
    )

    log_dir = Path(log_dir)
    run = wandb.init(project=project, name=name)

    last_epoch = -1
    print(f"Monitoring {log_dir} for training logs...")

    while True:
        log_files = sorted(log_dir.glob("training_log*.txt"))
        if not log_files:
            time.sleep(poll_interval)
            continue

        for log_file in log_files:
            entries = parse_training_log(log_file)
            for entry in entries:
                if entry["epoch"] <= last_epoch:
                    continue
                last_epoch = entry["epoch"]

                log_dict = {"epoch": entry["epoch"]}
                if "train_loss" in entry:
                    log_dict["train/loss"] = entry["train_loss"]
                if "val_loss" in entry:
                    log_dict["val/loss"] = entry["val_loss"]
                if "pseudo_dice" in entry:
                    for i, v in enumerate(entry["pseudo_dice"]):
                        log_dict[f"val/pseudo_dice_class{i}"] = v
                if "ema_pseudo_dice" in entry:
                    for i, v in enumerate(entry["ema_pseudo_dice"]):
                        log_dict[f"val/ema_pseudo_dice_class{i}"] = v

                wandb.log(log_dict, step=entry["epoch"])

        time.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--project", default="vesuvius-surface-nnunet")
    parser.add_argument("--name", default="nnunet_training")
    parser.add_argument("--poll_interval", type=int, default=30)
    args = parser.parse_args()
    monitor(args.log_dir, args.project, args.name, args.poll_interval)
