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
    current = None

    with open(log_path) as f:
        for line in f:
            line = line.strip()

            ep_match = re.search(r"Epoch (\d+)\s*$", line)
            if ep_match:
                if current is not None:
                    entries.append(current)
                current = {"epoch": int(ep_match.group(1))}
                continue

            if current is None:
                continue

            tl = re.search(r"train_loss\s+([-\d.eE+]+)", line)
            if tl:
                current["train_loss"] = float(tl.group(1))

            vl = re.search(r"val_loss\s+([-\d.eE+]+)", line)
            if vl:
                current["val_loss"] = float(vl.group(1))

            pd = re.search(r"Pseudo dice\s+\[([^\]]+)\]", line)
            if pd:
                vals = [float(x.replace("np.float32(", "").replace(")", "").strip())
                        for x in pd.group(1).split(",")]
                current["pseudo_dice"] = vals

            lr = re.search(r"Current learning rate:\s+([-\d.eE+]+)", line)
            if lr:
                current["lr"] = float(lr.group(1))

            et = re.search(r"Epoch time:\s+([-\d.]+)\s*s", line)
            if et:
                current["epoch_time"] = float(et.group(1))

    if current is not None:
        entries.append(current)

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
                if "train_loss" not in entry:
                    continue
                last_epoch = entry["epoch"]

                log_dict = {"epoch": entry["epoch"]}
                if "train_loss" in entry:
                    log_dict["train/loss"] = entry["train_loss"]
                if "val_loss" in entry:
                    log_dict["val/loss"] = entry["val_loss"]
                if "lr" in entry:
                    log_dict["train/lr"] = entry["lr"]
                if "epoch_time" in entry:
                    log_dict["train/epoch_time_s"] = entry["epoch_time"]
                if "pseudo_dice" in entry:
                    for i, v in enumerate(entry["pseudo_dice"]):
                        cls = ["background", "surface"][i] if i < 2 else f"class{i}"
                        log_dict[f"val/pseudo_dice_{cls}"] = v
                    if len(entry["pseudo_dice"]) >= 2:
                        log_dict["val/surface_dice"] = entry["pseudo_dice"][1]

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
