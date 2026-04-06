#!/usr/bin/env python3
"""
Train 3D U-Net for Vesuvius Challenge Surface Detection.

Usage:
    torchrun --nproc_per_node=8 scripts/train_unet3d.py --config configs/unet3d_config.yaml
    torchrun --nproc_per_node=2 scripts/train_unet3d.py --config configs/unet3d_config.yaml  # quick test
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.utils import load_config, set_seed
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train 3D U-Net")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])

    cfg = load_config(args.config)
    if args.resume:
        cfg["checkpoint"]["resume_from"] = args.resume

    trainer = Trainer(cfg, local_rank)
    trainer.train()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
