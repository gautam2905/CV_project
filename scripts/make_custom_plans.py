#!/usr/bin/env python3
"""
Generate custom nnU-Net plans files for patch-size 192 and 256, derived from
nnUNetResEncUNetMPlans. The only changes are:

  * plans_name
  * configurations.3d_fullres.patch_size
  * configurations.3d_fullres.batch_size

Because the spacing / normalization / data_identifier are unchanged, the
existing preprocessed cache (nnUNetPlans_3d_fullres) is reused — no new
preprocessing needed.

Usage:
    python scripts/make_custom_plans.py
"""
import json
import os
from pathlib import Path

PREPROCESSED = Path(os.environ.get(
    "nnUNet_preprocessed",
    "/raid/home/vikram_govt/Dikshant/gautam/cv/nnUNet_data/nnUNet_preprocessed",
)) / "Dataset200_VesuviusSurface"

BASE = PREPROCESSED / "nnUNetResEncUNetMPlans.json"


def derive(patch: int, batch: int, new_name: str) -> Path:
    with open(BASE) as f:
        plans = json.load(f)
    plans["plans_name"] = new_name
    cfg = plans["configurations"]["3d_fullres"]
    cfg["patch_size"] = [patch, patch, patch]
    cfg["batch_size"] = batch
    # Keep data_identifier the same so preprocessed cache is reused
    out = PREPROCESSED / f"{new_name}.json"
    with open(out, "w") as f:
        json.dump(plans, f, indent=2)
    return out


if __name__ == "__main__":
    # batch_size must be >= num_gpus for nnU-Net DDP. We use 4 GPUs for 192
    # runs and 2 GPUs for 256 (tight VRAM at patch 256³).
    p192 = derive(192, 4, "nnUNetResEncUNetMPlans_patch192")
    p256 = derive(256, 2, "nnUNetResEncUNetMPlans_patch256")
    print(f"Wrote {p192}")
    print(f"Wrote {p256}")
