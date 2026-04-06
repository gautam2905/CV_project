#!/usr/bin/env python3
"""
Setup nnU-Net: convert data to nnU-Net format and run plan+preprocess.

Usage:
    conda activate dikshant
    pip install nnunetv2  # if not already installed
    python scripts/setup_nnunet.py
"""
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nnunet.nnunet_config import setup_nnunet_env, DATASET_ID

CONDA_BIN = os.path.join(os.path.dirname(sys.executable))


def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  {cmd}")
    print(f"{'='*60}\n")
    env = os.environ.copy()
    result = subprocess.run(cmd, shell=True, env=env)
    if result.returncode != 0:
        print(f"ERROR: {desc} exited with code {result.returncode}")
        return result.returncode
    print(f"OK: {desc}")
    return 0


def main():
    setup_nnunet_env()

    nnunet_pp = os.path.join(CONDA_BIN, "nnUNetv2_plan_and_preprocess")
    if not os.path.exists(nnunet_pp):
        print(f"ERROR: nnUNetv2_plan_and_preprocess not found at {nnunet_pp}")
        print("Install nnunetv2 first: pip install nnunetv2")
        sys.exit(1)

    print("\n--- Step 1: Convert data to nnU-Net format ---")
    from src.nnunet.data_converter import convert_to_nnunet_format
    convert_to_nnunet_format()

    print("\n--- Step 2: Verify dataset integrity + plan with default planner ---")
    run_cmd(
        f"{nnunet_pp} -d {DATASET_ID} --verify_dataset_integrity -c 3d_fullres",
        "Verify + plan with default planner"
    )

    print("\n--- Step 3: Plan with ResEncL planner ---")
    run_cmd(
        f"{nnunet_pp} -d {DATASET_ID} -c 3d_fullres -pl nnUNetPlannerResEncL",
        "Plan + preprocess with ResEncUNet L Plans"
    )

    print("\n--- Step 4: Plan with ResEncXL planner ---")
    run_cmd(
        f"{nnunet_pp} -d {DATASET_ID} -c 3d_fullres -pl nnUNetPlannerResEncXL",
        "Plan + preprocess with ResEncUNet XL Plans"
    )

    print("\n--- Setup complete! ---")
    print("You can now run training with scripts/train_nnunet.sh")


if __name__ == "__main__":
    main()
