#!/bin/bash
# ============================================================
# Vesuvius Challenge — Full Training Pipeline
#
# Runs everything end-to-end:
#   1. nnU-Net data setup + preprocessing
#   2. 3D U-Net training (8 GPUs, DDP)
#   3. nnU-Net training (8 models in parallel)
#
# Usage:
#   conda activate dikshant
#   nohup bash run_all.sh > logs/run_all.log 2>&1 &
#   tail -f logs/run_all.log
# ============================================================

set -e

PROJECT_DIR="/raid/home/vikram_govt/Dikshant/gautam/cv"
CONDA_BIN="/raid/home/vikram_govt/anaconda3/envs/dikshant/bin"
PYTHON="${CONDA_BIN}/python"
TORCHRUN="${CONDA_BIN}/torchrun"
NNUNET_TRAIN="${CONDA_BIN}/nnUNetv2_train"
NNUNET_PP="${CONDA_BIN}/nnUNetv2_plan_and_preprocess"

cd "${PROJECT_DIR}"
mkdir -p logs checkpoints

# --- Environment ---
export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_DIR}/nnUNet_data/nnUNet_results"
export nnUNet_USE_BLOSC2=1
export nnUNet_compile=false
export WANDB_API_KEY="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"
export OMP_NUM_THREADS=2

DATASET=200
CONFIG="3d_fullres"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "============================================================"
echo "  Vesuvius Training Pipeline"
echo "  Started: $(timestamp)"
echo "============================================================"

# ============================================================
# PHASE 1: nnU-Net data conversion + preprocessing
# ============================================================
echo ""
echo "[$(timestamp)] PHASE 1: nnU-Net setup"
echo "------------------------------------------------------------"

# Convert data to nnU-Net format
${PYTHON} -c "
import sys, os
sys.path.insert(0, '.')
from src.nnunet.nnunet_config import setup_nnunet_env
from src.nnunet.data_converter import convert_to_nnunet_format
setup_nnunet_env()
convert_to_nnunet_format()
"

echo "[$(timestamp)] Data conversion done. Starting plan+preprocess..."

# Default planner (needed as base for all plans)
echo "[$(timestamp)] Planning with default planner..."
${NNUNET_PP} -d ${DATASET} --verify_dataset_integrity -c ${CONFIG}

echo "[$(timestamp)] Default plan done. Planning with ResEncL..."
${NNUNET_PP} -d ${DATASET} -c ${CONFIG} -pl nnUNetPlannerResEncL

echo "[$(timestamp)] ResEncL plan done. Planning with ResEncXL..."
${NNUNET_PP} -d ${DATASET} -c ${CONFIG} -pl nnUNetPlannerResEncXL

echo "[$(timestamp)] ResEncXL plan done. Planning with ResEncM..."
${NNUNET_PP} -d ${DATASET} -c ${CONFIG} -pl nnUNetPlannerResEncM

echo "[$(timestamp)] PHASE 1 COMPLETE: All plans + preprocessing done."

# ============================================================
# PHASE 2: Launch ALL training in parallel
# ============================================================
echo ""
echo "[$(timestamp)] PHASE 2: Launching training jobs"
echo "------------------------------------------------------------"

# --- 3D U-Net on GPUs that nnU-Net won't use (we'll use 4 GPUs for UNet) ---
echo "[$(timestamp)] Starting 3D U-Net (4 GPUs: 0-3)..."
CUDA_VISIBLE_DEVICES=0,1,2,3 ${TORCHRUN} --nproc_per_node=4 \
    scripts/train_unet3d.py --config configs/unet3d_config.yaml \
    > logs/unet3d_train.log 2>&1 &
PID_UNET=$!
echo "  3D U-Net PID: ${PID_UNET}"

# --- nnU-Net: primary models on GPUs 4-7 ---
echo "[$(timestamp)] Starting nnU-Net LPlans 4000ep (GPU 4)..."
CUDA_VISIBLE_DEVICES=4 ${NNUNET_TRAIN} ${DATASET} ${CONFIG} all \
    -p nnUNetResEncUNetLPlans \
    -tr nnUNetTrainer_4000epochs \
    --npz \
    > logs/nnunet_LPlans_4000_all.log 2>&1 &
PID_NNUNET_L=$!
echo "  LPlans PID: ${PID_NNUNET_L}"

echo "[$(timestamp)] Starting nnU-Net XLPlans 250ep (GPU 5)..."
CUDA_VISIBLE_DEVICES=5 ${NNUNET_TRAIN} ${DATASET} ${CONFIG} all \
    -p nnUNetResEncUNetXLPlans \
    -tr nnUNetTrainer_250epochs \
    --npz \
    > logs/nnunet_XLPlans_250_all.log 2>&1 &
PID_NNUNET_XL=$!
echo "  XLPlans PID: ${PID_NNUNET_XL}"

echo "[$(timestamp)] Starting nnU-Net MPlans 4000ep (GPU 6)..."
CUDA_VISIBLE_DEVICES=6 ${NNUNET_TRAIN} ${DATASET} ${CONFIG} all \
    -p nnUNetResEncUNetMPlans \
    -tr nnUNetTrainer_4000epochs \
    --npz \
    > logs/nnunet_MPlans_4000_all.log 2>&1 &
PID_NNUNET_M=$!
echo "  MPlans PID: ${PID_NNUNET_M}"

echo "[$(timestamp)] Starting nnU-Net default plans 1000ep (GPU 7)..."
CUDA_VISIBLE_DEVICES=7 ${NNUNET_TRAIN} ${DATASET} ${CONFIG} all \
    --npz \
    > logs/nnunet_default_1000_all.log 2>&1 &
PID_NNUNET_DEF=$!
echo "  Default PID: ${PID_NNUNET_DEF}"

echo ""
echo "[$(timestamp)] All training jobs launched!"
echo "  3D U-Net (GPUs 0-3):  PID ${PID_UNET}"
echo "  nnU-Net LPlans (GPU 4):  PID ${PID_NNUNET_L}"
echo "  nnU-Net XLPlans (GPU 5): PID ${PID_NNUNET_XL}"
echo "  nnU-Net MPlans (GPU 6):  PID ${PID_NNUNET_M}"
echo "  nnU-Net Default (GPU 7): PID ${PID_NNUNET_DEF}"
echo ""
echo "Waiting for all jobs to finish..."

# ============================================================
# PHASE 3: Wait and report
# ============================================================

# Wait for each job and report status
wait ${PID_UNET} 2>/dev/null && echo "[$(timestamp)] 3D U-Net DONE (success)" || echo "[$(timestamp)] 3D U-Net DONE (exit code: $?)"
wait ${PID_NNUNET_XL} 2>/dev/null && echo "[$(timestamp)] nnU-Net XLPlans DONE (success)" || echo "[$(timestamp)] nnU-Net XLPlans DONE (exit code: $?)"
wait ${PID_NNUNET_DEF} 2>/dev/null && echo "[$(timestamp)] nnU-Net Default DONE (success)" || echo "[$(timestamp)] nnU-Net Default DONE (exit code: $?)"
wait ${PID_NNUNET_M} 2>/dev/null && echo "[$(timestamp)] nnU-Net MPlans DONE (success)" || echo "[$(timestamp)] nnU-Net MPlans DONE (exit code: $?)"
wait ${PID_NNUNET_L} 2>/dev/null && echo "[$(timestamp)] nnU-Net LPlans DONE (success)" || echo "[$(timestamp)] nnU-Net LPlans DONE (exit code: $?)"

echo ""
echo "============================================================"
echo "  ALL TRAINING COMPLETE"
echo "  Finished: $(timestamp)"
echo "============================================================"
echo ""
echo "Checkpoints:"
echo "  3D U-Net:     ${PROJECT_DIR}/checkpoints/"
echo "  nnU-Net:      ${PROJECT_DIR}/nnUNet_data/nnUNet_results/"
echo ""

# List final checkpoints
echo "--- 3D U-Net checkpoints ---"
ls -lh ${PROJECT_DIR}/checkpoints/*.pth 2>/dev/null || echo "  (none found)"
echo ""
echo "--- nnU-Net results ---"
find ${PROJECT_DIR}/nnUNet_data/nnUNet_results/ -name "checkpoint_final.pth" -o -name "checkpoint_best.pth" 2>/dev/null || echo "  (none found)"
