#!/bin/bash
# ============================================================
# Vesuvius Challenge — FINAL DELIVERABLE (winner-aligned)
#
# 4-model ensemble mirroring the 1st-place Kaggle solution:
#   M1 : existing MPlans 4000ep at patch 128  (no new training)
#   M2 : fine-tune M1 at patch 192, 250 ep     (4 GPUs)
#   M3 : fine-tune M2 at patch 256, 250 ep     (8 GPUs, starts after M2)
#   M4 : from-scratch patch 192, 4000 ep        (4 GPUs, runs in parallel)
#
# Usage:
#   nohup bash run_final.sh > logs/run_final.log 2>&1 &
#   tail -f logs/run_final.log
# ============================================================
set -e

PROJECT_DIR="/raid/home/vikram_govt/Dikshant/gautam/cv"
CONDA_BIN="/raid/home/vikram_govt/anaconda3/envs/vesuvius/bin"
PYTHON="${CONDA_BIN}/python"
NNUNET_TRAIN="${CONDA_BIN}/nnUNetv2_train"

cd "${PROJECT_DIR}"
mkdir -p logs checkpoints

export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_DIR}/nnUNet_data/nnUNet_results"
export nnUNet_USE_BLOSC2=1
export nnUNet_compile=false
export WANDB_API_KEY="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"
export OMP_NUM_THREADS=2

DATASET=200
CONFIG="3d_fullres"
FOLD=all

M1_CKPT="${nnUNet_results}/Dataset200_VesuviusSurface/nnUNetTrainer_4000epochs__nnUNetResEncUNetMPlans__3d_fullres/fold_all/checkpoint_latest.pth"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
echo "============================================================"
echo "  Vesuvius Final Deliverable — Winner-Aligned"
echo "  Started: $(timestamp)"
echo "============================================================"

# --- Ensure custom plans files exist ---
if [ ! -f "${nnUNet_preprocessed}/Dataset200_VesuviusSurface/nnUNetResEncUNetMPlans_patch192.json" ]; then
    echo "[$(timestamp)] Generating custom plans (192/256)..."
    ${PYTHON} scripts/make_custom_plans.py
fi

# --- Sanity: M1 checkpoint exists ---
if [ ! -f "${M1_CKPT}" ]; then
    echo "ERROR: M1 checkpoint not found at ${M1_CKPT}" >&2
    exit 1
fi
echo "[$(timestamp)] M1 checkpoint: ${M1_CKPT}"

# ============================================================
# STAGE 1: M2 (4 GPUs: 0-3) + M4 (4 GPUs: 4-7) in parallel
# ============================================================
echo ""
echo "[$(timestamp)] STAGE 1: launching M2 (192-ft) on GPUs 0-3 and M4 (192-scratch) on GPUs 4-7"

CUDA_VISIBLE_DEVICES=0,1,2,3 ${NNUNET_TRAIN} ${DATASET} ${CONFIG} ${FOLD} \
    -p nnUNetResEncUNetMPlans_patch192 \
    -tr nnUNetTrainer_250epochs \
    -pretrained_weights ${M1_CKPT} \
    -num_gpus 4 \
    --npz \
    > logs/final_M2_192ft.log 2>&1 &
PID_M2=$!
echo "  M2 PID ${PID_M2} (GPUs 0-3)"

CUDA_VISIBLE_DEVICES=4,5,6,7 ${NNUNET_TRAIN} ${DATASET} ${CONFIG} ${FOLD} \
    -p nnUNetResEncUNetMPlans_patch192 \
    -tr nnUNetTrainer_4000epochs \
    -num_gpus 4 \
    --npz \
    > logs/final_M4_192scratch.log 2>&1 &
PID_M4=$!
echo "  M4 PID ${PID_M4} (GPUs 4-7)"

# wandb monitors
${PYTHON} -m src.nnunet.wandb_monitor \
    --log_dir ${nnUNet_results}/Dataset200_VesuviusSurface/nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans_patch192__3d_fullres/fold_all \
    --project vesuvius-surface-nnunet \
    --name M2_192ft_250ep \
    > logs/wandb_M2.log 2>&1 &
PID_WB_M2=$!

${PYTHON} -m src.nnunet.wandb_monitor \
    --log_dir ${nnUNet_results}/Dataset200_VesuviusSurface/nnUNetTrainer_4000epochs__nnUNetResEncUNetMPlans_patch192__3d_fullres/fold_all \
    --project vesuvius-surface-nnunet \
    --name M4_192scratch_4000ep \
    > logs/wandb_M4.log 2>&1 &
PID_WB_M4=$!

# Wait ONLY for M2 (M4 keeps running in background through Stage 2)
wait ${PID_M2} && echo "[$(timestamp)] M2 DONE" || echo "[$(timestamp)] M2 FAILED ($?)"
kill ${PID_WB_M2} 2>/dev/null || true

# ============================================================
# STAGE 2: M3 (fine-tune from M2) on GPUs 0-3 while M4 continues on 4-7
# ============================================================
M2_CKPT="${nnUNet_results}/Dataset200_VesuviusSurface/nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans_patch192__3d_fullres/fold_all/checkpoint_final.pth"
if [ ! -f "${M2_CKPT}" ]; then
    M2_CKPT="${nnUNet_results}/Dataset200_VesuviusSurface/nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans_patch192__3d_fullres/fold_all/checkpoint_latest.pth"
fi

echo ""
echo "[$(timestamp)] STAGE 2: launching M3 (256-ft from M2) on GPUs 0-1 (2-GPU DDP to fit 40GB VRAM)"
echo "  M2 checkpoint: ${M2_CKPT}"

CUDA_VISIBLE_DEVICES=0,1 ${NNUNET_TRAIN} ${DATASET} ${CONFIG} ${FOLD} \
    -p nnUNetResEncUNetMPlans_patch256 \
    -tr nnUNetTrainer_250epochs \
    -pretrained_weights ${M2_CKPT} \
    -num_gpus 2 \
    --npz \
    > logs/final_M3_256ft.log 2>&1 &
PID_M3=$!
echo "  M3 PID ${PID_M3} (GPUs 0-3)"

${PYTHON} -m src.nnunet.wandb_monitor \
    --log_dir ${nnUNet_results}/Dataset200_VesuviusSurface/nnUNetTrainer_250epochs__nnUNetResEncUNetMPlans_patch256__3d_fullres/fold_all \
    --project vesuvius-surface-nnunet \
    --name M3_256ft_250ep \
    > logs/wandb_M3.log 2>&1 &
PID_WB_M3=$!

# ============================================================
# STAGE 3: wait for both M3 and M4 to finish
# ============================================================
wait ${PID_M3} && echo "[$(timestamp)] M3 DONE" || echo "[$(timestamp)] M3 FAILED ($?)"
kill ${PID_WB_M3} 2>/dev/null || true

wait ${PID_M4} && echo "[$(timestamp)] M4 DONE" || echo "[$(timestamp)] M4 FAILED ($?)"
kill ${PID_WB_M4} 2>/dev/null || true

echo ""
echo "============================================================"
echo "  ALL TRAINING COMPLETE — $(timestamp)"
echo "============================================================"
find ${nnUNet_results}/Dataset200_VesuviusSurface -name "checkpoint_final.pth" -o -name "checkpoint_latest.pth"
echo ""
echo "Next: open notebooks/inference.ipynb to run 4-model ensemble + 5-step post-processing"
