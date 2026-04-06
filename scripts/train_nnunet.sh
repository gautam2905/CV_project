#!/bin/bash
# ============================================================
# nnU-Net Training for Vesuvius Surface Detection
#
# Usage:
#   conda activate dikshant
#   bash scripts/train_nnunet.sh
# ============================================================

set -e

CONDA_BIN="/raid/home/vikram_govt/anaconda3/envs/dikshant/bin"
NNUNET_TRAIN="${CONDA_BIN}/nnUNetv2_train"

# nnU-Net environment variables
export nnUNet_raw="/raid/home/vikram_govt/Dikshant/gautam/cv/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="/raid/home/vikram_govt/Dikshant/gautam/cv/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="/raid/home/vikram_govt/Dikshant/gautam/cv/nnUNet_data/nnUNet_results"
export nnUNet_USE_BLOSC2=1
export nnUNet_compile=false
export WANDB_API_KEY="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"

DATASET=200
CONFIG="3d_fullres"
LOGDIR="/raid/home/vikram_govt/Dikshant/gautam/cv/logs"
mkdir -p "${LOGDIR}"

echo "============================================================"
echo "  nnU-Net Training Launcher"
echo "  Dataset: ${DATASET}"
echo "  Config:  ${CONFIG}"
echo "  Binary:  ${NNUNET_TRAIN}"
echo "============================================================"

# --- PRIMARY MODEL: ResEncUNet L Plans, 4000 epochs, fold_all ---
echo "[GPU 0] Starting LPlans 4000 epochs (fold_all)..."
CUDA_VISIBLE_DEVICES=0 nohup ${NNUNET_TRAIN} ${DATASET} ${CONFIG} all \
    -p nnUNetResEncUNetLPlans \
    -tr nnUNetTrainer_4000epochs \
    --npz \
    > ${LOGDIR}/nnunet_LPlans_4000_all.log 2>&1 &
PID_L=$!
echo "  PID: $PID_L"

# --- SECONDARY MODEL: ResEncUNet XL Plans, 250 epochs, fold_all ---
echo "[GPU 1] Starting XLPlans 250 epochs (fold_all)..."
CUDA_VISIBLE_DEVICES=1 nohup ${NNUNET_TRAIN} ${DATASET} ${CONFIG} all \
    -p nnUNetResEncUNetXLPlans \
    -tr nnUNetTrainer_250epochs \
    --npz \
    > ${LOGDIR}/nnunet_XLPlans_250_all.log 2>&1 &
PID_XL=$!
echo "  PID: $PID_XL"

# --- TERTIARY MODEL: ResEncUNet M Plans, 4000 epochs, fold_all ---
echo "[GPU 2] Starting MPlans 4000 epochs (fold_all)..."
CUDA_VISIBLE_DEVICES=2 nohup ${NNUNET_TRAIN} ${DATASET} ${CONFIG} all \
    -p nnUNetResEncUNetMPlans \
    -tr nnUNetTrainer_4000epochs \
    --npz \
    > ${LOGDIR}/nnunet_MPlans_4000_all.log 2>&1 &
PID_M=$!
echo "  PID: $PID_M"

# --- 5-FOLD CV with LPlans for validation metrics (GPUs 3-7) ---
for FOLD in 0 1 2 3 4; do
    GPU=$((FOLD + 3))
    echo "[GPU ${GPU}] Starting LPlans 4000 epochs (fold ${FOLD})..."
    CUDA_VISIBLE_DEVICES=${GPU} nohup ${NNUNET_TRAIN} ${DATASET} ${CONFIG} ${FOLD} \
        -p nnUNetResEncUNetLPlans \
        -tr nnUNetTrainer_4000epochs \
        --npz \
        > ${LOGDIR}/nnunet_LPlans_4000_fold${FOLD}.log 2>&1 &
    echo "  PID: $!"
done

echo ""
echo "All training jobs launched!"
echo "Monitor with: tail -f ${LOGDIR}/nnunet_*.log"
echo "Primary model PID: $PID_L (GPU 0)"

wait
echo "All training jobs finished!"
