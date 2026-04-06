#!/bin/bash
# ============================================================
# nnU-Net Training for Vesuvius Surface Detection
#
# Usage:
#   conda activate dikshant
#   bash scripts/train_nnunet.sh
#
# This launches multiple nnU-Net training runs in parallel
# across available GPUs.
# ============================================================

set -e

# nnU-Net environment variables
export nnUNet_raw="/raid/home/vikram_govt/Dikshant/gautam/cv/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="/raid/home/vikram_govt/Dikshant/gautam/cv/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="/raid/home/vikram_govt/Dikshant/gautam/cv/nnUNet_data/nnUNet_results"
export nnUNet_USE_BLOSC2=1
export nnUNet_compile=false

DATASET=200
CONFIG="3d_fullres"

echo "============================================================"
echo "  nnU-Net Training Launcher"
echo "  Dataset: ${DATASET}"
echo "  Config:  ${CONFIG}"
echo "============================================================"

# --- PRIMARY MODEL: ResEncUNet L Plans, 4000 epochs, fold_all ---
echo "[GPU 0] Starting LPlans 4000 epochs (fold_all)..."
CUDA_VISIBLE_DEVICES=0 nohup nnUNetv2_train ${DATASET} ${CONFIG} all \
    -p nnUNetResEncUNetLPlans \
    -tr nnUNetTrainer_4000epochs \
    --npz \
    > logs/nnunet_LPlans_4000_all.log 2>&1 &
PID_L=$!
echo "  PID: $PID_L"

# --- SECONDARY MODEL: ResEncUNet XL Plans, 250 epochs, fold_all ---
echo "[GPU 1] Starting XLPlans 250 epochs (fold_all)..."
CUDA_VISIBLE_DEVICES=1 nohup nnUNetv2_train ${DATASET} ${CONFIG} all \
    -p nnUNetResEncUNetXLPlans \
    -tr nnUNetTrainer_250epochs \
    --npz \
    > logs/nnunet_XLPlans_250_all.log 2>&1 &
PID_XL=$!
echo "  PID: $PID_XL"

# --- TERTIARY MODEL: ResEncUNet M Plans, 4000 epochs, fold_all ---
echo "[GPU 2] Starting MPlans 4000 epochs (fold_all)..."
CUDA_VISIBLE_DEVICES=2 nohup nnUNetv2_train ${DATASET} ${CONFIG} all \
    -p nnUNetResEncUNetMPlans \
    -tr nnUNetTrainer_4000epochs \
    --npz \
    > logs/nnunet_MPlans_4000_all.log 2>&1 &
PID_M=$!
echo "  PID: $PID_M"

# --- 5-FOLD CV with LPlans for validation metrics (GPUs 3-7) ---
for FOLD in 0 1 2 3 4; do
    GPU=$((FOLD + 3))
    echo "[GPU ${GPU}] Starting LPlans 4000 epochs (fold ${FOLD})..."
    CUDA_VISIBLE_DEVICES=${GPU} nohup nnUNetv2_train ${DATASET} ${CONFIG} ${FOLD} \
        -p nnUNetResEncUNetLPlans \
        -tr nnUNetTrainer_4000epochs \
        --npz \
        > logs/nnunet_LPlans_4000_fold${FOLD}.log 2>&1 &
    echo "  PID: $!"
done

echo ""
echo "All training jobs launched!"
echo "Monitor with: tail -f logs/nnunet_*.log"
echo "Primary model PID: $PID_L (GPU 0)"
echo ""
echo "To also monitor with wandb, run:"
echo "  python -m src.nnunet.wandb_monitor --log_dir nnUNet_data/nnUNet_results/Dataset200_VesuviusSurface/nnUNetTrainer_4000epochs__nnUNetResEncUNetLPlans__3d_fullres --name LPlans_4000ep"

wait
echo "All training jobs finished!"
