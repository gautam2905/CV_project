import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_ID = 200
DATASET_NAME = "Dataset200_VesuviusSurface"
NUM_CLASSES = 3

NNUNET_RAW = BASE_DIR / "nnUNet_data" / "nnUNet_raw"
NNUNET_PREPROCESSED = BASE_DIR / "nnUNet_data" / "nnUNet_preprocessed"
NNUNET_RESULTS = BASE_DIR / "nnUNet_data" / "nnUNet_results"

DATA_ROOT = BASE_DIR / "data"


def setup_nnunet_env():
    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS)
    os.environ["nnUNet_USE_BLOSC2"] = "1"
    os.environ["nnUNet_compile"] = "false"

    for d in [NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"nnUNet_raw:          {NNUNET_RAW}")
    print(f"nnUNet_preprocessed: {NNUNET_PREPROCESSED}")
    print(f"nnUNet_results:      {NNUNET_RESULTS}")
