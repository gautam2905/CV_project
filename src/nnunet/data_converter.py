import csv
import json
from pathlib import Path

from src.nnunet.nnunet_config import (
    DATASET_ID, DATASET_NAME, NNUNET_RAW, DATA_ROOT, NUM_CLASSES
)


def convert_to_nnunet_format():
    ds_dir = NNUNET_RAW / DATASET_NAME
    images_dir = ds_dir / "imagesTr"
    labels_dir = ds_dir / "labelsTr"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_ROOT / "train.csv"
    train_images_dir = DATA_ROOT / "train_images"
    train_labels_dir = DATA_ROOT / "train_labels"

    count = 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["id"]
            img_src = (train_images_dir / f"{sample_id}.tif").resolve()
            lbl_src = (train_labels_dir / f"{sample_id}.tif").resolve()

            if not img_src.exists() or not lbl_src.exists():
                continue

            img_dst = images_dir / f"{sample_id}_0000.tif"
            lbl_dst = labels_dir / f"{sample_id}.tif"

            if img_dst.exists() or img_dst.is_symlink():
                img_dst.unlink()
            if lbl_dst.exists() or lbl_dst.is_symlink():
                lbl_dst.unlink()

            img_dst.symlink_to(img_src)
            lbl_dst.symlink_to(lbl_src)

            spacing_json = images_dir / f"{sample_id}_0000.json"
            with open(spacing_json, "w") as sf:
                json.dump({"spacing": [1.0, 1.0, 1.0]}, sf)

            count += 1

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {
            "background": 0,
            "surface": 1,
            "papyrus": 2,
        },
        "numTraining": count,
        "file_ending": ".tif",
        "overwrite_image_reader_writer": "TiffIO",
    }

    with open(ds_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Converted {count} cases to nnU-Net format at {ds_dir}")
    print(f"dataset.json written with {NUM_CLASSES} classes")
    return ds_dir
