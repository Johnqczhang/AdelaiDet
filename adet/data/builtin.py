import os

from .datasets.mots import register_mots_instances


_PREDEFINED_SPLITS_MOTS = {
    "mots_train": ("mot/data/MOTS-train", "mot/annotations/mots_train.json"),
    "mots_train_02": ("mot/data/MOTS-train", "mot/annotations/mots_train_02.json"),
    "mots_val_02": ("mot/data/MOTS-train", "mot/annotations/mots_val_02.json"),
    "mots_train_05": ("mot/data/MOTS-train", "mot/annotations/mots_train_05.json"),
    "mots_val_05": ("mot/data/MOTS-train", "mot/annotations/mots_val_05.json"),
    "mots_train_09": ("mot/data/MOTS-train", "mot/annotations/mots_train_09.json"),
    "mots_val_09": ("mot/data/MOTS-train", "mot/annotations/mots_val_09.json"),
    "mots_train_11": ("mot/data/MOTS-train", "mot/annotations/mots_train_11.json"),
    "mots_val_11": ("mot/data/MOTS-train", "mot/annotations/mots_val_11.json"),
    "kitti_mots_train": ("mot/data/kitti/training", "mot/annotations/kitti_mots_train.json"),
    "kitti_mots_val": ("mot/data/kitti/training", "mot/annotations/kitti_mots_val.json"),
}

metadata_mots = {
    "thing_classes": ["pedestrian"]
}

metadata_kitti_mots = {
    "thing_classes": ["pedestrian", "car"],
}


def register_all_coco(root="datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_MOTS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_mots_instances(
            key,
            metadata_kitti_mots if "kitti" in key else metadata_mots,
            os.path.join(root, json_file),
            os.path.join(root, image_root),
        )


register_all_coco()
