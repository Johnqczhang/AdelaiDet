import os

from detectron2.data.datasets.register_coco import register_coco_instances

from .datasets.text import register_text_instances
from .datasets.mots import register_mots_instances

# register plane reconstruction

_PREDEFINED_SPLITS_PIC = {
    "pic_person_train": ("pic/image/train", "pic/annotations/train_person.json"),
    "pic_person_val": ("pic/image/val", "pic/annotations/val_person.json"),
}

metadata_pic = {
    "thing_classes": ["person"]
}

_PREDEFINED_SPLITS_TEXT = {
    "totaltext_train": ("totaltext/train_images", "totaltext/train.json"),
    "totaltext_val": ("totaltext/test_images", "totaltext/test.json"),
    "ctw1500_word_train": ("CTW1500/ctwtrain_text_image", "CTW1500/annotations/train_ctw1500_maxlen100_v2.json"),
    "ctw1500_word_test": ("CTW1500/ctwtest_text_image","CTW1500/annotations/test_ctw1500_maxlen100.json"),
    "syntext1_train": ("syntext1/images", "syntext1/annotations/train.json"),
    "syntext2_train": ("syntext2/images", "syntext2/annotations/train.json"),
    "mltbezier_word_train": ("mlt2017/images","mlt2017/annotations/train.json"),
}

metadata_text = {
    "thing_classes": ["text"]
}

_PREDEFINED_SPLITS_MOTS = {
    "mots_train": ("mots/train", "mots/annotations/mots_train.json"),
    "mots_train_02": ("mots/train", "mots/annotations/mots_train_02.json"),
    "mots_val_02": ("mots/train", "mots/annotations/mots_val_02.json"),
    "mots_train_05": ("mots/train", "mots/annotations/mots_train_05.json"),
    "mots_val_05": ("mots/train", "mots/annotations/mots_val_05.json"),
    "mots_train_09": ("mots/train", "mots/annotations/mots_train_09.json"),
    "mots_val_09": ("mots/train", "mots/annotations/mots_val_09.json"),
    "mots_train_11": ("mots/train", "mots/annotations/mots_train_11.json"),
    "mots_val_11": ("mots/train", "mots/annotations/mots_val_11.json"),
    "kitti_mots_train": ("kitti/training", "mots/annotations/kitti_mots_train.json"),
    "kitti_mots_val": ("kitti/training", "mots/annotations/kitti_mots_val.json"),
    "kitti_mots_train_full": ("kitti/training", "mots/annotations/kitti_mots_train_full.json"),
}

metadata_mots = {
    "thing_classes": ["pedestrian"]
}

metadata_kitti_mots = {
    "thing_classes": ["pedestrian", "car"],
}


def register_all_coco(root="datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_PIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            metadata_pic,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TEXT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_text_instances(
            key,
            metadata_text,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_MOTS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_mots_instances(
            key,
            metadata_kitti_mots if "kitti" in key else metadata_mots,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


register_all_coco()