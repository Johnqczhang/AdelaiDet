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
    "rects_train": ("ReCTS/ReCTS_train_images", "ReCTS/annotations/rects_train.json"),
    "rects_val": ("ReCTS/ReCTS_val_images", "ReCTS/annotations/rects_val.json"),
    "rects_test": ("ReCTS/ReCTS_test_images", "ReCTS/annotations/rects_test.json"),
    "art_train": ("ArT/rename_artimg_train", "ArT/annotations/abcnet_art_train.json"), 
    "lsvt_train": ("LSVT/rename_lsvtimg_train", "LSVT/annotations/abcnet_lsvt_train.json"), 
    "chnsyn_train": ("ChnSyn/syn_130k_images", "ChnSyn/annotations/chn_syntext.json"),
}

metadata_text = {
    "thing_classes": ["text"]
}

_PREDEFINED_SPLITS_MOTS = {
    "mots_train": ("mot/mots/train", "mot/annotations/mots_train_full.json"),
    "mots_train_02": ("mot/mots/train", "mot/annotations/mots_train_02.json"),
    "mots_val_02": ("mot/mots/train", "mot/annotations/mots_val_02.json"),
    "mots_train_05": ("mot/mots/train", "mot/annotations/mots_train_05.json"),
    "mots_val_05": ("mot/mots/train", "mot/annotations/mots_val_05.json"),
    "mots_train_09": ("mot/mots/train", "mot/annotations/mots_train_09.json"),
    "mots_val_09": ("mot/mots/train", "mot/annotations/mots_val_09.json"),
    "mots_train_11": ("mot/mots/train", "mot/annotations/mots_train_11.json"),
    "mots_val_11": ("mot/mots/train", "mot/annotations/mots_val_11.json"),
    "kitti_mots_train": ("mot/kitti/training", "mot/annotations/kitti_mots_train.json"),
    "kitti_mots_val": ("mot/kitti/training", "mot/annotations/kitti_mots_val.json"),
    "kitti_mots_train_full": ("mot/kitti/training", "mot/annotations/kitti_mots_train_full.json"),
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
