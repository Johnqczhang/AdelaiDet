import argparse
import tqdm

from detectron2.utils.logger import setup_logger
from adet.config import get_cfg
from train_net import Trainer


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    min_size = min(args.input_height, args.input_width)
    max_size = max(args.input_height, args.input_width)
    cfg.INPUT.MIN_SIZE_TRAIN = min_size
    cfg.INPUT.MAX_SIZE_TRAIN = max_size
    cfg.INPUT.HFLIP_TRAIN = False
    cfg.INPUT.RANDOM_FLIP = "none"
    cfg.INPUT.AUGMENT_HSV = False
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.OUTPUT_DIR = "train_dir/debug"
    cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
    cfg.freeze()

    setup_logger(cfg.OUTPUT_DIR, name="adet")

    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="MOTS Demo")
    parser.add_argument(
        "--config-file",
        default="configs/FCOS-Detection/ht21/R_50_iou.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input-height", type=int, default=800, help="The height of the input image",
    )
    parser.add_argument(
        "--input-width", type=int, default=1422, help="The width of the input image",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class Counter(Trainer):
    def train(self):
        num_imgs = len(self.data_loader.dataset.dataset)
        filenames_set = set()

        for _ in tqdm.tqdm(range(num_imgs)):
            data = next(self._trainer._data_loader_iter)
            assert isinstance(data, list) and len(data) == 1
            filename = data[0]["file_name"]
            assert filename not in filenames_set
            filenames_set.add(filename)
            self._trainer.model(data)

        dataset_name = self.cfg.DATASETS.TRAIN
        cnt_amb = self._trainer.model.proposal_generator.fcos_outputs.cnt_ambiguous
        cnt_amb = {
            k: v for k, v in cnt_amb.items() if sum(list(v.values())) > 0
        }
        cnt_amb = {k: {kk: vv for kk, vv in v.items() if vv > 0} for k, v in cnt_amb.items()}
        print(f"Dataset: {dataset_name}")
        print(f"# locations: {cnt_amb}")
        # ratio_amb = {
        #     k: {n: num for n, num in v.items() if n > 0}
        #     for k, v in cnt_amb.items()
        # }
        # ratio_amb = {
        #     k: {n: num / sum(list(v.values())) for n, num in v.items()}
        #     for k, v in ratio_amb.items()
        # }
        # print(f"Ratio: {ratio_amb}")
        cnt_pos = self._trainer.model.proposal_generator.fcos_outputs.cnt_pos_per_level
        in_features = self.cfg.MODEL.FCOS.IN_FEATURES
        cnt_pos = {
            k: {in_features[i]: num for i, num in enumerate(v)}
            for k, v in cnt_pos.items()
        }
        soi = self._trainer.model.proposal_generator.fcos_outputs.sizes_of_interest
        for i, s in enumerate(soi):
            nums = {k: v[in_features[i]] for k, v in cnt_pos.items()}
            if sum(list(nums.values())) == 0:
                continue
            print(f"positive locations at {in_features[i]} within size {s}: {nums}")


if __name__ == "__main__":
    args = get_parser().parse_args()
    print(f"Arguments: {str(args)}")
    cfg = setup(args)

    counter = Counter(cfg)
    counter.train()
