from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json


class MOTSEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir):
        super().__init__(dataset_name, cfg, distributed, output_dir)
        self.cat_ids = None
        if cfg.MODEL.FCOS.NUM_CLASSES == 80:
            if "kitti" in dataset_name:
                self.cat_ids = [0, 2]
                self.cat_id_map = {2: 1}
            else:
                self.cat_ids = [0]
                self.cat_id_map = {}

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                if self.cat_ids is not None:
                    inds = sum([instances.pred_classes == cid for cid in self.cat_ids]) > 0
                    instances = instances[inds]
                    for k, v in self.cat_id_map.items():
                        inds = instances.pred_classes == k
                        instances.pred_classes[inds] = v
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)
