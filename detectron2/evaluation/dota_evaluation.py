import logging
import os
import tempfile
from collections import OrderedDict

import torch

from detectron2.data import MetadataCatalog
from detectron2.utils import comm

from .dota_utils import merge_and_eval
from .evaluator import DatasetEvaluator


def _select_eval_class_names(class_names, cfg):
    if cfg is None or not cfg.MODEL.ROI_HEADS.LEARN_INCREMENTALLY:
        return class_names

    num_base_class = cfg.MODEL.ROI_HEADS.NUM_BASE_CLASSES
    num_novel_class = cfg.MODEL.ROI_HEADS.NUM_NOVEL_CLASSES

    if cfg.MODEL.ROI_HEADS.TRAIN_ON_BASE_CLASSES:
        max_eval_class = num_base_class
    else:
        max_eval_class = num_base_class + num_novel_class

    return class_names[:max_eval_class]


class DOTADetectionEvaluator(DatasetEvaluator):
    """
    DOTA evaluator adapted from maskrcnn-benchmark logic:
    collect predictions in test loop -> dump class-wise txt -> merge_and_eval.
    """

    def __init__(self, dataset_name, cfg=None):
        self._logger = logging.getLogger(__name__)
        self._cpu_device = torch.device("cpu")
        self._class_names = list(
            _select_eval_class_names(MetadataCatalog.get(dataset_name).thing_classes, cfg)
        )
        self.cfg = cfg

    def reset(self):
        self._predictions = []
        self._file_list = []

    def process(self, inputs, outputs):
        for input_item, output_item in zip(inputs, outputs):
            image_id = str(input_item["image_id"])
            instances = output_item["instances"].to(self._cpu_device)
            self._file_list.append(image_id)
            self._predictions.append(instances)

    def evaluate(self):
        gathered_file_list = comm.gather(self._file_list, dst=0)
        gathered_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return {}

        file_list = []
        predictions = []
        for file_list_per_rank, predictions_per_rank in zip(gathered_file_list, gathered_predictions):
            file_list.extend(file_list_per_rank)
            predictions.extend(predictions_per_rank)

        class_txt = [[] for _ in self._class_names]
        for file_name, prediction in zip(file_list, predictions):
            result = {
                "boxes": prediction.pred_boxes.tensor,
                "labels": prediction.pred_classes,
                "scores": prediction.scores,
            }
            for box, label, score in zip(result["boxes"], result["labels"], result["scores"]):
                polygon = torch.tensor([box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]])
                bbox_str = " ".join([str(x) for x in polygon.tolist()])
                label_index = int(label.item())
                if label_index < 0 or label_index >= len(class_txt):
                    continue
                class_txt[label_index].append(f"{file_name} {score.item()} {bbox_str}\n")

        with tempfile.TemporaryDirectory(prefix="dota_eval_") as temp_dir:
            path = os.path.join(temp_dir, "DOTA_result")
            os.makedirs(path, exist_ok=True)
            for class_name, txt in zip(self._class_names, class_txt):
                with open(f"{path}/{class_name}.txt", "w") as file:
                    file.writelines(txt)

            print_msg, ap_list = merge_and_eval(path, path, self._class_names)
            self._logger.info("\n" + "\n".join(print_msg))
            if self.cfg.OUTPUT_DIR:
                with open(os.path.join(self.cfg.OUTPUT_DIR, "result.txt"), "w") as f:
                    f.write("\n".join(print_msg))

        mean_ap = float(ap_list.mean() * 100.0) if len(ap_list) > 0 else 0.0
        ret = OrderedDict()
        ret["bbox"] = {
            "AP": mean_ap,
            "AP50": mean_ap,
            "AP-LIST": [float(x) * 100.0 for x in ap_list.tolist()],
        }
        return ret