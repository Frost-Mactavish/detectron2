# -*- coding: utf-8 -*-

import logging
import os
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache

import numpy as np
import torch

from detectron2.data import MetadataCatalog
from detectron2.utils import comm

from .evaluator import DatasetEvaluator
from .pascal_voc_evaluation import voc_ap


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


@lru_cache(maxsize=None)
def _parse_xml_record(filename):
    tree = ET.parse(filename)
    objects = []

    def _get_text_or_default(node, tag, default):
        element = node.find(tag)
        if element is None or element.text is None:
            return default
        return element.text

    for obj in tree.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        objects.append(
            {
                "name": _get_text_or_default(obj, "name", ""),
                "difficult": int(_get_text_or_default(obj, "difficult", "0")),
                "bbox": [
                    int(bbox.find("xmin").text),
                    int(bbox.find("ymin").text),
                    int(bbox.find("xmax").text),
                    int(bbox.find("ymax").text),
                ],
            }
        )
    return objects


class DIORDetectionEvaluator(DatasetEvaluator):
    """
    Faster Pascal VOC-protocol evaluator for DIOR.
    It computes AP50 only.
    """

    def __init__(self, dataset_name, cfg=None):
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self._anno_file_template = os.path.join(meta.dirname, "Annotations", "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
        self._class_names = _select_eval_class_names(meta.thing_classes, cfg)
        assert meta.year in [2007, 2012], meta.year
        self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._annotations = None
        self.cfg = cfg

    def reset(self):
        self._predictions = defaultdict(list)

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                xmin += 1
                ymin += 1
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )

    def _load_annotations(self):
        if self._annotations is not None:
            return self._annotations
        with open(self._image_set_path, "r") as f:
            image_names = [x.strip() for x in f.readlines()]
        self._annotations = {
            image_name: _parse_xml_record(self._anno_file_template.format(image_name))
            for image_name in image_names
        }
        return self._annotations

    def _eval_class_ap50(self, cls_name, detections, annotations):
        class_recs = {}
        npos = 0
        for image_name, rec in annotations.items():
            records = [obj for obj in rec if obj["name"] == cls_name]
            bbox = np.array([x["bbox"] for x in records])
            difficult = np.array([x["difficult"] for x in records]).astype(np.bool_)
            det = [False] * len(records)
            npos += sum(~difficult)
            class_recs[image_name] = {"bbox": bbox, "difficult": difficult, "det": det}

        if len(detections) == 0:
            return 0.0

        splitlines = [x.strip().split(" ") for x in detections]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            image_id = image_ids[d]
            if image_id not in class_recs:
                fp[d] = 1.0
                continue
            R = class_recs[image_id]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R["bbox"].astype(float)

            if BBGT.size > 0:
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                ih = np.maximum(iymax - iymin + 1.0, 0.0)
                inters = iw * ih

                uni = (
                    (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                    + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                    - inters
                )
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > 0.5:
                if not R["difficult"][jmax]:
                    if not R["det"][jmax]:
                        tp[d] = 1.0
                        R["det"][jmax] = 1
                    else:
                        fp[d] = 1.0
            else:
                fp[d] = 1.0

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        if npos == 0:
            return 0.0
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, self._is_2007)
        return ap * 100

    def evaluate(self):
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return

        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)

        annotations = self._load_annotations()
        ap50_list = []
        ap50_per_class = OrderedDict()

        for cls_id, cls_name in enumerate(self._class_names):
            ap50 = self._eval_class_ap50(cls_name, predictions.get(cls_id, []), annotations)
            ap50_list.append(ap50)
            ap50_per_class[cls_name] = float(ap50)

        print_msg = []
        mAP50 = float(np.mean(ap50_list)) if len(ap50_list) > 0 else 0.0
        max_name_len = max(len(name) for name in self._class_names) + 1 if len(self._class_names) > 0 else 0

        for cls_name, ap50 in ap50_per_class.items():
            print_msg.append("{:<{width}}: {:.1f}".format(cls_name, ap50, width=max_name_len))
        print_msg.append(f"mAP: {mAP50:.1f}")
        self._logger.info("\n" + "\n".join(print_msg))

        if self.cfg.OUTPUT_DIR:
            with open(os.path.join(self.cfg.OUTPUT_DIR, "result.txt"), "w") as f:
                f.write("\n".join(print_msg))

        ret = OrderedDict()
        ret["bbox"] = {"AP50": mAP50, "AP-LIST": [float(x) for x in ap50_list]}
        return ret
