# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fvcore.common.file_io import PathManager
import os
import numpy as np
import xml.etree.ElementTree as ET
from random import shuffle
from typing import Sequence, Optional

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


__all__ = ["register_pascal_voc"]


# fmt: off
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
# fmt: on


def load_voc_instances(dirname: str, split: str, class_names: Sequence[str] = CLASS_NAMES):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: class names used for category-id mapping
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=str)
    fileids = np.atleast_1d(fileids)
    class_to_idx = {name: index for index, name in enumerate(class_names)}

    # shuffle(CLASS_NAMES)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
        jpg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")
        png_file = os.path.join(dirname, "JPEGImages", fileid + ".png")
        if PathManager.exists(jpg_file):
            image_file = jpg_file
        elif PathManager.exists(png_file):
            image_file = png_file
        else:
            raise FileNotFoundError(
                "Image file not found for '{}'. Tried: '{}' and '{}'".format(
                    fileid, jpg_file, png_file
                )
            )

        tree = ET.parse(anno_file)

        r = {
            "file_name": image_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls not in class_to_idx:
                raise ValueError(
                    "Class '{}' not found in provided class_names for dataset '{}' split '{}'".format(
                        cls, dirname, split
                    )
                )
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": class_to_idx[cls], "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_pascal_voc(
    name,
    dirname,
    split,
    year,
    class_names: Optional[Sequence[str]] = None,
):
    class_names = list(class_names) if class_names is not None else CLASS_NAMES
    DatasetCatalog.register(
        name, lambda: load_voc_instances(dirname, split, class_names)
    )
    MetadataCatalog.get(name).set(
        thing_classes=class_names, dirname=dirname, year=year, split=split
    )
