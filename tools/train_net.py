# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
import shutil
from collections import OrderedDict
import torch
import warnings

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    DIORDetectionEvaluator,
    SemSegEvaluator,
    DOTADetectionEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

warnings.filterwarnings("ignore", category=UserWarning)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a iOD research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        elif evaluator_type == "cityscapes":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name, cfg)
        elif evaluator_type == "dior":
            return DIORDetectionEvaluator(dataset_name, cfg)
        elif evaluator_type == "dota":
            return DOTADetectionEvaluator(dataset_name, cfg)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)

    task = args.config_file.split("/")[-2]
    cfg.TASK = task
    # modify cfg for multi-step incremental training
    if args.step >= 1 and not args.eval_only:
        root = f"log/{args.dataset}/{task}"
        cls_per_step = cfg.MODEL.ROI_HEADS.NUM_NOVEL_CLASSES
        dst_img_store = f"{root}/image_store.pth"

        basename = os.path.basename(args.config_file)
        if basename == "target.yaml":
            if args.step == 1:
                shutil.copy(cfg.WG.IMAGE_STORE_LOC, dst_img_store)
            else:
                cfg.MODEL.WEIGHTS = f"{root}/STEP{args.step-1}/FT/model_final.pth"
            cfg.OUTPUT_DIR = f"{root}/STEP{args.step}/INC"
        elif basename == "ft.yaml":
            if args.step > 1:
                cfg.MODEL.WEIGHTS = f"{root}/STEP{args.step}/INC/model_final.pth"
            cfg.OUTPUT_DIR = f"{root}/STEP{args.step}/FT"

        cfg.MODEL.ROI_HEADS.NUM_BASE_CLASSES += cls_per_step * (args.step - 1)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg.MODEL.ROI_HEADS.NUM_BASE_CLASSES + cls_per_step
        cfg.WG.IMAGE_STORE_LOC = dst_img_store

    # cfg will be modified later in warp training
    # cfg.freeze()

    return cfg


def append_old_new_map_log(args, cfg, res):
    def _to_percent(value):
        value = float(value)
        return value * 100.0 if value <= 1.0 else value

    num_old_classes = int(cfg.TASK.split("-")[0])
    bbox_result = None
    if isinstance(res, dict) and "bbox" in res:
        bbox_result = res.get("bbox", {})
    elif isinstance(res, dict):
        for dataset_result in res.values():
            if isinstance(dataset_result, dict) and "bbox" in dataset_result:
                bbox_result = dataset_result.get("bbox", {})
                break

    ap_list = []
    if isinstance(bbox_result, dict):
        ap_list = bbox_result.get("AP-LIST", bbox_result.get("AP_LIST", []))

    if len(ap_list) >= num_old_classes and num_old_classes > 0:
        old_list = ap_list[:num_old_classes]
        new_list = ap_list[num_old_classes:]
        map_old = sum(old_list) / len(old_list) if len(old_list) > 0 else 0.0
        map_new = sum(new_list) / len(new_list) if len(new_list) > 0 else 0.0
        map_all = float(bbox_result.get("AP50", bbox_result.get("AP", 0.0)))
        map_old = _to_percent(map_old)
        map_new = _to_percent(map_new)
        map_all = _to_percent(map_all)

        os.makedirs("log", exist_ok=True)
        with open(os.path.join("log", "result.txt"), "a") as f:
            f.write(f"{args.dataset} Task {cfg.TASK} Step {args.step}\n")
            f.write(f"mAP Old: {map_old:.1f}, mAP New: {map_new:.1f}, mAP: {map_all:.1f}\n\n")
    else:
        logging.getLogger(__name__).warning(
            "Skip old/new mAP logging because AP_LIST is missing or shorter than num_old_classes. "
            "len(AP_LIST/AP-LIST)=%s, num_old_classes=%s",
            len(ap_list),
            num_old_classes,
        )


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
            append_old_new_map_log(args, cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
