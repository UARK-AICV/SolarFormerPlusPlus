# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.file_io import PathManager

from detectron2.data.datasets.coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

SMART_PLANT_CATEGORIES = [
    {"color": [0,127.5,127.5], "isthing": 1, 'supercategory': 'chicken', 'id': 1, 'name': 'normal'}, 
    {"color": [127.5,127.5,0], "isthing": 1, 'supercategory': 'chicken', 'id': 2, 'name': 'defect'},
]

_PREDEFINED_SPLITS_SMART_PLANT_PANOPTIC = {
    "smart_plant_overlap_train_panoptic": (
        # This is the original panoptic annotation directory
        "smart_plant/front_2_class_overlap/panoptic_train2017",
        "smart_plant/front_2_class_overlap/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "smart_plant/front_2_class_overlap/panoptic_semseg_train2017",
    ),
    "smart_plant_overlap_val_panoptic": (
        "smart_plant/front_2_class_overlap/panoptic_val2017",
        "smart_plant/front_2_class_overlap/annotations/panoptic_val2017.json",
        "smart_plant/front_2_class_overlap/panoptic_semseg_val2017",
    ),
    "smart_plant_single_train_panoptic": (
        "smart_plant/front_2_class_new_COCO/panoptic_train2017",
        "smart_plant/front_2_class_new_COCO/annotations/panoptic_train2017.json",
        "smart_plant/front_2_class_new_COCO/panoptic_semseg_train2017",
    ),
    "smart_plant_single_val_panoptic": (
        "smart_plant/front_2_class_new_COCO/panoptic_val2017",
        "smart_plant/front_2_class_new_COCO/annotations/panoptic_val2017.json",
        "smart_plant/front_2_class_new_COCO/panoptic_semseg_val2017",
    ),
}


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in SMART_PLANT_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in SMART_PLANT_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in SMART_PLANT_CATEGORIES]
    stuff_colors = [k["color"] for k in SMART_PLANT_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(SMART_PLANT_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def register_smartplant_panoptic_datasets(root):
    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_SMART_PLANT_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file

        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            #_get_builtin_metadata("coco_panoptic_standard"),
            get_metadata(),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            instances_json,
        )



_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_smartplant_panoptic_datasets(_root)
