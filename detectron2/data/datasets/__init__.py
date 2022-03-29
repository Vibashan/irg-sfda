# Copyright (c) Facebook, Inc. and its affiliates.
from .coco import load_coco_json, load_sem_seg, register_coco_instances, convert_to_coco_json
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .lvis import load_lvis_json, register_lvis_instances, get_lvis_instances_meta
from .pascal_voc import load_voc_instances, register_pascal_voc
from . import builtin as _builtin  # ensure the builtin datasets are registered

from .clipart import load_clipart_instances, register_clipart
from .watercolor import load_watercolor_instances, register_watercolor
from .cityscape import load_cityscape_instances, register_cityscape
from .foggy import load_foggy_instances, register_foggy
from .sim10k import load_sim10k_instances, register_sim10k
from .kitti import load_kitti_instances, register_kitti
from .cityscape_car import load_cityscape_car_instances, register_cityscape_car


__all__ = [k for k in globals().keys() if not k.startswith("_")]
