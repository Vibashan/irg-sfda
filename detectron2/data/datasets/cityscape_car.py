# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

import pdb

__all__ = ["load_cityscape_car_instances", "register_cityscape_car"]


# fmt: off
CLASS_NAMES = ( "car", "background")
SYNTH_CLASS_NAMES = ( 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
# fmt: on

def load_cityscape_car_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]], synthetic: bool):
    """
    Load cityscape_car detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local

    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    annotation_synth_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations_synth/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text
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

            if cls in class_names:
                instances.append(
                    {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
                )
        r["annotations"] = instances

        if synthetic:
            anno_file = os.path.join(annotation_synth_dirname, fileid + ".txt")
            instances = []
            for line in open(anno_file).readlines():
                cls, x1, y1, x2, y2 = line.strip().split(',')
                bbox = [float(v) for v in [x1, y1, x2, y2]]
                if cls in SYNTH_CLASS_NAMES:
                    instances.append({"category_id": SYNTH_CLASS_NAMES.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS})
            r["annotations_synth"] = instances
        dicts.append(r)
    return dicts


def register_cityscape_car(name, dirname, split, year, class_names=CLASS_NAMES, synthetic=False):
    DatasetCatalog.register(name, lambda: load_cityscape_car_instances(dirname, split, class_names, synthetic))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )
