# Copyright (c) Facebook, Inc. and its affiliates.
from .cityscapes_evaluation import CityscapesInstanceEvaluator, CityscapesSemSegEvaluator
from .coco_evaluation import COCOEvaluator
from .rotated_coco_evaluation import RotatedCOCOEvaluator
from .evaluator import DatasetEvaluator, DatasetEvaluators, inference_context, inference_on_dataset, inference_on_corruption_dataset
from .lvis_evaluation import LVISEvaluator
from .panoptic_evaluation import COCOPanopticEvaluator
from .pascal_voc_evaluation import PascalVOCDetectionEvaluator
from .sem_seg_evaluation import SemSegEvaluator
from .testing import print_csv_format, verify_results

from .clipart_evaluation import ClipartDetectionEvaluator
from .watercolor_evaluation import WatercolorDetectionEvaluator
from .cityscape_evaluation import CityscapeDetectionEvaluator
from .foggy_evaluation import FoggyDetectionEvaluator
from .cityscape_car_evaluation import CityscapeCarDetectionEvaluator
from .sim10k_evaluation import Sim10kDetectionEvaluator

__all__ = [k for k in globals().keys() if not k.startswith("_")]
