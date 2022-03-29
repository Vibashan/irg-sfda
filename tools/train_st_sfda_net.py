#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
import copy
import torch.optim as optim
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
    ClipartDetectionEvaluator,
    WatercolorDetectionEvaluator,
    CityscapeDetectionEvaluator,
    FoggyDetectionEvaluator,
    CityscapeCarDetectionEvaluator,
)

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

import pdb
import cv2
from pynvml import *
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.data.detection_utils import convert_image_to_rgb

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
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
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if evaluator_type == "clipart":
        return ClipartDetectionEvaluator(dataset_name)
    if evaluator_type == "watercolor":
        return WatercolorDetectionEvaluator(dataset_name)
    if evaluator_type == "cityscape":
        return CityscapeDetectionEvaluator(dataset_name)
    if evaluator_type == "foggy":
        return FoggyDetectionEvaluator(dataset_name)
    if evaluator_type == "cityscape_car":
        return CityscapeCarDetectionEvaluator(dataset_name)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

# =====================================================
# ================== Pseduo-labeling ==================
# =====================================================
def threshold_bbox(proposal_bbox_inst, thres=0.7, proposal_type="roih"):
    if proposal_type == "rpn":
        valid_map = proposal_bbox_inst.objectness_logits > thres

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
            valid_map
        ]
    elif proposal_type == "roih":
        valid_map = proposal_bbox_inst.scores > thres

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
        new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

    return new_proposal_inst

def process_pseudo_label(proposals_rpn_k, cur_threshold, proposal_type, psedo_label_method=""):
    list_instances = []
    num_proposal_output = 0.0
    for proposal_bbox_inst in proposals_rpn_k:
        # thresholding
        if psedo_label_method == "thresholding":
            proposal_bbox_inst = threshold_bbox(
                proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
            )
        else:
            raise ValueError("Unkown pseudo label boxes methods")
        num_proposal_output += len(proposal_bbox_inst)
        list_instances.append(proposal_bbox_inst)
    num_proposal_output = num_proposal_output / len(proposals_rpn_k)
    return list_instances, num_proposal_output

@torch.no_grad()
def update_teacher_model(model_student, model_teacher, keep_rate=0.996):
    if comm.get_world_size() > 1:
        student_model_dict = {
            key[7:]: value for key, value in model_student.state_dict().items()
        }
    else:
        student_model_dict = model_student.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in model_teacher.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                student_model_dict[key] *
                (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))

    return new_teacher_dict

def visualize_proposals(cfg, batched_inputs, proposals, box_size, proposal_dir, metadata):
        from detectron2.utils.visualizer import Visualizer

        for input, prop in zip(batched_inputs, proposals):
            img = input["image_weak"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), None)
            #v_gt = Visualizer(img, None)
            #v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            #anno_img = v_gt.get_image()
            v_pred = Visualizer(img, metadata)
            if proposal_dir == "rpn":
                v_pred = v_pred.overlay_instances( boxes=prop.proposal_boxes[0:int(box_size)].tensor.cpu().numpy())
            if proposal_dir == "roih":
                v_pred = v_pred.draw_instance_predictions(prop)
            vis_img = v_pred.get_image()

            save_path = os.path.join(cfg.OUTPUT_DIR, proposal_dir) 
            save_img_path = os.path.join(cfg.OUTPUT_DIR, proposal_dir, input['file_name'].split('/')[-1]) 
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_img_path, vis_img)


def test_sfda(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        cfg.defrost()
        cfg.SOURCE_FREE.TYPE = False
        cfg.freeze()
        test_data_loader = build_detection_test_loader(cfg, dataset_name)
        test_metadata = MetadataCatalog.get(dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, test_data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
            #pdb.set_trace()
            cls_names = test_metadata.get("thing_classes")
            cls_aps = results_i['bbox']['class-AP50']
            for i in range(len(cls_aps)):
                logger.info("AP for {}: {}".format(cls_names[i], cls_aps[i]))
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def train_sfda(cfg, model_student, model_teacher, resume=False):
    
    checkpoint = copy.deepcopy(model_teacher.state_dict())

    # norm_params = []
    # for name, p in model_student.named_parameters():
    #     if "norm" in name:
    #         norm_params.append(p)
    #     else:
    #         p.requires_grad = False
    # optimizer = optim.SGD(norm_params, lr=0.0001, momentum=0.9, weight_decay=0.0001)
    # #pdb.set_trace()
    model_teacher.eval()
    model_student.train()

    #optimizer = optim.SGD(model_student.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    optimizer = build_optimizer(cfg, model_student)
    scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(model_student, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)

    #pdb.set_trace()

    data_loader = build_detection_train_loader(cfg)

    total_epochs = 10
    len_data_loader = len(data_loader.dataset.dataset.dataset)
    start_iter, max_iter = 0, len_data_loader
    max_sf_da_iter = total_epochs*max_iter
    logger.info("Starting training from iteration {}".format(start_iter))

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, len_data_loader, max_iter=max_sf_da_iter)
    writers = default_writers(cfg.OUTPUT_DIR, max_sf_da_iter) if comm.is_main_process() else []

    model_teacher.eval()
    test_sfda(cfg, model_teacher)

    with EventStorage(start_iter) as storage:
        for epoch in range(1, total_epochs+1):
            cfg.defrost()
            cfg.SOURCE_FREE.TYPE = True
            cfg.freeze()
            data_loader = build_detection_train_loader(cfg)
            model_teacher.eval()
            model_student.train()
            for data, iteration in zip(data_loader, range(start_iter, max_iter)):
                storage.iter = iteration
                optimizer.zero_grad()

                with torch.no_grad():
                    _, teacher_features, teacher_proposals, teacher_results = model_teacher(data, mode="train")

                teacher_pseudo_proposals, num_rpn_proposal = process_pseudo_label(teacher_proposals, 0.9, "rpn", "thresholding")
                teacher_pseudo_results, num_roih_proposal = process_pseudo_label(teacher_results, 0.9, "roih", "thresholding")

                #visualize_proposals(cfg, data, teacher_proposals, num_rpn_proposal, "rpn", metadata)
                #visualize_proposals(cfg, data, teacher_results, num_roih_proposal, "roih", metadata)

                loss_dict = model_student(data, cfg, model_teacher, teacher_features, teacher_proposals, teacher_pseudo_results, mode="train")

                losses = sum(loss_dict.values())
                #losses = loss_dict["feat_consistency"]
                # if not torch.isfinite(losses).all():
                #     print(loss_dict)
                #     continue
                assert torch.isfinite(losses).all(), loss_dict

                #loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                #losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                #scheduler.step()

                # if (
                #     cfg.TEST.EVAL_PERIOD > 0
                #     and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                #     and iteration != max_iter - 1
                # ):
                #     test_sfda(cfg, model)
                #     # Compared to "train_net.py", the test results are not dumped to EventStorage
                #     comm.synchronize()
                if iteration - start_iter > 5 and ((iteration + 1) % 50 == 0 or iteration == max_iter - 1):
                    print("epoch: ", epoch, "lr:", optimizer.param_groups[0]["lr"], ''.join(['{0}: {1}, '.format(k, v.item()) for k,v in loss_dict.items()]))
                    # for writer in writers:
                    #     writer.write()
                    # if (iteration + 1) % 1000 == 0:
                    #     new_teacher_dict = update_teacher_model(model_student, model_teacher, keep_rate=0.9)
                    #     model_teacher.load_state_dict(new_teacher_dict)
                
                
                # new_teacher_dict = update_teacher_model(model_student, model_teacher, keep_rate=0.996)
                # model_teacher.load_state_dict(new_teacher_dict)

                # if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                #     for writer in writers:
                #         writer.write()
                periodic_checkpointer.step(iteration)

            new_teacher_dict = update_teacher_model(model_student, model_teacher, keep_rate=0.9)
            model_teacher.load_state_dict(new_teacher_dict)

            if epoch == 1 or epoch == 10:
                model_student.eval()
                print("Student model testing@", epoch)
                test_sfda(cfg, model_student)

                #new_teacher_dict = update_teacher_model(model_student, model_teacher, keep_rate=0.996)
                #model_teacher.load_state_dict(new_teacher_dict)
                model_teacher.eval()
                print("Teacher model testing@", epoch)
                test_sfda(cfg, model_teacher)

                torch.save(model_teacher.state_dict(), cfg.OUTPUT_DIR + "/model_teacher_{}.pth".format(epoch))
                torch.save(model_student.state_dict(), cfg.OUTPUT_DIR + "/model_student_{}.pth".format(epoch))
    
    model_student.eval()
    print("Student model testing@", epoch)
    test_sfda(cfg, model_student)

    #new_teacher_dict = update_teacher_model(model_student, model_teacher, keep_rate=0.996)
    #model_teacher.load_state_dict(new_teacher_dict)
    model_teacher.eval()
    print("Teacher model testing@", epoch)
    test_sfda(cfg, model_teacher)



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    # while 1:
    #     nvmlInit()
    #     h = nvmlDeviceGetHandleByIndex(0)
    #     info = nvmlDeviceGetMemoryInfo(h)
    #     if info.free > info.used:
    #         break
    
    cfg = setup(args)

    model_student = build_model(cfg)
    cfg.defrost()
    cfg.MODEL.META_ARCHITECTURE = "teacher_sfda_RCNN"
    cfg.freeze()
    model_teacher = build_model(cfg)
    logger.info("Model:\n{}".format(model_student))

    DetectionCheckpointer(model_student, save_dir=cfg.OUTPUT_DIR).load(args.model_dir)
    DetectionCheckpointer(model_teacher, save_dir=cfg.OUTPUT_DIR).load(args.model_dir)
    logger.info("Trained model has been sucessfully loaded")
    return train_sfda(cfg, model_student, model_teacher)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
