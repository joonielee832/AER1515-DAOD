import numpy as np
import cv2 as cv
import random
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.colors as mplc

# import some common detectron2 utilities
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.layers import nms

from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances

from contextlib import ExitStack, contextmanager

from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)

def visualize(cfg, model):
  print(cfg.DATASETS.TEST)
  for _, dataset_name in enumerate(cfg.DATASETS.TEST):
      metadata = MetadataCatalog.get(dataset_name)
      dataset_dicts = DatasetCatalog.get(dataset_name)
      data_loader = build_detection_test_loader(cfg, dataset_name)

      with ExitStack() as stack:
          if isinstance(model, nn.Module):
              stack.enter_context(inference_context(model))
          stack.enter_context(torch.no_grad())
          count = 0
          for _, inputs in enumerate(data_loader):
              if count == 4:
                break
              img = cv.imread(inputs[0]["file_name"])

              # get output:
              output = model(inputs)
              predictions = output[0]["instances"].to("cpu")
              print(predictions)
              # threshold on predictions:
              #predictions = threshold_predictions(predictions)

              print(predictions)

              if len(predictions.pred_boxes) == 0:
                  # no detections
                  continue
              count += 1
              # find gt in dataset dict:
              gt_dict = next((item for item in dataset_dicts if item["file_name"] == inputs[0]["file_name"]), None)
              if gt_dict != None:
                
                  # visualize outputs
                  img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                  visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)

                  # visualize prediction:
                  masks = None
                  boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
                  scores = predictions.scores if predictions.has("scores") else None
                  classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
                  labels = create_text_labels(classes, scores, metadata.get("thing_classes", None), None, prepend="PRED_")
                  keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

                  num_instances = len(boxes)
                  # only use r/g for predictions
                  colors = [(random.uniform(0, 0), random.uniform(0, 1), random.uniform(0, 1)) for _ in range(num_instances)]
                  alpha = 0.5

                  visualizer.overlay_instances(
                      masks=masks,
                      boxes=boxes,
                      labels=labels,
                      keypoints=keypoints,
                      assigned_colors=colors,
                      alpha=alpha,
                  )

                  # visualize ground truth:
                  annos = gt_dict.get("annotations", None)
                  if annos:
                      masks = None
                      if "keypoints" in annos[0]:
                          keypts = [x["keypoints"] for x in annos]
                          keypts = np.array(keypts).reshape(len(annos), -1, 3)
                      else:
                          keypts = None

                      boxes = [
                          BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYXY_ABS)
                          if len(x["bbox"]) == 4
                          else x["bbox"]
                          for x in annos
                      ]

                      category_ids = [x["category_id"] for x in annos]
                      names = metadata.get("thing_classes", None)
                      labels = create_text_labels(
                          category_ids,
                          scores=None,
                          class_names=names,
                          is_crowd=[x.get("iscrowd", 0) for x in annos],
                          prepend="GT_"
                      )

                      num_instances = len(boxes)

                      # only use b/r/g for predictions
                      colors = [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 0)) for _ in range(num_instances)]
                      vis = visualizer.overlay_instances(
                          labels=labels, 
                          boxes=boxes, 
                          masks=masks, 
                          keypoints=keypts, 
                          assigned_colors=colors
                      )
                      plt.imshow(vis.get_image())
                      plt.show()
                      #cv.imshow(dataset_name, vis.get_image())
                      #cv.waitKey(0) 
              
                      # closing all open windows 
                      #cv.destroyAllWindows() 
              
def threshold_predictions(predictions, thres=0.5):
    image_shape = predictions.image_size
    new_predictions = Instances(image_shape)

    #theshold based on confidence
    valid_map = predictions.scores > thres
    new_bbox_loc = predictions.pred_boxes.tensor[valid_map, :]

    # nms_bboxes, nms_classes, nms_scores, nms_masks = nms_predictions(
    #     predictions.pred_classes[valid_map],
    #     predictions.scores[valid_map],
    #     new_bbox_loc,
    #     None,
    #     image_shape)
    new_predictions.pred_boxes = Boxes(new_bbox_loc)
    new_predictions.pred_classes = predictions.pred_classes[valid_map]
    new_predictions.scores = predictions.scores[valid_map]

    # # threshiold based on nms:
    # new_predictions.pred_boxes = Boxes(nms_bboxes)
    # new_predictions.pred_classes = nms_classes
    # new_predictions.scores = nms_scores
    if predictions.has("pred_keypoints") and predictions.pred_keypoints != None:
        new_predictions.keypoint = predictions.pred_keypoints[valid_map] 

    return new_predictions


def nms_predictions(classes, scores, bboxes, masks, image_shape, iou_th=.5):
    he, wd = image_shape[0], image_shape[1]
    boxes_list = [[x[0] / wd, x[1] / he, x[2] / wd, x[3] / he]
                  for x in bboxes]
    scores_list = [x for x in scores]
    labels_list = [x for x in classes]
    nms_bboxes, nms_scores, nms_classes = nms(
        boxes=[boxes_list], 
        scores=[scores_list], 
        labels=[labels_list], 
        weights=None,
        iou_thr=iou_th
    )
    nms_masks = []
    for s in nms_scores:
        nms_masks.append(masks[scores.index(s)])
    nms_scores, nms_classes, nms_masks = zip(
        *sorted(
            zip(nms_scores, nms_classes, nms_masks), 
            reverse=True))
    return nms_classes, nms_scores, nms_masks


# def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
#     if proposal_type == "rpn":
#         valid_map = proposal_bbox_inst.objectness_logits > thres

#         # create instances containing boxes and gt_classes
#         image_shape = proposal_bbox_inst.image_size
#         new_proposal_inst = Instances(image_shape)

#         # create box
#         new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
#         new_boxes = Boxes(new_bbox_loc)

#         # add boxes to instances
#         new_proposal_inst.gt_boxes = new_boxes
#         new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
#             valid_map
#         ]
#     elif proposal_type == "roih":
#         valid_map = proposal_bbox_inst.scores > thres

#         # create instances containing boxes and gt_classes
#         image_shape = proposal_bbox_inst.image_size
#         new_proposal_inst = Instances(image_shape)

#         # create box
#         new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
#         new_boxes = Boxes(new_bbox_loc)

#         # add boxes to instances
#         new_proposal_inst.gt_boxes = new_boxes
#         new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
#         new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

#     return new_proposal_inst


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def create_text_labels(classes, scores, class_names, is_crowd=None, prepend=""):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [(prepend + class_names[i]) for i in classes]
            #labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels



def jitter(color):
    """
    Randomly modifies given color to produce a slightly different color than the color given.

    Args:
        color (tuple[double]): a tuple of 3 elements, containing the RGB values of the color
            picked. The values in the list are in the [0.0, 1.0] range.

    Returns:
        jittered_color (tuple[double]): a tuple of 3 elements, containing the RGB values of the
            color after being jittered. The values in the list are in the [0.0, 1.0] range.
    """
    color = mplc.to_rgb(color)
    vec = np.random.rand(3)
    # better to do it in another color space
    vec = vec / np.linalg.norm(vec) * 0.5
    res = np.clip(vec + color, 0, 1)
    return tuple(res)