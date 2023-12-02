import numpy as np
import cv2 as cv
import random
import torch
from torch import nn
import matplotlib.colors as mplc

# import some common detectron2 utilities
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import DatasetCatalog
from detectron2.structures import BoxMode

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
              count += 1
              if count == 4:
                break
              img = cv.imread(inputs[0]["file_name"])

              # get output:
              output = model(inputs)
              predictions = output[0]["instances"].to("cpu")

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

                      # only use b/r/g for predictions
                      colors = [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 0)) for _ in range(num_instances)]
                      vis = visualizer.overlay_instances(
                          labels=labels, 
                          boxes=boxes, 
                          masks=masks, 
                          keypoints=keypts, 
                          assigned_colors=colors
                      )
                  
                      cv.imshow(dataset_name, vis.get_image())
                      cv.waitKey(0) 
              
                      # closing all open windows 
                      cv.destroyAllWindows() 
              



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