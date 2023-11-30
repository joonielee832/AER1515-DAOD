import functools
import json
import logging
import multiprocessing as mp
import numpy as np
import os
from itertools import chain
import pycocotools.mask as mask_util
from PIL import Image

from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from matplotlib import pyplot as plt
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass 


logger = logging.getLogger(__name__)

def _get_acdc_files(root_dir, split_dir):
    files = []
    weather = split_dir.split("/", 1)[0]
    split = split_dir.split("/", 1)[1]
    # create dirs:
    image_dir = os.path.join(root_dir, "rgb_anon")
    gt_dir = os.path.join(root_dir, "gt")
    gt_panoptic_dir = os.path.join(root_dir, "gt_panoptic")

    split_img_dir = os.path.join(image_dir, split_dir)
    split_gt_dir = os.path.join(gt_dir, split_dir)
    json_file = os.path.join(gt_panoptic_dir,  split + "_gt_panoptic.json")
    split_panoptic_dir = os.path.join(gt_panoptic_dir, split_dir)
    for folder in PathManager.ls(split_img_dir):
        folder_img_dir = os.path.join(split_img_dir, folder)
        folder_gt_dir = os.path.join(split_gt_dir, folder)
        folder_panoptic_dir = os.path.join(split_panoptic_dir, folder)
        for basename in PathManager.ls(folder_img_dir):
            if not "cropped" in basename:
                continue
            
            image_file = os.path.join(folder_img_dir, basename)

            suffix = 'rgb_anon_cropped.png'
            basename = basename.split(suffix)[0]

            label_file = os.path.join(folder_gt_dir, basename + "gt_labelIds.png")
            instance_file = os.path.join(folder_panoptic_dir, basename + "gt_panoptic.png")

            files.append((image_file, instance_file, label_file, json_file))
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def load_acdc_cropped_instances(root_dir, split_dir, from_json=True, to_polygons=True):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/acdc/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/acdc/gtFine/train".
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (acdc's format).

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    files = _get_acdc_files(root_dir, split_dir)

    logger.info("Preprocessing acdc annotations ...")
    # This is still not fast: all workers will execute duplicate works and will
    # take up to 10m on a 8GPU server.
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))

    ret = pool.map(
        functools.partial(_acdc_files_to_dict, from_json=from_json, to_polygons=to_polygons),
        files,
    )
    logger.info("Loaded {} images from {}".format(len(ret), root_dir))
    pool.close()

    # # Map cityscape ids to contiguous ids
    from cityscapesscripts.helpers.labels import labels

    labels = [l for l in labels if l.hasInstances and not l.ignoreInEval]
    dataset_id_to_contiguous_id = {l.id: idx for idx, l in enumerate(labels)}
    for dict_per_image in ret:
        for anno in dict_per_image["annotations"]:
            anno["category_id"] = dataset_id_to_contiguous_id[anno["category_id"]]
    return ret


def _acdc_files_to_dict(files, from_json, to_polygons):
    """
    Parse acdc annotation files to a instance segmentation dataset dict.

    Args:
        files (tuple): consists of (image_file, instance_id_file, label_id_file, json_file)
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (acdc's format).

    Returns:
        A dict in Detectron2 Dataset format.
    """

    image_file, instance_id_file, label_id_file, json_file = files

    annos = []

    if from_json:
        from cityscapesscripts.helpers.labels import labels
        from shapely.geometry import MultiPolygon, Polygon

        with PathManager.open(json_file, "r") as f:
            jsonobj = json.load(f)

        basename = os.path.basename(image_file)
        with PathManager.open(image_file, "rb") as f:
            image = np.asarray(Image.open(f), order="F")

        ret = {
            "file_name": image_file,
            "image_id": basename,
            "height": image.shape[0],
            "width": image.shape[1],
        }

        logger.info(image_file)

        for file in jsonobj["annotations"]:
            if file["image_id"] == basename.split("_rgb_anon_cropped.png",1)[0]:
             
                for obj in file["segments_info"]:

                    anno = {}
                    anno["iscrowd"] = obj["iscrowd"]
                    anno["category_id"] = obj["category_id"]

                    label = labels[int(obj["category_id"])]

                    has_instances = label.hasInstances
                    ignore_in_eval = label.ignoreInEval

                    if not has_instances or ignore_in_eval:
                        continue

                    bbox = obj["bbox"]
                    # cropped images are sized in half, giving us 540x960
                    # we remove 28 from the bottom and 210 off each side to give us 512x512
                    # resulting in the transformation needed for the bbox:
                    xmin = int(bbox[0]/2) - 224
                    ymin = int(bbox[1]/2)
                    width = int(bbox[2]/2)
                    height = int(bbox[3]/2)

                    xmax = xmin + width
                    ymax = ymin + height

                    # check to make sure that new labels are not out of range and crop:
                    xmin = min(max(0, xmin), 512)
                    ymin = min(max(0, ymin), 512)
                    xmax = min(max(0, xmax), 512)
                    ymax = min(max(0, ymax), 512)

                    # detection is no longer in the image at all:
                    if xmin == xmax or ymin == ymax:
                        continue
                    
                    img = cv2.imread(image_file)
                    
                    cv2.rectangle(img,(xmin,ymin),(xmin + width, ymin + height),(0,255,0),2)
                    #     # font 
                    # font = cv2.FONT_HERSHEY_SIMPLEX 
                      
                    # # org 
                    # org = (xmin,ymin) 
                      
                    # # fontScale 
                    # fontScale = 1
                      
                    # # Blue color in BGR 
                    # color = (255, 0, 0) 
                      
                    # # Line thickness of 2 px 
                    # thickness = 2

                    # string = str([anno["category_id"]])
                    # img = cv2.putText(img, string, org, font,  
                    #           fontScale, color, thickness, cv2.LINE_AA) 
                    
                    cv2.imshow(image_file,img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                    anno["bbox"] = (xmin, ymin, xmax, ymax)
                    anno["bbox_mode"] = BoxMode.XYXY_ABS

                    annos.append(anno)

                break
       
    else:

        with PathManager.open(instance_id_file, "rb") as f:
            rgb_inst_image = np.asarray(Image.open(f), order="F")

        # This is in the COCO Panoptic Segmentation Format - we can find the IDs using the following calculation:
        # ids=R+G*256+B*256^2
        # Assuming this is coming in as RGB:

        inst_image = rgb_inst_image[:,:,0] + rgb_inst_image[:,:,1]*256 + rgb_inst_image[:,:,1]*65536


        # ids < 24 are stuff labels (filtering them first is about 5% faster)
        flattened_ids = np.unique(inst_image[inst_image >= 24])
        flattened_ids = np.unique(inst_image)

        ret = {
            "file_name": image_file,
            "image_id": os.path.basename(image_file),
            "height": inst_image.shape[0],
            "width": inst_image.shape[1],
        }

        for instance_id in flattened_ids:

            anno = {}

            # TODO - verify if iscrowd is still true
            anno["iscrowd"] = instance_id < 1000

            mask = np.asarray(inst_image == instance_id, dtype=np.uint8, order="F")

            inds = np.nonzero(mask)
            ymin, ymax = inds[0].min(), inds[0].max()
            xmin, xmax = inds[1].min(), inds[1].max()
            anno["bbox"] = (xmin, ymin, xmax, ymax)

            # use the label image to find the label:

            with PathManager.open(label_id_file, "rb") as f:
                label_id_image = np.asarray(Image.open(f), order="F")

            # mask out the index:
            label_id_image = np.multiply(label_id_image, mask)

            # mask the label img and get the most common label:
            anno["category_id"] = np.max(label_id_image)

            img = cv2.imread(instance_id_file)
            cv2.rectangle(img,(xmin,ymin),(xmax, ymax),(0,255,0),2)
                # font 
            font = cv2.FONT_HERSHEY_SIMPLEX 
              
            # org 
            org = (xmin,ymin) 
              
            # fontScale 
            fontScale = 1
              
            # Blue color in BGR 
            color = (255, 0, 0) 
              
            # Line thickness of 2 px 
            thickness = 2

            string = str([anno["category_id"]])
            img = cv2.putText(img, string, org, font,  
                      fontScale, color, thickness, cv2.LINE_AA) 
             
            cv2.imshow('image',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if xmax <= xmin or ymax <= ymin:
                continue
            anno["bbox_mode"] = BoxMode.XYXY_ABS
            if to_polygons:
                # This conversion comes from D4809743 and D5171122,
                # when Mask-RCNN was first developed.
                contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
                    -2
                ]
                polygons = [c.reshape(-1).tolist() for c in contours if len(c) >= 3]
                # opencv's can produce invalid polygons
                if len(polygons) == 0:
                    continue
                anno["segmentation"] = polygons
            else:
                anno["segmentation"] = mask_util.encode(mask[:, :, None])[0]
            annos.append(anno)
    ret["annotations"] = annos
    return ret
