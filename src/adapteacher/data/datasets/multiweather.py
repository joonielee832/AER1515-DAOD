# Copyright (c) Facebook, Inc. and its affiliates.
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

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass


logger = logging.getLogger(__name__)

def _get_multiweather_files(image_dir, gt_dir):
    files = []
    # scan through the directory
    cities = PathManager.ls(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        for basename in PathManager.ls(city_img_dir):
            image_file = os.path.join(city_img_dir, basename)

            suffix = 'leftImg8bit'
            basename = basename.split(suffix)[0]

            instance_file = os.path.join(city_gt_dir, basename + "gtFine_instanceIds.png")
            label_file = os.path.join(city_gt_dir, basename + "gtFine_labelIds.png")

            files.append((image_file, instance_file, label_file))
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def load_multiweather_instances(image_dir, gt_dir, to_polygons=True):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    files = _get_multiweather_files(image_dir, gt_dir)

    logger.info("Preprocessing multiweather annotations ...")
    # This is still not fast: all workers will execute duplicate works and will
    # take up to 10m on a 8GPU server.
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))

    ret = pool.map(
        functools.partial(_multiweather_files_to_dict, to_polygons=to_polygons),
        files,
    )
    logger.info("Loaded {} images from {}".format(len(ret), image_dir))
    pool.close()

    # Map cityscape ids to contiguous ids
    from cityscapesscripts.helpers.labels import labels

    labels = [l for l in labels if l.hasInstances and not l.ignoreInEval]
    dataset_id_to_contiguous_id = {l.id: idx for idx, l in enumerate(labels)}
    for dict_per_image in ret:
        for anno in dict_per_image["annotations"]:
            anno["category_id"] = dataset_id_to_contiguous_id[anno["category_id"]]
    return ret


def load_multiweather_semantic(image_dir, gt_dir):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".

    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    # gt_dir is small and contain many small files. make sense to fetch to local first
    gt_dir = PathManager.get_local_path(gt_dir)
    for image_file, _, label_file, json_file in _get_multiweather_files(image_dir, gt_dir):
        label_file = label_file.replace("labelIds", "labelTrainIds")

        with PathManager.open(json_file, "r") as f:
            jsonobj = json.load(f)
        ret.append(
            {
                "file_name": image_file,
                "sem_seg_file_name": label_file,
                "height": jsonobj["imgHeight"],
                "width": jsonobj["imgWidth"],
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    return ret


def _multiweather_files_to_dict(files, to_polygons):
    """
    Parse cityscapes annotation files to a instance segmentation dataset dict.

    Args:
        files (tuple): consists of (image_file, instance_id_file, label_id_file)
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        A dict in Detectron2 Dataset format.
    """
    from cityscapesscripts.helpers.labels import id2label, name2label

    image_file, instance_id_file, _, = files

    annos = []

    # See also the official annotation parsing scripts at
    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/instances2dict.py  # noqa
    with PathManager.open(instance_id_file, "rb") as f:
        inst_image = np.asarray(Image.open(f), order="F")
    # ids < 24 are stuff labels (filtering them first is about 5% faster)
    flattened_ids = np.unique(inst_image[inst_image >= 24])

    ret = {
        "file_name": image_file,
        "image_id": os.path.basename(image_file),
        "height": inst_image.shape[0],
        "width": inst_image.shape[1],
    }

    for instance_id in flattened_ids:
        # For non-crowd annotations, instance_id // 1000 is the label_id
        # Crowd annotations have <1000 instance ids
        label_id = instance_id // 1000 if instance_id >= 1000 else instance_id
        label = id2label[label_id]
        if not label.hasInstances or label.ignoreInEval:
            continue

        anno = {}
        anno["iscrowd"] = instance_id < 1000
        anno["category_id"] = label.id

        mask = np.asarray(inst_image == instance_id, dtype=np.uint8, order="F")

        inds = np.nonzero(mask)
        ymin, ymax = inds[0].min(), inds[0].max()
        xmin, xmax = inds[1].min(), inds[1].max()
        anno["bbox"] = (xmin, ymin, xmax, ymax)
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
