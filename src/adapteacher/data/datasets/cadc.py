import functools
import json
import logging
import multiprocessing as mp
import numpy as np
import yaml
import os
from itertools import chain
import pycocotools.mask as mask_util
from PIL import Image

from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt
import cv2  # noqa

CITYSCAPES_LABELS = {
  "Traffic_Guidance_Objects": 18,
  "Pedestrian" : 24,
  "Car" : 26,
  "Truck" : 27,
}

logger = logging.getLogger(__name__)

def _get_cadc_files(root_dir, split_dir, name):
    
    # Split the folders based on train and val - 
    # if train, take the first 80%
    # if val, take the last 20%
    train = True
    if 'train' in name:
        train = True
    
    files = []
    # create dirs:
    split_dir = os.path.join(root_dir, split_dir)
    run_folders = PathManager.ls(split_dir)
    
    # remove the calibration folder
    run_folders.remove("calib")

    # Sort folders for consistency
    run_folders.sort()


    folder_len = len(run_folders)
    
    if train:
        run_folders = run_folders[0 : int(folder_len * .8)]
    else:
        run_folders = run_folders[int(folder_len * .8) : folder_len]
        
    print(run_folders)

    for run_folder in run_folders:
        camera_dir = os.path.join(split_dir, run_folder + "/labeled")
        annotation = os.path.join(split_dir, run_folder + "/3d_ann.json")
        for camera in PathManager.ls(camera_dir):
            if camera =="lidar_points" or camera == "novatel":
                continue
            camera_id = camera.split("_0",1)[1]
            img_dir = os.path.join(camera_dir, camera + "/data")
            for img in PathManager.ls(img_dir):
                image_file = os.path.join(img_dir, img)
                files.append((image_file, annotation, camera_id))
    assert len(files), "No images found in {}".format(split_dir)
    return files

def _get_cadc_intrinsics(root_dir, split_dir):
    calib = {}
    calib_path = os.path.join(root_dir, split_dir + "/calib")
    # Get calibrations
    calib['extrinsics'] = yaml.load(open(calib_path + '/extrinsics.yaml'), yaml.SafeLoader)
    calib['CAM00'] = yaml.load(open(calib_path + '/00.yaml'), yaml.SafeLoader)
    calib['CAM01'] = yaml.load(open(calib_path + '/01.yaml'), yaml.SafeLoader)
    calib['CAM02'] = yaml.load(open(calib_path + '/02.yaml'), yaml.SafeLoader)
    calib['CAM03'] = yaml.load(open(calib_path + '/03.yaml'), yaml.SafeLoader)
    calib['CAM04'] = yaml.load(open(calib_path + '/04.yaml'), yaml.SafeLoader)
    calib['CAM05'] = yaml.load(open(calib_path + '/05.yaml'), yaml.SafeLoader)
    calib['CAM06'] = yaml.load(open(calib_path + '/06.yaml'), yaml.SafeLoader)
    calib['CAM07'] = yaml.load(open(calib_path + '/07.yaml'), yaml.SafeLoader)

    return calib

def load_cadc_instances(root_dir, split_dir, name):
    """
    Args:
        root_dir (str): path to the raw dataset. e.g., "~/datasets/cadcd".
        split_dir (str): path to the desired split. e.g., "../2018_03_07".

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    files = _get_cadc_files(root_dir, split_dir, name)
    intrinsics = _get_cadc_intrinsics(root_dir, split_dir)

    logger.info("Preprocessing acdc annotations ...")
    # This is still not fast: all workers will execute duplicate works and will
    # take up to 10m on a 8GPU server.
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))

    ret = pool.map(
        functools.partial(_cadc_files_to_dict, intrinsics=intrinsics),
        files,
    )
    logger.info("Loaded {} images from {}".format(len(ret), root_dir))
    pool.close()
    return ret


def _cadc_files_to_dict(files, intrinsics):
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

    image_file, annotations_json, cam = files

    img_basename = os.path.basename(image_file)
    frame = int(img_basename.split('.png')[0])
    print((frame))

    with PathManager.open(image_file, "rb") as f:
        image = np.asarray(Image.open(f), order="F")

    img_w = image.shape[1]
    img_h = image.shape[0]

    annos = []

    ret = {
        "file_name": image_file,
        "image_id": img_basename,
        "height": image.shape[0],
        "width": image.shape[1],
    }

    annotations_data = None
    with open(annotations_json) as f:
      annotations_data = json.load(f)

    # Projection matrix from camera to image frame
    T_IMG_CAM = np.eye(4)
    T_IMG_CAM[0:3,0:3] = np.array(intrinsics['CAM0' + cam]['camera_matrix']['data']).reshape(-1, 3)
    T_IMG_CAM = T_IMG_CAM[0:3,0:4]; # remove last row

    T_CAM_LIDAR = np.linalg.inv(np.array(intrinsics['extrinsics']['T_LIDAR_CAM0' + cam]))
    
    labels = []

    for cuboid in annotations_data[frame]['cuboids']:
        label = CITYSCAPES_LABELS[cuboid['label']]
        anno = {}

        # TODO REMOVE
        if label not in labels:
           labels += [label]

        T_Lidar_Cuboid = np.eye(4)
        T_Lidar_Cuboid[0:3,0:3] = R.from_euler('z', cuboid['yaw'], degrees=False).as_matrix()
        T_Lidar_Cuboid[0][3] = cuboid['position']['x']
        T_Lidar_Cuboid[1][3] = cuboid['position']['y']
        T_Lidar_Cuboid[2][3] = cuboid['position']['z']

        width = cuboid['dimensions']['x']
        length = cuboid['dimensions']['y']
        height = cuboid['dimensions']['z']

        front_right_bottom = np.array([[1,0,0,length/2],[0,1,0,-width/2],[0,0,1,-height/2],[0,0,0,1]])
        front_right_top = np.array([[1,0,0,length/2],[0,1,0,-width/2],[0,0,1,height/2],[0,0,0,1]])
        front_left_bottom = np.array([[1,0,0,length/2],[0,1,0,width/2],[0,0,1,-height/2],[0,0,0,1]])
        front_left_top = np.array([[1,0,0,length/2],[0,1,0,width/2],[0,0,1,height/2],[0,0,0,1]])

        back_right_bottom = np.array([[1,0,0,-length/2],[0,1,0,-width/2],[0,0,1,-height/2],[0,0,0,1]])
        back_right_top = np.array([[1,0,0,-length/2],[0,1,0,-width/2],[0,0,1,height/2],[0,0,0,1]])
        back_left_bottom = np.array([[1,0,0,-length/2],[0,1,0,width/2],[0,0,1,-height/2],[0,0,0,1]])
        back_left_top = np.array([[1,0,0,-length/2],[0,1,0,width/2],[0,0,1,height/2],[0,0,0,1]])

        # Project to image
        tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, front_right_bottom))
        if tmp[2][3] < 0:
          continue
        f_r_b = np.matmul(T_IMG_CAM, tmp)
        tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, front_right_top))
        if tmp[2][3] < 0:
          continue
        f_r_t = np.matmul(T_IMG_CAM, tmp)
        tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, front_left_bottom))
        if tmp[2][3] < 0:
          continue
        f_l_b = np.matmul(T_IMG_CAM, tmp)
        tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, front_left_top))
        if tmp[2][3] < 0:
          continue
        f_l_t = np.matmul(T_IMG_CAM, tmp)

        tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, back_right_bottom))
        if tmp[2][3] < 0:
          continue
        b_r_b = np.matmul(T_IMG_CAM, tmp)
        tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, back_right_top))
        if tmp[2][3] < 0:
          continue
        b_r_t = np.matmul(T_IMG_CAM, tmp)
        tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, back_left_bottom))
        if tmp[2][3] < 0:
          continue
        b_l_b = np.matmul(T_IMG_CAM, tmp)
        tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, back_left_top))
        if tmp[2][3] < 0:
          continue
        b_l_t = np.matmul(T_IMG_CAM, tmp)

        # Make sure the 
        # Remove z
        f_r_b_coord = (int(f_r_b[0][3]/f_r_b[2][3]), int(f_r_b[1][3]/f_r_b[2][3]))
        f_r_t_coord = (int(f_r_t[0][3]/f_r_t[2][3]), int(f_r_t[1][3]/f_r_t[2][3]))
        f_l_b_coord = (int(f_l_b[0][3]/f_l_b[2][3]), int(f_l_b[1][3]/f_l_b[2][3]))
        f_l_t_coord = (int(f_l_t[0][3]/f_l_t[2][3]), int(f_l_t[1][3]/f_l_t[2][3]))
        if f_r_b_coord[0] < 0 or f_r_b_coord[0] > img_w or f_r_b_coord[1] < 0 or f_r_b_coord[1] > img_h:
          continue
        if f_r_t_coord[0] < 0 or f_r_t_coord[0] > img_w or f_r_t_coord[1] < 0 or f_r_t_coord[1] > img_h:
          continue
        if f_l_b_coord[0] < 0 or f_l_b_coord[0] > img_w or f_l_b_coord[1] < 0 or f_l_b_coord[1] > img_h:
          continue
        if f_l_t_coord[0] < 0 or f_l_t_coord[0] > img_w or f_l_t_coord[1] < 0 or f_l_t_coord[1] > img_h:
          continue

        b_r_b_coord = (int(b_r_b[0][3]/b_r_b[2][3]), int(b_r_b[1][3]/b_r_b[2][3]))
        b_r_t_coord = (int(b_r_t[0][3]/b_r_t[2][3]), int(b_r_t[1][3]/b_r_t[2][3]))
        b_l_b_coord = (int(b_l_b[0][3]/b_l_b[2][3]), int(b_l_b[1][3]/b_l_b[2][3]))
        b_l_t_coord = (int(b_l_t[0][3]/b_l_t[2][3]), int(b_l_t[1][3]/b_l_t[2][3]))
        if b_r_b_coord[0] < 0 or b_r_b_coord[0] > img_w or b_r_b_coord[1] < 0 or b_r_b_coord[1] > img_h:
          continue
        if b_r_t_coord[0] < 0 or b_r_t_coord[0] > img_w or b_r_t_coord[1] < 0 or b_r_t_coord[1] > img_h:
          continue
        if b_l_b_coord[0] < 0 or b_l_b_coord[0] > img_w or b_l_b_coord[1] < 0 or b_l_b_coord[1] > img_h:
          continue
        if b_l_t_coord[0] < 0 or b_l_t_coord[0] > img_w or b_l_t_coord[1] < 0 or b_l_t_coord[1] > img_h:
          continue


        xmin = min(f_l_b_coord[0], f_l_t_coord[0], b_l_b_coord[0],  b_l_t_coord[0])
        xmax = max(f_r_b_coord[0], f_r_t_coord[0], b_r_b_coord[0],  b_r_t_coord[0])
        ymin = min(f_r_t_coord[1], f_l_t_coord[1], b_r_t_coord[1],  b_l_t_coord[1])
        ymax = max(f_r_b_coord[1], f_l_b_coord[1], b_r_b_coord[1],  b_l_b_coord[1])
        anno["bbox"] = (xmin, ymin, xmax, ymax)
        anno["category_id"] = label

        # TODO REMOVE 
        print(image_file)
        img = cv2.imread(image_file)
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
        string = cuboid['label'] + str(cuboid["points_count"])
        img = cv2.putText(img, string, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
        

        # Draw  12 lines
        # Front
        cv2.line(img, f_r_b_coord, f_r_t_coord, [0, 0, 255], thickness=2, lineType=8, shift=0)
        cv2.line(img, f_r_b_coord, f_l_b_coord, [0, 0, 255], thickness=2, lineType=8, shift=0)
        cv2.line(img, f_l_b_coord, f_l_t_coord, [0, 0, 255], thickness=2, lineType=8, shift=0)
        cv2.line(img, f_l_t_coord, f_r_t_coord, [0, 0, 255], thickness=2, lineType=8, shift=0)
        # back
        cv2.line(img, b_r_b_coord, b_r_t_coord, [0, 0, 100], thickness=2, lineType=8, shift=0)
        cv2.line(img, b_r_b_coord, b_l_b_coord, [0, 0, 100], thickness=2, lineType=8, shift=0)
        cv2.line(img, b_l_b_coord, b_l_t_coord, [0, 0, 100], thickness=2, lineType=8, shift=0)
        cv2.line(img, b_l_t_coord, b_r_t_coord, [0, 0, 100], thickness=2, lineType=8, shift=0)
        # connect front to back
        cv2.line(img, f_r_b_coord, b_r_b_coord, [0, 0, 150], thickness=2, lineType=8, shift=0)
        cv2.line(img, f_r_t_coord, b_r_t_coord, [0, 0, 150], thickness=2, lineType=8, shift=0)
        cv2.line(img, f_l_b_coord, b_l_b_coord, [0, 0, 150], thickness=2, lineType=8, shift=0)
        cv2.line(img, f_l_t_coord, b_l_t_coord, [0, 0, 150], thickness=2, lineType=8, shift=0)

        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if xmax <= xmin or ymax <= ymin:
            continue
        anno["bbox_mode"] = BoxMode.XYXY_ABS

        annos.append(anno)

    ret["annotations"] = annos

    print(labels)
    return ret
