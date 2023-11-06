"""Organize all 7 weather conditions from Multi-Weather dataset into one common folder to maintain same folder structure as Cityscapes dataset."""
import os
import os.path as osp
import shutil
import argparse
from tqdm import tqdm

"""
Steps:
1. Duplicate Cityscapes dataset folder structure
2. Duplicate ground truth labels for all conditions
3. Move around images

Original structure:
    - Cityscapes_night
    ...
    - Cityscapes_overcast   (original Cityscapes train set)
        - gtFine
            - train
                - aachen
                ...
        - leftImg8bit
            - train
                - aachen
                ...
    - Cityscapes_wet_drops
        - leftImg8bit
            - train
                - aachen
                ...
                
New structure:
    - leftImg8bit
        - train
            - aachen
            ...
    - gtFine
        - train
            - aachen
            ...
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Organize Multi-Weather dataset into Cityscapes format")
    parser.add_argument("--input", type=str, default="/home/data/cityscapes_multiweather_original", help="Path to Multi-Weather dataset")
    parser.add_argument("--out", type=str, default="/home/data/cityscapes_multiweather", help="Output directory")
    args = parser.parse_args()
    return args

def main(in_path, out_path):
    prefix = "Cityscapes_"
    conditions = sorted([
        "night", "night_drops", "overcast", "overcast_drops", "snow", "snow_drops", "wet", "wet_drops"
    ])
    
    #? Create output directory
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(osp.join(out_path, "leftImg8bit", "train"), exist_ok=True)
    os.makedirs(osp.join(out_path, "gtFine", "train"), exist_ok=True)
    
    for _dir in sorted(os.listdir(in_path)):
        assert _dir.startswith(prefix), f"Output directory must start with {prefix}"
        assert _dir[len(prefix):] in conditions, f"Output directory must be one of {conditions}"
        
        #? Copy images and/or labels
        imgTrain_dir = osp.join(in_path, _dir, "leftImg8bit", "train")
        assert osp.isdir(imgTrain_dir), f"Directory {imgTrain_dir} does not exist"
        scenes = sorted(os.listdir(imgTrain_dir))
        
        if _dir.endswith("overcast"):
            copy_labels = True
            gtTrain_dir = osp.join(in_path, _dir, "gtFine", "train")
            assert osp.isdir(gtTrain_dir), f"Directory {gtTrain_dir} does not exist"
        else:
            copy_labels = False
        
        for scene in tqdm(scenes, desc=f"Copying images and/or labels from {_dir}"):
            imgSrc_dir = osp.join(imgTrain_dir, scene)
            imgDst_dir = osp.join(out_path, "leftImg8bit", "train", scene)
            os.makedirs(imgDst_dir, exist_ok=True)
            
            for img in os.listdir(imgSrc_dir):
                imgSrc = osp.join(imgSrc_dir, img)
                filename, ext = osp.splitext(osp.basename(img))
                imgDst = osp.join(imgDst_dir, f"{filename}_{_dir.split('_')[-1]}" + ext)
                if not osp.isfile(imgDst):
                    shutil.copy(imgSrc, imgDst)
                
            if copy_labels:
                gtSrc_dir = osp.join(gtTrain_dir, scene)
                gtDst_dir = osp.join(out_path, "gtFine", "train", scene)
                os.makedirs(gtDst_dir, exist_ok=True)
                
                for gt in os.listdir(gtSrc_dir):
                    gtSrc = osp.join(gtSrc_dir, gt)
                    gtDst = osp.join(gtDst_dir, gt)
                    if not osp.isfile(gtDst):
                        shutil.copy(gtSrc, gtDst)
                    
if __name__ == "__main__":
    cfg = parse_args()
    main(cfg.input, cfg.out)