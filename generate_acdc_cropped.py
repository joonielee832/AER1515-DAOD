# first make a copy of acdc to acdc_cropped - place it in the same place as acdc:
# your dir should look like AER1515-DAOD/datasets/acdc_cropped

import cv2

from os import listdir, path


basepath = "datasets/acdc_cropped/rgb_anon"

for condition in listdir(basepath):
  condition_dir = basepath + "/" + condition
  for split in listdir(condition_dir):
    print("Cropping " + split + " split of acdc " + condition)
    split_dir = condition_dir + "/" + split
    for set in listdir(split_dir):
      set_dir = split_dir + "/" + set
      for image in listdir(set_dir):
        image_dir = set_dir + "/" + image
        cropped_dir = path.splitext(image_dir)[0]+'_cropped.png' 
        img = cv2.imread(image_dir)
        resized = cv2.resize(img, (0,0), fx=0.5, fy=0.5) # should reduce from 1024 x 2048 to 512 x 1024
        crop = resized[:512, 224:736]
        cv2.imwrite(cropped_dir, crop) 


    