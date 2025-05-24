'''

CityscapeSem_format
- train_imgs/
- train_gt/
- test_imgs/
- test_gt/
'''

import os
import cv2
import numpy as np
from os.path import join
from tqdm import tqdm


split='train'
raw_ann_path='../data/datasets/argi_vision/Agriculture-Vision-2021/{}/labels'.format(split)
img_path='../data/datasets/argi_vision/Agriculture-Vision-2021/{}/images/rgb'.format(split)
gt_path='../data/datasets/argi_vision/argi_vision_cityscapes/{}_gt'.format(split)

def main():
    
    for img_name in tqdm(sorted(os.listdir(img_path))):
        img = cv2.imread(join(img_path, img_name))
        base_img_name = os.path.splitext(img_name)[0]
        h,w = img.shape[:2]
        sem_seg_gt = np.ones((h,w), dtype=np.uint8) * 255

        for class_id, class_name in enumerate(sorted(os.listdir(raw_ann_path))):
            class_path = join(raw_ann_path,class_name)
            
            #print(class_name, class_id)
            for ann_name in sorted(os.listdir(class_path)):
                if base_img_name in ann_name:
                    ann_path = join(class_path, ann_name)
                    ann = cv2.imread(ann_path)

                    sem_seg_gt[ann[:,:,2] == 255] = class_id


        gt_name = base_img_name + '_labelIds.png'
        output_path = join(gt_path, gt_name)
        cv2.imwrite(output_path, sem_seg_gt)


main()
