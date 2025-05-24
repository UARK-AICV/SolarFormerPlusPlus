import cv2
import numpy as np


def main():
    img_path = '../data/datasets/smart_plant/front_2_class_new_COCO/val2017/4_Trim_1_16_1.jpg'
    img_path = '/media/Seagate16T/tqminh/AllData/ChickenDefectDetection/data/datasets/smart_plant/front_2_class_new_COCO/val2017/4_Trim_1_794_4.jpg'
    img_path = '/media/Seagate16T/tqminh/AllData/ChickenDefectDetection/data/datasets/smart_plant/front_2_class_overlap/train2017/4_Trim_1_1_1___10_Trim_982_5.jpg'
    img = cv2.imread(img_path)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_channel = lab[:,:,1]
    th = cv2.threshold(a_channel,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    #th = cv2.threshold(a_channel,127,255,cv2.THRESH_OTSU)[1]
    masked = cv2.bitwise_and(img, img, mask = th)    # contains dark background
    m1 = masked.copy()
    m1[th==0]=(255,255,255)

    cv2.imwrite('../data/viz/1_highres_m2f/4_Trim_1_1_1___10_Trim_982_5.jpg', m1)


main()

