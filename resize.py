import os,sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

root_path = '/home/ajay/pipeline/'
folder = root_path + 'trail_data'
files = os.listdir(folder)
save_path = root_path + 'trail_resized_data'

for file in files:
    img = cv2.imread(folder+'/'+file)
    resized = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
    cv2.imwrite(save_path+'/'+file, resized)

folder1 = root_path + 'eval_data'
files1 = os.listdir(folder1)
new_path1 = root_path + 'eval_resized_data'

for file1 in files1:
    img1 = cv2.imread(folder1+'/'+file1)
    resized1 = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
    cv2.imwrite(new_path1+'/'+file1, resized1)   