import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

root_path = '/home/ajay/Downloads/'
img_fold = root_path + 'split'
files = os.listdir(img_fold)

folder = root_path + 'split_thresh'
files1 = os.listdir(folder)
save_path = root_path + 'char'

def sorting(array):
	array.sort()
	return array


for (file, file1) in zip(files, files1):
	img = cv2.imread(img_fold+'/'+file)
	img1 = cv2.imread(folder+'/'+file1, 0)
	ret,thresh1 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
	mxarea = ((thresh1.shape[0])*(thresh1.shape[1]))/5
	mnarea = ((thresh1.shape[0])*(thresh1.shape[1]))/21
	contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	points=[]
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		area = w*h
		if(mnarea < area < mxarea):
			points.append((x,y,w,h))
	if(len(points)==5):
		ans = sorting(points)
		for i in range(5):
			x,y,w,h = ans[i]
			cv2.imwrite(save_path+ '/' +str(file[i]) + '.jpg',img[y:y+h+4,x:x+w+4])

