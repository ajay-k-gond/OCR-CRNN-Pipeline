import os,sys
import numpy as np
import cv2

def build_train_set(root_path, folder):
	files = os.listdir(folder)
	letters = ['0','4','5','6','7','9','B','F','H','J','K','L','M','N','P','R','T','W','V','X']
	images = []
	labels = []
	for file in files:
	    img = cv2.imread(folder+'/'+file)
	    codes = file.split('_')
	    label = (codes[0]+codes[1])
	    indcs = list(map(lambda x: letters.index(x), label))
	    label = np.zeros([10,20])
	    for i in range(10):
	        label[i,indcs[i]]=1
	    labels.append(label)
	    images.append(img)
	labels = np.array(labels)
	images = np.array(images)

	return images, labels

