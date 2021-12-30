import os,sys
import numpy as np
import cv2

root_path = '/home/ajay/pipeline/'
folder = root_path + 'trail_resized_thresh_data'
files = os.listdir(folder)
file = folder+'/'+files[0]
save_path = root_path + 'character'

img = cv2.imread(file, 0)
ret,thresh = cv2.threshold(img,120,240,cv2.THRESH_BINARY)
contours,hierachy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

i=0
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	
	if w>4 and h>5:
		#save individual images

		cv2.imwrite(save_path+ '/' +str(i) + '.jpg',thresh[y:y+h,x:x+w])
		i=i+1