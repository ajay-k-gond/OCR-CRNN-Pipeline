import os,sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

root_path = '/home/ajay/pipeline/'
folder = root_path + 'trail_resized_data'
files = os.listdir(folder)
new_path = root_path + 'trail_resized_thresh_data'

for file in files:
	img = cv2.imread(folder+'/'+file)
	morph = img.copy()

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
	morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
	morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

	# take morphological gradient
	gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)
	#cv2_imshow(gradient_image)

	# split the gradient image into channels
	image_channels = np.split(np.asarray(gradient_image), 3, axis=2)

	channel_height, channel_width, _ = image_channels[0].shape

	# apply Otsu threshold to each channel
	for i in range(0,3):
		_, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
		image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))

	# merge the channels
	image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)

	# save the denoised image
	cv2.imwrite(new_path+'/'+file, image_channels)
	




