from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Reshape
from param import letters
import numpy as np
#print(letters)

from train_set import build_train_set
root_path = '/home/ajay/pipeline/'
folder = root_path + 'trail_resized_thresh_data'
eval_path = root_path + 'eval_resized_thresh_data'

visible = Input(shape=(128,128,3))

conv0 = Conv2D(64, kernel_size=3, activation='relu')(visible)
pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)
conv1 = Conv2D(32, kernel_size=3, activation='relu')(pool0)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(16, kernel_size=3, activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat = Flatten()(pool2)
hidden1 = Dense(2048, activation='relu')(flat)
output = Dense(200, activation='sigmoid')(hidden1)
output = Reshape((10,20))(output)

model = Model(visible, output)
sumry = model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

images, labels = build_train_set(root_path, folder)
eval_img, eval_label = build_train_set(root_path, eval_path)
model.fit(images, labels, steps_per_epoch=3, epochs = 10)
save_path = root_path + 'weights/'
model.save_weights(save_path+'CNN_model_weights.h5')

img1 = images[0]
val = np.reshape(img1,(1,128,128,3))
a1 = model.predict(val)
a2 = np.argmax(a1, axis=2)
outval = ""
for i in range(len(a2[0])):
  outval = outval+ str(letters[a2[0][i]])
print(outval)