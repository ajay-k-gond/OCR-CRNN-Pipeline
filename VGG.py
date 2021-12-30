from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Reshape
from keras.layers import Activation
import numpy as np
from param import letters

from train_set import build_train_set
root_path = '/home/ajay/pipeline/'
folder = root_path + 'trail_resized_thresh_data'

visible = Input(shape=(128,128,3))

conv0 = Conv2D(64, kernel_size=3, activation='relu')(visible)
conv1 = Conv2D(64, kernel_size=3, activation='relu')(conv0)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(128, kernel_size=3, activation='relu')(pool1)
conv3 = Conv2D(128, kernel_size=3, activation='relu')(conv2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, kernel_size=3, activation='relu')(pool3)
conv5 = Conv2D(256, kernel_size=3, activation='relu')(conv4)
pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

conv6 = Conv2D(512, kernel_size=3, activation='relu')(pool5)
conv7 = Conv2D(512, kernel_size=3, activation='relu')(conv6)
pool7 = MaxPooling2D(pool_size=(2, 2))(conv7)

flat = Flatten()(pool7)
hidden1 = Dense(2048, activation='relu')(flat)
final_cnn = Dense(200, activation='sigmoid')(hidden1)
output = Reshape((10,20))(final_cnn)

model = Model(visible, output)
sumry = model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

images, labels = build_train_set(root_path, folder)
model.fit(images, labels, steps_per_epoch=32, epochs = 5)

save_path = root_path + 'weights/'
model.save_weights(save_path+'VGG_model_weights.h5')


#val_img = images[0]
#val = np.reshape(val_img,(1,128,128,3))
#a1 = model.predict(val)
#a2 = np.argmax(a1, axis=2)
#outval = ""
#for i in range(len(a2[0])):
#  outval = outval+ str(letters[a2[0][i]])
#print(outval)

