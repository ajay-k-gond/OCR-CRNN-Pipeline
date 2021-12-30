from train_set import build_train_set
import numpy as np
from param import *

root_path = '/home/ajay/pipeline/'
train_folder = root_path+ 'train_resized_thresh_data'
valid_folder = root_path+ 'eval_resized_thresh_data'
train_image, train_label = build_train_set(root_path, train_folder)
valid_image, valid_label = build_train_set(root_path, valid_folder)

from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation,Flatten
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM
K.set_learning_phase(0)
img_w = 128
img_h = 128
num_classes = 20

input_shape = (img_w, img_h, 3)     # (128, 128, 1)

    # Make Networkw
inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 128, 1)

# Convolution layer (VGG)
inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(inputs)  # (None, 128, 128, 64)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 64, 64)

inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  # (None, 64, 64, 128)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 32, 128)

inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # (None, 32, 32, 256)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)  # (None, 32, 32, 256)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, 32, 16, 256)

inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 512)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 16, 512)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, 32, 8, 512)

inner = Conv2D(50, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(inner)  # (None, 32, 8, 50)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
print(inner.shape)
# CNN to RNN
inner = Reshape(target_shape=((10,1280)), name='reshape')(inner)  # (None, 10,1280)
inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 10, 64)

# RNN layer
lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)  # (None, 10, 256)
lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (lstm_1b)

lstm1_merged = add([lstm_1, reversed_lstm_1b])  # (None, 10, 512)
lstm1_merged = BatchNormalization()(lstm1_merged)

lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
lstm_2b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)
reversed_lstm_2b= Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (lstm_2b)

lstm2_merged = concatenate([lstm_2, reversed_lstm_2b])  # (None, 10, 1024)
lstm_merged = BatchNormalization()(lstm2_merged)
#flat = Flatten()(lstm_merged)

# transforms RNN output to character activations:
inner = Dense(100, kernel_initializer='he_normal',name='dense2')(lstm_merged)
inner = Dense(num_classes, kernel_initializer='he_normal',name='dense3')(inner)
y_pred = Activation('softmax', name='softmax')(inner)

model = Model(inputs,y_pred)
model.compile(optimizer='adam', loss='categorical_crossentropy')

model.summary()

model.fit(train_image, train_label,batch_size = 64, epochs = 1,validation_data = (valid_image,valid_label))
save_path = root_path + 'weights/'
model.save_weights(save_path+'CRNN_without_CTC-Loss_model_weights.h5')
#val_img = train_image[0]
#val_img = np.reshape(val_img,(1,128,128,1))
#pred = model.predict(val_img)
#indc = np.argmax(pred, axis=2)
#pred_text = ''
#for i in range(len(indc[0])):
#    pred_text = pred_text+str(letters[indc[0][i]])
#print(pred_text)
