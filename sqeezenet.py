from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, warnings,BatchNormalization
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Reshape, Dense
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.utils import get_file
from keras.utils import layer_utils
import tensorflow as tf
import keras
from param import letters

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

from train_set import build_train_set
root_path = '/home/ajay/pipeline/'
folder = root_path + 'trail_resized_thresh_data'
valid_folder = root_path + 'eval_resized_thresh_data'

def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    
    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x

def SqueezeNet(input_shape,include_top=False,classes=200):

    img_input = Input(shape=input_shape)
    
    x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    
    x = Dropout(0.5, name='drop9')(x)

    x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    #x = GlobalAveragePooling2D()(x)
    #x = Activation('sigmoid', name='loss')(x)
    x = Flatten()(x)
    x = Dense(200,activation=None)(x)
    x = keras.layers.Reshape((10, 20))(x)
    x = Activation('softmax', name='loss')(x)
    
    inputs = img_input
    model = Model(inputs, x, name='squeezenet')
    return model

model=SqueezeNet(input_shape=(128,128,3))
sumry = model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

images, labels = build_train_set(root_path, folder)
valid_img, valid_label = build_train_set(root_path, valid_folder)
model.fit(images, labels, steps_per_epoch=5, epochs = 20)
save_path = root_path + 'weights/'
model.save_weights(save_path+'Sqeezenet_model_weights.h5')

#img1 = valid_img[0]
#val = np.reshape(img1,(1,128,128,3))
#a1 = model.predict(val)
#a2 = np.argmax(a1, axis=2)
#outval = ""
#for i in range(len(a2[0])):
#  outval = outval+ str(letters[a2[0][i]])
#print(outval)





