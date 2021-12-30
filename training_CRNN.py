from keras import backend as K
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint
from dataset_generation import TextImageGenerator
from CRNN_model import get_Model
from param import *
K.set_learning_phase(0)

root_path = '/home/ajay/pipeline/'
train_path = root_path + 'trail_resized_thresh_data/'
#val_path = root_path + 'trail_val_resized_thresh_data/'

model = get_Model(training=True)

train_file_path = train_path
tiger_train = TextImageGenerator(train_file_path, img_w, img_h, batch_size, downsample_factor)
tiger_train.build_data()

#valid_file_path = val_path
#tiger_val = TextImageGenerator(valid_file_path, img_w, img_h, val_batch_size, downsample_factor)
#tiger_val.build_data()

ada = Adadelta()
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)
model.fit_generator(generator=tiger_train.next_batch(),
                    steps_per_epoch=int(tiger_train.n / batch_size), 
                    epochs=30)#, validation_data=tiger_val.next_batch(), validation_steps=int(tiger_val.n / val_batch_size))

save_path = root_path + 'weights/'
model.save_weights(save_path+'CRNN_model_weights.h5')


