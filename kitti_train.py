'''
Train PreCNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/).
Based on code related to PredNet - Lotter et al. 2016 (https://arxiv.org/abs/1605.08104 https://github.com/coxlab/prednet).
'''

import os
import numpy as np
np.random.seed(123)
from six.moves import cPickle

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from precnet import PreCNet
from data_utils import SequenceGenerator
from kitti_settings import *


save_model = True  # if weights will be saved
model_file = os.path.join(WEIGHTS_DIR, 'precnet_kitti_model.h5')

# Data files
train_file = os.path.join(DATA_DIR, 'X_kitti_train_bic.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_kitti_train_bic.hkl')
val_file = os.path.join(DATA_DIR, 'X_kitti_val_bic.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_kitti_val_bic.hkl')

# Training parameters
nb_epoch = 1000
batch_size = 4
samples_per_epoch = 500
N_seq_val = 100  # number of sequences to use for validation

# Model parameters
n_channels, im_height, im_width = (3, 128, 160)
input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)

stack1=60
stack2=120
R_stack3=240
stack_sizes = (n_channels, stack1, stack2)
R_stack_sizes = (stack1, stack2, R_stack3)
Ahat_filt_sizes = (3, 3, 3)
R_filt_sizes = (3, 3, 3)

alpha=0.0
layer_loss_weights = np.array([1, alpha, alpha])

layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
nt = 10  # number of timesteps used for sequences in training
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
time_loss_weights[0] = 0


precnet = PreCNet(stack_sizes, R_stack_sizes, Ahat_filt_sizes, R_filt_sizes, output_mode='error', return_sequences=True)

inputs = Input(shape=(nt,) + input_shape)
errors = precnet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
model = Model(inputs=inputs, outputs=final_errors)
model.compile(loss='mean_absolute_error', optimizer='adam')

train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)

lr_schedule = lambda epoch: 0.001 if epoch < 900 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 900 epochs
callbacks = [LearningRateScheduler(lr_schedule)]
model_file_rep=os.path.join(WEIGHTS_DIR, 'precnet_kitti_model.{epoch:02d}.h5')
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=model_file, monitor='val_loss', save_best_only=True))
    callbacks.append(ModelCheckpoint(filepath=model_file_rep, monitor='val_loss', save_best_only=False, mode='auto', period=10))


history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,
                validation_data=val_generator, validation_steps=N_seq_val / batch_size)

        
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
#plt.show()
plt.savefig('training_curves_kitti')
