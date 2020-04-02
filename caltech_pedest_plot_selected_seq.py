'''
Plot predictions of a selected sequence from Caltech Pedestrian Dataset outputted by trained PreCNet.
Based on code related to PredNet - Lotter et al. 2016
'''

import os
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.models import load_model

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from precnet import PreCNet
from data_utils import SequenceGenerator

#choose model (trained on kitti/bdd_large(2M)/bdd_small(41K))
#from kitti_settings import *
from bdd_large_settings import *
#from bdd_small_settings import *

import tensorflow as tf
import hickle as hkl



batch_size = 1

start_img=18648
end_img=18658
nt = end_img-start_img+1

data_file = os.path.join(DATA_DIR, 'X_pedest_test.hkl')
source_file = os.path.join(DATA_DIR, 'sources_pedest_test.hkl')

#choose model (trained on kitti/bdd_large(2M)/bdd_small(41K))
#model_file = os.path.join(WEIGHTS_DIR, 'precnet_kitti_model.1000.h5')
model_file = os.path.join(WEIGHTS_DIR, 'precnet_bdd100k_model.10000.h5')
#model_file = os.path.join(WEIGHTS_DIR, 'precnet_bdd100k_model.1000.h5')

X = hkl.load(data_file)
X=X.astype(np.float32) / 255
sources = hkl.load(source_file)

train_model=load_model(model_file,custom_objects = {'PreCNet': PreCNet})


# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_precnet = PreCNet(weights=train_model.layers[1].get_weights(), **layer_config)

input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_precnet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)


X_test_tmp=[]
X_test=[]
for i in range(start_img,end_img+1):
    X_test_tmp.append(X[i])

X_test.append(X_test_tmp)
X_test=np.array(X_test)

X_hat = test_model.predict(X_test, batch_size)
if data_format == 'channels_first':
    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_selected_plots_pedest/')
from PIL import Image

if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)


for t in range(nt):
    if t==0:
        act_tmp=X_test[0,t]
        pred_tmp=X_hat[0,t]

    else:
        act_tmp=np.concatenate((act_tmp, X_test[0,t]), 1)
        pred_tmp=np.concatenate((pred_tmp, X_hat[0,t]), 1)

conc_im=np.concatenate((act_tmp, pred_tmp), 0)
im = Image.fromarray((conc_im * 255).astype(np.uint8))
im.save(plot_save_dir +  'plot_caltech' + str(start_img) + '-' + str(end_img) + '.png')

