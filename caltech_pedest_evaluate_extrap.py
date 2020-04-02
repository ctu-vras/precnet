'''
Evaluate multiple frame prediction performance of trained PreCNet on Caltech Pedestrian Dataset sequences.
Based on code related to PredNet - Lotter et al. 2016
Calculates mean-squared error, SSIM, PSNR and plots predictions.
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


n_plot = 40 #number of figures with predictions
batch_size = 1
nt = 25 #number of timesteps
extrap_start_time = 10 

test_file = os.path.join(DATA_DIR, 'X_pedest_test.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_pedest_test.hkl')

#choose model (trained on kitti/bdd_large(2M)/bdd_small(41K))
#model_file = os.path.join(WEIGHTS_DIR, 'precnet_kitti_model.1000.h5')
model_file = os.path.join(WEIGHTS_DIR, 'precnet_bdd100k_model.10000.h5')
#model_file = os.path.join(WEIGHTS_DIR, 'precnet_bdd100k_model.1000.h5')

train_model=load_model(model_file,custom_objects = {'PreCNet': PreCNet}, compile=False)


# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
layer_config['extrap_start_time'] = extrap_start_time
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_precnet = PreCNet(weights=train_model.layers[1].get_weights(), **layer_config)

input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_precnet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)

test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format)
X_test = test_generator.create_all()
X_hat = test_model.predict(X_test, batch_size)
if data_format == 'channels_first':
    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

# Compare MSE of PreCNet predictions vs. using last frame.  Write results to prediction_scores.txt
mse_model = []
mse_last_seen = []
ssim_model = []
ssim_last_seen = []
psnr_model = []
psnr_last_seen = []

#for copy last seen frame evaluation
xtest_last_seen = tf.convert_to_tensor(X_test[:, extrap_start_time-1])

for t in range(extrap_start_time,nt):
#MSE
    mse_model.append(np.mean( (X_test[:, t] - X_hat[:, t])**2 ))   # look at all timesteps except the first
    mse_last_seen.append(np.mean( (X_test[:, extrap_start_time-1] - X_test[:, t])**2 ))

    xtest = tf.convert_to_tensor(X_test[:, t])
    xhat = tf.convert_to_tensor(X_hat[:, t])

#SSIM
    sess_ssim = tf.Session()
    ssim_array = tf.image.ssim(xtest, xhat, 1)
    ssim_run = sess_ssim.run(ssim_array)
    ssim_model.append(np.mean(ssim_run))

    sess_ssim = tf.Session()
    ssim_array = tf.image.ssim(xtest_last_seen, xtest, 1)
    ssim_run = sess_ssim.run(ssim_array)
    ssim_last_seen.append(np.mean(ssim_run))

#PSNR
    sess_psnr = tf.Session()
    psnr_array = tf.image.psnr(xtest, xhat, 1)
    psnr_run = sess_psnr.run(psnr_array)
    psnr_model.append(np.mean(psnr_run))

    sess_psnr = tf.Session()
    psnr_array = tf.image.psnr(xtest_last_seen, xtest, 1)
    psnr_run = sess_psnr.run(psnr_array)
    psnr_last_seen.append(np.mean(psnr_run))

if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
f = open(RESULTS_SAVE_DIR + 'extrap_scores_pedest_model.txt', 'w')

for i in range(0,6):
    if i==0:
        print("Model_MSE:")
        f.write("Model_MSE:")
        tmp_array=mse_model
    if i==1:
        print("Last_seen_MSE:")
        f.write("Last_seen_MSE:")
        tmp_array=mse_last_seen
    if i==2:
        print("Model_SSIM:")
        f.write("Model_SSIM:")
        tmp_array=ssim_model
    if i==3:
        print("Last_seen_SSIM:")
        f.write("Last_seen_SSIM:")
        tmp_array=ssim_last_seen
    if i==4:
        print("Model_PSNR:")
        f.write("Model_PSNR:")
        tmp_array=psnr_model
    if i==5:
        print("Last_seen_PSNR:")
        f.write("Last_seen_PSNR:")
        tmp_array=psnr_last_seen

    for it in range(nt - extrap_start_time):
        print(" %f" % tmp_array[it])
        f.write(" %f" % tmp_array[it])
    print("\n")
    f.write("\n")

f.close()



# Plot some predictions
aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
plt.figure(figsize = (nt, 2*aspect_ratio))
gs = gridspec.GridSpec(2, nt)
gs.update(wspace=0., hspace=0.)
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'extrap_plots_pedest/')
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
for i in plot_idx:
    for t in range(nt):
        plt.subplot(gs[t])
        plt.imshow(X_test[i,t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Actual', fontsize=10)

        plt.subplot(gs[t + nt])
        plt.imshow(X_hat[i,t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Predicted', fontsize=10)

    plt.savefig(plot_save_dir +  'plot_PC_' + str(i) + '.png')
    plt.clf()

