'''
Code for processing BDD100K dataset - small train subset 40K (Yu et al. 2020, https://bdd-data.berkeley.edu/)
Based on code related to PredNet - Lotter et al. 2016 (https://arxiv.org/abs/1605.08104 https://github.com/coxlab/prednet).
Method of resizing was specified (bicubic). 
'''

import os, pdb
import imageio, random
import hickle as hkl
import h5py
import numpy as np
from scipy.misc import imresize
from scipy.misc import toimage
from kitti_settings import *


desired_sz = (128, 160)
# change 30fps to 10 fps
offset = 0
shift = 3
sources_name = "sources_bdd100k_train_40K.hkl"
#use the same sequences as were used during training
sources = hkl.load(DATA_DIR+sources_name)

sel_sequences = []
for seq in sources:
    if seq not in sel_sequences:
        sel_sequences.append(seq)


X=[]
for sequence in sel_sequences:
    vid = imageio.get_reader(DATA_DIR+"raw_bdd100k_dataset/bdd100k/videos/"+"train/"+sequence, 'ffmpeg', fps=30)
    
    for i, im in enumerate(vid):
        # change 30fps to 10 fps
        if (i-offset) % shift == 0:
            target_ds = float(desired_sz[0]) / im.shape[0]
            im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))), 'bicubic')
            d = int((im.shape[1] - desired_sz[1]) / 2)
            im = im[:, d:d + desired_sz[1]]
            X.append(im)

X = np.array(X)


hkl.dump(X, os.path.join(DATA_DIR, 'X_bdd100k_train_40K' + '.hkl'))




