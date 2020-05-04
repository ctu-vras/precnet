'''
Code for processing BDD100K dataset - large train subset 2M (Yu et al. 2020, https://bdd-data.berkeley.edu/)
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
sources_name = "sources_bdd100k_train0-4999.hkl"
#use the same sequences as were used during training
sources = hkl.load(DATA_DIR+sources_name)

sel_sequences = []
for seq in sources:
    if seq not in sel_sequences:
        sel_sequences.append(seq)
X=[]

for seq_idx, sequence in enumerate(sel_sequences):
#save into 5 hkl files (memory optimization)
    if seq_idx % 1000 == 0 and seq_idx>0:
        X = np.array(X)
        hkl.dump(X, os.path.join(DATA_DIR, 'raw_bdd100k_dataset/X_bdd100k_train' + str(seq_idx - 1000) + '-' + str(seq_idx - 1) + '.hkl'))
        X = []

    vid = imageio.get_reader(DATA_DIR+"raw_bdd100k_dataset/bdd100k/videos/"+"train/"+sequence, 'ffmpeg', fps=30)
    print(seq_idx)

    
    for i, im in enumerate(vid):
        # change 30fps to 10 fps
        if (i-offset) % shift == 0:
            target_ds = float(desired_sz[0]) / im.shape[0]
            im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))), 'bicubic')
            d = int((im.shape[1] - desired_sz[1]) / 2)
            im = im[:, d:d + desired_sz[1]]
            X.append(im)

    if seq_idx == 4999:
        X = np.array(X)
        hkl.dump(X, os.path.join(DATA_DIR, 'raw_bdd100k_dataset/X_bdd100k_train' + str(seq_idx - 1000 +1) + '-' + str(seq_idx) + '.hkl'))
        X = []

#merge all 5 hkl files
X=hkl.load(os.path.join(DATA_DIR, 'raw_bdd100k_dataset/X_bdd100k_train0-999.hkl'))
print('0-999')
print(X.shape)

X_tmp=hkl.load(os.path.join(DATA_DIR, 'raw_bdd100k_dataset/X_bdd100k_train1000-1999.hkl'))
X=np.append(X,X_tmp,axis=0)
print('1000-1999')
print(X.shape)

X_tmp=hkl.load(os.path.join(DATA_DIR, 'raw_bdd100k_dataset/X_bdd100k_train2000-2999.hkl'))
X=np.append(X,X_tmp,axis=0)
print('2000-2999')
print(X.shape)

X_tmp=hkl.load(os.path.join(DATA_DIR, 'raw_bdd100k_dataset/X_bdd100k_train3000-3999.hkl'))
X=np.append(X,X_tmp,axis=0)
print('3000-3999')
print(X.shape)

X_tmp=hkl.load(os.path.join(DATA_DIR, 'raw_bdd100k_dataset/X_bdd100k_train4000-4999.hkl'))
X=np.append(X,X_tmp,axis=0)
print('4000-4999')
print(X.shape)

hkl.dump(X, os.path.join(DATA_DIR, 'X_bdd100k_train0-4999' + '.hkl'))




