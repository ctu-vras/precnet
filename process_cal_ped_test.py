'''
Code for downloading and processing Caltech Pedestrian Dataset - test part (P. Dollar et al. 2009, http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)
Based on code related to PredNet - Lotter et al. 2016
Method of resizing was specified (bicubic) 
'''

import os
import numpy as np
from scipy.misc import imresize, imread
import hickle as hkl
from kitti_settings import *


desired_im_sz = (128, 160)

IMGS_DIR = DATA_DIR+'raw_cal_ped_dataset/imgs/'


# Create image datasets.
# Processes images and saves them
def process_data():
    im_list = []
    source_list = []  # corresponds to recording that image came from
    for setn in ['set06','set07','set08','set09','set10']:
        im_dir = os.path.join(IMGS_DIR,setn)
        #sort in this order 'set06/V000', 'set06/V001',.., 'set10/V011'
        folders = sorted(list(os.walk(im_dir, topdown=False))[-1][-2])
        for seq in folders:
            seq_dir = im_dir + '/' + seq
            files = list(os.walk(seq_dir, topdown=False))[-1][-1]
            im_list_orig_fps = [seq_dir + '/' + f for f in sorted(files)]
            source_list_orig_fps = [setn + '/' + seq] * len(files)
            #change 30fps to 10 fps
            offset=0
            shift=3
            im_list += [im_list_orig_fps[i] for i in list(range(offset,len(im_list_orig_fps),shift))]
            source_list += [source_list_orig_fps[i] for i in list(range(offset,len(source_list_orig_fps),shift))]


    print( 'Creating test data: ' + str(len(im_list)) + ' images')
    X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
    for i, im_file in enumerate(im_list):
        im = imread(im_file)
        X[i] = process_im(im, desired_im_sz)


    hkl.dump(X, os.path.join(DATA_DIR, 'X_pedest_test' + '.hkl'))
    hkl.dump(source_list, os.path.join(DATA_DIR, 'sources_pedest_test' + '.hkl'))


# resize and crop image
def process_im(im, desired_sz):
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))), 'bicubic')
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    return im


if __name__ == '__main__':
    process_data()
