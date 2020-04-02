#!/bin/bash

#Code for downloading selected parts of bdd100k  (Yu et al. 2018, https://bdd-data.berkeley.edu/)

mkdir -p datasets/raw_bdd100k_dataset
cd datasets/raw_bdd100k_dataset

#val part (randomly selected)
wget http://dl.yf.io/bdd-data/bdd100k/video_parts/bdd100k_videos_val_00.zip
unzip bdd100k_videos_val_00.zip
rm bdd100k_videos_val_00.zip

#train part (randomly selected)
wget http://dl.yf.io/bdd-data/bdd100k/video_parts/bdd100k_videos_train_01.zip
unzip bdd100k_videos_train_01.zip
rm bdd100k_videos_train_01.zip

wget http://dl.yf.io/bdd-data/bdd100k/video_parts/bdd100k_videos_train_07.zip
unzip bdd100k_videos_train_07.zip
rm bdd100k_videos_train_07.zip

wget http://dl.yf.io/bdd-data/bdd100k/video_parts/bdd100k_videos_train_10.zip
unzip bdd100k_videos_train_10.zip
rm bdd100k_videos_train_10.zip

wget http://dl.yf.io/bdd-data/bdd100k/video_parts/bdd100k_videos_train_18.zip
unzip bdd100k_videos_train_18.zip
rm bdd100k_videos_train_18.zip

wget http://dl.yf.io/bdd-data/bdd100k/video_parts/bdd100k_videos_train_27.zip
unzip bdd100k_videos_train_27.zip
rm bdd100k_videos_train_27.zip

wget http://dl.yf.io/bdd-data/bdd100k/video_parts/bdd100k_videos_train_38.zip
unzip bdd100k_videos_train_38.zip
rm bdd100k_videos_train_38.zip

wget http://dl.yf.io/bdd-data/bdd100k/video_parts/bdd100k_videos_train_53.zip
unzip bdd100k_videos_train_53.zip
rm bdd100k_videos_train_53.zip

wget http://dl.yf.io/bdd-data/bdd100k/video_parts/bdd100k_videos_train_57.zip
unzip bdd100k_videos_train_57.zip
rm bdd100k_videos_train_57.zip

wget http://dl.yf.io/bdd-data/bdd100k/video_parts/bdd100k_videos_train_60.zip
unzip bdd100k_videos_train_60.zip
rm bdd100k_videos_train_60.zip

wget http://dl.yf.io/bdd-data/bdd100k/video_parts/bdd100k_videos_train_68.zip
unzip bdd100k_videos_train_68.zip
rm bdd100k_videos_train_68.zip

