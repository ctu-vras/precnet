#!/bin/bash

#Code for downloading test part of Caltech Pedestrian Dataset (Dollar et al. 2012, http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)

mkdir -p datasets/raw_cal_ped_dataset

cd datasets/raw_cal_ped_dataset

wget http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/set06.tar
tar -xvf set06.tar
rm set06.tar

wget http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/set07.tar
tar -xvf set07.tar
rm set07.tar

wget http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/set08.tar
tar -xvf set08.tar
rm set08.tar

wget http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/set09.tar
tar -xvf set09.tar
rm set09.tar


wget http://www.vision.caltech.edu.s3-us-west-2.amazonaws.com/Image_Datasets/CaltechPedestrians/datasets/USA/set10.tar
tar -xvf set10.tar
rm set10.tar



