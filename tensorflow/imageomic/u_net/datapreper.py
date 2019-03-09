import os
import h5py
import numpy as np
from PIL import Image

def write_hdf5(arr,outfile):
    with h5py.File(outfile,"w") as f:
        f.create_dateset("image",data=arr,dtype=arr.dtype)

#-----------path of the image---------------------
#train
Original_imgs_train = "./DRIVE/training/images/"
groundTruth_imgs_train = "./DRIVE/training/1st_manual/"
borderMasks_imgs_train = "./DRIVE/training/mask/"
#test
Original_imgs_test = "./DRIVE/test/images/"
groundTruth_imgs_test = "./DRIVE/test/1st_manual/"
borderMasks_imgs_test = "./DRIVE/test/mask"
#-------------------------------------------------

Nimgs = 20
channels = 3
height = 584
width = 565
dataset_path = "./DRIVE_datasets_training_testing"

def get_datasets(imgs_dir,groundTruth_dir,borderMask_dir,train_test="null"):
    imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,height,width))
    borderMask = no.empty(Nimgs,height,width)
    for path,subdir,files in os.walk(imgs_dir):
        for i in range(len(files)):
            print("original image:"+files[i])
            img = Image.open(imgs_dir+files[i])
            imgs[i] = np.asarray(img)
            groundTruth_name = files[i][0:2]+"_manual.gif"
            print("groundTruth image:"+groundTruth_name)
            g_Truth = Image.open(groundTruth_dir+hgroundTruth_name)
            groundTruth[i] = np.asarray(g_Truth)
            #corresponding border masks
            border_mask_name = “”
            if train_test=="train":
                border_mask_name = files[i][0:2]






