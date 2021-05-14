from tensorflow import data
from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from glob import glob
from keras_unet.utils import get_patches
import os

# only for load keras model
def f1():
    pass
def mean_iou():
    pass
def tversky_loss():
    pass


""" choose size of image and stride. Height and Width of image are same. size mode (stride) needs to equal zero """
size_stride = [224, 224]
img_test_orig = glob(pathname='orig_test_dataset/img/*.tiff')
mask_test_orig = glob(pathname='orig_test_dataset/mask/*.tif')

""" sort the lists and load keras model"""
img_test_orig.sort()
mask_test_orig.sort()
model_ = models.load_model(filepath='model/weights.60-0.36.hdf5',
                           custom_objects={'tversky_loss': tversky_loss, 'f1': f1, 'mean_iou': mean_iou})

""" function to get RGB and mask tiles 224x224"""
def get_tiles(img_, mask_, size):
    x = np.array(Image.open(img_))
    y = np.array(Image.open(mask_))
    print("img shape: ", str(x.shape))
    print("mask shape: ", str(y.shape))
    x_crops = get_patches(img_arr=x,  # required - array of images to be cropped
                          size=size[0],  # default is 224
                          stride=size[1])  # default is 56
    print("img_crops shape: ", str(x_crops.shape))
    y_crops = get_patches(img_arr=y.reshape(y.shape[0], y.shape[1], 1),
                          size=size[0],
                          stride=size[1])
    print("img_crops shape: ", str(y_crops.shape))
    return x_crops, y_crops


""" function to get predictions from keras model"""
def get_pred(img_tile_):
    img_tile_ = img_tile_/255.
    _test = data.Dataset.from_tensor_slices(img_tile_)

    final = _test.batch(4)
    return final


""" build folders"""
if not os.path.exists(path='img'):
    os.makedirs('img')
if not os.path.exists(path='mask'):
    os.makedirs('mask')
if not os.path.exists(path='predict'):
    os.makedirs('predict')

""" create images, masks and predictions tiles"""
for img, mask in zip(img_test_orig, mask_test_orig):
    name_i = os.path.basename(img).split('.')[0]
    img_tile, mask_tile = get_tiles(img, mask, size_stride)
    predic_ = model_.predict(get_pred(img_tile))
    n = len(predic_)
    for k in range(n):
        plt.imsave(f'img/{name_i}_0{k}.png', img_tile[k, ])
        plt.imsave(f'mask/{name_i}_0{k}.png', mask_tile[k, :, :, 0], cmap='gray')
        plt.imsave(f'predict/{name_i}_0{k}.png', predic_[k, :, :, 0], cmap='gray')
