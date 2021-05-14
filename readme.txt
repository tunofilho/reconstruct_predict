Predictions / Masks / Images tiles creation Project

This project aims to create road segmentation predictions from keras model (U-Net Architecture - backbone ResNet).
Input:
    .model/xxxxx.hdf5 - model trained
    .orig_test_dataset/img/xxx.tiff - RGB images (24bits) from Massachusetts test dataset (https://www.cs.toronto.edu/~vmnih/data/)
    .orig_test_dataset/mask/xxx.tif - Mask images (8bits) from Massachusetts test dataset (https://www.cs.toronto.edu/~vmnih/data/)

Output:
    .img/xxxxx_0yyy.png - RGB tiles (Height = Width)
    .mask/xxxxx_0yyy.png - Mask tiles (Height = Width)
    .predict/xxxxx_0yyy.png - Prediction tiles (Height = Width) ** without threshold rate on predictions.

Obs: It is possible to create overlapping among tiles by changing the stride variable.
