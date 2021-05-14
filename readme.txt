Predictions / Masks / Images tiles creation Project

This project has the aim to create road segmentation predictions from keras model (U-Net Architecture - backbone ResNet).
Input:
    .model/xxxxx.hdf5 - model trained
    .orig_test_dataset/img/xxx.tiff - RGB images (24bits) from Massachusetts test dataset (https://www.cs.toronto.edu/~vmnih/data/)
    .orig_test_dataset/mask/xxx.tif - Mask images (8bits) from Massachusetts test dataset (https://www.cs.toronto.edu/~vmnih/data/)

Output:
    .img/xxxxx_0yyy.png - RGB tiles (Height = Width)
    .mask/xxxxx_0yyy.png - Mask tiles (Height = Width)
    .predict/xxxxx_0yyy.png - Prediction tiles (Height = Width) ** without threshold rate on predictions.

Obs: It is possible create overlap among tiles changing the stride variable.

Requirements:
tensorflow==2.3.1
pillow>=8.2
keras-unet==0.1.2
matplotlib>=3.4
numpy>=1.18