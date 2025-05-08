from ivclab.utils import imread, imshow, calc_mse, calc_psnr
from ivclab.signal import rgb2gray, lowpass_filter, downsample, upsample
import matplotlib.pyplot as plt
import numpy as np

# read images
img_lena = imread('D:/Pycharm/ivclab/data/lena.tif')
img_lena_gray = rgb2gray(img_lena)

img_smandril = imread('D:/Pycharm/ivclab/data/smandril.tif')
img_smandril_gray = imread('D:/Pycharm/ivclab/data/smandril_rec.tif')

mse = calc_mse(img_smandril, img_smandril_gray)
psnr = calc_psnr(img_smandril, img_smandril_gray)
print(f'MSE of lena.tif is {mse:.3f}')
print(f'PSNR of lena.tif is {psnr:.3f} dB')


# plot images
fig, axs = plt.subplots(2,2)
imshow(axs[0][0], img_lena, title='Original Lena Image')
imshow(axs[0][1], img_lena_gray, title='Compressed Lena Image')
imshow(axs[1][0], img_smandril, title='Original Smandril Image')
imshow(axs[1][1], img_smandril_gray, title='Compressed Smandril Image')

plt.show()
