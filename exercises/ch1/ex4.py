import numpy as np
from ivclab.image.yuv420codec import yuv420compression
from ivclab.utils import imread, calc_psnr

# read images
img_lena = imread('D:/Pycharm/ivclab/data/lena.tif')
img_sail = imread('D:/Pycharm/ivclab/data/sail.tif')

img_lena_rec = yuv420compression(img_lena)
img_sail_rec = yuv420compression(img_sail)

psnr_lena = calc_psnr(img_lena, img_lena_rec)
psnr_sail = calc_psnr(img_sail, img_sail_rec)

print(f'PSNR of lena is {psnr_lena:.3f} dB\n')
print(f'PSNR of sail is {psnr_sail:.3f} dB\n')