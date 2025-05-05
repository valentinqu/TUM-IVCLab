from ivclab.utils import imread, imshow, calc_psnr
import matplotlib.pyplot as plt

# read images
img_lena = imread('D:/Pycharm/ivclab/data/lena.tif')
img_lena_compressed = imread('D:/Pycharm/ivclab/data/lena_compressed.tif')
img_monarch = imread('D:/Pycharm/ivclab/data/monarch.tif')
img_monarch_compressed = imread('D:/Pycharm/ivclab/data/monarch_compressed.tif')

# YOUR CODE STARTS HERE

# Compute the PSNR values for Lena and Monarch compression
psnr_lena = calc_psnr(img_lena, img_lena_compressed)
psnr_monarch = calc_psnr(img_monarch, img_monarch_compressed)

# YOUR CODE ENDS HERE

# print metrics
print(f'PSNR of lena.tif is {psnr_lena:.3f} dB\n')
print(f'PSNR of monarch.tif is {psnr_monarch:.3f} dB\n')

# plot images
fig, axs = plt.subplots(2,2)
imshow(axs[0][0], img_lena, title='Original Lena Image')
imshow(axs[0][1], img_lena_compressed, title='Compressed Lena Image')
imshow(axs[1][0], img_monarch, title='Original Monarch Image')
imshow(axs[1][1], img_monarch_compressed, title='Compressed Monarch Image')

plt.show()

