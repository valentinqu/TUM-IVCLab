from ivclab.utils import imread, imshow, calc_mse, calc_psnr
from ivclab.signal import rgb2gray, lowpass_filter, downsample, upsample
import matplotlib.pyplot as plt
import numpy as np

# read images
img_lena = imread('D:/Pycharm/ivclab/data/lena.tif')
img_lena_gray = rgb2gray(img_lena)

img_smandril = imread('D:/Pycharm/ivclab/data/smandril.tif')
img_smandril_gray = rgb2gray(img_smandril)

img_satpic1 = imread('D:/Pycharm/ivclab/data/satpic1.bmp')

compressed_bpp = 8
#E1-1 d.
mse_smandril = calc_mse(img_smandril, img_smandril_gray)
psnr_smandril = calc_psnr(img_smandril, img_smandril_gray)
print(f"MSE_smandril: {mse_smandril:.2f}")
print(f"PSNR_smandril: {psnr_smandril:.2f} dB")


#E1-2 b.
kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32)

kernel = kernel / np.sum(kernel)

H = np.fft.fft2(kernel)
H_shifted = np.fft.fftshift(H)
H_magnitude = np.abs(H_shifted)

#E1-2 c.
img_satpic1_filtered = lowpass_filter(img_satpic1, kernel)

#E1-2 d.
img_satpic1_filtered_ds_h = downsample(img_satpic1_filtered)
img_satpic1_filtered_ds = downsample(img_satpic1_filtered_ds_h.transpose(1, 0, 2)).transpose(1, 0, 2)

img_satpic1_filtered_us_h = upsample(img_satpic1_filtered_ds)
img_satpic1_filtered_us = upsample(img_satpic1_filtered_us_h.transpose(1, 0, 2)).transpose(1, 0, 2)

#E1-2 e.
img_rec_pre = lowpass_filter(img_satpic1_filtered_us, kernel)
img_satpic1_ds_nopre = downsample(img_satpic1)
img_satpic1_ds_nopre = downsample(img_satpic1_ds_nopre.transpose(1, 0, 2)).transpose(1, 0, 2)
img_satpic1_us_nopre = upsample(img_satpic1_ds_nopre)
img_satpic1_us_nopre = upsample(img_satpic1_us_nopre.transpose(1, 0, 2)).transpose(1, 0, 2)
img_rec_nopre = lowpass_filter(img_satpic1_us_nopre, kernel)

psnr_with_prefilter = calc_psnr(img_satpic1, img_rec_pre)
psnr_without_prefilter = calc_psnr(img_satpic1, img_rec_nopre)

print(f"PSNR with prefiltering: {psnr_with_prefilter:.2f} dB")
print(f"PSNR without prefiltering: {psnr_without_prefilter:.2f} dB")


# plot images
fig, axs = plt.subplots(2,2)
imshow(axs[0][0], img_lena, title='Original Lena Image')
imshow(axs[0][1], img_lena_gray, title='Compressed Lena Image')
imshow(axs[1][0], img_smandril, title='Original Smandril Image')
imshow(axs[1][1], img_smandril_gray, title='Compressed Smandril Image')

plt.show()


plt.figure()
plt.plot([compressed_bpp], [psnr_smandril], 'ro', label='Grayscale compression')
plt.plot([compressed_bpp], [psnr_lena_rec], 'go', label='Grayscale compression')
plt.plot([compressed_bpp], [psnr_monarch_rec], 'bo', label='Grayscale compression')
plt.title('Rate–Distortion Point: RGB to Grayscale')
plt.xlabel('Bit per Pixel (bpp)')
plt.ylabel('PSNR (dB)')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(6, 5))
plt.title("Frequency Response of Lowpass Filter")
plt.matshow(H_magnitude, fignum=1, cmap='viridis')
plt.colorbar(label='Magnitude')
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
imshow(axs[0], img_satpic1, title='Original Satpic Image')
imshow(axs[1], img_satpic1_filtered, title='Filtered Satpic Image')
plt.show()
