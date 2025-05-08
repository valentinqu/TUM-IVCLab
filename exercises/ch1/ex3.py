from ivclab.utils import imread, calc_psnr, imshow
from ivclab.signal import FilterPipeline, lowpass_filter
import matplotlib.pyplot as plt
import numpy as np

image = imread('D:/Pycharm/ivclab/data/satpic1.bmp')

kernel = np.asarray(
    [[1,2,1],
     [2,4,2],
     [1,2,1]]
)

H = np.fft.fft2(kernel)
H_shifted = np.fft.fftshift(H)
H_magnitude = np.abs(H_shifted)

plt.figure(figsize=(6, 5))
plt.title("Frequency Response of Lowpass Filter")
plt.matshow(H_magnitude, fignum=1, cmap='viridis')
plt.colorbar(label='Magnitude')
plt.show()

filtered = lowpass_filter(image, kernel)

fig, axs = plt.subplots(1,2)
imshow(axs[0], image, title='original image')
imshow(axs[1], filtered, title='image with lowpass filter')

plt.show()

pipeline = FilterPipeline(kernel=kernel)

recon_image_not_pre = pipeline.filter_img(image, False)
recon_image_pre = pipeline.filter_img(image, True)

psnr_not_pre = calc_psnr(image, recon_image_not_pre)
psnr_pre = calc_psnr(image, recon_image_pre)

print(f"Reconstructed image, not prefiltered, PSNR = {psnr_not_pre:.2f} dB")
print(f"Reconstructed image, prefiltered, PSNR = {psnr_pre:.2f} dB")

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
imshow(axs[0], recon_image_not_pre, title='Filtered Image')
imshow(axs[1], recon_image_pre, title='Downsampled (1/4 size)')
plt.show()


