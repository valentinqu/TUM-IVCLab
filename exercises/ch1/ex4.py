import numpy as np
from ivclab.utils import imread
from ivclab.signal import rgb2ycbcr
from scipy.signal import resample

image = imread('data/sail.tif')

# RGB → YCbCr
ycbcr = rgb2ycbcr(image)
Y, Cb, Cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]

# Assume Cb is a chroma plane
H, W = Cb.shape

# Pad manually
Cb_padded = np.pad(Cb, ((4, 4), (4, 4)), mode='symmetric')

# Resample
Cb_half_w = resample(Cb_padded, num=Cb_padded.shape[1] // 2, axis=1)  # horizontal
Cb_half_hw = resample(Cb_half_w, num=Cb_half_w.shape[0] // 2, axis=0)  # vertical

# Crop manually
Cb_result = np.round(Cb_half_hw[2:-2, 2:-2])