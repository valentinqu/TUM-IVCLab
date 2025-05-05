import numpy as np
from scipy.signal import decimate, resample
from ivclab.signal import rgb2gray, rgb2ycbcr, ycbcr2rgb

def yuv420compression(image: np.ndarray):
    """
    Steps:
    1. Convert an image from RGB to YCbCr
    2. Compress the image
        A. Pad the image with 4 pixels symmetric pixels on each side
        B. Downsample only Cb and Cr channels with prefiltering (use scipy.signal.decimate for it)
        C. Crop the image 2 pixels from each side to get rid of padding
    3. Apply rounding to Y, Cb and Cr channels
    4. Decompress the image
        A. Pad the image with 4 pixels symmetric pixels on each side
        B. Upsample Cb and Cr channels (use scipy.signal.resample for it)
        C. Crop the image 2 pixels from each side to get rid of padding
    5. Convert the YCbCr image back to RGB

    image: np.array of shape [H, W, C]

    returns 
        output_image: np.array of shape [H, W, C]
    """
    # Cast image to floating point
    image = image * 1.0

    # YOUR CODE STARTS HERE
    ycbcr = rgb2ycbcr(image)
    Y, Cb, Cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]

    pad = ((4, 4), (4, 4))
    Cb_pad = np.pad(Cb, pad, mode='symmetric')
    Cr_pad = np.pad(Cr, pad, mode='symmetric')

    Cb_ds = decimate(decimate(Cb_pad, 2, axis=1, ftype='iir'), 2, axis=0, ftype='iir')
    Cr_ds = decimate(decimate(Cr_pad, 2, axis=1, ftype='iir'), 2, axis=0, ftype='iir')

    Y_crop = Y[2:-2, 2:-2]

    Y_round = np.round(Y_crop)
    Cb_ds = np.round(Cb_ds)
    Cr_ds = np.round(Cr_ds)

    Cb_pad2 = np.pad(Cb_ds, ((4, 4), (4, 4)), mode='symmetric')
    Cr_pad2 = np.pad(Cr_ds, ((4, 4), (4, 4)), mode='symmetric')

    target_shape = Y_round.shape
    Cb_us = resample(resample(Cb_pad2, 2 * Cb_pad2.shape[0], axis=0), 2 * Cb_pad2.shape[1], axis=1)
    Cr_us = resample(resample(Cr_pad2, 2 * Cr_pad2.shape[0], axis=0), 2 * Cr_pad2.shape[1], axis=1)

    Cb_crop = Cb_us[2:-2, 2:-2]
    Cr_crop = Cr_us[2:-2, 2:-2]

    ycbcr_rec = np.stack([Y_round, Cb_crop, Cr_crop], axis=-1)
    output = ycbcr2rgb(ycbcr_rec)
    #YOUR CODE ENDS HERE

    # Cast output to integer again
    output = np.round(output).astype(np.uint8)
    return output