import numpy as np
from scipy.signal import decimate, resample
from ivclab.signal.color import rgb2ycbcr, ycbcr2rgb

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
    # Convert RGB to YCbCr
    yuv_image = rgb2ycbcr(image)

    # Split the channels
    Y, Cb, Cr = yuv_image[:, :, 0], yuv_image[:, :, 1], yuv_image[:, :, 2]

    # Pad the image with 4 symmetric pixels on each side
    Y_padded = np.pad(Y, 2, mode='reflect')
    Cb_padded = np.pad(Cb, 2, mode='reflect')
    Cr_padded = np.pad(Cr, 2, mode='reflect')

    # Downsample Cb and Cr channels using decimate
    Cb_downsampled = decimate(Cb_padded, 2, axis=0)
    Cb_downsampled = decimate(Cb_downsampled, 2, axis=1)
    Cr_downsampled = decimate(Cr_padded, 2, axis=0)
    Cr_downsampled = decimate(Cr_downsampled, 2, axis=1)

    # Crop 2 pixels from each side
    Y_cropped = Y_padded[2:-2, 2:-2]
    Cb_cropped = Cb_downsampled[2:-2, 2:-2]
    Cr_cropped = Cr_downsampled[2:-2, 2:-2]

    # Rounding Y, Cb, and Cr channels
    Y_rounded = np.round(Y_cropped)
    Cb_rounded = np.round(Cb_cropped)
    Cr_rounded = np.round(Cr_cropped)

    # Pad the rounded channels with 4 symmetric pixels on each side
    Y_padded = np.pad(Y_rounded, 2, mode='reflect')
    Cb_padded = np.pad(Cb_rounded, 2, mode='reflect')
    Cr_padded = np.pad(Cr_rounded, 2, mode='reflect')

    # Upsample Cb and Cr channels using resample
    Cb_upsampled = resample(Cb_padded, Cb_padded.shape[0]*2, axis=0)
    Cb_upsampled = resample(Cb_upsampled, Cb_padded.shape[1]*2, axis=1)
    Cr_upsampled = resample(Cr_padded, Cr_padded.shape[0]*2, axis=0)
    Cr_upsampled = resample(Cr_upsampled, Cr_padded.shape[1]*2, axis=1)

    # Crop 2 pixels from each side of upsampled channels
    Y_final = Y_padded[2:-2, 2:-2]
    Cb_final = Cb_upsampled[2:-2, 2:-2]
    Cr_final = Cr_upsampled[2:-2, 2:-2]

    # Recombine channels
    yuv_reconstructed = np.stack([Y_final, Cb_final, Cr_final], axis=-1)

    # Convert back to RGB
    output = ycbcr2rgb(yuv_reconstructed)
    #YOUR CODE ENDS HERE

    # Cast output to integer again
    output = np.round(output).astype(np.uint8)
    return output