import numpy as np

def calc_mse(orig: np.array, rec: np.array):
    """
    Computes the Mean Squared Error by taking the square of
    the difference between orig and rec, and averaging it
    over all the pixels.

    orig: np.array of shape [H, W, C]
    rec: np.array of shape [H, W, C]

    returns 
        mse: a scalar value
    """
    # YOUR CODE STARTS HERE
    diff = orig.astype(np.float32) - rec.astype(np.float32)
    mse = np.mean(np.square(diff))
    # YOUR CODE ENDS HERE
    return mse

def calc_psnr(orig: np.array, rec:np.array, maxval=255):
    """
    Computes the Peak Signal Noise Ratio by computing
    the MSE and using it in the formula from the lectures.

    > **_ Warning _**: Assumes the signals are in the 
    range [0, 255] by default

    orig: np.array of shape [H, W, C]
    rec: np.array of shape [H, W, C]

    returns 
        psnr: a scalar value
    """
    # YOUR CODE STARTS HERE
    mse = calc_mse(orig, rec)
    if mse == 0:
        psnr = float('inf')  # Perfect reconstruction
    else:
        psnr = 10 * np.log10((maxval ** 2) / mse)
    # YOUR CODE ENDS HERE
    return psnr