import numpy as np

def rgb2gray(image: np.array):
    """
    Computes the grayscale version of the image. 

    image: np.array of shape [H, W, C]

    returns 
        output_image: np.array of shape [H, W, 1]
    """
    output_image = np.mean(image, axis=-1, keepdims=True)
    return output_image

def rgb2ycbcr(image: np.array):
    """
    Converts an RGB image to its YCbCr version. 

    image: np.array of shape [H, W, 3]

    returns 
        output_image: np.array of shape [H, W, 3]
    """
    output_image = np.zeros_like(image)
    # YOUR CODE STARTS HERE
    matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])
    offset = np.array([0, 128, 128])

    output_image = image @ matrix.T + offset
    # YOUR CODE ENDS HERE
    return output_image

def ycbcr2rgb(image: np.array):
    """
    Converts an YCbCr image to its RGB version. 

    image: np.array of shape [H, W, 3]

    returns 
        output_image: np.array of shape [H, W, 3]
    """
    output_image = np.zeros_like(image)
    # YOUR CODE STARTS HERE
    matrix = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])
    offset = np.array([0, 128, 128])
    output_image = (image - offset) @ matrix.T
    output_image = np.clip(output_image, 0, 255)
    # YOUR CODE ENDS HERE
    return output_image