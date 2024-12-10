import numpy as np
from einops import rearrange

class ZeroRunCoder:

    def __init__(self, end_of_block=4000, block_size = 64):
        self.EOB = end_of_block
        self.block_size = block_size

    def encode(self, flat_patch_img: np.array):
        """
        This function gets a flattened patched image and produces a list of 
        symbols that applies a zero run encoding of the input where sequential
        blocks of zeroes (e.g. [... 0 0 0 0 0 ...]) are replaced with a marker zero
        and the number of additional zeroes (e.g. [... 0 4 ...]). The original sequence
        is processed in blocks of block_size and every encoding of a block ends with an
        end of block symbol. If all the original values are zero until the end of block,
        then no marker is necessary and we can put an EOB symbol directly.

        flat_patch_img: np.array of shape [H_patch, W_patch, C, Block_size]

        returns:
            encoded: List of symbols that represent the original elements
        
        """
        flat_img = rearrange(flat_patch_img, 'h w c p -> (h w c) p', p=self.block_size)
        # YOUR CODE STARTS HERE
        raise NotImplementedError()
        # YOUR CODE ENDS HERE
        return encoded
    
    def decode(self, encoded, patch_shape):
        """
        This function gets an encoding and the original shape to decode the elements 
        of the original array. It acts as the inverse function of the decoder.

        encoded: List of symbols that represent the original elements
        patch_shape: List of 3 numbers that represent number of H_patch, W_patch and C

        returns:
            flat_patch_img: np.array of shape [H_patch, W_patch, C, Block_size]
        
        """
        
        # YOUR CODE STARTS HERE
        raise NotImplementedError()
        # YOUR CODE ENDS HERE

        flat_patch_img = rearrange(
            flat_img, 
            '(h w c) p -> h w c p', 
            h = original_shape[0], w = original_shape[1], 
            c = original_shape[2], p=self.block_size)
        return flat_patch_img

        
        