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
        flat_img = rearrange(flat_patch_img, 'h w c p-> (h w c) p', p=self.block_size)
        # YOUR CODE STARTS HERE
        encoded = []

        for block in flat_img:
            i = 0
            while i < self.block_size:
                if block[i] == 0:
                    run_length = 0
                    # 统计连续 0 的数量
                    while i < self.block_size and block[i] == 0:
                        run_length += 1
                        i += 1
                    if i == self.block_size:
                        encoded.append(self.EOB)
                    else:
                        encoded.append(0)
                        encoded.append(run_length)
                else:
                    encoded.append(block[i])
                    i += 1
            if block[-1] != 0:
                encoded.append(self.EOB)

        # YOUR CODE ENDS HERE
        encoded = np.array(encoded, dtype=np.int32)
        return encoded
    
    def decode(self, encoded, original_shape):
        """
        This function gets an encoding and the original shape to decode the elements 
        of the original array. It acts as the inverse function of the encoder.

        encoded: List of symbols that represent the original elements
        original_shape: List of 3 numbers that represent number of H_patch, W_patch and C

        returns:
            flat_patch_img: np.array of shape [H_patch, W_patch, C, Block_size]
        
        """
        
        # YOUR CODE STARTS HERE
        decoded_blocks = []
        block = []
        i = 0

        while i < len(encoded):
            symbol = encoded[i]
            if symbol == self.EOB:
                while len(block) < self.block_size:
                    block.append(0)
                decoded_blocks.append(block)
                block = []
                i += 1
            elif symbol == 0:
                run_length = encoded[i + 1]
                block.extend([0] * run_length)
                i += 2
            else:
                block.append(symbol)
                i += 1

        flat_img = np.array(decoded_blocks, dtype=np.int32)
        flat_patch_img = rearrange(
            flat_img,
            '(h w c) p -> h w c p',
            h=original_shape[0], w=original_shape[1], c=original_shape[2], p=self.block_size)
        # YOUR CODE ENDS HERE
        flat_patch_img = rearrange(
            flat_img,
            '(h w c) p -> h w c p', 
            h = original_shape[0], w = original_shape[1], 
            c = original_shape[2], p=self.block_size)
        return flat_patch_img

            
            