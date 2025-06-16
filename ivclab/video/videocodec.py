import numpy as np
from ivclab.entropy import ZeroRunCoder
from ivclab.image import IntraCodec
from ivclab.entropy import HuffmanCoder, stats_marg
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
from ivclab.video import MotionCompensator
from ivclab.utils import calc_psnr
import matplotlib.pyplot as plt


class VideoCodec:

    def __init__(self, 
                 quantization_scale = 1.0,
                 bounds = (-1000, 4000),
                 end_of_block = 4000,
                 block_shape = (8,8),
                 search_range = 4
                 ):
        
        self.quantization_scale = quantization_scale
        self.bounds = bounds
        self.end_of_block = end_of_block
        self.block_shape = block_shape
        self.search_range = search_range

        self.intra_codec = IntraCodec(quantization_scale=quantization_scale, bounds=bounds, end_of_block=end_of_block, block_shape=block_shape)
        self.residual_codec = IntraCodec(quantization_scale=quantization_scale, bounds=bounds, end_of_block=end_of_block, block_shape=block_shape)
        self.motion_comp = MotionCompensator(search_range=search_range)
        self.motion_huffman = HuffmanCoder(lower_bound= -((2*search_range + 1)**2 - 1)//2)

        self.decoder_recon = None
    
    def encode_decode(self, frame, frame_num=0):

        if frame_num == 0:
        # YOUR CODE STARTS HERE
            bitstream, residual_bitsize = self.intra_codec.intra_encode(frame, return_bpp=True, is_source_rgb=True)
            self.decoder_recon = self.intra_codec.intra_decode(bitstream, frame.shape)
            motion_bitsize = 0
        else:
            # 1. Convert to YCbCr
            frame_ycbcr = rgb2ycbcr(frame)
            ref_ycbcr = rgb2ycbcr(self.decoder_recon)

            # 2. Motion Estimation (only on Y channel)
            motion_vector = self.motion_comp.compute_motion_vector(ref_ycbcr[..., 0], frame_ycbcr[..., 0])
            motion_vector_flat = motion_vector.ravel()
            self.motion_huffman.train(motion_vector_flat, np.arange(self.motion_huffman.lower_bound, -self.motion_huffman.lower_bound + 1))

            # 3. Motion Compensation to get predicted frame
            predicted = self.motion_comp.reconstruct_with_motion_vector(ref_ycbcr, motion_vector)

            # 4. Compute Residual: 当前帧 - 预测帧
            residual = frame_ycbcr - predicted

            # 5. Encode residual with IntraCodec
            residual_stream, residual_bitsize = self.residual_codec.intra_encode(residual, return_bpp=True,
                                                                                 is_source_rgb=False)

            # 6. Encode motion vectors with Huffman
            motion_stream, motion_bitsize = self.motion_huffman.encode(motion_vector.ravel())

            # 7. Decode过程：motion + residual → 当前帧
            motion_vector_decoded = self.motion_huffman.decode(motion_stream, motion_vector.size).reshape(
                motion_vector.shape)
            predicted = self.motion_comp.reconstruct_with_motion_vector(ref_ycbcr, motion_vector_decoded)
            recon_residual = self.residual_codec.intra_decode(residual_stream, residual.shape)
            recon_ycbcr = predicted + recon_residual
            self.decoder_recon = ycbcr2rgb(recon_ycbcr)
            bitstream = (residual_stream, motion_stream)
        # YOUR CODE ENDS HERE
        bitsize = residual_bitsize + motion_bitsize
        return self.decoder_recon, bitstream, bitsize
