import numpy as np

class MotionCompensator:

    def __init__(self, search_range=4):
        self.search_range = search_range

    def compute_motion_vector(self, ref_image, image):

        """
        Computes the motion vector that describes the motion between the single channeled reference image and the current image.
        The motion vector is represented as a 2D numpy array with shape [H / 8, W / 8, 1], where the x and y displacements
        are converted to a single value using the formula: motion_vector = y_displacement * (2 * search_range + 1) + x_displacement. Notice that
        displacements can take any value in the range [-search_range, search_range]. We compute the motion vectors only for
        the 8x8 non-overlapping blocks. Compute the closest indice using sum of the squared differences (SSD) between the reference block and the current block.

        ref_image: np.array of shape [H, W]
        image: np.array of shape [H, W]

        returns:
            motion_vector: np.array of shape [H / 8, W / 8, 1]
        """
        # YOUR CODE STARTS HERE
        H, W = ref_image.shape
        motion_vector = np.zeros((H // 8, W // 8, 1), dtype=int)
        R = self.search_range
        offset = (2 * R + 1)

        for i in range(0, H, 8):
            for j in range(0, W, 8):
                best_ssd = np.inf
                best_dx, best_dy = 0, 0
                target_block = image[i:i + 8, j:j + 8]

                for dy in range(-R, R + 1):
                    for dx in range(-R, R + 1):
                        ref_i = i + dy
                        ref_j = j + dx
                        if 0 <= ref_i < H - 7 and 0 <= ref_j < W - 7:
                            ref_block = ref_image[ref_i:ref_i + 8, ref_j:ref_j + 8]
                            ssd = np.sum((ref_block - target_block) ** 2)
                            if ssd < best_ssd:
                                best_ssd = ssd
                                best_dx, best_dy = dx, dy

                mv_value = best_dy * offset + best_dx
                motion_vector[i // 8, j // 8, 0] = mv_value
        # YOUR CODE ENDS HERE

        return motion_vector.astype(int)
    
    def reconstruct_with_motion_vector(self, ref_image, motion_vector):

        """
        Reconstructs the current image using the reference image and the motion vector. The motion vector is used to
        displace the 8x8 blocks in the reference image to their corresponding positions in the current image.

        ref_image: np.array of shape [H, W, C]
        motion_vector: np.array of shape [H / 8, W / 8, 1]

        returns:
            image: np.array of shape [H, W, C]
        """

        image = np.zeros_like(ref_image)

        # YOUR CODE STARTS HERE
        H, W, C = ref_image.shape
        R = self.search_range
        offset = 2 * R + 1
        image = np.zeros_like(ref_image)

        for i in range(0, H, 8):
            for j in range(0, W, 8):
                mv_value = motion_vector[i // 8, j // 8, 0]
                dx = mv_value % offset
                dy = mv_value // offset
                dx -= R
                dy -= R

                ref_i = i + dy
                ref_j = j + dx

                if 0 <= ref_i < H - 7 and 0 <= ref_j < W - 7:
                    image[i:i + 8, j:j + 8, :] = ref_image[ref_i:ref_i + 8, ref_j:ref_j + 8, :]
        # YOUR CODE ENDS HERE
        return image