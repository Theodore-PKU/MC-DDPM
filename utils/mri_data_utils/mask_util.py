import contextlib
import numpy as np


ACC4 = {
    "mask_type": "random",
    "center_fractions": [0.08],
    "accelerations": [4]
}
ACC8 = {
    "mask_type": "random",
    "center_fractions": [0.04],
    "accelerations": [8]
}


@contextlib.contextmanager
def temp_seed(rng, seed):
    state = rng.get_state()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.set_state(state)


class MaskFunc(object):
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the under-sampling factor.
    """

    def __init__(self, center_fractions, accelerations):
        """
        :param center_fractions: list of float, fraction of low-frequency columns to be
                retained. If multiple values are provided, then one of these
                numbers is chosen uniformly each time.
        :param accelerations: list of int, amount of under-sampling. This should have
                the same length as center_fractions. If multiple values are
                provided, then one of these is chosen uniformly each time.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError("number of center fractions should match number of accelerations.")

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random

    def choose_acceleration(self):
        """
        Choose acceleration based on class parameters.
        """
        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        return center_fraction, acceleration


class RandomMaskFunc(MaskFunc):
    """
    RandomMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the RandomMaskFunc object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def __call__(self, shape, seed=None):
        """
        Create the mask.

        :param shape: (iterable[int]), the shape of the mask to be created.
        :param seed: (int, optional), seed for the random number generator. Setting
                the seed ensures the same mask is generated each time for the
                same shape. The random state is reset afterwards.
        :return torch.Tensor, a mask of the specified shape. Its shape should be
                (2, height, width) and the two channels are the same.
        """
        with temp_seed(self.rng, seed):
            num_cols = shape[-1]
            center_fraction, acceleration = self.choose_acceleration()

            # create the mask
            num_low_freqs = int(round(num_cols * center_fraction))
            prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
            mask_location = self.rng.uniform(size=num_cols) < prob
            pad = (num_cols - num_low_freqs + 1) // 2
            mask_location[pad: pad + num_low_freqs] = True
            mask = np.zeros(shape, dtype=np.float32)
            mask[..., mask_location] = 1.0

        return mask


def create_mask_for_mask_type(mask_type_str, center_fractions, accelerations):
    if mask_type_str == "random":
        return RandomMaskFunc(center_fractions, accelerations)
    else:
        raise Exception(f"{mask_type_str} not supported")
