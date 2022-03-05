import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import contextlib
import torch as th
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# --------data reading--------

def read_datafile(data_dir, file_name, slice_index):
    """
    Read mri data from .h5 file.
    :param data_dir: str, directory saving data.
    :param file_name: str, file name of selected data.
    :param slice_index: int, index of selected slice.
    :return: tuple of (str, numpy.array of complex64, numpy.array of float32),
        acquisition, raw kspace with shape larger than (320, 320), and rss image with shape of (320, 320).
    """
    file_path = os.path.join(data_dir, file_name)
    data = h5py.File(file_path, mode="r")
    acquisition = data.attrs["acquisition"]
    kspace_raw = np.array(data["kspace"])[slice_index]
    image_rss = np.array(data["reconstruction_rss"])[slice_index]
    return acquisition, kspace_raw, image_rss


# --------show images--------

def show_single_image(image):
    """
    Show image of real or complex (computing the magnitude).
    :param image: numpy.array of float or complex with shape of (h, w).
    :return: None
    """
    if image.dtype.name.startswith("complex"):
        image = np.abs(image)
    plt.imshow(image, "gray")


# --------FFT transform--------

def ifftc_np_from_raw_data(kspace_raw):
    """
    Inverse orthogonal FFT2 transform raw kspace data to feasible complex image, numpy.array to numpy.array.
    :param kspace_raw: numpy.array of complex with shape of (h, w), raw kspace data from .h5 file.
    :return: numpy.array of complex with shape of (h, w), transformed image, keep dtype.
    """
    transformed_image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_raw), norm="ortho"))
    transformed_image = transformed_image.astype(kspace_raw.dtype)
    return transformed_image


def fftc_np(image):
    """
    Orthogonal FFT2 transform image to kspace data, numpy.array to numpy.array.
    :param image: numpy.array of complex with shape of (h, w), mri image.
    :return: numpy.array of complex with shape of (h, w), kspace data with center low-frequency, keep dtype.
    """
    kspace = np.fft.fftshift(np.fft.fft2(image, norm="ortho"))
    kspace = kspace.astype(image.dtype)
    return kspace


def ifftc_np(kspace):
    """
    Inverse orthogonal FFT2 transform kspace data to image, numpy.array to numpy.array.
    :param kspace: numpy.array of complex with shape of (h, w) and center low-frequency, kspace data.
    :return: numpy.array of complex with shape of (h, w), transformed image, keep dtype.
    """
    image = np.fft.ifft2(np.fft.ifftshift(kspace), norm="ortho")
    image = image.astype(kspace.dtype)
    return image


# not used
# def fftshift_th(x):
#     """
#     Similar to np.fft.fftshift but applies to th.Tensor.
#     :param x: th.Tensor of real with shape of (h, w, 2).
#     :return: th.Tensor of real with shape of (h, w, 2), shifted along axis of (h, w).
#     """
#     shift = [x.shape[-3] // 2, x.shape[-2] // 2]
#     return th.roll(x, shifts=shift, dims=(-3, -2))
#
#
# def ifftshift_th(x):
#     """
#     Similar to np.fft.ifftshift but applies to th.Tensor.
#     :param x: th.Tensor of real with shape of (h, w, 2).
#     :return: th.Tensor of real with shape of (h, w, 2), shifted along axis of (h, w).
#     """
#     shift = [(x.shape[-3] + 1) // 2, (x.shape[-2] + 1) // 2]
#     return th.roll(x, shifts=shift, dims=(-3, -2))


def fftc_th(image):
    """
    Orthogonal FFT2 transform image to kspace data, th.Tensor to th.Tensor.

    :param image: th.Tensor of real with shape of (..., 2, h, w), mri image.
    :return: th.Tensor of real with shape of (..., 2, h, w), kspace data with center low-frequency, keep dtype.
    """
    image = image.permute(0, 2, 3, 1).contiguous()
    kspace = th.fft.fftshift(th.fft.fft2(th.view_as_complex(image), norm="ortho"), dim=(-1, -2))
    kspace = th.view_as_real(kspace).permute(0, 3, 1, 2).contiguous()
    return kspace


def ifftc_th(kspace):
    """
    Inverse orthogonal FFT2 transform kspace data to image, th.Tensor to th.Tensor.

    :param kspace: th.Tensor of real with shape of (..., 2, h, w), kspace data with center low-frequency.
    :return: th.Tensor of real with shape of (..., 2, h, w), mri image, keep dtype.
    """
    kspace = kspace.permute(0, 2, 3, 1).contiguous()
    image = th.fft.ifft2(th.fft.ifftshift(th.view_as_complex(kspace), dim=(-1, -2)), norm="ortho")
    image = th.view_as_real(image).permute(0, 3, 1, 2).contiguous()
    return image


# --------dtype transform--------

def complex2real_np(x):
    """
    Change a complex numpy.array to a real array with two channels.
    :param x: numpy.array of complex with shape of (h, w).
    :return: numpy.array of real with shape of (2, h, w).
    """
    return np.stack([x.real, x.imag])


def real2complex_np(x):
    """
    Change a real numpy.array with two channels to a complex array.
    :param x: numpy.array of real with shape of (2, h, w).
    :return: numpy.array of complex64 with shape of (h, w).
    """
    x_real = x[0]
    x_imag = x[1]
    x = np.zeros_like(x[0, ...], dtype=np.complex64)
    x.real = x_real
    x.imag = x_imag
    return x


def np2th(x):
    return th.tensor(x)


def th2np(x):
    return x.detach().cpu().numpy()


def np_comlex_to_th_real2c(x):
    """
    Transform numpy.array of complex to th.Tensor of real with 2 channels.
    :param x: numpy.array of complex with shape of (h, w).
    :return: th.Tensor of real with 2 channels with shape of (h, w, 2).
    """
    return np2th(complex2real_np(x).transpose((1, 2, 0)))


def th_real2c_to_np_complex(x):
    """
    Transform th.Tensor of real with 2 channels to numpy.array of complex.
    :param x: th.Tensor of real with 2 channels with shape of (h, w, 2).
    :return: numpy.array of complex with shape of (h, w).
    """
    return real2complex_np(th2np(x.permute(2, 0, 1)))


def th2np_magnitude(x):
    """
    Compute the magnitude of torch.Tensor with shape of (c, 2, h, w).

    :param x: th.Tensor of real with 2 channels with shape of (c, 2, h, w).
    :return: numpy.array of real with shape of (c, h, w).
    """
    x = th2np(x)
    return np.sqrt(x[:, 0, ...] ** 2 + x[:, 1, ...] ** 2)


# --------generate mask--------

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
    subsampling outer k-space regions based on the undersampling factor.
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


# -------- evaluation metric --------

def compute_mse(arr1, arr2):
    """
    Compute MSE
    """
    return np.mean((arr1 - arr2) ** 2)


def compute_psnr(arr1, arr2):
    """
    Compute PSNR.
    """
    return peak_signal_noise_ratio(arr1, arr2, data_range=arr1.max())


def compute_nmse(gt, pred):
    """
    Compute Normalized Mean Squared Error (NMSE)
    """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def compute_ssim(gt, pred):
    """
    Compute Structural Similarity (SSIM), gt and pred are with shape of (C, H, W).
    """
    return structural_similarity(
        gt, pred,
        data_range=gt.max(),
        channel_axis=0
    )