import numpy as np
import torch as th


# -------- FFT transform --------

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


# -------- dtype transform --------

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
    complex_x = np.zeros_like(x[0, ...], dtype=np.complex64)
    complex_x.real, complex_x.imag = x[0], x[1]
    return complex_x


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
    Compute the magnitude of torch.Tensor with shape of (b, 2, h, w).

    :param x: th.Tensor of real with 2 channels with shape of (b, 2, h, w).

    :return: numpy.array of real with shape of (b, h, w).
    """
    x = th2np(x)
    return np.sqrt(x[:, 0, ...] ** 2 + x[:, 1, ...] ** 2)
