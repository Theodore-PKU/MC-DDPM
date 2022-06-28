import matplotlib.pyplot as plt
import numpy as np

import torch as th


def show_single_image(image):
    """
    Show image of real or complex (computing the magnitude).

    :param image: numpy.array of float or complex with shape of (h, w).

    :return: None
    """
    if image.dtype.name.startswith("complex"):
        image = np.abs(image)
    plt.imsave("temp.png", image, cmap="gray")


def center_crop_image(image, size):
    """
    Crop the center part of image at last two axes.

    :param image: any type array of shape (..., h, w).
    :param size: tuple or list of int, two elements, (h1, w1).

    :return: the same type array of shape (..., h1, w1).
    """
    h, w = image.shape[-2:]
    h1, w1 = size
    if h < h1 or w < w1:
        raise ValueError("the value of size is not applicable.")
    up_index = (h - h1) // 2
    down_index = up_index + h1
    left_index = (w - w1) // 2
    right_index = left_index + w1
    return image[..., up_index: down_index, left_index: right_index]


def magnitude(image):
    return th.sqrt(image[:, 0, ...] ** 2 + image[:, 1, ...] ** 2).unsqueeze(1)


def tile_image(batch_image, ncols, nrows):
    assert ncols * nrows == batch_image.shape[0]
    _, channels, height, width = batch_image.shape
    batch_image = batch_image.view(nrows, ncols, channels, height, width)
    batch_image = batch_image.permute(2, 0, 3, 1, 4)
    batch_image = batch_image.contiguous().view(channels, nrows * height, ncols * width)
    return batch_image
