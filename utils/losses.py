"""
This code started out as a PyTorch port of Guided Diffusion Model:
https://github.com/openai/guided-diffusion/tree/912d5776a64a33e3baf3cff7eb1bcba9d9b9354c
"""

import numpy as np
import torch as th
import torch.nn.functional as F


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def noise_loss(model_output, noise, weight=None):
    """
    Compute DDPM training loss.

    :param model_output: th.Tensor of shape (b, c, h, w), epsilon_{theta, t}(x_t).
    :param noise: th.Tensor of shape (b, c, h, w), epsilon_{t}.
    :param weight: th.Tensor of shape (b, c, h, w) or (b, 1, 1, 1), used to compute mse loss.
    :return: mse loss, th.Tensor.
    """
    if weight is None:
        return F.mse_loss(model_output, noise)
    else:
        loss = th.mean((model_output - noise) ** 2 / weight)
        return loss


def score_function_loss(model_output, noise, bar_beta_t, weight=None):
    """
    Compute score-function training loss.

    :param model_output: th.Tensor of shape (b, c, h, w), s_{theta, t}(x_t).
    :param noise: th.Tensor of shape (b, c, h, w), epsilon_{t}.
    :param bar_beta_t: th.Tensor of shape (b, c, h, w) or (b, 1, 1, 1), diffusion process parameter.
    :param weight: th.Tensor of shape (b, c, h, w) or (b, 1, 1, 1), used to compute mse loss.
    :return: mse loss, th.Tensor.
    """
    if weight is None:
        return F.mse_loss(bar_beta_t * model_output, - noise)
    else:
        loss = th.mean((bar_beta_t * model_output + noise) ** 2 / weight)
        return loss


def slice_score_matching_loss(model, model_output, model_input, eta=1e-3):
    """
    Compute slice score matching loss.

    :param model: the model for training.
    :param model_output: th.Tensor of shape (b, c, h, w), s_{theta, t}(x_t).
    :param model_input: th.Tensor of shape (b, c, h, w), x_t.
    :param eta: float, control the norm of the random vector v.
    :return: slice score matching loss.
    """
    # this implement need to be checked.
    v = th.randn_like(model_output) * eta
    j = th.sum(v * model_output)
    j.backward()
    grad = model_input.grad
    loss = th.sum(model_output ** 2) + 2. * th.sum(grad * v)

    v = th.randn_like(model_output)
    loss = 2. * th.sum((model(model_input + eta * v) - model_output) * v / eta)
    loss = th.sum(model_output ** 2) + loss

    loss = loss / th.prod(th.tensor(model_input.shape, dtype=model_input.dtype, device=model_input.device))

    return loss
