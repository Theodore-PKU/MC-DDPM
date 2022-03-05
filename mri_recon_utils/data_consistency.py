from .data_utils import (
    fftc_th, ifftc_th
)


def direct_project_dc(x_t, y, mask_c):
    k_t = fftc_th(x_t)
    k_t = k_t * mask_c + y
    return ifftc_th(k_t)


def weight_project_dc(x_t, y_t, mask, mask_c, eta):
    k_t = fftc_th(x_t)
    k_t = k_t * mask_c + eta * y_t + (1. - eta) * mask * k_t
    return ifftc_th(k_t)


def score_func_bias_dc(x_t, y, mask, delta_square, eta_t_square):
    k_t = fftc_th(x_t)
    score_func = ifftc_th(y - mask * k_t) / (delta_square + eta_t_square)
    return score_func


def ddpm_bias_dc(x_t, y, mask, delta_square, bar_alpha_t, bar_beta_t):
    k_t = fftc_th(x_t)
    coeff = 1 / (bar_alpha_t * (bar_beta_t ** 2 / bar_alpha_t ** 2 + delta_square))
    score_func = ifftc_th(y - mask * k_t / bar_alpha_t) * coeff
    return score_func

