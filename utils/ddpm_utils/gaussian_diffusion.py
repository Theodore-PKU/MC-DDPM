"""
This code started out as a PyTorch port of Guided Diffusion Model:
https://github.com/openai/guided-diffusion/tree/912d5776a64a33e3baf3cff7eb1bcba9d9b9354c

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th

from models.nn import mean_flat
# from .losses import normal_kl, discretized_gaussian_log_likelihood
from utils.dist_util import dev


# ytxie: This function is used in `get_named_alpha_beta_shedule`.
# ytxie: It is only used for ddpm shedule.
def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_named_alpha_beta_shedule(dp_name, schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined alpha and beta schedule for the given name.

    :param dp_name: string, diffusion process type, "ddpm" or "score_based".
    :param schedule_name: string, "linear" or "cosine".
    :param num_diffusion_timesteps: int.
    :return: alpha and beta: 1-d array.
    """
    alpha = beta = None
    if dp_name == "score_based":
        alpha = np.array([1.] * num_diffusion_timesteps, dtype=np.float64)
        if schedule_name == "linear":
            # ???
            beta = np.array(
                [(1. / num_diffusion_timesteps) ** 0.5] * num_diffusion_timesteps,
                dtype=np.float64
            )
        else:
            raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

    elif dp_name == "ddpm":
        if schedule_name == "linear":
            scale = 1000 / num_diffusion_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02  # 0.03, 0.04 are also OK
            beta = np.linspace(
                beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
            )
        elif schedule_name == "cosine":
            beta = betas_for_alpha_bar(
                num_diffusion_timesteps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        else:
            raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

        alpha = np.sqrt(1 - beta)
        beta = np.sqrt(beta)

    return alpha, beta


class DiffusionType(enum.Enum):
    """
    Which type of diffusion process is used.
    """
    DDPM = enum.auto()  # ddpm, \alpha^2 + \beta^2 = 1
    SCORE = enum.auto()  # score-based, \alpha = 1


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    EPSILON = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts score function


class ModelVarType(enum.Enum):
    """
    What is used as the variance when sampling.
    """
    DEFAULT = enum.auto()  # use \beta
    TILDE_BETA = enum.auto()  # use \tilde{\beta}
    LEARNED = enum.auto()  # use leared delta through range parameter


class PredictorType(enum.Enum):
    """
    Which type of Predictor is used when sampling.
    """
    DDPM = enum.auto()  # use ddpm predictor (sampling)
    SDE = enum.auto()  # use sde predictor
    DDIM = enum.auto()  # use ddim predictor (sampling)


class CorrectorType(enum.Enum):
    """
    Which type of Corrector is used when sampling.
    """
    LANGEVIN = enum.auto()  # use langevin corrector (sampling)
    NONE = enum.auto()  # do not use any corrector


class LossType(enum.Enum):
    """
    Which type of loss function is used.
    """
    MSE = enum.auto  # use raw MSE loss


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    :param alphas: a 1-D numpy array of alphas for each diffusion timestep, starting at 1 and going to T.
    :param betas: a 1-D numpy array of betas for each diffusion timestep, starting at 1 and going to T.
    :param diffusion_type: a DiffusionType determing which diffusion model is used.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param predictor_type: a PredictorType determing which predictor is used.
    :param corrector_type: a CorrectorType determing which corrector is used.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the model so that they are always scaled like
        in the original paper (0 to 1000).
    :param sampling_kwargs: hyper-paramters used for predictor or corrector.
    """

    def __init__(
            self,
            alphas,
            betas,
            diffusion_type,
            model_mean_type,
            model_var_type,
            predictor_type,
            corrector_type,
            loss_type,
            rescale_timesteps=False,
            sampling_kwargs=None,
    ):
        self.diffusion_type = diffusion_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.predictor_type = predictor_type
        self.corrector_type = corrector_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.sampling_kwargs = sampling_kwargs

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas  # \beta_t
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all()

        alphas = np.array(alphas, dtype=np.float64)
        self.alphas = alphas  # \alpha_t
        if self.diffusion_type == DiffusionType.SCORE:
            assert (alphas == 1.).all()
        elif self.diffusion_type == DiffusionType.DDPM:
            assert (alphas > 0).all()
        else:
            raise ValueError("diffusion type is either ddpm or score-based")

        self.num_timesteps = int(betas.shape[0])

        self.betas_square = betas ** 2  # \beta_t^2
        self.bar_alphas = np.cumprod(alphas, axis=0)  # \bar{\alpha}_{t}
        self.bar_alphas_prev = np.append(1.0, self.bar_alphas[:-1])  # \bar{\alpha}_{t-1}
        # self.bar_alphas_next = np.append(self.bar_alphas[1:], 0.0)
        self.bar_alphas_square = self.bar_alphas ** 2
        self.bar_alphas_square_prev = self.bar_alphas_prev ** 2
        self.bar_betas_square = self.bar_alphas_square * np.cumsum(self.betas_square / self.bar_alphas_square, axis=0)
        self.bar_betas = np.sqrt(self.bar_betas_square)
        self.log_bar_betas_square = np.log(self.bar_betas_square)
        self.bar_betas_prev = np.append(0.0, self.bar_betas[:-1])
        self.bar_betas_square_prev = np.append(0.0, self.bar_betas_square[:-1])
        self.recip_bar_alphas = 1. / self.bar_alphas
        assert self.bar_alphas_prev.shape == (self.num_timesteps,)
        assert self.bar_betas_prev.shape == (self.num_timesteps,)
        assert self.bar_betas_square_prev.shape == (self.num_timesteps,)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_std = self.betas * self.bar_betas_prev / self.bar_betas  # \tilde{\beta}_t
        self.posterior_variance = self.posterior_std ** 2  # \tilde{\beta}_t^2
        # log calculation clipped because the posterior variance is 0 when t = 1 (\bar{\beta}_{0} = 0)
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )   # \log \tilde{\beta}_t^2
        # the following two coefs are used to compute \tilde{\mu}_t
        self.posterior_mean_coef_x0 = self.bar_alphas_prev * self.betas_square / self.bar_betas_square
        self.posterior_mean_coef_xt = self.alphas * self.bar_betas_square_prev / self.bar_betas_square

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs, x_0.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.bar_alphas, t, x_start.shape) * x_start
        variance = _extract_into_tensor(self.bar_betas_square, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_bar_betas_square, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t):
        """
        Diffuse the data for a given number of diffusion steps. In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A noisy version of x_start and noise.
        """
        noise = th.randn_like(x_start)
        x_t = _extract_into_tensor(self.bar_alphas, t, x_start.shape) * x_start + \
              _extract_into_tensor(self.bar_betas, t, x_start.shape) * noise
        return x_t, noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0).
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef_x0, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef_xt, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_eps_std(self, model, x_t, t, model_kwargs=None):
        """
        Apply the model to compute "epsilon" item and std parameter in predictor or corrector.

        :param model: the model, which takes a signal and a batch of timesteps as input.
        :param x_t: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param model_kwargs: if not None, a dict of extra keyword arguments to pass to the model.
            This can be used for conditioning.
        :return: (eps, std)
        """
        if model_kwargs is None:
            model_kwargs = {}
        assert t.shape == (x_t.shape[0], )
        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        model_var_values = None

        if self.model_mean_type == ModelMeanType.EPSILON:
            if self.model_var_type == ModelVarType.LEARNED:
                eps, model_var_values = th.split(model_output, x_t.shape[1], dim=1)
            else:
                eps = model_output
        elif self.model_mean_type == ModelMeanType.SCORE:
            assert False, "code of score estimation has not been completed"
        else:
            raise NotImplementedError(self.model_mean_type)

        # compute model variance
        if self.model_var_type == ModelVarType.DEFAULT:
            # \delta_t = \beta_t
            std = _extract_into_tensor(self.betas, t, x_t.shape)
            # model_variance = _extract_into_tensor(np.square(self.betas), t, x_t.shape)
            # model_log_variance = _extract_into_tensor(np.log(np.square(self.betas)), t, x_t.shape)
        elif self.model_var_type == ModelVarType.TILDE_BETA:
            # \delta_t = \tilde{\beta}_{t}
            std = _extract_into_tensor(self.posterior_std, t, x_t.shape)
            # model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
            # model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        elif self.model_var_type == ModelVarType.LEARNED:
            assert self.model_mean_type == ModelMeanType.EPSILON and model_var_values is not None
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
            max_log = _extract_into_tensor(np.log(self.betas_square), t, x_t.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            std = th.exp(0.5 * model_log_variance)
        else:
            raise NotImplementedError(self.model_var_type)

        return eps, std

    def ddpm_predictor(self, model, x_t, t, model_kwargs=None, clip=False):
        """
        DDPM-Predictor

        :param model: the model to sample from.
        :param x_t: the current tensor at x_t.
        :param t: the value of t, starting at T for the first diffusion step.
        :param model_kwargs: if not None, a dict of extra keyword arguments to pass to the model.
            This can be used for conditioning.
        :param clip: if True, clip the x_start prediction to [-1, 1].
        :return: a random sample from the model.
        """
        assert self.sampling_kwargs is None, "in ddpm-predictor, no hyper-parameter"

        eps, std = self.p_eps_std(model, x_t, t, model_kwargs=model_kwargs)

        # compute model mean from model output
        pred_xstart = _extract_into_tensor(self.recip_bar_alphas, t, x_t.shape) * \
                      (x_t - _extract_into_tensor(self.bar_betas, t, x_t.shape) * eps)
        pred_xstart = self._clip(pred_xstart, clip=clip)
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)

        # compute one sample of x_{t-1}
        noise = th.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))  # no noise when t == 0
        sample = model_mean + nonzero_mask * std * noise

        return sample

    def ddim_predictor(self, model, x_t, t, model_kwargs=None, clip=False):
        """
        DDIM-Predictor

        :param model: the model to sample from.
        :param x_t: the current tensor at x_t.
        :param t: the value of t, starting at T for the first diffusion step.
        :param model_kwargs: if not None, a dict of extra keyword arguments to pass to the model.
            This can be used for conditioning.
        :param clip: if True, clip the x_start prediction to [-1, 1].
        :return: a random sample from the model.
        """
        if self.sampling_kwargs is None:
            eta = 0.0
        else:
            assert "eta" in self.sampling_kwargs.keys(), "in ddim-predictor, eta is a hyper-parameter"
            eta = self.sampling_kwargs["eta"]

        eps, std = self.p_eps_std(model, x_t, t, model_kwargs=model_kwargs)

        # compute model mean
        pred_xstart = _extract_into_tensor(self.recip_bar_alphas, t, x_t.shape) * \
                      (x_t - _extract_into_tensor(self.bar_betas, t, x_t.shape) * eps)
        pred_xstart = self._clip(pred_xstart, clip=clip)

        eps = (x_t - _extract_into_tensor(self.bar_alphas, t, x_t.shape) * pred_xstart) / \
              _extract_into_tensor(self.bar_betas, t, x_t.shape)

        # this code is according to guided-diffusion code
        bar_alpha = _extract_into_tensor(self.bar_alphas_square, t, x_t.shape)
        bar_alpha_prev = _extract_into_tensor(self.bar_alphas_square_prev, t, x_t.shape)
        sigma = eta * th.sqrt((1 - bar_alpha_prev) / (1 - bar_alpha)) * th.sqrt(1 - bar_alpha / bar_alpha_prev)

        mean_pred = pred_xstart * th.sqrt(bar_alpha_prev) + th.sqrt(1 - bar_alpha_prev - sigma ** 2) * eps

        noise = th.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        return sample

    def sample_loop(self, model, shape, model_kwargs=None, clip=False, noise=None):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param model_kwargs: if not None, a dict of extra keyword arguments to pass to the model.
            This can be used for conditioning.
        :param clip: if True, clip x_start predictions to [-1, 1].
        :param noise: if specified, the noise from the encoder to sample. Should be of the same shape as `shape`.
        :return: a non-differentiable batch of samples.
        """
        device = dev()
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        elif self.diffusion_type == DiffusionType.DDPM:
            img = th.randn(*shape, device=device)
        elif self.diffusion_type == DiffusionType.SCORE:
            assert False, "code fo score-based model has not been completed"
        else:
            raise NotImplementedError(self.diffusion_type)
        indices = list(range(self.num_timesteps))[::-1]

        if self.predictor_type == PredictorType.DDPM:
            predictor = self.ddpm_predictor
        elif self.predictor_type == PredictorType.DDIM:
            predictor = self.ddim_predictor
        elif self.predictor_type == PredictorType.SDE:
            assert False, "code of sde-predictor has not been completed"
        else:
            raise NotImplementedError(self.predictor_type)

        if self.corrector_type == CorrectorType.LANGEVIN:
            assert False, "code of langevin-corrector has not been completed"
        elif self.corrector_type == CorrectorType.NONE:
            corrector = None
        else:
            raise NotImplementedError(self.corrector_type)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                img = predictor(model, img, t, model_kwargs=model_kwargs, clip=clip)
                if corrector is not None:
                    assert False, "code of corrector has not been completed"

        return img

    def training_losses(self, model, x_start, t, model_kwargs=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to pass to the model.
            This can be used for conditioning.
        :return: a dict with the key "loss" containing a tensor of shape [N].
            Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}

        x_t, noise = self.q_sample(x_start, t)

        terms = {}

        if self.loss_type == LossType.MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
            target = noise
            if self.model_mean_type == ModelMeanType.EPSILON:
                if self.model_var_type == ModelVarType.LEARNED:
                    assert False, "code of learned variance has not been completed"
                else:
                    model_output = model_output
            elif self.model_mean_type == ModelMeanType.SCORE:
                assert False, "code of score estimation has not been completed"
            else:
                raise NotImplementedError(self.model_mean_type)

            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            terms["loss"] = terms["mse"]

        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def _clip(self, x, clip=False):
        if clip:
            x = x.clamp(-1, 1)
        return x


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original process to divide up.
    :param section_counts: either a list of numbers, or a string containing comma-separated numbers,
        indicating the step count per section. As a special case,
        use "ddimN" where N is a number of steps to use the striding from the DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """
    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        # compute new alphas and betas
        last_bar_alpha = 1.0
        last_bar_beta = 0.0
        new_betas = []
        new_alphas = []
        for i, (bar_alpha, bar_beta) in enumerate(zip(base_diffusion.bar_alphas, base_diffusion.bar_betas)):
            if i in self.use_timesteps:
                alpha = bar_alpha / last_bar_alpha
                new_alphas.append(alpha)
                last_bar_alpha = bar_alpha
                beta = np.sqrt(bar_beta ** 2 - alpha ** 2 * last_bar_beta ** 2)
                last_bar_beta = bar_beta
                new_betas.append(beta)
                self.timestep_map.append(i)
        kwargs["alphas"] = np.array(new_alphas)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_eps_std(
            self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_eps_std(self._wrap_model(model), *args, **kwargs)

    def training_losses(
            self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
