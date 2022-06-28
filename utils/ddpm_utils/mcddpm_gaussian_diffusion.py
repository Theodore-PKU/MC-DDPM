from utils.ddpm_utils.gaussian_diffusion import *
from utils.ddpm_utils.gaussian_diffusion import _extract_into_tensor, _WrappedModel


class KspaceGaussianDiffusion(GaussianDiffusion):
    """
    Utilities for training and sampling diffusion models of kspace mri reconstruction.
    We assume the model's input is kspace_c and output is also kspace_c for under-sampled part.

    :param beta_scale: float, to scale the variance.
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

    def __init__(self, beta_scale, *args, **kwargs):
        # we add a new attribute `_mask_c` so that many methods keep the same args.
        # ususally it is a [N x 2 x ...] tensor of 0-1 values indicate the under-sampled position.
        # its value is obtained by model_kwargs["mask_c"]
        self._mask_c = None
        self.beta_scale = beta_scale
        super().__init__(*args, **kwargs)

    def q_sample(self, x_start, t):
        """
        Diffuse the data for a given number of diffusion steps. In other words, sample from q(x_t | x_0).
        x_start and x_t are all under-sampled kspace data. In the position of mask, the value is 0.

        :param x_start: the initial data batch, kspace_c for under-sampled part.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A noisy version of x_start (under-sampled position), kspace_c for under-sampled part, and noise.

        Note: self._mask_c needs to be defined beforehand.
        """
        x_t, noise = super().q_sample(x_start, t)
        return x_t * self._mask_c, noise * self._mask_c

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0).

        Note: self._mask_c needs to be defined beforehand.
        """
        posterior_mean, posterior_variance, posterior_log_variance_clipped = (
            super().q_posterior_mean_variance(x_start, x_t, t)
        )
        return (
            posterior_mean * self._mask_c,
            posterior_variance * self._mask_c,
            posterior_log_variance_clipped * self._mask_c
        )

    def p_eps_std(self, model, x_t, t, model_kwargs):
        """
        Apply the model to compute "epsilon" item and std parameter in predictor or corrector.

        :param model: the model, which takes a signal and a batch of timesteps as input.
        :param x_t: the [N x C x ...] tensor at time t, kspace_c for under-sampled part.
        :param t: a 1-D Tensor of timesteps.
        :param model_kwargs: a dict of extra keyword arguments to pass to the model containing mask information.
        :return: (eps, std), eps and std only contain under-sampled part, sampled-part is all 0.

        Note: self._mask_c needs to be defined beforehand.
        """
        assert model_kwargs is not None, "model_kwargs contains the condtions"
        eps, std = super().p_eps_std(model, x_t, t, model_kwargs=model_kwargs)
        return eps * self._mask_c, std * self._mask_c

    def ddim_predictor(self, model, x_t, t, model_kwargs, clip=False):
        """
        DDIM-Predictor

        :param model: the model to sample from.
        :param x_t: the current tensor at x_t, kspace_c for under-sampled part.
        :param t: the value of t, starting at T for the first diffusion step.
        :param model_kwargs: a dict of extra keyword arguments to pass to the model containing mask information.
        :param clip: if True, clip the x_start prediction to [-1, 1].
        :return: a random sample from the model, kspace_c for under-sampled part.

        Note: self._mask_c needs to be defined beforehand.
        """
        if self.sampling_kwargs is None:
            eta = 0.0
        else:
            assert "eta" in self.sampling_kwargs.keys(), "in ddim-predictor, eta is a hyper-parameter"
            eta = self.sampling_kwargs["eta"]
        assert model_kwargs is not None, "model_kwargs contains the condtions"
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
        # we only modifies this line, others are kept unchanged or not important.
        sigma = eta * th.sqrt((1 - bar_alpha_prev) / (1 - bar_alpha)) \
                * th.sqrt(1 - bar_alpha / bar_alpha_prev) * self._mask_c

        mean_pred = pred_xstart * th.sqrt(bar_alpha_prev) + th.sqrt(1 - bar_alpha_prev - sigma ** 2) * eps

        noise = th.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        return sample

    def sample_loop(self, model, shape, model_kwargs, clip=False, noise=None):
        """
        Generate samples from the model. When noise is not sampled from N(0, 1), it should be not None.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param model_kwargs: a dict of extra keyword arguments to pass to the model containing mask information.
        :param clip: if True, clip x_start predictions to [-1, 1].
        :param noise: if specified, the noise from the encoder to sample. Should be of the same shape as `shape`.
        :return: a non-differentiable batch of samples, kspace_c for under-sampled part.
        """
        device = dev()
        assert isinstance(shape, (tuple, list))

        # the only modifed code is here. when generating noise, we multiply it with mask_c
        # -------- modified part --------
        assert model_kwargs is not None, "model_kwargs contains the condtions"
        self._mask_c = model_kwargs["mask_c"]
        if noise is not None:
            kspace_c = noise * self._mask_c  # when noise is not sampled from N(0, 1), it should be specified.
        elif self.diffusion_type == DiffusionType.DDPM:
            # need modify.
            noise = th.randn(*shape, device=device)
            # kspace_c = 0.5 * noise * self._mask_c
            kspace_c = self.beta_scale * noise * self._mask_c
        elif self.diffusion_type == DiffusionType.SCORE:
            assert False, "code fo score-based model has not been completed"
        else:
            raise NotImplementedError(self.diffusion_type)
        # -------- modified part --------

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

        # we change variable name `img` to `kspace_c`
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                kspace_c = predictor(model, kspace_c, t, model_kwargs=model_kwargs, clip=clip)
                if corrector is not None:
                    assert False, "code of corrector has not been completed"
        return kspace_c

    def training_losses(self, model, x_start, t, model_kwargs):
        assert model_kwargs is not None, "model_kwargs contains the condtions"
        self._mask_c = model_kwargs["mask_c"]
        return super().training_losses(model, x_start, t, model_kwargs=model_kwargs)


class KspaceSpacedDiffusion(KspaceGaussianDiffusion):
    """
    A diffusion process for kspace mri reconstruction which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.

    Note: the code is the same to class SpacedDiffusion.
    """
    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = KspaceGaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
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
