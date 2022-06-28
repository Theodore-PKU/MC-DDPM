from models.mcddpm_unet import KspaceModel
from utils.ddpm_utils.mcddpm_gaussian_diffusion import KspaceSpacedDiffusion, space_timesteps
import utils.ddpm_utils.mcddpm_gaussian_diffusion as mcgd


# -------- gaussian diffusion --------

def diffusion_defaults():
    """
    Defaults for kspace diffusion model training.
    """
    return dict(
        diffusion_type="ddpm",
        diffusion_steps=1000,
        noise_schedule="cosine",
        beta_scale=0.5,
        timestep_respacing="",
        predict_epsilon=True,
        learn_sigma=False,
        use_tilde_beta=False,
        predictor_type="ddpm",
        use_corrector=False,
        only_use_mse=True,
        rescale_timesteps=False,
    )


def create_gaussian_diffusion(
        *,
        diffusion_type,
        diffusion_steps,
        noise_schedule,
        beta_scale,
        timestep_respacing,
        predict_epsilon,
        learn_sigma,
        use_tilde_beta,
        predictor_type,
        use_corrector,
        only_use_mse,
        rescale_timesteps,
):
    alphas, betas = mcgd.get_named_alpha_beta_shedule(diffusion_type, noise_schedule, diffusion_steps)
    betas = betas * beta_scale

    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    if diffusion_type == "ddpm":
        diffusion_type = mcgd.DiffusionType.DDPM
    elif diffusion_type == "score":
        diffusion_type = mcgd.DiffusionType.SCORE
    else:
        raise NotImplementedError(diffusion_type)

    if predict_epsilon:
        model_mean_type = mcgd.ModelMeanType.EPSILON
    else:
        model_mean_type = mcgd.ModelMeanType.SCORE

    if learn_sigma:
        model_var_type = mcgd.ModelVarType.LEARNED
    elif use_tilde_beta:
        model_var_type = mcgd.ModelVarType.TILDE_BETA
    else:
        model_var_type = mcgd.ModelVarType.DEFAULT

    sampling_kwargs = None
    if predictor_type == "ddpm":
        predictor_type = mcgd.PredictorType.DDPM
    elif predictor_type == "ddim":
        predictor_type = mcgd.PredictorType.DDIM
    elif predictor_type == "sde":
        predictor_type = mcgd.PredictorType.SDE
    else:
        raise NotImplementedError(predictor_type)

    if use_corrector:
        corrector_type = mcgd.CorrectorType.LANGEVIN
    else:
        corrector_type = mcgd.CorrectorType.NONE

    if only_use_mse:
        loss_type = mcgd.LossType.MSE
    else:
        raise NotImplementedError(only_use_mse)

    return KspaceSpacedDiffusion(
        beta_scale=beta_scale,
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        alphas=alphas,
        betas=betas,
        diffusion_type=diffusion_type,
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        predictor_type=predictor_type,
        corrector_type=corrector_type,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        sampling_kwargs=sampling_kwargs,
    )


# -------- model --------

def model_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=320,
        model_channels=128,
        learn_sigma=False,
        num_res_blocks=2,
        attention_resolutions="20",
        dropout=0,
        channel_mult="",
        use_checkpoint=False,
        use_fp16=True,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
    )


def create_model(
        *,
        image_size,
        model_channels,
        learn_sigma,
        num_res_blocks,
        attention_resolutions,
        dropout,
        channel_mult,
        use_checkpoint,
        use_fp16,
        num_heads,
        num_head_channels,
        num_heads_upsample,
        use_scale_shift_norm,
        resblock_updown,
        use_new_attention_order,
):
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))  # image_size is fixed.

    if channel_mult == "":
        # 320, 160, 80, 40, 20
        channel_mult = (1, 1, 2, 2, 4)
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    return KspaceModel(
        image_size=image_size,
        in_channels=2,
        model_channels=model_channels,
        out_channels=(2 if not learn_sigma else 4),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


# -------- training process --------

def training_setting_defaults():
    """
    Defaults for training.
    """
    return dict(
        batch_size=1,
        microbatch=-1,
        lr=1e-4,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        model_save_dir="",
        resume_checkpoint="",
        use_fp16=True,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=20.0,
        weight_decay=0.0,
        lr_anneal_steps=0,
        run_time=23.8,
        debug_mode=False,
        max_step=100000000,
    )


def schedule_sampler_setting_defaults():
    """
    Defaults for schedule sampler setting.
    """
    return dict(
        name="uniform",
    )


# -------- test process --------

def test_setting_defaults():
    """
    Defaults for training.
    """
    return dict(
        image_size=320,
        batch_size=1,
        log_interval=1,
        model_save_dir="",
        resume_checkpoint="",
        output_dir="",
        use_fp16=True,
        debug_mode=False,
        num_samples_per_mask=1,
    )
