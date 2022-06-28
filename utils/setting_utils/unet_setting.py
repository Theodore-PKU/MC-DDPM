from models.guided_ddpm_plain_unet import UNetModel


# -------- model --------

def model_defaults():
    """
    Defaults for mean model training.
    :return: a dict that contains parameters setting.
    """
    return dict(
        image_size=320,
        model_channels=128,
        num_res_blocks=2,
        attention_resolutions="20",
        dropout=0,
        channel_mult="",
        use_checkpoint=False,
        use_fp16=True,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        resblock_updown=True,
        use_new_attention_order=False,
    )


def create_model(
        image_size,
        model_channels,
        num_res_blocks,
        attention_resolutions,
        dropout,
        channel_mult,
        use_checkpoint,
        use_fp16,
        num_heads,
        num_head_channels,
        num_heads_upsample,
        resblock_updown,
        use_new_attention_order,
):
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    if channel_mult == "":
        # 320, 160, 80, 40, 20
        channel_mult = (1, 1, 2, 2, 4)
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    return UNetModel(
        image_size=image_size,
        in_channels=2,
        model_channels=model_channels,
        out_channels=2,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        conv_resample=True,
        dims=2,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
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


# -------- test process --------

def test_setting_defaults():
    """
    Defaults for training.
    """
    return dict(
        batch_size=1,  # this value should not be changed.
        microbatch=1,
        log_interval=10,
        model_save_dir="",
        resume_checkpoint="",
        output_dir="",
        use_fp16=True,
        debug_mode=False,
    )
