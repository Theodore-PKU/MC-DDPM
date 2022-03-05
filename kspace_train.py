"""
Train a diffusion model on mri kspace.
"""

import argparse
import os
from utils import dist_util, logger
from utils.resample import create_named_schedule_sampler
from utils.script_util import args_to_dict, add_dict_to_argparser, load_args_dict, save_args_dict
from mri_recon_utils.kspace_train_util import KspaceTrainLoop
from mri_recon_utils.kspace_script_util import (
    model_defaults,
    create_model,
    diffusion_defaults,
    create_gaussian_diffusion,
    training_dataset_defaults,
    create_training_dataset,
    schedule_sampler_setting_defaults,
    training_setting_defaults,
)


def main():
    args = create_argparser().parse_args()

    is_distributed, rank = dist_util.setup_dist()
    logger.configure(args.log_dir, rank, is_distributed, is_write=True)
    logger.log("making device configuration...")

    # when args.resume_checkpoint is not "", all args dicts is loaded from pickle files.

    logger.log("creating model...")
    if args.resume_checkpoint:
        model_args = load_args_dict(os.path.join(args.log_dir, "model_args.pkl"))
    else:
        model_args = args_to_dict(args, model_defaults().keys())
        save_args_dict(model_args, os.path.join(args.log_dir, "model_args.pkl"))
    model = create_model(**model_args)
    model.to(dist_util.dev())

    logger.log("creating diffusion...")
    if args.resume_checkpoint:
        diffusion_args = load_args_dict(os.path.join(args.log_dir, "diffusion_args.pkl"))
    else:
        diffusion_args = args_to_dict(args, diffusion_defaults().keys())
        save_args_dict(diffusion_args, os.path.join(args.log_dir, "diffusion_args.pkl"))
    diffusion = create_gaussian_diffusion(**diffusion_args)

    logger.log("creating data loader...")
    if args.resume_checkpoint:
        data_args = load_args_dict(os.path.join(args.log_dir, "data_args.pkl"))
    else:
        data_args = args_to_dict(args, training_dataset_defaults().keys())
        save_args_dict(data_args, os.path.join(args.log_dir, "data_args.pkl"))
    data = create_training_dataset(**data_args)

    logger.log("creating schedule_sampler...")
    if args.resume_checkpoint:
        schedule_sampler_args = load_args_dict(os.path.join(args.log_dir, "schedule_sampler_args.pkl"))
    else:
        schedule_sampler_args = args_to_dict(args, schedule_sampler_setting_defaults().keys())
        save_args_dict(schedule_sampler_args, os.path.join(args.log_dir, "schedule_sampler_args.pkl"))
    schedule_sampler = create_named_schedule_sampler(**schedule_sampler_args, diffusion=diffusion)

    logger.log("training...")
    if args.resume_checkpoint:
        training_args = load_args_dict(os.path.join(args.log_dir, "training_args.pkl"))
    else:
        training_args = args_to_dict(args, training_setting_defaults().keys())
        save_args_dict(training_args, os.path.join(args.log_dir, "training_args.pkl"))
    KspaceTrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        schedule_sampler=schedule_sampler,
        **training_args,
    ).run_loop()

    logger.log("complete training.\n")


def create_argparser():
    defaults = dict(
        log_dir="logs",
        local_rank=0,
    )
    defaults.update(model_defaults())
    defaults.update(diffusion_defaults())
    defaults.update(training_dataset_defaults())
    defaults.update(training_setting_defaults())
    defaults.update(schedule_sampler_setting_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
