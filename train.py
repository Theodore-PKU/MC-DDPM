import argparse
import os
from utils import dist_util, logger
from utils.ddpm_utils.resample import create_named_schedule_sampler
from utils.script_util import args_to_dict, add_dict_to_argparser, load_args_dict, save_args_dict
from utils.setting_utils import (
    dataset_setting, mcddpm_setting, unet_setting,
)
from utils.train_utils.mcddpm_train_util import MCDDPMTrainLoop
from utils.train_utils.unet_train_util import UNetTrainLoop


def main():
    args = create_argparser().parse_args()

    # distributed setting
    is_distributed, rank = dist_util.setup_dist()
    logger.configure(args.log_dir, rank, is_distributed, is_write=True)
    logger.log("making device configuration...")

    if args.method_type == "mcddpm":
        method_setting = mcddpm_setting
    elif args.method_type == "unet":
        method_setting = unet_setting
    else:
        raise ValueError

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir, exist_ok=True)

    # create or load model
    # when args.resume_checkpoint is not "", model_args will be loaded from saved pickle file.
    logger.log("creating model...")
    if args.resume_checkpoint:
        model_args = load_args_dict(os.path.join(args.model_save_dir, "model_args.pkl"))
    else:
        model_args = args_to_dict(args, method_setting.model_defaults().keys())
        save_args_dict(model_args, os.path.join(args.model_save_dir, "model_args.pkl"))
    model = method_setting.create_model(**model_args)
    model.to(dist_util.dev())

    logger.log("creating data loader...")
    if args.resume_checkpoint:
        data_args = load_args_dict(os.path.join(args.model_save_dir, "data_args.pkl"))
    else:
        data_args = args_to_dict(args, dataset_setting.training_dataset_defaults().keys())
        save_args_dict(data_args, os.path.join(args.model_save_dir, "data_args.pkl"))
    data = dataset_setting.create_training_dataset(**data_args)

    logger.log("training...")
    if args.resume_checkpoint:
        training_args = load_args_dict(os.path.join(args.model_save_dir, "training_args.pkl"))
        training_args["resume_checkpoint"] = args.resume_checkpoint
    else:
        training_args = args_to_dict(args, method_setting.training_setting_defaults().keys())
        save_args_dict(training_args, os.path.join(args.model_save_dir, "training_args.pkl"))
    if args.method_type == "mcddpm":
        logger.log("creating diffusion...")
        if args.resume_checkpoint:
            diffusion_args = load_args_dict(os.path.join(args.model_save_dir, "diffusion_args.pkl"))
        else:
            diffusion_args = args_to_dict(args, method_setting.diffusion_defaults().keys())
            save_args_dict(diffusion_args, os.path.join(args.model_save_dir, "diffusion_args.pkl"))
        diffusion = method_setting.create_gaussian_diffusion(**diffusion_args)

        logger.log("creating schedule_sampler...")
        if args.resume_checkpoint:
            schedule_sampler_args = load_args_dict(os.path.join(args.model_save_dir, "schedule_sampler_args.pkl"))
        else:
            schedule_sampler_args = args_to_dict(args, method_setting.schedule_sampler_setting_defaults().keys())
            save_args_dict(schedule_sampler_args, os.path.join(args.model_save_dir, "schedule_sampler_args.pkl"))
        schedule_sampler = create_named_schedule_sampler(**schedule_sampler_args, diffusion=diffusion)
        MCDDPMTrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            schedule_sampler=schedule_sampler,
            **training_args,
        ).run_loop()
    elif args.method_type == "unet":
        UNetTrainLoop(
            model=model,
            data=data,
            **training_args,
        ).run_loop()

    logger.log("complete training.\n")


def create_argparser():
    defaults = dict(
        method_type="mcddpm",
        log_dir="logs",
        local_rank=0,
    )
    defaults.update(mcddpm_setting.model_defaults())
    defaults.update(mcddpm_setting.diffusion_defaults())
    defaults.update(mcddpm_setting.training_setting_defaults())
    defaults.update(mcddpm_setting.schedule_sampler_setting_defaults())
    defaults.update(dataset_setting.training_dataset_defaults())

    parser_temp = argparse.ArgumentParser()
    add_dict_to_argparser(parser_temp, defaults)
    args_temp = parser_temp.parse_args()
    if args_temp.method_type == "mcddpm":
        defaults.update(mcddpm_setting.model_defaults())
        defaults.update(mcddpm_setting.diffusion_defaults())
        defaults.update(mcddpm_setting.training_setting_defaults())
        defaults.update(mcddpm_setting.schedule_sampler_setting_defaults())
    elif args_temp.method_type == "unet":
        defaults.update(unet_setting.model_defaults())
        defaults.update(unet_setting.training_setting_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
