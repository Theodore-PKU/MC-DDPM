import argparse
import os
from utils import dist_util, logger
from utils.script_util import args_to_dict, add_dict_to_argparser, load_args_dict, save_args_dict
from utils.setting_utils import (
    dataset_setting, mcddpm_setting, unet_setting,
)
from utils.test_utils.mcddpm_test_util import MCDDPMTestLoop
from utils.test_utils.unet_test_util import UNetTestLoop


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

    # create or load model
    # when args.resume_checkpoint is not "", model_args will be loaded from saved pickle file.
    logger.log("creating model...")
    model_args = load_args_dict(os.path.join(args.model_save_dir, "model_args.pkl"))
    model = method_setting.create_model(**model_args)
    model.to(dist_util.dev())

    logger.log("creating data loader...")
    data_args = args_to_dict(args, dataset_setting.test_dataset_defaults().keys())
    data = dataset_setting.create_test_dataset(**data_args)

    logger.log("test...")
    test_args = args_to_dict(args, method_setting.test_setting_defaults().keys())
    if args.method_type == "mcddpm":
        logger.log("creating diffusion...")
        diffusion_args = args_to_dict(args, method_setting.diffusion_defaults().keys())
        diffusion = method_setting.create_gaussian_diffusion(**diffusion_args)
        MCDDPMTestLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            **test_args,
        ).run_loop()
    elif args.method_type == "unet":
        UNetTestLoop(
            model=model,
            data=data,
            **test_args,
        ).run_loop()

    logger.log("complete test.\n")


def create_argparser():
    defaults = dict(
        method_type="mcddpm",
        log_dir="logs",
        local_rank=0,
    )
    defaults.update(mcddpm_setting.model_defaults())
    defaults.update(mcddpm_setting.diffusion_defaults())
    defaults.update(mcddpm_setting.test_setting_defaults())
    defaults.update(unet_setting.test_setting_defaults())
    defaults.update(dataset_setting.test_dataset_defaults())

    parser_temp = argparse.ArgumentParser()
    add_dict_to_argparser(parser_temp, defaults)
    args_temp = parser_temp.parse_args()
    if args_temp.method_type == "mcddpm":
        defaults.update(mcddpm_setting.model_defaults())
        defaults.update(mcddpm_setting.diffusion_defaults())
        defaults.update(mcddpm_setting.test_setting_defaults())
    elif args_temp.method_type == "unet":
        defaults.update(unet_setting.model_defaults())
        defaults.update(unet_setting.test_setting_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
