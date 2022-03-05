"""
Test a diffusion model on mri kspace.
"""

import argparse
import os
from utils import dist_util, logger
from utils.script_util import args_to_dict, add_dict_to_argparser, load_args_dict
from mri_recon_utils.kspace_test_util import KspaceTestLoop
from mri_recon_utils.kspace_script_util import (
    model_defaults,
    create_model,
    diffusion_defaults,
    create_gaussian_diffusion,
    test_dataset_defaults,
    create_test_dataset,
    test_setting_defaults,
)


def main():
    args = create_argparser().parse_args()

    is_distributed, rank = dist_util.setup_dist()
    logger.configure(args.log_dir, rank, is_distributed, is_write=True)
    logger.log("making device configuration...")

    logger.log("creating model...")
    model_args = load_args_dict(os.path.join(args.log_dir, "model_args.pkl"))
    model = create_model(**model_args)
    model.to(dist_util.dev())

    logger.log("creating diffusion...")
    # gaussian diffusion needs to be specify.
    diffusion_args = args_to_dict(args, diffusion_defaults().keys())
    diffusion = create_gaussian_diffusion(**diffusion_args)

    logger.log("creating data loader...")
    data_args = args_to_dict(args, test_dataset_defaults().keys())
    data = create_test_dataset(**data_args)

    logger.log("test...")
    test_args = args_to_dict(args, test_setting_defaults().keys())
    KspaceTestLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        **test_args,
    ).run_loop(is_distributed)

    logger.log("complete test.\n")


def create_argparser():
    defaults = dict(
        log_dir="logs",
        local_rank=0,
    )
    defaults.update(model_defaults())
    defaults.update(diffusion_defaults())
    defaults.update(test_dataset_defaults())
    defaults.update(test_setting_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
