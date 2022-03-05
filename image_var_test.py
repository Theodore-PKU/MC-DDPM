"""
Test a var model on mri image.
"""

import argparse
from utils.script_util import args_to_dict, add_dict_to_argparser, load_args_dict
from mri_recon_utils.image_var_utils import *
from mri_recon_utils.kspace_script_util import test_dataset_defaults, create_test_dataset
from mri_recon_utils.image_mean_utils import create_mean_model


def main():
    args = create_argparser().parse_args()

    is_distributed, rank = dist_util.setup_dist()
    logger.configure(args.log_dir, rank, is_distributed, is_write=True)
    logger.log("making device configuration...")

    assert args.resume_checkpoint, "resume_checkpoint should be specified"

    logger.log("creating model...")
    model_args = load_args_dict(os.path.join(args.log_dir, "model_args.pkl"))
    model = create_var_model(**model_args)
    model.to(dist_util.dev())

    logger.log("load mean model...")
    mean_model_args = load_args_dict(args.mean_model_args_path)
    mean_model = create_mean_model(**mean_model_args)
    mean_model.load_state_dict(
        th.load(
            args.mean_model_path, map_location=dist_util.dev()
        )
    )
    mean_model.to(dist_util.dev())
    if mean_model_args['use_fp16']:
        mean_model.convert_to_fp16()

    logger.log("creating data loader...")
    data_args = args_to_dict(args, test_dataset_defaults().keys())
    data = create_test_dataset(**data_args)

    logger.log("testing...")
    test_args = args_to_dict(args, test_setting_defaults().keys())
    ImageVarTestLoop(
        mean_model=mean_model,
        model=model,
        data=data,
        **test_args,
    ).run_loop()

    logger.log("complete testing.\n")


def create_argparser():
    defaults = dict(
        log_dir="logs",
        mean_model_path="",
        mean_model_args_path="",
        local_rank=0,
    )
    defaults.update(var_model_defaults())
    defaults.update(test_dataset_defaults())
    defaults.update(test_setting_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
