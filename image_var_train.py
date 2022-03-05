"""
Train a var model on mri image.
"""

import argparse
from utils.script_util import args_to_dict, add_dict_to_argparser, load_args_dict, save_args_dict
from mri_recon_utils.image_var_utils import *
from mri_recon_utils.kspace_script_util import training_dataset_defaults, create_training_dataset
from mri_recon_utils.image_mean_utils import create_mean_model


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
        model_args = args_to_dict(args, var_model_defaults().keys())
        save_args_dict(model_args, os.path.join(args.log_dir, "model_args.pkl"))
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
    if args.resume_checkpoint:
        data_args = load_args_dict(os.path.join(args.log_dir, "data_args.pkl"))
    else:
        data_args = args_to_dict(args, training_dataset_defaults().keys())
        save_args_dict(data_args, os.path.join(args.log_dir, "data_args.pkl"))
    data = create_training_dataset(**data_args)

    logger.log("training...")
    if args.resume_checkpoint:
        training_args = load_args_dict(os.path.join(args.log_dir, "training_args.pkl"))
    else:
        training_args = args_to_dict(args, training_setting_defaults().keys())
        save_args_dict(training_args, os.path.join(args.log_dir, "training_args.pkl"))
    ImageVarTrainLoop(
        mean_model=mean_model,
        model=model,
        data=data,
        **training_args,
    ).run_loop()

    logger.log("complete training.\n")


def create_argparser():
    defaults = dict(
        log_dir="logs",
        mean_model_path="",
        mean_model_args_path="",
        local_rank=0,
    )
    defaults.update(var_model_defaults())
    defaults.update(training_dataset_defaults())
    defaults.update(training_setting_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
