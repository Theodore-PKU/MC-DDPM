"""
Train a mean model on mri image.
"""

import argparse
from utils.script_util import args_to_dict, add_dict_to_argparser, load_args_dict, save_args_dict
from mri_recon_utils.image_mean_utils import *
from mri_recon_utils.kspace_script_util import training_dataset_defaults, create_training_dataset


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
        model_args = args_to_dict(args, mean_model_defaults().keys())
        save_args_dict(model_args, os.path.join(args.log_dir, "model_args.pkl"))
    model = create_mean_model(**model_args)
    model.to(dist_util.dev())

    logger.log("creating data loader...")
    if args.resume_checkpoint:
        data_args = load_args_dict(os.path.join(args.log_dir, "data_args.pkl"))
    else:
        data_args = args_to_dict(args, training_dataset_defaults().keys())
        save_args_dict(data_args, os.path.join(args.log_dir, "data_args.pkl"))
    data = create_training_dataset(**data_args)

    logger.log("training...")
    training_args = args_to_dict(args, training_setting_defaults().keys())
    save_args_dict(training_args, os.path.join(args.log_dir, "training_args.pkl"))
    ImageMeanTrainLoop(
        model=model,
        data=data,
        **training_args,
    ).run_loop()

    logger.log("complete training.\n")


def create_argparser():
    defaults = dict(
        log_dir="logs",
        local_rank=0,
    )
    defaults.update(mean_model_defaults())
    defaults.update(training_dataset_defaults())
    defaults.update(training_setting_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
