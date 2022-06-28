from utils.mri_data_utils.mask_util import ACC4, ACC8
from utils.dataset_utils import fastmri


def training_dataset_defaults():
    """
    Defaults for training fastmri dataset.
    """
    return dict(
        dataset="fastmri",
        data_dir="../datasets/fastmri/knee_singlecoil_train",
        data_info_list_path="",
        batch_size=1,
        acceleration=4,
        random_flip=False,
        is_distributed=True,
        num_workers=0,
    )


def create_training_dataset(
        dataset,
        data_dir,
        data_info_list_path,
        batch_size,
        acceleration,
        random_flip,
        is_distributed,
        num_workers,
):
    if acceleration == 4:
        acc_factor = ACC4
    elif acceleration == 8:
        acc_factor = ACC8
    else:
        acc_factor = {"mask_type": None, "center_fractions": None, "accelerations": None}

    if dataset == "fastmri":
        load_data = fastmri.load_data
    else:
        raise ValueError
    return load_data(
        data_dir=data_dir,
        data_info_list_path=data_info_list_path,
        batch_size=batch_size,
        random_flip=random_flip,
        is_distributed=is_distributed,
        is_train=True,
        post_process=None,
        num_workers=num_workers,
        **acc_factor,
    )


def test_dataset_defaults():
    """
    Defaults for test fastmri dataset.
    """
    return dict(
        dataset="fastmri",
        data_dir="../dataset/fastmri/knee_singlecoil_val",
        data_info_list_path="",
        batch_size=1,
        acceleration=4,
        random_flip=False,
        is_distributed=True,
        num_workers=0,
    )


def create_test_dataset(
        dataset,
        data_dir,
        data_info_list_path,
        batch_size,
        acceleration,
        random_flip,
        is_distributed,
        num_workers,
):
    if acceleration == 4:
        acc_factor = ACC4
    elif acceleration == 8:
        acc_factor = ACC8
    else:
        acc_factor = {"mask_type": None, "center_fractions": None, "accelerations": None}

    if dataset == "fastmri":
        load_data = fastmri.load_data
    else:
        raise ValueError
    return load_data(
        data_dir=data_dir,
        data_info_list_path=data_info_list_path,
        batch_size=batch_size,
        random_flip=random_flip,
        is_distributed=is_distributed,
        is_train=False,
        post_process=None,
        num_workers=num_workers,
        **acc_factor,
    )
