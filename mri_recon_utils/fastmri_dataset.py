import pickle
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .data_utils import *


ACC4 = {
    "mask_type": "random",
    "center_fractions": [0.08],
    "accelerations": [4]
}
ACC8 = {
    "mask_type": "random",
    "center_fractions": [0.04],
    "accelerations": [8]
}


def load_data(
        data_dir,
        data_info_list_path,
        batch_size,
        random_flip=False,
        is_distributed=False,
        is_train=False,
        mask_type=None,
        center_fractions=None,
        accelerations=None,
        post_process=None,
        num_workers=0,
):
    if not data_dir:
        raise ValueError("unspecified dta directory.")

    # read data infomation which is saved in a list.
    with open(data_info_list_path, "rb") as f:
        data_info_list = pickle.load(f)

    dataset = ImageDataset(
        data_dir,
        data_info_list,
        random_flip=random_flip,
        mask_type=mask_type,
        center_fractions=center_fractions,
        accelerations=accelerations,
        mask_seed_fixed=not is_train,
        post_process=post_process,
    )

    if is_train:
        data_sampler = None
        if is_distributed:
            data_sampler = DistributedSampler(dataset)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(data_sampler is None) and is_train,
            sampler=data_sampler,
            num_workers=num_workers,
            drop_last=is_train,
            pin_memory=True,
        )
        # return loader
        while True:
            yield from loader

    else:
        for kspace_c, args_dict in dataset:
            kspace_c = np2th(kspace_c).unsqueeze(0).repeat(batch_size, 1, 1, 1)
            for k, v in args_dict.items():
                if isinstance(v, np.ndarray):
                    args_dict[k] = np2th(v).unsqueeze(0).repeat(batch_size, *tuple([1] * len(v.shape)))
            yield kspace_c, args_dict


class ImageDataset(Dataset):
    """
    Dataset for MRI.

    :param data_dir: str, the directory saving data.
    :param data_info_list: list, containing elements like (file_name, index). A .h5 file contains multi slices.
    :param random_flip: bool, wheter to flip image.
    :param mask_type: str or None, if None under-sampled mask will not be used. Usually the value is "random".
    :param center_fractions: list of float, under-sampled mask center part fraction.
    :param accelerations: list of int, acceleration factor.
    :param post_process: function, used to post-processe image, image_zf, kspace, kspace_zf and mask.
    """

    def __init__(
            self,
            data_dir,
            data_info_list,
            random_flip,
            mask_type,
            center_fractions,
            accelerations,
            mask_seed_fixed,
            post_process,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.random_flip = random_flip
        self.mask_type = mask_type
        if self.mask_type is not None:
            self.mask_func = create_mask_for_mask_type(self.mask_type, center_fractions, accelerations)
        self.mask_seed_fixed = mask_seed_fixed
        self.post_process = post_process
        self.data_info_list = data_info_list

    def __len__(self):
        return len(self.data_info_list)

    def __getitem__(self, idx):
        file_name, index = self.data_info_list[idx]
        acquisition, kspace_raw, image_rss = read_datafile(self.data_dir, file_name, index)

        image_raw = ifftc_np_from_raw_data(kspace_raw)
        image = center_crop_image(image_raw, size=(320, 320))  # crop to 320x320 at center

        # use image_rss as the manitude and keep the phase unchanged
        # after processing, image is different to original one.
        image_magnitude = np.abs(image)
        image_phase_real = image.real / (image_magnitude + 1e-20)
        image_phase_imag = image.imag / (image_magnitude + 1e-20)
        image.real = image_phase_real * image_rss
        image.imag = image_phase_imag * image_rss
        image = image.astype(image_raw.dtype)

        # compute kspace of new image
        kspace = fftc_np(image)
        if self.mask_func is not None:
            # generate mask
            if self.mask_seed_fixed:
                mask = self.mask_func(kspace.shape, seed=seed_from_file_name_slice_index(file_name, index))
            else:
                mask = self.mask_func(kspace.shape)  # may add seed number.
            kspace_zf = kspace * mask + 0.0  # the + 0.0 removes the sign of the zeros
        else:
            # if mask is not used, ksapce_zf equals to kspace
            mask = None
            kspace_zf = kspace

        # compute image_zf using kspace_zf, zero-filling reconstruction
        image_zf = ifftc_np(kspace_zf)
        # compute scale coeff by zero-filling reconstruction
        scale_coeff = 1. / np.max(np.abs(image_zf))
        # rescale image_zf, kspace_zf, image, and kspace
        image_zf = image_zf * scale_coeff
        kspace_zf = kspace_zf * scale_coeff
        kspace = kspace * scale_coeff
        image = image * scale_coeff

        # post-process
        if self.post_process:
            image, image_zf, kspace, kspace_zf, mask = self.post_process(image, image_zf, kspace, kspace_zf, mask)
        # change image, image_zf, kspace, kspace_zf and mask to two channel.
        image = complex2real_np(image)
        image_zf = complex2real_np(image_zf)
        kspace = complex2real_np(kspace)
        kspace_zf = complex2real_np(kspace_zf)
        if mask is not None:
            mask = np.stack([mask, mask])

        kspace_c = kspace - kspace_zf
        # args_dict containing some information about this slice image
        args_dict = {
            "image": image,
            "image_zf": image_zf,
            "kspace": kspace,
            "kspace_zf": kspace_zf,
            "mask": mask,
            "mask_c": 1. - mask,
            "scale_coeff": scale_coeff,
            "acquisition": acquisition,
            "file_name": file_name,
            "slice_index": index,
        }

        return kspace_c, args_dict


def center_crop_image(image, size):
    """
    Crop the center part of image at last two axes.
    :param image: any type array of shape (..., h, w).
    :param size: tuple or list of int, two elements, (h1, w1).
    :return: the same type array of shape (..., h1, w1).
    """
    h, w = image.shape[-2:]
    h1, w1 = size
    if h < h1 or w < w1:
        raise ValueError("the value of size is not applicable.")
    up_index = (h - h1) // 2
    down_index = up_index + h1
    left_index = (w - w1) // 2
    right_index = left_index + w1
    return image[..., up_index: down_index, left_index: right_index]


def seed_from_file_name_slice_index(file_name, slice_index):
    return int(file_name[4:-3]) * 100 + slice_index
