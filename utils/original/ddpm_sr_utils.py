"""
Dataset and helpers for DDPM super-resolution sampling
"""

import os
from PIL import Image
import pickle
import blobfile as bf
import numpy as np
import torch as th
import torch.nn.functional as F


# ytxie: This function is to crop iamges at the center location and is not used directly.
# ytxie: The parameter `image_size` is of int type.
def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    # return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]  # source
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]  # new


class DDPMSRDataset:
    """
    This dataset class is used to generate multi-samples.
    """
    def __init__(
            self,
            data_dir,
            to_sample_images_dict_path,
            batch_size,
            small_size,
            large_size,
            class_cond,
            output_dir,
            num_samples_per_image,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.small_size = small_size
        self.large_size = large_size
        self.class_cond = class_cond
        self.output_dir = output_dir

        self.dataset = []
        if data_dir == "":
            raise ValueError("unspecified data directory")
        if to_sample_images_dict_path == "":
            raise ValueError("unspecified dict file path of images to sample")
        with open(to_sample_images_dict_path, "rb") as f:
            to_sample_images_info_dict = pickle.load(f)

        # Get all image files info from a dict
        for class_name, info_dict in to_sample_images_info_dict.items():
            y_label = info_dict["y"]
            for file_name in info_dict["file_names"]:
                # count samples having generated and judge whether to sample this image file
                count = self._count_have_sampled_images(file_name)
                num_samples = num_samples_per_image - count
                if num_samples <= 0:
                    continue

                # save useful info
                image_info_dict = dict(
                    file_name=file_name,
                    y=y_label,
                    num_samples=num_samples,
                    num_have_samples=count,
                )
                self.dataset.append(image_info_dict)

        self.dataset = self.dataset[:5]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_info_dict = self.dataset[idx]
        model_kwargs = self._get_input_batch(image_info_dict)
        return model_kwargs, image_info_dict

    def _count_have_sampled_images(self, file_name):
        high_res_file_path = os.path.join(self.output_dir, file_name, "high_res_samples")
        count = 0
        if os.path.exists(high_res_file_path):
            for entry in sorted(bf.listdir(high_res_file_path)):
                if entry.endswith(".png") and entry.startswith("sample"):
                    count += 1
        return count

    def _get_input_batch(self, image_info_dict):
        file_name = image_info_dict["file_name"]
        label = image_info_dict["y"],
        out_dict = {}
        if self.class_cond:
            out_dict["y"] = th.tensor(label, dtype=th.int64).repeat(self.batch_size)

        file_path = os.path.join(self.data_dir, file_name + ".JPEG")
        with bf.BlobFile(file_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        arr = center_crop_arr(pil_image, self.large_size)
        arr = arr.astype(np.float32) / 127.5 - 1
        arr = np.transpose(arr, [2, 0, 1])
        arr = th.from_numpy(arr).float().unsqueeze(0)
        arr = F.interpolate(arr, self.small_size, mode="area")
        out_dict["low_res"] = arr.repeat(self.batch_size, 1, 1, 1)
        return out_dict


def save_images(low_res, arr, output_dir, file_name, count):
    """
    Used to save ddpm sr samples.
    :param low_res: `low_res` value in model input `model_kwargs`
    :param arr: numpy.array, generated samples.
    :param output_dir: str, where to save output samples.
    :param file_name: str, indicates which image file.
    :param count: int, the number of samples in output_dir/file_name/high_res_samples/ before generating.
    :return: None
    """
    images_path = os.path.join(output_dir, file_name, "high_res_samples")
    if not os.path.exists(images_path):
        os.makedirs(images_path, exist_ok=True)

    # save low-resolution image
    low_res = low_res[0].squeeze()
    low_res = ((low_res + 1) * 127.5).clamp(0, 255).to(th.uint8)
    low_res = low_res.permute(1, 2, 0).contiguous().cpu().numpy()
    low_res = Image.fromarray(low_res)
    low_res_path = os.path.join(output_dir, file_name, f"low_res.png")
    low_res.save(low_res_path)

    # save sr samples
    for i, image in enumerate(arr):
        image = Image.fromarray(image)
        image_path = os.path.join(images_path, f"sample_{i + count}.png")
        image.save(image_path)
