import torch.distributed as dist

from utils.test_util import TestLoop
from utils import dist_util, logger
from utils.debug_util import *
from utils.script_util import save_args_dict
from mri_recon_utils.data_utils import *


class KspaceTestLoop(TestLoop):

    def __init__(self, max_num_files=-1, num_samples_per_mask=1, *args, **kwargs):
        self.max_num_files = max_num_files
        self.num_samples_per_mask = num_samples_per_mask
        super().__init__(*args, **kwargs)

    def run_loop(self, is_distributed):
        count = -1
        curr_file_name = ""
        for _, batch_kwargs in self.data:
            file_name = batch_kwargs["file_name"]
            if file_name != curr_file_name:
                count += 1
                curr_file_name = file_name
            if count == self.max_num_files:
                break
            slice_index = batch_kwargs["slice_index"]
            samples_path = os.path.join(self.output_dir, file_name, f"slice_{slice_index}")
            if os.path.exists(samples_path):
                if os.path.exists(os.path.join(samples_path, "slice_information.pkl")):
                    logger.log(f"have sampled for {file_name} slice {slice_index}")
                    continue

            cond = {
                k: batch_kwargs[k].to(dist_util.dev()) for k in ["kspace_zf", "image_zf", "mask_c"]
            }
            all_images = []
            while len(all_images) * self.batch_size < self.num_samples_per_mask:
                sample = self.sample(cond)
                self.step += 1
                if self.debug_mode or self.step <= 10:
                    show_gpu_usage(f"step: {self.step}, device: {dist.get_rank()}", idx=dist.get_rank())
                if self.step % 100 == 0:
                    logger.log(f"have run {self.step} steps")
                # # gather samples in different devices
                # if is_distributed:
                #     all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                #     dist.all_gather(all_samples, sample)  # gather not supported with NCCL
                #
                #     all_noise = [th.zeros_like(noise) for _ in range(dist.get_world_size())]
                #     dist.all_gather(all_noise, noise)  # gather not supported with NCCL
                # else:
                #     all_samples = [sample]
                #     all_noise = [noise]
                # for sample in all_samples:
                #     all_images.append(sample.cpu().numpy())
                # for noise in all_noise:
                #     all_init_noise.append(noise.cpu().numpy())
                all_images.append(sample.cpu().numpy())

            # gather all samples and save them
            arr = np.concatenate(all_images, axis=0)
            arr = arr[: self.num_samples_per_mask]

            if is_distributed and dist.get_rank() != 0:
                continue
            else:
                self.save_samples(arr, file_name, slice_index, batch_kwargs)
                logger.log(f"complete sampling for {file_name} slice {slice_index}")

        dist.barrier()

    def sample(self, cond):
        sample = self.diffusion.sample_loop(
            self.model,
            (self.batch_size, 2, 320, 320),
            cond,
            clip=False
        )
        kspace = sample + cond["kspace_zf"]
        # permute operation is for fft operation.
        image_full = ifftc_th(kspace)
        return image_full

    def save_samples(self, arr, file_name, slice_index, batch_kwargs):
        samples_path = os.path.join(self.output_dir, file_name, f"slice_{slice_index}")
        if not os.path.exists(samples_path):
            os.makedirs(samples_path, exist_ok=True)

        # save mask, image, image_zf
        mask = batch_kwargs["mask"][0, 0, ...]
        # transform to magnitude images
        image = np.abs(real2complex_np(batch_kwargs["image"][0]))
        image_zf = np.abs(real2complex_np(batch_kwargs["image_zf"][0]))
        plt.imsave(
            fname=os.path.join(samples_path, "full_recon.png"),
            arr=image, cmap="gray"
        )
        plt.imsave(
            fname=os.path.join(samples_path, "zf_recon.png"),
            arr=image_zf, cmap="gray"
        )
        plt.imsave(
            fname=os.path.join(samples_path, "mask.png"),
            arr=mask, cmap="gray"
        )

        # save some samples, less than 5
        for i in range(min(5, len(arr))):
            sample = np.abs(real2complex_np(arr[i]))
            plt.imsave(
                fname=os.path.join(samples_path, f"sample_{i + 1}.png"),
                arr=sample, cmap="gray"
            )

        # save all information
        np.savez(os.path.join(samples_path, f"all_samples"), arr)  # arr is not magnitude images
        saved_args = {
            "scale_coeff": batch_kwargs["scale_coeff"],
            "slice_index": batch_kwargs["slice_index"],
            "image": batch_kwargs["image"][0:1, ...],
        }
        save_args_dict(saved_args, os.path.join(samples_path, "slice_information.pkl"))
