"""
All functions used in mri image mean training and testing
"""

import functools
import blobfile as bf
import time
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from utils import dist_util, logger
from utils.fp16_util import MixedPrecisionTrainer
from utils.debug_util import *
from models.plain_unet import UNetModel
from .data_utils import *

# -------- training --------

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


# ytxie: This is the main part.
# ytxie: The parameter `microbatch` is used to train the model with smaller batch_size and accumulate gradient.
# ytxie: We add parameter `max_step` to control the iteration steps and `run_time` to save the last model.
class TrainLoop:
    """
    This class contains the training details.
    """
    def __init__(
            self,
            *,
            model,
            data,  # iterate output is (kspace_c, args_dict)
            batch_size,
            microbatch,  # ytxie: if we don't use microbatch, set it as 0 or a negative integer.
            lr,
            log_interval,
            save_interval,
            model_save_dir="",
            resume_checkpoint="",  # ytxie: resume_checkpoint file name
            use_fp16=False,
            fp16_scale_growth=1e-3,
            initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
            weight_decay=0.0,
            lr_anneal_steps=0,
            max_step=100000000,
            run_time=23.8,  # ytxie: hours, if we don't use run_time to control, set it as 0 or a negative value.
            debug_mode=False,  # whether to show gpu usage.
    ):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.max_step = max_step
        self.run_time = run_time
        self.resume_checkpoint = resume_checkpoint
        self.model_save_dir = model_save_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.debug_mode = debug_mode
        self.save_last = True

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        logger.log(f"This model contains {count_parameters_in_M(self.model)}M parameters")

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            initial_lg_loss_scale=initial_lg_loss_scale
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                # ytxie: Maybe this parameters is used in pytorch1.7, However in 1.6 version,
                # ytxie: it seems to ought be removed.
                # find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.log(
                    "Warning!"
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    # ytxie: We use the simplest method to load model parameters.
    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            model_checkpoint = os.path.join(self.model_save_dir, self.resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {model_checkpoint}...")
                self.model.load_state_dict(
                    th.load(
                        model_checkpoint, map_location=dist_util.dev()
                    )
                )
            self.step = self.resume_step

        dist_util.sync_params(self.model.parameters())

    # ytxie: We use the simplest method to load optimizer state.
    def _load_optimizer_state(self):
        # ytxie: The format `{self.resume_step:06}` may need to be changed.
        opt_checkpoint = bf.join(
            self.model_save_dir, f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    # ytxie: This function wraps the whole training process.
    def run_loop(self):
        start_time = time.time()
        # ytxie: When lr_anneal_steps > 0, it seems like maximum steps.
        while (
            not self.lr_anneal_steps
            or self.step < self.lr_anneal_steps
        ):
            batch, label = self.data_process(next(self.data))
            self.run_step(batch, label)
            self.step += 1

            if self.debug_mode and self.step % self.log_interval == 0:
                show_gpu_usage(f"step: {self.step}, device: {dist.get_rank()}", idx=dist.get_rank())
            elif not self.debug_mode and self.step % self.save_interval == 0:
                show_gpu_usage(f"step: {self.step}, device: {dist.get_rank()}", idx=dist.get_rank())
            elif not self.debug_mode and self.step == self.log_interval:
                show_gpu_usage(f"step: {self.step}, device: {dist.get_rank()}", idx=dist.get_rank())

            # save last model
            if self.run_time > 0 and time.time() - start_time > self.run_time * 3600 and self.save_last:
                self.save()
                self.save_last = False

            if self.step % self.log_interval == 0:
                logger.write_kv(self.step)
                logger.clear_kv()

            if self.step % self.log_interval == 0 and self.step <= 10 * self.log_interval:
                logger.log(f"have trained {self.step} steps")

            if self.step % self.save_interval == 0:
                self.save()
                # ytxie: How this code works? If this condition is not satisfies and
                # ytxie: `lr_anneal_steps` is 0, `run_loop` will continue running.
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            if self.step >= self.max_step:
                break

        # Save the last checkpoint if it wasn't already saved.
        if self.step % self.save_interval != 0:
            self.save()

    def data_process(self, data):
        return data

    def run_step(self, batch, label):
        self.forward_backward(batch, label)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()

    def forward_backward(self, batch, label):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro_input = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_label = label[i: i + self.microbatch].to(dist_util.dev())
            last_batch = (i + self.microbatch) >= batch.shape[0]
            micro_output = self.ddp_model(micro_input)

            compute_loss = functools.partial(
                th.nn.functional.mse_loss,
                micro_output,
                micro_label,
            )

            if last_batch or not self.use_ddp:
                loss = compute_loss()
            else:
                with self.ddp_model.no_sync():
                    loss = compute_loss()

            logger.log_kv("loss", loss)
            self.mp_trainer.backward(loss)

            self._post_process(micro_input, micro_label, micro_output, i)

    def _post_process(self, micro_input, micro_label, micro_output, i):
        """
        This function should be reloaded.
        :param micro_input:
        :param micro_label:
        :param micro_output:
        :param i:
        :return:
        """
        pass

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        # ytxie: In training process, step + resume_step will not be larger than lr_annela_steps.
        frac_done = self.step / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def save(self):
        """
        Save the model and the optimizer state.
        """
        if dist.get_rank() == 0:
            state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
            logger.log(f"saving model at step {self.step}...")
            filename = f"model{self.step:06d}.pt"
            with bf.BlobFile(bf.join(self.model_save_dir, filename), "wb") as f:
                th.save(state_dict, f)

            with bf.BlobFile(
                bf.join(self.model_save_dir, f"opt{self.step:06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


# ytxie: We keep the filename form.
def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


class ImageMeanTrainLoop(TrainLoop):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def data_process(self, data):
        ksapce_c, args_dict = data
        image_zf = args_dict["image_zf"]
        image = args_dict["image"]
        return image_zf, image

    def _post_process(self, micro_input, micro_label, micro_output, i):
        if self.step % self.save_interval == 0 and i == 0:
            ncols = len(micro_input)

            input_tiled_images = tile_image(magnitude(micro_input), ncols=ncols, nrows=1)
            label_tiled_images = tile_image(magnitude(micro_label), ncols=ncols, nrows=1)
            output_tiled_images = tile_image(magnitude(micro_output), ncols=ncols, nrows=1)
            # image alignment:
            # image_zf | image | recon_mean
            all_tiled_images = th.cat(
                [input_tiled_images, label_tiled_images, output_tiled_images, ], dim=1
            )
            logger.get_current().write_image(
                'train_images',
                all_tiled_images,
                self.step
            )


def magnitude(image):
    return th.sqrt(image[:, 0, ...] ** 2 + image[:, 1, ...] ** 2).unsqueeze(1)


def tile_image(batch_image, ncols, nrows):
    assert ncols * nrows == batch_image.shape[0]
    _, channels, height, width = batch_image.shape
    batch_image = batch_image.view(nrows, ncols, channels, height, width)
    batch_image = batch_image.permute(2, 0, 3, 1, 4)
    batch_image = batch_image.contiguous().view(channels, nrows * height, ncols * width)
    return batch_image


def training_setting_defaults():
    """
    Defaults for training.
    """
    return dict(
        batch_size=1,
        microbatch=-1,
        lr=1e-4,
        log_interval=10,
        save_interval=10000,
        model_save_dir="",
        resume_checkpoint="",
        use_fp16=True,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=20.0,
        weight_decay=0.0,
        lr_anneal_steps=0,
        run_time=23.8,
        debug_mode=False,
        max_step=100000000,
    )

# -------- training --------


# -------- testing --------

class TestLoop:
    def __init__(
            self,
            *,
            model,
            data,
            batch_size,  # in fact we will not use this paramter.
            microbatch,
            model_save_dir="",
            resume_checkpoint="",  # ytxie: resume_checkpoint file name
            output_dir="",
            use_fp16=True,
            debug_mode=False,
    ):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.model_save_dir = model_save_dir
        self.resume_checkpoint = resume_checkpoint
        self.output_dir = output_dir
        self.use_fp16 = use_fp16
        self.debug_mode = debug_mode

        self.global_batch = self.batch_size * dist.get_world_size()  # seems to be useless.

        self.step = 0

        self._load_parameters()
        logger.log(f"This model contains {count_parameters_in_M(self.model)}M parameters")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    # ytxie: We use the simplest method to load model parameters.
    def _load_parameters(self):
        model_checkpoint = os.path.join(self.model_save_dir, self.resume_checkpoint)
        logger.log(f"loading model from checkpoint: {model_checkpoint}...")
        self.model.load_state_dict(
            th.load(
                model_checkpoint,
                map_location="cpu",
            )
        )
        self.model.to(dist_util.dev())
        if self.use_fp16:
            self.model.convert_to_fp16()

        self.model.eval()

    # ytxie: This function wraps the whole test process.
    def run_loop(self):
        for data_item in self.data:
            self.forward_backward(data_item)
            self.step += 1
            if self.debug_mode and self.step % 10 == 0 and self.step > 100:
                show_gpu_usage(f"step: {self.step}, device: {dist.get_rank()}", idx=dist.get_rank())

            if self.step <= 100 and self.step % 10 == 0:
                show_gpu_usage(f"step: {self.step}, device: {dist.get_rank()}", idx=dist.get_rank())
                logger.log(f"have test {self.step} steps")

        dist.barrier()

    def forward_backward(self, data_item):
        batch, label = data_item
        for i in range(0, batch.shape[0], self.microbatch):
            micro_input = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_label = label[i: i + self.microbatch].to(dist_util.dev())
            with th.no_grad():
                micro_output = self.model(micro_input)

            self._post_process(micro_input, micro_label, micro_output, i)

    def _post_process(self, micro_input, micro_label, micro_output, i):
        pass


class ImageMeanTestLoop(TestLoop):

    def __init__(
            self,
            num_output_file,  # use to control the number of saved volumes
            num_test_file,  # restrict the number of testing files
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert self.batch_size == 1
        self.num_output_file = num_output_file
        self.num_test_file = num_test_file
        # _zf means that image_zf is used
        self.slice_metrics_zf = {
            metric: [] for metric in ["psnr", "ssim", "mse", "nmse"]
        }
        self.volume_metrics_zf = {
            metric: [] for metric in ["psnr", "ssim", "mse", "nmse"]
        }
        # _base means that kspace_zf will not be used to refine reconstruction.
        self.slice_metrics_base = {
            metric: [] for metric in ["psnr", "ssim", "mse", "nmse"]
        }
        self.volume_metrics_base = {
            metric: [] for metric in ["psnr", "ssim", "mse", "nmse"]
        }
        # _refine means that kspace_zf will be used to refine reconstruction.
        self.slice_metrics_refine = {
            metric: [] for metric in ["psnr", "ssim", "mse", "nmse"]
        }
        self.volume_metrics_refine = {
            metric: [] for metric in ["psnr", "ssim", "mse", "nmse"]
        }

    def run_loop(self):
        curr_file_name = ""
        image_batch = []
        image_zf_batch = []
        kspace_zf_batch = []
        mask_batch = []
        scale_coeff_batch = []
        slice_index_batch = []
        for data_item in self.data:
            ksapce_c, args_dict = data_item

            if args_dict["file_name"] != curr_file_name:
                if curr_file_name == "":
                    curr_file_name = args_dict["file_name"]
                    image_batch.append(args_dict["image"])
                    image_zf_batch.append(args_dict["image_zf"])
                    kspace_zf_batch.append(args_dict["kspace_zf"])
                    mask_batch.append(args_dict["mask"])
                    scale_coeff_batch.append(args_dict["scale_coeff"])
                    slice_index_batch.append(args_dict["slice_index"])
                else:
                    self.run_for_volume(
                        curr_file_name,
                        image_batch,
                        image_zf_batch,
                        kspace_zf_batch,
                        mask_batch,
                        scale_coeff_batch,
                        slice_index_batch,
                    )

                    if self.step == self.num_test_file:
                        break

                    # for next file and the first slice
                    curr_file_name = args_dict["file_name"]
                    image_batch = [args_dict["image"]]
                    image_zf_batch = [args_dict["image_zf"]]
                    kspace_zf_batch = [args_dict["kspace_zf"]]
                    mask_batch = [args_dict["mask"]]
                    scale_coeff_batch = [args_dict["scale_coeff"]]
                    slice_index_batch = [args_dict["slice_index"]]

            else:
                # collect slices
                image_batch.append(args_dict["image"])
                image_zf_batch.append(args_dict["image_zf"])
                kspace_zf_batch.append(args_dict["kspace_zf"])
                mask_batch.append(args_dict["mask"])
                scale_coeff_batch.append(args_dict["scale_coeff"])
                slice_index_batch.append(args_dict["slice_index"])

        # if -1 run the last file.
        if self.num_test_file == -1:
            self.run_for_volume(
                curr_file_name,
                image_batch,
                image_zf_batch,
                kspace_zf_batch,
                mask_batch,
                scale_coeff_batch,
                slice_index_batch,
            )

        dist.barrier()
        self.metrics_average()

    def run_for_volume(
            self,
            curr_file_name,
            image_batch,
            image_zf_batch,
            kspace_zf_batch,
            mask_batch,
            scale_coeff_batch,
            slice_index_batch,
    ):
        self.step += 1

        # run test main part
        self.forward_backward(
            curr_file_name,
            th.cat(image_batch, dim=0),
            th.cat(image_zf_batch, dim=0),
            th.cat(kspace_zf_batch, dim=0),
            th.cat(mask_batch, dim=0),
            scale_coeff_batch,
            slice_index_batch
        )

        if self.step <= 10:
            show_gpu_usage(f"step: {self.step}, device: {dist.get_rank()}", idx=dist.get_rank())

        logger.log(f"have trained {self.step} step")

    def forward_backward(
            self,
            curr_file_name,
            image_batch,
            image_zf_batch,
            kspace_zf_batch,
            mask_batch,
            scale_coeff_batch,
            slice_index_batch,
    ):
        output_batch = []
        for i in range(0, image_zf_batch.shape[0], self.microbatch):
            micro_input = image_zf_batch[i: i + self.microbatch].to(dist_util.dev())
            with th.no_grad():
                micro_output = self.model(micro_input)
            output_batch.append(micro_output)

        # obtain output for the whole volume and compute refine output
        output_batch = th.cat(output_batch, dim=0)
        assert len(output_batch) == len(image_zf_batch)
        mask_batch = mask_batch.to(dist_util.dev())
        kspace_zf_batch = kspace_zf_batch.to(dist_util.dev())
        image_batch = image_batch.to(dist_util.dev())
        image_zf_batch = image_zf_batch.to(dist_util.dev())
        output_refine_batch = ifftc_th(fftc_th(output_batch) * (1. - mask_batch) + kspace_zf_batch)

        self.save_to_tensorboard(image_zf_batch, image_batch, output_batch, output_refine_batch)

        # rescale image value
        for i in range(len(image_batch)):
            scale_coeff = scale_coeff_batch[i]
            image_batch[i] = image_batch[i] / scale_coeff
            output_batch[i] = output_batch[i] / scale_coeff
            output_refine_batch[i] = output_refine_batch[i] / scale_coeff
            image_zf_batch[i] = image_zf_batch[i] / scale_coeff

        self.compute_metrics(image_batch, image_zf_batch,
                             self.slice_metrics_zf, self.volume_metrics_zf)
        self.compute_metrics(image_batch, output_batch,
                             self.slice_metrics_base, self.volume_metrics_base)
        self.compute_metrics(image_batch, output_refine_batch,
                             self.slice_metrics_refine, self.volume_metrics_refine)

        self._post_process(
            curr_file_name,
            image_batch,
            image_zf_batch,
            output_batch,
            output_refine_batch,
            mask_batch,
            slice_index_batch,
            kspace_zf_batch,
        )

    def compute_metrics(
            self,
            image_batch,
            output_batch,
            slices_metrics_dict,
            volumes_metrics_dict
    ):
        for i in range(len(image_batch)):
            gt = th2np_magnitude(image_batch[i:i+1])
            pred = th2np_magnitude(output_batch[i:i+1])
            slices_metrics_dict["mse"].append(compute_mse(gt, pred))
            slices_metrics_dict["nmse"].append(compute_nmse(gt, pred))
            slices_metrics_dict["psnr"].append(compute_psnr(gt, pred))
            slices_metrics_dict["ssim"].append(compute_ssim(gt, pred))

        gt = th2np_magnitude(image_batch)
        pred = th2np_magnitude(output_batch)
        volumes_metrics_dict["mse"].append(compute_mse(gt, pred))
        volumes_metrics_dict["nmse"].append(compute_nmse(gt, pred))
        volumes_metrics_dict["psnr"].append(compute_psnr(gt, pred))
        volumes_metrics_dict["ssim"].append(compute_ssim(gt, pred))

    def save_to_tensorboard(self, image_zf_batch, image_batch, output_batch, output_refine_batch):
        if self.step <= self.num_output_file:
            # save images to tensorboard
            nrows = len(image_zf_batch)
            image_zf_tiled_images = tile_image(
                magnitude(image_zf_batch), ncols=1, nrows=nrows
            )
            image_tiled_images = tile_image(
                magnitude(image_batch), ncols=1, nrows=nrows
            )
            output_tiled_images = tile_image(
                magnitude(output_batch), ncols=1, nrows=nrows
            )
            output_refine_batch = tile_image(
                magnitude(output_refine_batch), ncols=1, nrows=nrows
            )
            # image alignment:
            # image_zf | image | recon_mean | recon_refine
            all_tiled_images = th.cat(
                [image_zf_tiled_images, image_tiled_images, output_tiled_images, output_refine_batch],
                dim=2
            )
            logger.get_current().write_image(
                'test_output',
                all_tiled_images,
                self.step
            )

    def _post_process(
            self,
            curr_file_name,
            image_batch,
            image_zf_batch,
            output_batch,
            output_refine_batch,
            mask_batch,
            slice_index_batch,
            kspace_zf_batch
    ):
        if self.step <= self.num_output_file:
            num_slices = len(slice_index_batch)
            for i in range(num_slices):
                dir_path = os.path.join(self.output_dir, curr_file_name, f"slice_{slice_index_batch[i]}")
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)

                mask = th2np(mask_batch[i, 0, ...])
                image = th2np_magnitude(image_batch[i:i+1])[0]
                image_zf = th2np_magnitude(image_zf_batch[i:i+1])[0]
                output = th2np_magnitude(output_batch[i:i+1])[0]
                output_refine = th2np_magnitude(output_refine_batch[i:i+1])[0]

                plt.imsave(
                    fname=os.path.join(dir_path, "mask.png"),
                    arr=mask, cmap="gray"
                )
                plt.imsave(
                    fname=os.path.join(dir_path, "image.png"),
                    arr=image, cmap="gray"
                )
                plt.imsave(
                    fname=os.path.join(dir_path, "image_zf.png"),
                    arr=image_zf, cmap="gray"
                )
                plt.imsave(
                    fname=os.path.join(dir_path, "output.png"),
                    arr=output, cmap="gray"
                )
                plt.imsave(
                    fname=os.path.join(dir_path, "output_refine.png"),
                    arr=output_refine, cmap="gray"
                )

            def write_volume_metric(statement, metrics_dict, index):
                f.write(f"{statement:13}")
                for metric in ["psnr", "ssim", "nmse", "mse"]:
                    if metric != "mse":
                        f.write(f"{metrics_dict[metric][index]:10.6f}")
                    else:
                        f.write(f"{metrics_dict[metric][index]:12.4e}\n")

            def write_slice_metric(statement, metrics_dict, index1, index2):
                f.write(f"slice_{index1:2} {statement:13}")
                for metric in ["psnr", "ssim", "nmse", "mse"]:
                    if metric != "mse":
                        f.write(f"{metrics_dict[metric][index2]:10.6f}")
                    else:
                        f.write(f"{metrics_dict[metric][index2]:12.4e}\n")

            # save evaluation metrics to .txt file
            with open(os.path.join(self.output_dir, curr_file_name, "metrics.txt"), "w") as f:
                f.write("volume evaluation metrics: psnr ssim nmse mse\n")
                write_volume_metric("zf recon", self.volume_metrics_zf, -1)
                write_volume_metric("base recon", self.volume_metrics_base, -1)
                write_volume_metric("refine recon", self.volume_metrics_refine, -1)

                f.write("\nslice evaluation metrics: psnr ssim nmse mse\n")
                for i in range(num_slices):
                    write_slice_metric("zf recon", self.slice_metrics_zf, slice_index_batch[i], i - num_slices)
                    write_slice_metric("base recon", self.slice_metrics_base, slice_index_batch[i], i - num_slices)
                    write_slice_metric("refine recon", self.slice_metrics_refine, slice_index_batch[i], i - num_slices)
                    f.write("\n")

    def metrics_average(self):
        def log_stat_info(statement, metrics_dict):
            logger.log("")
            logger.log(statement)
            for metric in ["psnr", "ssim", "nmse", "mse"]:
                if metric != "mse":
                    logger.log(f"{metric}: {np.mean(metrics_dict[metric]):.6f} +/- "
                               f"({np.std(metrics_dict[metric]):.6f})")
                else:
                    logger.log(f"{metric}: {np.mean(metrics_dict[metric]):.4e} +/- "
                               f"({np.std(metrics_dict[metric]):.4e})")

        log_stat_info("slice metrics zf-recon", self.slice_metrics_zf)
        log_stat_info("slice metrics base-recon", self.slice_metrics_base)
        log_stat_info("slice metrics refine-recon", self.slice_metrics_refine)
        log_stat_info("volume metrics zf-recon", self.volume_metrics_zf)
        log_stat_info("volume metrics base-recon", self.volume_metrics_base)
        log_stat_info("volume metrics refine-recon", self.volume_metrics_refine)

        def print_stat_info(statement, metrics_dict, f):
            f.write(statement)
            for metric in ["psnr", "ssim", "nmse", "mse"]:
                if metric != "mse":
                    f.write(f"{metric:4}: {np.mean(metrics_dict[metric]):.6f} +/- "
                            f"({np.std(metrics_dict[metric]):.6f})\n")
                else:
                    f.write(f"{metric:4}: {np.mean(metrics_dict[metric]):.4e} +/- "
                            f"({np.std(metrics_dict[metric]):.4e})\n\n")

        with open(os.path.join(self.output_dir, "metrics_average.txt"), "w") as f:
            print_stat_info("slice metrics zf-recon\n", self.slice_metrics_zf, f)
            print_stat_info("slice metrics base-recon\n", self.slice_metrics_base, f)
            print_stat_info("slice metrics refine-recon\n", self.slice_metrics_refine, f)
            print_stat_info("volume metrics zf-recon\n", self.volume_metrics_zf, f)
            print_stat_info("volume metrics base-recon\n", self.volume_metrics_base, f)
            print_stat_info("volume metrics refine-recon\n", self.volume_metrics_refine, f)


def test_setting_defaults():
    """
    Defaults for training.
    """
    return dict(
        num_output_file=0,
        num_test_file=-1,
        batch_size=1,  # this value should not be changed.
        microbatch=-1,
        model_save_dir="",
        resume_checkpoint="",
        output_dir="",
        use_fp16=True,
        debug_mode=False,
    )

# -------- testing --------


# -------- model script util --------

def mean_model_defaults():
    """
    Defaults for mean model training.
    :return: a dict that contains parameters setting.
    """
    return dict(
        model_channels=128,
        num_res_blocks=2,
        attention_resolutions="20",
        dropout=0,
        channel_mult="",
        use_checkpoint=False,
        use_fp16=True,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        resblock_updown=True,
        use_new_attention_order=False,
    )


def create_mean_model(
    model_channels,
    num_res_blocks,
    attention_resolutions,
    dropout,
    channel_mult,
    use_checkpoint,
    use_fp16,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    resblock_updown,
    use_new_attention_order,
):
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(320 // int(res))

    if channel_mult == "":
        # 320, 160, 80, 40, 20
        channel_mult = (1, 1, 2, 2, 4)
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    return UNetModel(
        image_size=320,
        in_channels=2,
        model_channels=model_channels,
        out_channels=2,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        conv_resample=True,
        dims=2,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )
