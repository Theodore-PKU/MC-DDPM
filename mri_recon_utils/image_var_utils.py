"""
All functions used in mri image var training and testing
"""

import functools
import torch.distributed as dist
from .image_mean_utils import TrainLoop, TestLoop, magnitude, tile_image
from models.plain_unet import VarianceModel
from utils import dist_util, logger
from .data_utils import *


# -------- training --------

class ImageVarTrainLoop(TrainLoop):

    def __init__(self, mean_model, *args, **kwargs):
        self.mean_model = mean_model
        super().__init__(*args, **kwargs)

    def data_process(self, data):
        ksapce_c, args_dict = data
        image_zf = args_dict["image_zf"]
        image = args_dict["image"]
        return image_zf, image

    def forward_backward(self, batch, label):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro_input = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_label = label[i: i + self.microbatch].to(dist_util.dev())
            last_batch = (i + self.microbatch) >= batch.shape[0]

            with th.no_grad():
                mean_output = self.mean_model(micro_input)
            # transform to magnitude images
            mean_output = magnitude(mean_output) * 2 - 1
            micro_input = magnitude(micro_input) * 2 - 1
            micro_label = magnitude(micro_label) * 2 - 1
            # input is (image_zf, mean_output)
            micro_input = th.cat([micro_input, mean_output], dim=1)
            micro_output = self.ddp_model(micro_input)

            # label is square error
            micro_label = (micro_label - mean_output) ** 2

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
        if self.step % (self.log_interval * 10) == 0 and i == 0:
            ncols = len(micro_input)

            zf_tiled_images = tile_image(micro_input[:, 0:1, ...], ncols=ncols, nrows=1)
            mean_tiled_images = tile_image(micro_input[:, 1:, ...], ncols=ncols, nrows=1)

            micro_label[th.where(micro_label < 0)] = 0.
            micro_label = th.sqrt(micro_label)
            for i in range(len(micro_label)):
                micro_label[i] = micro_label[i] / th.max(micro_label[i])

            micro_output[th.where(micro_output < 0)] = 0.
            micro_output = th.sqrt(micro_output)
            for i in range(len(micro_output)):
                micro_output[i] = micro_output[i] / th.max(micro_output[i])

            label_tiled_images = tile_image(micro_label, ncols=ncols, nrows=1)
            output_tiled_images = tile_image(micro_output, ncols=ncols, nrows=1)
            # image alignment:
            # image_zf | mean image | label std | output std
            all_tiled_images = th.cat(
                [zf_tiled_images, mean_tiled_images, label_tiled_images, output_tiled_images, ], dim=1
            )
            logger.get_current().write_image(
                'train_images',
                all_tiled_images,
                self.step
            )


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
        initial_lg_loss_scale=16.0,
        weight_decay=0.0,
        lr_anneal_steps=0,
        run_time=23.8,
        debug_mode=False,
        max_step=100000000,
    )

# -------- training --------


# -------- testing --------

class ImageVarTestLoop(TestLoop):

    def __init__(self, mean_model, num_test_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.batch_size == 1
        self.mean_model = mean_model
        self.num_test_file = num_test_file

    def run_loop(self):
        curr_file_name = ""
        for data_item in self.data:
            ksapce_c, args_dict = data_item

            if args_dict["file_name"] != curr_file_name:
                if curr_file_name != "":
                    self.step += 1
                    logger.log(f"have test {self.step} steps")
                else:
                    curr_file_name = args_dict["file_name"]

            if self.step == self.num_test_file:
                break

            self.forward_backward(args_dict)

        dist.barrier()

    def forward_backward(self, args_dict):
        image_zf = args_dict["image_zf"].to(dist_util.dev())
        with th.no_grad():
            mean_output = self.mean_model(image_zf)  # mean_model output is 2 channels
        # transform to magnitude images
        mean_output = magnitude(mean_output) * 2 - 1

        image_zf = magnitude(image_zf) * 2 - 1
        # input is (image_zf, mean_output)
        micro_input = th.cat([image_zf, mean_output], dim=1)
        with th.no_grad():
            micro_output = self.model(micro_input)

        micro_output[th.where(micro_output < 0)] = 0.
        micro_output = th2np(micro_output[0, 0, ...])  # since we assert batch_size is 1
        micro_output = np.sqrt(micro_output)

        file_name = args_dict["file_name"]
        slice_index = args_dict["slice_index"]
        output_path = os.path.join(self.output_dir, file_name, f"slice_{slice_index}")
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        np.savez(
            os.path.join(output_path, "model_std"), micro_output
        )
        plt.imsave(
            fname=os.path.join(output_path, "sample_std.png"),
            arr=micro_output / np.max(micro_output), cmap="gray"
        )


def test_setting_defaults():
    """
    Defaults for training.
    """
    return dict(
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

def var_model_defaults():
    """
    Defaults for mean model training.
    :return: a dict that contains parameters setting.
    """
    return dict(
        last_layer_type="none",
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


def create_var_model(
        last_layer_type,
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

    return VarianceModel(
        image_size=320,
        in_channels=1,
        last_layer_type=last_layer_type,
        model_channels=model_channels,
        out_channels=1,
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

# -------- model script util --------
