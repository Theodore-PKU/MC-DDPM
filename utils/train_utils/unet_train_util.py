import functools

from utils import dist_util, logger
from utils.train_utils.base_train_util import TrainLoop

from utils.mri_data_utils.transform_util import *
from utils.mri_data_utils.image_util import magnitude, tile_image


class UNetTrainLoop(TrainLoop):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_process(self, batch):
        ksapce_c, args_dict = batch
        image_zf = args_dict["image_zf"]
        image = args_dict["image"]
        return image_zf, image

    def forward_backward(self, batch):
        batch, label = self.batch_process(batch)
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
