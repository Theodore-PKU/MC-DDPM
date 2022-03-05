import os
import torch as th
import torch.distributed as dist

from . import dist_util, logger
from .debug_util import *


class TestLoop:

    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            batch_size,
            model_save_dir="",
            resume_checkpoint="",
            output_dir="",
            use_fp16=False,
            debug_mode=False,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.model_save_dir = model_save_dir
        self.resume_checkpoint = resume_checkpoint
        assert resume_checkpoint != "", "the model for test must be specified."
        self.output_dir = output_dir
        self.use_fp16 = use_fp16
        self.debug_mode = debug_mode

        self._load_parameters()
        logger.log(f"This model contains {count_parameters_in_M(self.model)}M parameters")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.step = 0

    def _load_parameters(self):
        model_path = os.path.join(self.model_save_dir, self.resume_checkpoint)
        logger.log(f"loading model from checkpoint: {model_path}...")
        self.model.load_state_dict(
            th.load(
                model_path,
                map_location="cpu",
            )
        )
        self.model.to(dist_util.dev())
        if self.use_fp16:
            self.model.convert_to_fp16()

        self.model.eval()

    def run_loop(self):
        for _, batch_kwargs in self.data:
            batch_kwargs = {k: v.to(dist_util.dev()) for k, v in batch_kwargs.items()}
            self.sample(batch_kwargs)
            self.step += 1
            if self.debug_mode and self.step == 1:
                show_gpu_usage(f"step: {self.step}, device: {dist.get_rank()}", idx=dist.get_rank())
            if self.step % 10 == 0:
                logger.log(f"have run {self.step} steps")

        dist.barrier()

    def sample(self, batch_kwargs):
        """
        The sample process is defined in children class.
        """
        pass
