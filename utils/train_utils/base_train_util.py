import copy
import os
import time
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW


from utils import dist_util, logger
from utils.debug_util import *
from utils.fp16_util import MixedPrecisionTrainer
from models.nn import update_ema


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:

    def __init__(
            self,
            *,
            model,
            data,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            model_save_dir="",
            resume_checkpoint="",
            use_fp16=False,
            fp16_scale_growth=1e-3,
            initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
            weight_decay=0.0,
            lr_anneal_steps=0,
            max_step=100000000,
            run_time=-1,
            debug_mode=False,
    ):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.model_save_dir = model_save_dir
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.max_step = max_step
        self.run_time = run_time
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
            initial_lg_loss_scale=initial_lg_loss_scale,
        )

        self.opt = AdamW(self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
            )
        else:
            if dist.get_world_size() > 1:
                logger.log(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

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

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = os.path.join(self.model_save_dir, self.resume_checkpoint)
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = th.load(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        opt_checkpoint = bf.join(self.model_save_dir, f"opt{self.resume_step:06}.pt")
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        start_time = time.time()
        while (
                not self.lr_anneal_steps
                or self.step < self.lr_anneal_steps
        ):
            self.run_step(next(self.data))
            self.step += 1

            if (self.debug_mode and self.step % self.log_interval == 0) or \
                    (not self.debug_mode and self.step == self.log_interval) or \
                    (self.step % self.save_interval == 0):
                show_gpu_usage(f"step: {self.step}, device: {dist.get_rank()}", idx=dist.get_rank())

            if self.step % self.log_interval == 0 and self.step <= 10 * self.log_interval:
                logger.log(f"have trained {self.step} step")

            if self.step % self.log_interval == 0:
                logger.write_kv(self.step)
                logger.clear_kv()

            if self.step % self.save_interval == 0:
                self.save()

            if self.run_time > 0 and time.time() - start_time > self.run_time * 3600 and self.save_last:
                self.save()
                self.save_last = False

            if self.step >= self.max_step:
                break

        # Save the last checkpoint if it wasn't already saved.
        if self.step % self.save_interval != 0:
            self.save()

    def run_step(self, batch):
        self.forward_backward(batch)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()

    def forward_backward(self, batch):
        pass

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = self.step / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{self.step:06d}.pt"
                else:
                    filename = f"ema_{rate}_{self.step:06d}.pt"
                with bf.BlobFile(bf.join(self.model_save_dir, filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(self.model_save_dir, f"opt{self.step:06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{step:06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None
