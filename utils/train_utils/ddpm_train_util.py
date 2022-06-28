import functools

from utils import dist_util, logger
from utils.ddpm_utils.resample import LossAwareSampler, UniformSampler
from utils.train_utils.base_train_util import TrainLoop


class DDPMTrainLoop(TrainLoop):

    def __init__(self, diffusion, schedule_sampler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion = diffusion
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)

    def batch_process(self, batch):
        batch, cond = batch
        return batch, cond

    def forward_backward(self, batch):
        batch, cond = self.batch_process(batch)
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.log_kv(key, values.mean())
        # Log the quantiles (eight quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach()):
            quartile = int(8 * sub_t / diffusion.num_timesteps)
            logger.log_kv(f"{key}_q{quartile}", sub_loss)
