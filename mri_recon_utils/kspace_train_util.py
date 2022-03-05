from utils.train_util import *


class KspaceTrainLoop(TrainLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_step(self, batch, cond):
        # modify condition so that it only contains the information we need.
        cond = {
            k: cond[k] for k in ["kspace_zf", "image_zf", "mask_c"]
        }
        # others are unchanged
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
