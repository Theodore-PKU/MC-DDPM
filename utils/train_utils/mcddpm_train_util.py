from utils.train_utils.ddpm_train_util import *


class MCDDPMTrainLoop(DDPMTrainLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_process(self, batch):
        # modify condition so that it only contains the information we need.
        batch, cond = batch
        cond = {
            k: cond[k] for k in ["kspace_zf", "image_zf", "mask_c"]
        }
        return batch, cond
