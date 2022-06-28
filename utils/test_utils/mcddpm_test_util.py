from utils.test_utils.ddpm_test_util import DDPMTestLoop
from utils import dist_util
from utils.mri_data_utils.transform_util import *


class MCDDPMTestLoop(DDPMTestLoop):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self, batch_kwargs):
        cond = {
            k: batch_kwargs[k].to(dist_util.dev()) for k in ["kspace_zf", "image_zf", "mask_c"]
        }
        samples = []
        while len(samples) * self.batch_size < self.num_samples_per_mask:
            sample = self.diffusion.sample_loop(
                self.model,
                (self.batch_size, 2, self.image_size, self.image_size),
                cond,
                clip=False
            )
            kspace = sample + cond["kspace_zf"]
            # permute operation is for fft operation.
            sample = ifftc_th(kspace)
            samples.append(sample.cpu().numpy())

        # gather all samples and save them
        samples = np.concatenate(samples, axis=0)
        samples = samples[: self.num_samples_per_mask]
        return samples
