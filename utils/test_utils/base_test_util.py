import os
import torch.distributed as dist

from utils import dist_util, logger
from utils.debug_util import *
from utils.mri_data_utils.transform_util import *
from utils.mri_data_utils.metrics_util import *


class TestLoop:
    def __init__(
            self,
            *,
            model,
            data,
            batch_size,
            log_interval,
            model_save_dir="",
            resume_checkpoint="",
            output_dir="",
            use_fp16=False,
            debug_mode=False,
    ):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.model_save_dir = model_save_dir
        if self.model:
            assert resume_checkpoint != "", "the model for test must be specified."
        else:
            assert resume_checkpoint == "", "do not use any model."
        self.resume_checkpoint = resume_checkpoint
        self.output_dir = output_dir
        self.use_fp16 = use_fp16
        self.debug_mode = debug_mode

        self.step = 0

        if self.model:
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
            if self.debug_mode or self.step % self.log_interval == 0:
                show_gpu_usage(f"step: {self.step}, device: {dist.get_rank()}", idx=dist.get_rank())

            if self.step % self.log_interval == 0:
                logger.log(f"have test {self.step} steps")

        dist.barrier()

    def forward_backward(self, data_item):
        pass

    def compute_metrics_for_dataset(self):
        pass


def compute_metrics(gt, pred):
    """
    Compute metrics for one slice (b=1) or one volume (b>1).

    :param gt: ground truth, torch.Tensor with shape of (b, 2, h, w).
    :param pred: reconstruction, torch.Tensor with shape of (b, 2, h, w).
    """
    if isinstance(gt, th.Tensor):
        gt = th2np_magnitude(gt)
    if isinstance(pred, th.Tensor):
        pred = th2np_magnitude(pred)
    return {metric: METRICS_FUNC[metric](gt, pred) for metric in METRICS}


def refine_output(output, mask, kspace_zf):
    return ifftc_th(fftc_th(output) * (1. - mask) + kspace_zf)


def write_metric_to_file(file_name, metrics_dict, statement, mode="w"):
    with open(file_name, mode) as f:
        f.write(statement)
        for key in METRICS:
            f.write(f"{key}\t{metrics_dict[key]:10.4e}\n")


def write_average_metrics_to_file(file_name, metrics_dict, statement):
    with open(file_name, "w") as f:
        f.write(statement)
        for key in METRICS:
            f.write(f"{key:}\t{np.mean(metrics_dict[key]):10.4e} +/- "
                    f"{np.std(metrics_dict[key]):10.4e}\n")
