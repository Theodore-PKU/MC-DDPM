import matplotlib.pyplot as plt

from utils.script_util import save_args_dict, load_args_dict
from utils.test_utils.base_test_util import *
from utils.mri_data_utils.metrics_util import METRICS
from utils.mri_data_utils.transform_util import *


MAX_NUM_SAVED_SAMPLES = 5


class DDPMTestLoop(TestLoop):

    def __init__(self, diffusion, image_size, num_samples_per_mask=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion = diffusion
        self.image_size = image_size
        self.num_samples_per_mask = num_samples_per_mask

        self.slice_metrics_samples = {
            metric: [] for metric in METRICS
        }
        self.volume_metrics_samples = {
            metric: [] for metric in METRICS
        }
        self.slice_metrics_mean = {
            metric: [] for metric in METRICS
        }
        self.volume_metrics_mean = {
            metric: [] for metric in METRICS
        }

    def run_loop(self):
        super().run_loop()
        self.compute_metrics_for_dataset()

    def forward_backward(self, data_item):
        kspace_c, batch_kwargs = data_item
        file_name = batch_kwargs["file_name"]
        slice_index = batch_kwargs["slice_index"]
        samples_path = os.path.join(self.output_dir, file_name, f"slice_{slice_index}")
        if os.path.exists(samples_path):
            if os.path.exists(os.path.join(samples_path, "slice_information.pkl")):
                logger.log(f"have sampled for {file_name} slice {slice_index}")
                return
        else:
            os.makedirs(samples_path, exist_ok=True)

        samples = self.sample(batch_kwargs)
        self.save_samples(samples, samples_path, batch_kwargs)
        logger.log(f"complete sampling for {file_name} slice {slice_index}")

    def sample(self, batch_kwargs):
        """
        The sample process is defined in children class.
        """
        pass

    def save_samples(self, samples, samples_path, batch_kwargs):
        # save mask, image, image_zf
        mask = batch_kwargs["mask"][0, 0, ...]
        # transform to magnitude images
        image = np.abs(real2complex_np(batch_kwargs["image"][0]))
        image_zf = np.abs(real2complex_np(batch_kwargs["image_zf"][0]))
        for to_save_image, name in zip([mask, image, image_zf], ["mask", "image", "image_zf"]):
            plt.imsave(
                fname=os.path.join(samples_path, f"{name}.png"),
                arr=to_save_image, cmap="gray"
            )

        # save some samples, less than 5
        for i in range(min(MAX_NUM_SAVED_SAMPLES, len(samples))):
            sample = np.abs(real2complex_np(samples[i]))
            plt.imsave(
                fname=os.path.join(samples_path, f"sample_{i + 1}.png"),
                arr=sample, cmap="gray"
            )

        # save all information
        np.savez(os.path.join(samples_path, f"all_samples"), samples)  # arr is not magnitude images
        saved_args = {
            "scale_coeff": batch_kwargs["scale_coeff"],
            "slice_index": batch_kwargs["slice_index"],
            "image": batch_kwargs["image"][0:1, ...],
        }
        save_args_dict(saved_args, os.path.join(samples_path, "slice_information.pkl"))

    def compute_metrics_for_dataset(self):
        file_dirs = []
        for file_dir in os.listdir(self.output_dir):
            dir_path = os.path.join(self.output_dir, file_dir)
            if os.path.isdir(dir_path) and file_dir.endswith(".h5"):
                file_dirs.append(dir_path)

        for file_dir in file_dirs:
            # for each volume
            slice_dirs = []
            for slice_dir in os.listdir(file_dir):
                if slice_dir.startswith("slice_"):
                    slice_dirs.append(slice_dir)

            slice_dirs = sorted(slice_dirs, key=extract_slice_index)
            image_batch = []
            slice_index_batch = []
            output_mean_batch = []
            output_batch = []
            for slice_dir in slice_dirs:
                # for each slice
                slice_path = os.path.join(file_dir, slice_dir)
                all_samples = np.load(os.path.join(slice_path, "all_samples.npz"))["arr_0"]
                slice_args = load_args_dict(os.path.join(slice_path, "slice_information.pkl"))
                scale_coeff = slice_args["scale_coeff"]
                slice_index_batch.append(slice_args["slice_index"])
                image = th2np_magnitude(slice_args["image"]) / scale_coeff
                image_batch.append(image)
                # transform to magnitude image
                all_samples = np.sqrt(all_samples[:, 0, ...] ** 2 + all_samples[:, 1, ...] ** 2)
                sample_mean = np.mean(all_samples, axis=0)
                sample_std = np.std(all_samples, axis=0)
                plt.imsave(
                    fname=os.path.join(slice_path, "sample_mean.png"),
                    arr=sample_mean, cmap="gray"
                )
                plt.imsave(
                    fname=os.path.join(slice_path, "sample_std.png"),
                    arr=sample_std / np.max(sample_std), cmap="gray"
                )
                # sampe_std is still computed in 0-1 range.
                np.savez(
                    os.path.join(slice_path, "sample_std"), sample_std
                )
                all_samples = all_samples / scale_coeff
                sample_mean = sample_mean / scale_coeff
                output_mean_batch.append(sample_mean)
                output_batch.append(all_samples)

            image_batch = np.concatenate(image_batch, axis=0)
            output_mean_batch = np.stack(output_mean_batch, axis=0)
            output_batch = np.stack(output_batch, axis=0)

            # compute evaluation metrics
            for i in range(len(image_batch)):
                # slice: mean of samples
                gt = image_batch[i:i + 1]
                pred = output_mean_batch[i:i + 1]
                curr_slice_metrics = compute_metrics(gt, pred)
                for key in METRICS:
                    self.slice_metrics_mean[key].append(curr_slice_metrics[key])
                write_metric_to_file(
                    os.path.join(file_dir, slice_dirs[i], "slice_samples_mean_metrics.txt"),
                    curr_slice_metrics,
                    f"volume {file_dir}, {slice_dirs[i]}, mean of samples\n"
                )
                # slice: each sample
                for j in range(output_batch.shape[1]):
                    pred = output_batch[i, j:j + 1, ...]
                    curr_slice_metrics = compute_metrics(gt, pred)
                    for key in METRICS:
                        self.slice_metrics_samples[key].append(curr_slice_metrics[key])
                    if j == 0:
                        mode = "w"
                    else:
                        mode = "a"
                    write_metric_to_file(
                        os.path.join(file_dir, slice_dirs[i], f"slice_samples_metrics.txt"),
                        curr_slice_metrics,
                        f"volume {file_dir}, {slice_dirs[i]}, sample {j + 1}\n",
                        mode=mode
                    )
            # volume: mean of samples
            curr_volume_metrics = compute_metrics(image_batch, output_mean_batch)
            for key in METRICS:
                self.volume_metrics_mean[key].append(curr_volume_metrics[key])
            write_metric_to_file(
                os.path.join(file_dir, f"volume_samples_mean_metrics.txt"),
                curr_volume_metrics,
                f"volume {file_dir}, mean of samples\n"
            )
            # volume: each sample
            for j in range(output_batch.shape[1]):
                pred = output_batch[:, j, ...]
                curr_volume_metrics = compute_metrics(image_batch, pred)
                for key in METRICS:
                    self.volume_metrics_samples[key].append(curr_volume_metrics[key])
                write_metric_to_file(
                    os.path.join(file_dir, f"volume_sample_{j + 1}_metrics.txt"),
                    curr_volume_metrics,
                    f"volume {file_dir}, sample {j + 1}\n"
                )

        write_average_metrics_to_file(
            os.path.join(self.output_dir, "slice_sample_mean_average_metrics.txt"),
            self.slice_metrics_mean,
            "average of metrics for mean of samples of all slice data\n"
        )
        write_average_metrics_to_file(
            os.path.join(self.output_dir, "slice_samples_average_metrics.txt"),
            self.slice_metrics_samples,
            "average of metrics for samples of all slice data\n"
        )
        write_average_metrics_to_file(
            os.path.join(self.output_dir, "volume_sample_mean_average_metrics.txt"),
            self.volume_metrics_mean,
            "average of metrics for mean of samples of all volume data\n"
        )
        write_average_metrics_to_file(
            os.path.join(self.output_dir, "volume_samples_average_metrics.txt"),
            self.volume_metrics_samples,
            "average of metrics for samples of all volume data\n"
        )


def extract_slice_index(slice_index):
    return int(slice_index.split("_")[-1])
