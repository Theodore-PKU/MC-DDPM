from mri_recon_utils.data_utils import *
import argparse
from utils.script_util import add_dict_to_argparser, load_args_dict


def main():
    args = create_argparser().parse_args()

    slice_metrics_samples = {
        metric: [] for metric in ["psnr", "ssim", "mse", "nmse"]
    }
    volume_metrics_samples = {
        metric: [] for metric in ["psnr", "ssim", "mse", "nmse"]
    }
    slice_metrics_mean = {
        metric: [] for metric in ["psnr", "ssim", "mse", "nmse"]
    }
    volume_metrics_mean = {
        metric: [] for metric in ["psnr", "ssim", "mse", "nmse"]
    }

    file_dir = []
    for sub_dir in os.listdir(args.output_dir):
        dir_path = os.path.join(args.output_dir, sub_dir)
        if os.path.isdir(dir_path) and sub_dir.endswith(".h5"):
            file_dir.append(dir_path)

    for sub_dir in file_dir:
        # for each file
        slice_dirs_tmp = os.listdir(sub_dir)
        slice_dirs = []

        for slice_dir in slice_dirs_tmp:
            if slice_dir.startswith("slice_"):
                slice_dirs.append(slice_dir)

        slice_dirs = sorted(slice_dirs, key=extract_slice_index)
        image_batch = []
        slice_index_batch = []
        output_mean_batch = []
        output_batch = []
        for slice_dir in slice_dirs:
            # for each slice
            slice_path = os.path.join(sub_dir, slice_dir)
            all_samples = np.load(os.path.join(slice_path, "all_samples.npz"))["arr_0"]
            # all_samples2 = np.load(
            #     os.path.join("outputs/expe_4_pd8x/test_6_files", sub_dir.split("/")[-1], slice_dir, "all_samples.npz")
            # )["arr_0"]
            # all_samples = np.concatenate([all_samples, all_samples2], axis=0)[:40, ...]
            slice_args = load_args_dict(os.path.join(slice_path, "slice_information.pkl"))
            scale_coeff = slice_args["scale_coeff"]

            slice_index_batch.append(slice_args["slice_index"])
            # transform to magnitude image
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

        for i in range(len(image_batch)):
            gt = image_batch[i:i + 1]
            pred = output_mean_batch[i:i + 1]
            compute_metrics(gt, pred, slice_metrics_mean)

            for j in range(output_batch.shape[1]):
                pred = output_batch[i, j:j+1, ...]
                compute_metrics(gt, pred, slice_metrics_samples)

            write_slice_samples_metric(
                volume_dir=sub_dir,
                slice_index=slice_index_batch[i],
                metrics_dict=slice_metrics_samples,
                num_samples=output_batch.shape[1],
            )

        # write the metrics for the mean of samples at each slice
        write_slice_mean_metrics(
            volume_dir=sub_dir,
            metrics_dict=slice_metrics_mean,
            slice_index_batch=slice_index_batch,
        )

        # compute volume metrics for sample mean
        compute_metrics(image_batch, output_mean_batch, volume_metrics_mean)
        write_volume_mean_metrics(
            volume_dir=sub_dir,
            metrics_dict=volume_metrics_mean,
        )

        for j in range(output_batch.shape[1]):
            pred = output_batch[:, j, ...]
            compute_metrics(image_batch, pred, volume_metrics_samples)
        write_volume_sample_metrics(
            volume_dir=sub_dir,
            metrics_dict=volume_metrics_samples,
            num_samples=output_batch.shape[1],
        )

        print(f"{sub_dir} metrics computing complete.")

    with open(os.path.join(args.output_dir, "metrics_average.txt"), "w") as f:
        print_stat_info("volum samples\n", volume_metrics_samples, f)
        print_stat_info("volume mean\n", volume_metrics_mean, f)
        print_stat_info("slice samples\n", slice_metrics_samples, f)
        print_stat_info("slice mean\n", slice_metrics_mean, f)


def create_argparser():
    defaults = dict(
        # log_dir="logs",
        output_dir="",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def extract_slice_index(slice_index):
    return int(slice_index.split("_")[-1])


def compute_metrics(gt, pred, metrics_dict):
    metrics_dict["mse"].append(compute_mse(gt, pred))
    metrics_dict["nmse"].append(compute_nmse(gt, pred))
    metrics_dict["psnr"].append(compute_psnr(gt, pred))
    metrics_dict["ssim"].append(compute_ssim(gt, pred))


def write_volume_mean_metrics(volume_dir, metrics_dict):
    with open(os.path.join(volume_dir, "metrics_for_volume_sample_mean.txt"), "w") as f:
        f.write(f"psnr | ssim | nmse | mse\n")
        for metric in ["psnr", "ssim", "nmse", "mse"]:
            if metric != "mse":
                f.write(f"{metrics_dict[metric][-1]:10.6f}")
            else:
                f.write(f"{metrics_dict[metric][-1]:12.4e}\n")


def write_volume_sample_metrics(volume_dir, metrics_dict, num_samples):
    with open(os.path.join(volume_dir, "metrics_for_volume_sample.txt"), "w") as f:
        f.write(f"psnr | ssim | nmse | mse\n")
        for i in range(num_samples):
            f.write(f"sample {i:2}: ")
            for metric in ["psnr", "ssim", "nmse", "mse"]:
                if metric != "mse":
                    f.write(f"{metrics_dict[metric][i - num_samples]:10.6f}")
                else:
                    f.write(f"{metrics_dict[metric][i - num_samples]:12.4e}\n")

        f.write("\nstats: mean +/- (std)\n")
        for metric in ["psnr", "ssim", "nmse", "mse"]:
            if metric != "mse":
                f.write(f"{metric:4}: {np.mean(metrics_dict[metric][- num_samples:]):.6f} +/- "
                        f"({np.std(metrics_dict[metric]):.6f})\n")
            else:
                f.write(f"{metric:4}: {np.mean(metrics_dict[metric][- num_samples:]):.4e} +/- "
                        f"({np.std(metrics_dict[metric]):.4e})\n\n")


def write_slice_samples_metric(volume_dir, slice_index, metrics_dict, num_samples):
    with open(
            os.path.join(volume_dir, f"slice_{slice_index}", f"metrics_for_slice_{slice_index}_samples.txt"),
            "w"
    ) as f:
        f.write(f"psnr | ssim | nmse | mse\n")
        for i in range(num_samples):
            f.write(f"sample {i:2}: ")
            for metric in ["psnr", "ssim", "nmse", "mse"]:
                if metric != "mse":
                    f.write(f"{metrics_dict[metric][i - num_samples]:10.6f}")
                else:
                    f.write(f"{metrics_dict[metric][i - num_samples]:12.4e}\n")
        f.write("\nstats: mean +/- (std)\n")
        for metric in ["psnr", "ssim", "nmse", "mse"]:
            if metric != "mse":
                f.write(f"{metric:4}: {np.mean(metrics_dict[metric][- num_samples:]):.6f} +/- "
                        f"({np.std(metrics_dict[metric]):.6f})\n")
            else:
                f.write(f"{metric:4}: {np.mean(metrics_dict[metric][- num_samples:]):.4e} +/- "
                        f"({np.std(metrics_dict[metric]):.4e})\n\n")


def write_slice_mean_metrics(volume_dir, metrics_dict, slice_index_batch):
    with open(os.path.join(volume_dir, "metrics_for_slice_sample_mean.txt"), "w") as f:
        f.write(f"psnr | ssim | nmse | mse\n")
        num_slices = len(slice_index_batch)
        for i in range(num_slices):
            f.write(f"slice {slice_index_batch[i]:2}: ")
            for metric in ["psnr", "ssim", "nmse", "mse"]:
                if metric != "mse":
                    f.write(f"{metrics_dict[metric][i - num_slices]:10.6f}")
                else:
                    f.write(f"{metrics_dict[metric][i - num_slices]:12.4e}\n")


def print_stat_info(statement, metrics_dict, f):
    f.write(statement)
    for metric in ["psnr", "ssim", "nmse", "mse"]:
        if metric != "mse":
            f.write(f"{metric:4}: {np.mean(metrics_dict[metric]):.6f} +/- "
                    f"({np.std(metrics_dict[metric]):.6f})\n")
        else:
            f.write(f"{metric:4}: {np.mean(metrics_dict[metric]):.4e} +/- "
                    f"({np.std(metrics_dict[metric]):.4e})\n\n")


if __name__ == "__main__":
    main()
