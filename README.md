# MCDDPM 

## Introduction

This docs is helpful to reproduce our work MCDDPM, which has been accepted by MICCAI 2022. It includes the following parts:
1. Dataset
2. Preparation
3. Experiments

## Dataset

We use the single-coil knee data of fastmri to evaluate our MCDDPM method. The dataset can be downloaded from https://fastmri.med.nyu.edu. The data was extracted to `../datasets/fastmri/` and includes two sub-directories, `knee_singlecoil_train` and `knee_singlecoil_test`. We split the dataset into two parts, `pd` and `pdfs` for different sequences. In our experiments, we consider the acceleration factors of 4 and 8.

## Preparation

Before experiments, we generate some lists which contain fastmri data information and save them as `.pkl` files. They will be used for model training and test.

* for `pd` training

```shell
python utils/dataset_utils/gen_fastmri_data_info.py \
--data_dir ../datasets/fastmri/knee_singlecoil_train \
--data_info_dir data/fastmri \
--num_files -1 \
--num_pd_files -1 \
--num_pdfs_files 0 \
--data_info_file_name pd_train_info
```

* for `pd` test

```shell
python utils/dataset_utils/gen_fastmri_data_info.py \
--data_dir ../datasets/fastmri/knee_singlecoil_val \
--data_info_dir data/fastmri \
--num_files -1 \
--num_pd_files -1 \
--num_pdfs_files 0 \
--data_info_file_name pd_test_info
```

* for `pd` 6 file test

```shell
python utils/dataset_utils/gen_fastmri_data_info.py \
--data_dir ../datasets/fastmri/knee_singlecoil_val \
--data_info_dir data/fastmri \
--num_files -1 \
--num_pd_files 6 \
--num_pdfs_files 0 \
--data_info_file_name pd_test_6_file_info
```

* for `pdfs` training

```shell
python utils/dataset_utils/gen_fastmri_data_info.py \
--data_dir ../datasets/fastmri/knee_singlecoil_train \
--data_info_dir data/fastmri \
--num_files -1 \
--num_pd_files 0 \
--num_pdfs_files -1 \
--data_info_file_name pdfs_train_info
```

* for `pdfs` test

```shell
python utils/dataset_utils/gen_fastmri_data_info.py \
--data_dir ../datasets/fastmri/knee_singlecoil_val \
--data_info_dir data/fastmri \
--num_files -1 \
--num_pd_files 0 \
--num_pdfs_files -1 \
--data_info_file_name pdfs_test_info
```

* for `pdfs` 6 file test

```shell
python utils/dataset_utils/gen_fastmri_data_info.py \
--data_dir ../datasets/fastmri/knee_singlecoil_val \
--data_info_dir data/fastmri \
--num_files -1 \
--num_pd_files 0 \
--num_pdfs_files 6 \
--data_info_file_name pdfs_test_6_file_info
```

## Experiments

We conducted experiments of U-Net and MCDDPM. When training multiple gpus can be used, while only one gpu is used for test.

### U-Net

#### Training for U-Net
Take `pd4x` as an example.

Train for `pd4x`.

```shell
SCRIPT_FLAGS="--method_type unet \
--log_dir logs/fastmri/unet/pd4x"
DATASET_FLAGS="--dataset fastmri --data_dir ../datasets/fastmri/knee_singlecoil_train \
--data_info_list_path data/fastmri/pd_train_info.pkl \
--batch_size 16 --acceleration 4 --num_workers 6"
TRAIN_FLAGS="--microbatch 4 --log_interval 10 --save_interval 5000 --max_step 10000 \
--model_save_dir checkpoints/fastmri/unet/pd4x"

python train.py $SCRIPT_FLAGS $DATASET_FLAGS $TRAIN_FLAGS
```

We use the default model setting as shown in `utils/setting_utils/unet_setting.py`. We can add corresponding arguments to change default setting.

If train the model from last checkpoint, use argument `resume_checkpoint` and other arguments will be loaded from last checkpoint setting.

```shell
SCRIPT_FLAGS="--method_type unet \
--log_dir logs/fastmri/unet/pd4x"
TRAIN_FLAGS="--model_save_dir checkpoints/fastmri/unet/pd4x \
--resume_checkpoint model005000.pt"

python train.py $SCRIPT_FLAGS $TRAIN_FLAGS
```

When training for other data, such as `pd8x`, `pdfs4x` and `pdfs8x`, the following arguments should be specified:

1. --log_dir 
2. --data_info_list_path
3. --acceleration
4. --model_save_dir
5. --output_dir

Trained models along with other setting files will be saved in the sub-directory of `checkpoints`, which is specified by the argument `--model_save_dir`.


#### Test for U-Net

Take `pd4x` as an example and test 6 volumes. Run the following shell script to reproduce our test result for U-Net.

Test for `pd4x`.

```shell
SCRIPT_FLAGS="--method_type unet \
--log_dir logs/fastmri/unet/pd4x"
DATASET_FLAGS="--dataset fastmri --data_dir ../datasets/fastmri/knee_singlecoil_val \
--data_info_list_path data/fastmri/pd_test_6_file_info.pkl \
--acceleration 4 --num_workers 2"
TEST_FLAGS="--microbatch 10 \
--model_save_dir checkpoints/fastmri/unet/pd4x --resume_checkpoint model010000.pt \
--output_dir outputs/fastmri/unet/pd4x \
--debug_mode False"

python test.py $SCRIPT_FLAGS $DATASET_FLAGS $TEST_FLAGS
```

When test for other data, such as `pd8x`, `pdfs4x` and `pdfs8x`, the following arguments should be specified:

1. --log_dir 
2. --data_info_list_path
3. --acceleration
4. --model_save_dir
5. --output_dir

We list other shell script as follows.

Test for `pd8x`.

```shell
SCRIPT_FLAGS="--method_type unet \
--log_dir logs/fastmri/unet/pd8x"
DATASET_FLAGS="--dataset fastmri --data_dir ../datasets/fastmri/knee_singlecoil_val \
--data_info_list_path data/fastmri/pd_test_6_file_info.pkl \
--acceleration 8 --num_workers 2"
TEST_FLAGS="--microbatch 10 \
--model_save_dir checkpoints/fastmri/unet/pd8x --resume_checkpoint model010000.pt \
--output_dir outputs/fastmri/unet/pd8x \
--debug_mode False"

python test.py $SCRIPT_FLAGS $DATASET_FLAGS $TEST_FLAGS
```

Test for `pdfs4x`.

```shell
SCRIPT_FLAGS="--method_type unet \
--log_dir logs/fastmri/unet/pdfs4x"
DATASET_FLAGS="--dataset fastmri --data_dir ../datasets/fastmri/knee_singlecoil_val \
--data_info_list_path data/fastmri/pdfs_test_6_file_info.pkl \
--acceleration 4 --num_workers 2"
TEST_FLAGS="--microbatch 10 \
--model_save_dir checkpoints/fastmri/unet/pdfs4x --resume_checkpoint model010000.pt \
--output_dir outputs/fastmri/unet/pdfs4x \
--debug_mode False"

python test.py $SCRIPT_FLAGS $DATASET_FLAGS $TEST_FLAGS
```

Test for `pdfs8x`.

```shell
SCRIPT_FLAGS="--method_type unet \
--log_dir logs/fastmri/unet/pdfs8x"
DATASET_FLAGS="--dataset fastmri --data_dir ../datasets/fastmri/knee_singlecoil_val \
--data_info_list_path data/fastmri/pd_test_6_file_info.pkl \
--acceleration 8 --num_workers 2"
TEST_FLAGS="--microbatch 10 \
--model_save_dir checkpoints/fastmri/unet/pdfs8x --resume_checkpoint model010000.pt \
--output_dir outputs/fastmri/unet/pdfs8x \
--debug_mode False"

python test.py $SCRIPT_FLAGS $DATASET_FLAGS $TEST_FLAGS
```

The result of testing will be saved in the sub-directory of `outputs` which is specified by the argument `--output_dir`. The output directory contains reconstructions and evaluation metrics.

### MCDDPM

#### Training for MCDDPM
Take `pd4x` as an example.

Train for `pd4x`.

```shell
SCRIPT_FLAGS="--method_type mcddpm \
--log_dir logs/fastmri/mcddpm/pd4x"
DATASET_FLAGS="--dataset fastmri --data_dir ../datasets/fastmri/knee_singlecoil_train \
--data_info_list_path data/fastmri/pd_train_info.pkl \
--batch_size 16 --acceleration 4 --num_workers 6"
TRAIN_FLAGS="--microbatch 4 --log_interval 10 --save_interval 5000 --max_step 35000 \
--model_save_dir checkpoints/fastmri/mcddpm/pd4x"

python train.py $SCRIPT_FLAGS $DATASET_FLAGS $TRAIN_FLAGS
```

We use the default model setting, diffusion setting and schedule sampler setting as shown in `utils/setting_utils/mcddpm_setting.py`. We can add corresponding arguments to change default setting.

Trained models along with other setting files will be saved in the sub-directory of `checkpoints`, which is specified by the argument `--model_save_dir`.

If train the model from last checkpoint, use argument `resume_checkpoint` and other arguments will be loaded from last checkpoint setting.

```shell
SCRIPT_FLAGS="--method_type mcddpm \
--log_dir logs/fastmri/mcddpm/pd4x"
TRAIN_FLAGS="--model_save_dir checkpoints/fastmri/mcddpm/pd4x \
--resume_checkpoint model005000.pt"

python train.py $SCRIPT_FLAGS $TRAIN_FLAGS
```

#### Test for MCDDPM

Take `pd4x` as an example and test 6 volumes. Run the following shell script to reproduce our test result for MCDDPM.

```shell
SCRIPT_FLAGS="--method_type mcddpm \
--log_dir logs/fastmri/mcddpm/pd4x"
DATASET_FLAGS="--dataset fastmri --data_dir ../datasets/fastmri/knee_singlecoil_val \
--data_info_list_path data/fastmri/pd_test_6_file_info.pkl \
--batch_size 20 --acceleration 4 --num_workers 2"
TEST_FLAGS="--model_save_dir checkpoints/fastmri/mcddpm/pd4x --resume_checkpoint model035000.pt \
--output_dir outputs/fastmri/mcddpm/pd4x --num_samples_per_mask 20 \
--debug_mode False"

test.py $SCRIPT_FLAGS $DATASET_FLAGS $TEST_FLAGS
```

The argument `--num_samples_per_mask` is to control the number of construction for one slice. We can also add the argument `--timestep_respacing 500` to specify the sampling steps (default is 1000).

When training for other data, such as `pd8x`, `pdfs4x` and `pdfs8x`, the following arguments should be specified:

1. --log_dir 
2. --data_info_list_path
3. --acceleration
4. --model_save_dir
5. --output_dir

The specific settings for `pd8x`, `pdfs4x`, `pdfs8x` are similar to the part of U-Net above.

The result of testing will be saved in the sub-directory of `outpus` which is specified by the argument `--output_dir`. The output directory contains reconstructions and evaluation metrics.


### Trained Models

The trained models can be downloaded from https://drive.google.com/drive/folders/1cR4_6CX8tfGEHz_UytT5QbHKXSEOJOxX?usp=sharing to test the performance.
