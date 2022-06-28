module add anaconda/3
source activate pytorch1.9

python utils/dataset_utils/gen_fastmri_data_info.py \
--data_dir ../datasets/fastmri/knee_singlecoil_train \
--data_info_dir data/fastmri \
--num_files -1 \
--num_pd_files -1 \
--num_pdfs_files 0 \
--data_info_file_name pd_train_info

python utils/dataset_utils/gen_fastmri_data_info.py \
--data_dir ../datasets/fastmri/knee_singlecoil_val \
--data_info_dir data/fastmri \
--num_files -1 \
--num_pd_files -1 \
--num_pdfs_files 0 \
--data_info_file_name pd_test_info

python utils/dataset_utils/gen_fastmri_data_info.py \
--data_dir ../datasets/fastmri/knee_singlecoil_val \
--data_info_dir data/fastmri \
--num_files -1 \
--num_pd_files 1 \
--num_pdfs_files 0 \
--data_info_file_name pd_test_1_file_info

python utils/dataset_utils/gen_fastmri_data_info.py \
--data_dir ../datasets/fastmri/knee_singlecoil_val \
--data_info_dir data/fastmri \
--num_files -1 \
--num_pd_files 6 \
--num_pdfs_files 0 \
--data_info_file_name pd_test_6_file_info

python utils/dataset_utils/gen_fastmri_data_info.py \
--data_dir ../datasets/fastmri/knee_singlecoil_train \
--data_info_dir data/fastmri \
--num_files -1 \
--num_pd_files 0 \
--num_pdfs_files -1 \
--data_info_file_name pdfs_train_info

python utils/dataset_utils/gen_fastmri_data_info.py \
--data_dir ../datasets/fastmri/knee_singlecoil_val \
--data_info_dir data/fastmri \
--num_files -1 \
--num_pd_files 0 \
--num_pdfs_files -1 \
--data_info_file_name pdfs_test_info

python utils/dataset_utils/gen_fastmri_data_info.py \
--data_dir ../datasets/fastmri/knee_singlecoil_val \
--data_info_dir data/fastmri \
--num_files -1 \
--num_pd_files 0 \
--num_pdfs_files 1 \
--data_info_file_name pdfs_test_1_file_info

python utils/dataset_utils/gen_fastmri_data_info.py \
--data_dir ../datasets/fastmri/knee_singlecoil_val \
--data_info_dir data/fastmri \
--num_files -1 \
--num_pd_files 0 \
--num_pdfs_files 6 \
--data_info_file_name pdfs_test_6_file_info