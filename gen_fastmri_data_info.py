import os
import h5py
import numpy as np
from utils.script_util import *

CORPD_FBK = 4e-5
CORPDFS_FBK = 1.5e-5


def create_argparser():
    defaults = dict(
        data_dir="data/fastmri/train",
        data_info_dir="data/fastmri",
        min_value_pd=CORPD_FBK,
        min_value_pdfs=CORPDFS_FBK,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()
    file_list = []
    for entry in sorted(os.listdir(args.data_dir)):
        ext = entry.split(".")[-1]
        if "." in entry and ext == "h5":
            file_list.append(entry)

    data_info_list = []
    pd_data_info_list = []
    pdfs_data_info_list = []
    for file_name in file_list:
        file_path = os.path.join(args.data_dir, file_name)
        data = h5py.File(file_path, mode="r")
        image_rss = np.array(data["reconstruction_rss"])
        acquisition = data.attrs["acquisition"]
        # if acquisition == "CORPD_FBK":
        #     min_value = args.min_value_pd
        # else:
        #     min_value = args.min_value_pdfs
        # start = 0
        # for i in range(len(image_rss)):
        #     if np.mean(image_rss[i]) < min_value:
        #         start += 1
        #     else:
        #         break
        # end = len(image_rss)
        # for i in range(len(image_rss) - 1, -1, -1):
        #     if np.mean(image_rss[i]) < min_value:
        #         end -= 1
        #     else:
        #         break
        num_slice = len(image_rss)
        for i in range(5, num_slice - 5):
            data_info_list.append((file_name, i))
            if acquisition == "CORPD_FBK":
                pd_data_info_list.append((file_name, i))
            else:
                pdfs_data_info_list.append((file_name, i))

    with open(os.path.join(args.data_info_dir, "fastmri_test_info.pkl"), "wb") as f:
        pickle.dump(data_info_list, f)
        print("all slices", len(data_info_list))
    with open(os.path.join(args.data_info_dir, "fastmri_pd_test_info.pkl"), "wb") as f:
        pickle.dump(pd_data_info_list, f)
        print("pd slices", len(pd_data_info_list))
    with open(os.path.join(args.data_info_dir, "fastmri_pdfs_test_info.pkl"), "wb") as f:
        print("pdfs slices", len(pdfs_data_info_list))
        pickle.dump(pdfs_data_info_list, f)


if __name__ == "__main__":
    main()
