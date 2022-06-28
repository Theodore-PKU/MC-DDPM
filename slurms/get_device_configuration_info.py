import socket
import argparse


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()


def main(args):
    params_key = ["#SBATCH --nodelist=", "#SBATCH -N ", "#SBATCH --gres=gpu:"]
    params_val_length = {"#SBATCH --nodelist=": 5, "#SBATCH -N ": 1, "#SBATCH --gres=gpu:": 1}
    params = {}

    if args.slurm_file == "":
        nproc_per_node = 1
    else:
        with open(args.slurm_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                for key in params_key:
                    if key in line:
                        val = line.split(key)[1][: params_val_length[key]]
                        params[key] = val
                        break

        nproc_per_node = params["#SBATCH --gres=gpu:"]
    master_addr = socket.gethostbyname(socket.gethostname())
    master_port = _find_free_port()
    with open("../slurm_params.txt", 'w') as f:
        f.write("nproc_per_node={}\n".format(nproc_per_node))
        f.write("master_addr={}\n".format(master_addr))
        f.write("master_port={}\n".format(master_port))
    print("get device configuration.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slurm_file", default="", type=str)
    args = parser.parse_args()
    main(args)
