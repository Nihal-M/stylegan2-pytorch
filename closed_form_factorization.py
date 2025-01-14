import argparse

import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract factor/eigenvectors of latent spaces using closed form factorization"
    )

    parser.add_argument(
        "--out", type=str, default="factor.pt", help="name of the result factor file"
    )
    parser.add_argument('--save_dir', type=str, default='/ocean/projects/asc170022p/singla/MIMICCX-Chest-Explainer/stylegan2-pytorch/Experiment_MIMIC_CXR/Run1/', help='')
    parser.add_argument(
        "--ckpt",
        type=str,
        default='/ocean/projects/asc170022p/singla/MIMICCX-Chest-Explainer/stylegan2-pytorch/Experiment_MIMIC_CXR/Run1/checkpoint/030000.pt',
        help="path to the model checkpoint",
    )

    args = parser.parse_args()

    ckpt = torch.load(args.ckpt)
    modulate = {
        k: v
        for k, v in ckpt["g_ema"].items()
        if "modulation" in k and "to_rgbs" not in k and "weight" in k
    }

    weight_mat = []
    for k, v in modulate.items():
        weight_mat.append(v)

    W = torch.cat(weight_mat, 0)
    eigvec = torch.svd(W).V.to("cpu")

    torch.save({"ckpt": args.ckpt, "eigvec": eigvec}, args.save_dir + args.out)

