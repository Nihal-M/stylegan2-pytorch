import argparse

import torch
from torchvision import utils




if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")

    parser.add_argument(
        "-i", "--index", type=int, default=0, help="index of eigenvector"
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=5,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help='channel multiplier factor. config-f = 2, else = 1',
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default='/ocean/projects/asc170022p/singla/MIMICCX-Chest-Explainer/stylegan2-pytorch/Experiment_MIMIC_CXR/Run1/checkpoint/030000.pt',
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "-n", "--n_sample", type=int, default=20, help="number of samples created"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="truncation factor"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="factor",
        help="filename prefix to result samples",
    )

    parser.add_argument('--save_dir', type=str, default='/ocean/projects/asc170022p/singla/MIMICCX-Chest-Explainer/stylegan2-pytorch/Experiment_MIMIC_CXR/Run1/', help='')
    parser.add_argument('--arch', type=str, default='swagan', help='model architectures (stylegan2 | swagan)')
    args = parser.parse_args()

    eigvec = torch.load(args.save_dir + 'factor.pt')["eigvec"].to(args.device)
    ckpt = torch.load(args.ckpt)
    if args.arch == 'stylegan2':
        from model import Generator

    elif args.arch == 'swagan':
        from swagan import Generator
    
    g = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier).to(args.device)
    g.load_state_dict(ckpt["g_ema"], strict=False)
    

    trunc = g.mean_latent(4096)

    latent = torch.randn(args.n_sample, 512, device=args.device)
    latent = g.get_latent(latent)
    
    for index in range(0,100):
        direction = args.degree * eigvec[:, index].unsqueeze(0)

        img, _ = g(
            [latent],
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )
        img1, _ = g(
            [latent + direction],
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )
        img2, _ = g(
            [latent - direction],
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )

        grid = utils.save_image(
            torch.cat([img1, img, img2], 0),
            args.save_dir+ 'index' + f"{args.out_prefix}_index-{index}_degree-{args.degree}.png",
            normalize=True,
            range=(0, 1),
            nrow=args.n_sample,
        )
