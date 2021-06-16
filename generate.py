import argparse

import torch
from torchvision import utils
from tqdm import tqdm
import pdb
import numpy as np

def generate(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )
            
            utils.save_image(
                sample,
                args.save_dir + f"sample/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(0, 1),
            )
            sample =sample.cpu()
            sample = np.asarray(sample)
            np.save(args.save_dir + f"sample/{str(i).zfill(6)}.npy", sample)


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument('--save_dir', type=str, default='/ocean/projects/asc170022p/singla/MIMICCX-Chest-Explainer/stylegan2-pytorch/Experiment_MIMIC_CXR/Run1/', help='')
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument('--arch', type=str, default='swagan', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default='/ocean/projects/asc170022p/singla/MIMICCX-Chest-Explainer/stylegan2-pytorch/Experiment_MIMIC_CXR/Run1/checkpoint/030000.pt',
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8
    if args.arch == 'stylegan2':
        from model import Generator

    elif args.arch == 'swagan':
        from swagan import Generator
        
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    #g_ema.load_state_dict(checkpoint["g_ema"])
    g_ema.load_state_dict(checkpoint["g_ema"], strict=False)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
