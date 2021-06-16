import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import wandb
from dataset import MultiResolutionDataset
from dataset import XRayDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
import pdb
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
    print("No SummaryWriter")
    
    

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(args, loader, encoder, generator, discriminator, e_optim, d_optim, device, save_dir):
    loader = sample_data(loader)
    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    e_loss_val = 0
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        e_module = encoder.module
        g_module = generator.module
        d_module = discriminator.module

    else:
        e_module = encoder
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)
    
    #sample_z = torch.randn(args.n_sample, args.latent, device=device) #no need of noise sample
    
    requires_grad(generator, False) #no need of gradients to pass through Generator
    truncation = 0.7
    trunc = generator.mean_latent(4096).detach()
    trunc.requires_grad = False
    
    if SummaryWriter:
        logger = SummaryWriter(save_dir+'checkpoint') 
        
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img = next(loader)
        real_img = real_img.to(device)
        
        requires_grad(encoder, False)
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        
        latents = encoder(real_img)
        
        recon_img, _ = generator([latents],
                                 input_is_latent=True,
                                 truncation=truncation,
                                 truncation_latent=trunc,
                                 randomize_noise=False)
        
        recon_pred = discriminator(recon_img)
        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, recon_pred)
        
        
        #fake_pred = discriminator(fake_img)
        #real_pred = discriminator(real_img_aug)
        #d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = recon_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()


        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss
        
        # E update
        requires_grad(encoder, True)
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        
        real_img = real_img.detach()
        real_img.requires_grad = False
        
        latents = encoder(real_img)
        recon_img, _ = generator([latents], 
                                 input_is_latent=True,
                                 truncation=truncation,
                                 truncation_latent=trunc,
                                 randomize_noise=False)
        
        recon_l2_loss = F.mse_loss(recon_img, real_img)
        loss_dict["l2"] = recon_l2_loss * args.l2
        
        recon_pred = discriminator(recon_img)
        adv_loss = g_nonsaturating_loss(recon_pred) * args.adv
        loss_dict["adv"] = adv_loss
        
        e_loss = recon_l2_loss + adv_loss 
        loss_dict["e_loss"] = e_loss

        encoder.zero_grad()
        e_loss.backward()
        e_optim.step()

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        l2_loss_val = loss_reduced["l2"].mean().item()
        adv_loss_val = loss_reduced["adv"].mean().item()
        e_loss_val = loss_reduced["e_loss"].mean().item()
        
        if SummaryWriter :
            logger.add_scalar('E_loss/total', e_loss_val, i)
            logger.add_scalar('E_loss/l2', l2_loss_val, i)
            logger.add_scalar('E_loss/adv', adv_loss_val, i)
            logger.add_scalar('D_loss/adv', d_loss_val, i)
            logger.add_scalar('D_loss/r1', r1_val, i)    
            logger.add_scalar('D_loss/real_score', real_score_val, i)    
            logger.add_scalar('D_loss/fake_score', fake_score_val, i)    
            
        if get_rank() == 0:
            pbar.set_description(
                (
                    f"e: {e_loss_val:.4f}; \
                    real_score:{real_score_val:.4f} ;\
                    l2: {l2_loss_val:.4f};\
                    fake_score: {fake_score_val:.4f};\
                    adv: {adv_loss_val:.4f};\
                    d: {d_loss_val:.4f};\
                    r1: {r1_val:.4f};\
                    "
                )
            )
        
            if wandb and args.wandb:
                wandb.log(
                    {
                        "Encoder": e_loss_val,
                        "Discriminator": d_loss_val,
                        "R1": r1_val,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "l2": l2_loss_val,
                        "adv":adv_loss_val,
                    }
                )

            if i % 100 == 0:
                with torch.no_grad():
                    sample = torch.cat([real_img.detach(), recon_img.detach()])
                    utils.save_image(
                        sample,
                        save_dir+f"sample/{str(i).zfill(6)}.png",
                        nrow=int(args.batch),
                        normalize=True,
                        range=(0, 1),
                    )
                

            if i % 1000 == 0:
                torch.save(
                    {
                        "e": encoder.state_dict(),
                        "d": discriminator.state_dict(),
                        "e_optim": e_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                    },
                    save_dir+f"checkpoint/encoder_{str(i).zfill(6)}.pt",
                )
                

if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("path", default = '/ocean/projects/asc170022p/singla/Datasets/MIMIC-CXR/PA_AP_views_image_report.csv',type=str, help="path to the lmdb dataset")
    parser.add_argument('--save_dir', type=str, default='/ocean/projects/asc170022p/singla/MIMICCX-Chest-Explainer/stylegan2-pytorch/Experiment_MIMIC_CXR/Run_encoder/', help='')
    parser.add_argument('--arch', type=str, default='swagan', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--g_ckpt",
        type=str,
        default='/ocean/projects/asc170022p/singla/MIMICCX-Chest-Explainer/stylegan2-pytorch/Experiment_MIMIC_CXR/Run1/checkpoint/030000.pt',
        help="path to the checkpoints to resume training",
    )
    parser.add_argument(
        "--e_ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.3,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--adv", type=float, default=0.05) 

    args = parser.parse_args()
    n_gpu = torch.cuda.device_count()
    #n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    print("n_gpu: ",n_gpu)
    args.distributed = n_gpu > 1
    print("args.distributed: ",args.distributed)
    if args.distributed:
        for i in range(n_gpu):
            print(torch.cuda.get_device_name(device=i))
            print(torch.cuda.get_device_properties(i))
        print("args.local_rank: ",args.local_rank)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8  #Number for MLP for Z --> W

    args.start_iter = 0

    if args.arch == 'stylegan2':
        from model import Generator, Discriminator

    elif args.arch == 'swagan':
        from swagan_encoder import Generator, Discriminator, Encoder
    
    encoder = Encoder(args.size, args.latent).to(device)
    
    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    
    
    #g_ema = Generator(
    #    args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    #).to(device)
    #g_ema.eval()
    #accumulate(g_ema, generator, 0)

    #g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    e_optim = optim.Adam(
        encoder.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
    )
    
    #g_optim = optim.Adam(
    #    generator.parameters(),
    #    lr=args.lr * g_reg_ratio,
    #    betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    #)
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.g_ckpt is not None:
        print("load generator model:", args.g_ckpt)

        g_ckpt = torch.load(args.g_ckpt, map_location=lambda storage, loc: storage)
        g_args = g_ckpt['args']
        args.size = g_args.size
        args.latent = g_args.latent
        args.n_mlp = g_args.n_mlp
        args.channel_multiplier = g_args.channel_multiplier
        
        generator.load_state_dict(g_ckpt["g_ema"])
        discriminator.load_state_dict(g_ckpt["d"])

       
        d_optim.load_state_dict(g_ckpt["d_optim"])
    else:
        print("Error!!! Need generator checkpoint")
    
    if args.e_ckpt is not None:
        print("resume training:", args.e_ckpt)
        e_ckpt = torch.load(args.e_ckpt, map_location=lambda storage, loc: storage)

        encoder.load_state_dict(e_ckpt["e"])
        e_optim.load_state_dict(e_ckpt["e_optim"])
        discriminator.load_state_dict(e_ckpt["d"])
        d_optim.load_state_dict(e_ckpt["d_optim"])
        
        try:
            ckpt_name = os.path.basename(args.e_ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name.split('_')[-1])[0])
        except ValueError:
            pass 
        
        
    if args.distributed:
        encoder = nn.parallel.DistributedDataParallel(
            encoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    #dataset = MultiResolutionDataset(args.path, transform, args.size)
    #for chest xray
    dataset = XRayDataset(args.path, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    train(args, loader, encoder, generator, discriminator, e_optim, d_optim, device, save_dir = args.save_dir)
