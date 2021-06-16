import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix
from model import ModulatedConv2d, StyledConv, ConstantInput, PixelNorm, Upsample, Downsample, Blur, EqualLinear, ConvLayer, EqualConv2d

num_channel = 1
def get_haar_wavelet(in_channels):
    haar_wav_l = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]

    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h
    
    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh


class HaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        ll, lh, hl, hh = get_haar_wavelet(in_channels)
    
        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)
        
    def forward(self, input):
        ll = upfirdn2d(input, self.ll, down=2)
        lh = upfirdn2d(input, self.lh, down=2)
        hl = upfirdn2d(input, self.hl, down=2)
        hh = upfirdn2d(input, self.hh, down=2)
        
        return torch.cat((ll, lh, hl, hh), 1)
    
class InverseHaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', -lh)
        self.register_buffer('hl', -hl)
        self.register_buffer('hh', hh)
        
    def forward(self, input):
        ll, lh, hl, hh = input.chunk(4, 1)
        ll = upfirdn2d(ll, self.ll, up=2, pad=(1, 0, 1, 0))
        lh = upfirdn2d(lh, self.lh, up=2, pad=(1, 0, 1, 0))
        hl = upfirdn2d(hl, self.hl, up=2, pad=(1, 0, 1, 0))
        hh = upfirdn2d(hh, self.hh, up=2, pad=(1, 0, 1, 0))
        
        return ll + lh + hl + hh


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.iwt = InverseHaarTransform(3)
            self.upsample = Upsample(blur_kernel)
            self.dwt = HaarTransform(3)

        self.conv = ModulatedConv2d(in_channel, 3 * 4, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3 * 4, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.iwt(skip)
            skip = self.upsample(skip)
            skip = self.dwt(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        num_channel = 1
    ):
        super().__init__()

        self.size = size  #256

        self.style_dim = style_dim  #512

        layers = [PixelNorm()]  #MLP Z ---> W

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier, #512
            128: 128 * channel_multiplier, #256
            256: 64 * channel_multiplier, #128
            512: 32 * channel_multiplier, #64
            1024: 16 * channel_multiplier, #32
        }
        
        self.input = ConstantInput(self.channels[4])  #Input [n , 512]
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel  #conv: 512 x 512 x 3
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2)) - 1   #7
        self.num_layers = (self.log_size - 2) * 2 + 1  #11

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]           
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))
            '''
            layer idx:  0 [1, 1, 4, 4]
            layer idx:  1 [1, 1, 8, 8]
            layer idx:  2 [1, 1, 8, 8]
            layer idx:  3 [1, 1, 16, 16]
            layer idx:  4 [1, 1, 16, 16]
            layer idx:  5 [1, 1, 32, 32]
            layer idx:  6 [1, 1, 32, 32]
            layer idx:  7 [1, 1, 64, 64]
            layer idx:  8 [1, 1, 64, 64]
            layer idx:  9 [1, 1, 128, 128]
            layer idx:  10 [1, 1, 128, 128]
            '''

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )
            
            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )
            ''' 
            i: 3 512, 512, 3, upsample=True ; 512, 512, 3, upsample=False
            i: 4 512, 512, 3, upsample=True ; 512, 512, 3, upsample=False
            i: 5 512, 512, 3, upsample=True ; 512, 512, 3, upsample=False
            i: 6 512, 512, 3, upsample=True ; 512, 512, 3, upsample=False
            i: 7 512, 256, 3, upsample=True ; 256, 256, 3, upsample=False
            '''
            self.to_rgbs.append(ToRGB(out_channel, style_dim)) #512, 12, 1, upsample=False

            in_channel = out_channel

        self.iwt = InverseHaarTransform(3)

        self.n_latent = self.log_size * 2 - 2  #12

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        do_something = False,
    ):
        
        #styles: batch_size x 512
        try:
            print("len(styles): ", len(styles))
        except:pass
        try:
            print("styles.shape: ", styles.shape)
        except:pass
        
        if not input_is_latent: 
            styles = [self.style(s) for s in styles]
        #print("len(styles): ", len(styles))
        
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers #11 layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]
        if do_something:
            if truncation < 1:
                style_t = []

                for style in styles:
                    style_t.append(
                        truncation_latent + truncation * (style - truncation_latent)
                    )

                styles = style_t
                #print("len(styles) in trucntion: ", len(styles))

            if len(styles) < 2:
                inject_index = self.n_latent

                if styles[0].ndim < 3:
                    latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

                else:
                    latent = styles[0] #torch.Size([batch_size, 12, 512])

            else:
                if inject_index is None:
                    inject_index = random.randint(1, self.n_latent - 1)

                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

                latent = torch.cat([latent, latent2], 1)  #batch_size x 12 x 512: batch_size x 512: styles; 12 layers ; 16: batch size
        else:
            latent = styles
        #print("first latent: ", latent.shape)
        out = self.input(latent)
        #print("first out: ", out.shape)
        out = self.conv1(out, latent[:, 0], noise=noise[0])  #batch_size x 512 x 4 x 4
        #print("out after conv: ", out.shape)
        skip = self.to_rgb1(out, latent[:, 1])  #batch_size x 12 x 4 x 4
        #print("first rgb: ", skip.shape)
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1) # [upsample] 
            #print("out: ", out.shape, i)
            out = conv2(out, latent[:, i + 1], noise=noise2) 
            #print("out: ",out.shape, i)
            skip = to_rgb(out, latent[:, i + 2], skip) 
            #print("skip RGB: ",skip.shape, i)
            i += 2
            '''
            out:  torch.Size([16, 512, 8, 8]) 1
            out:  torch.Size([16, 512, 8, 8]) 1
            skip RGB:  torch.Size([16, 12, 8, 8]) 1
            out:  torch.Size([16, 512, 16, 16]) 3
            out:  torch.Size([16, 512, 16, 16]) 3
            skip:  torch.Size([16, 12, 16, 16]) 3
            out:  torch.Size([16, 512, 32, 32]) 5
            out:  torch.Size([16, 512, 32, 32]) 5
            skip:  torch.Size([16, 12, 32, 32]) 5
            out:  torch.Size([16, 512, 64, 64]) 7
            out:  torch.Size([16, 512, 64, 64]) 7
            skip:  torch.Size([16, 12, 64, 64]) 7
            out:  torch.Size([16, 256, 128, 128]) 9
            out:  torch.Size([16, 256, 128, 128]) 9
            skip:  torch.Size([16, 12, 128, 128]) 9
            '''

        image = self.iwt(skip) # torch.Size([16, 3, 256, 256])
        #print(image.shape)
        if return_latents:
            return image, latent # latent: torch.Size([16, 12, 512])

        else:
            return image, None


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        #print("Input: ", input.shape)
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        #print("out resblock: ", out.shape)
        return out
    
    
class FromRGB(nn.Module):
    def __init__(self, out_channel, downsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.downsample = downsample

        if downsample:
            self.iwt = InverseHaarTransform(3)
            self.downsample = Downsample(blur_kernel)
            self.dwt = HaarTransform(3)

        self.conv = ConvLayer(3 * 4, out_channel, 1)

    def forward(self, input, skip=None):
        if self.downsample:
            input = self.iwt(input)
            input = self.downsample(input)
            input = self.dwt(input)

        out = self.conv(input)

        if skip is not None:
            out = out + skip

        return input, out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        # size: 256
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier, #512
            128: 128 * channel_multiplier, #256
            256: 64 * channel_multiplier, #128
            512: 32 * channel_multiplier, #64
            1024: 16 * channel_multiplier, #32
        }
        self.dwt = HaarTransform(3)
        self.from_rgbs = nn.ModuleList()
        self.convs = nn.ModuleList()
        log_size = int(math.log(size, 2)) - 1  #7
        in_channel = channels[size]  #512

        for i in range(log_size, 2, -1): #7,6,5,4,3
            out_channel = channels[2 ** (i - 1)]
            self.from_rgbs.append(FromRGB(in_channel, downsample=i != log_size))
            self.convs.append(ConvBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        self.from_rgbs.append(FromRGB(channels[4]))

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)  #513 512 3
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),  #8192 , 512
            EqualLinear(channels[4], 1),  # 512, 1
        )
        

    def forward(self, input):
        input = self.dwt(input)
        out = None

        for from_rgb, conv in zip(self.from_rgbs, self.convs):
            input, out = from_rgb(input, out)
            out = conv(out)

        _, out = self.from_rgbs[-1](input, out)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        out = self.final_conv(out)
        out = out.view(batch, -1)
        out = self.final_linear(out)
        return out


class Encoder(nn.Module):
    def __init__(self, size, w_dim=512):
        super().__init__()
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
            1024: 16
        }        
        self.w_dim = w_dim  #512
        log_size = int(math.log(size, 2))  #8
        self.n_latents = log_size*2 - 4  #14
        convs = [ConvLayer(3, channels[size], 1)]
        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel
        convs.append(EqualConv2d(in_channel, self.n_latents*self.w_dim, 4, padding=0, bias=False))    
        self.convs = nn.Sequential(*convs)

    def forward(self, input):
        out = self.convs(input)
        return out.view(len(input), self.n_latents, self.w_dim)
    
    
    
    