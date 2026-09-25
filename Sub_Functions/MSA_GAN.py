import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
# from torchsummary import summary
from torchvision.utils import save_image
import torchvision
from torchvision.utils import make_grid
from torch.autograd import Variable
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from termcolor import cprint
from Sub_Functions.Attention import Network_3d

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AnimeDataset(Dataset):
    def __init__(self, img, device = "cpu"):
        self.device = device
        self.img = img

    def __len__(self):
        return len(self.img)

    def __getitem__(self, id):
        img = self.img[id]
        img = cv2.resize(img ,(64 ,64))
        img = np.moveaxis(img, 2, 0)
        img = torch.tensor(img, dtype = torch.float).to(self.device)

        return (img -127.5 ) /127.5


import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Utility: Sinusoidal timestep embeddings (for diffusion)
# ------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: (B,) integer or float
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb  # (B, dim)


# ------------------------------
# Simple UNet block for diffusion denoising
# ------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_ch, out_ch)
        )
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.nin_shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.nin_shortcut(x)


class AttnBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj_out = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        b, c, hgt, wid = q.shape
        q = q.reshape(b, c, hgt*wid).transpose(1, 2)  # (B, HW, C)
        k = k.reshape(b, c, hgt*wid)                  # (B, C, HW)
        attn = torch.bmm(q, k) / math.sqrt(c)
        attn = F.softmax(attn, dim=-1)
        v = v.reshape(b, c, hgt*wid).transpose(1, 2)  # (B, HW, C)
        h = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, hgt, wid)
        h = self.proj_out(h)
        return x + h


class SimpleUNet(nn.Module):
    """
    Lightweight UNet used as the denoiser eps_theta(x_t, t, context).
    Optionally takes a context tensor (from the temporal memory module) that is
    broadcast and concatenated across spatial dims.
    """
    def __init__(self, in_ch=64, base_ch=64, time_ch=128, context_ch: int = 0):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_ch),
            nn.Linear(time_ch, time_ch*4),
            nn.SiLU(),
            nn.Linear(time_ch*4, time_ch)
        )
        self.context_ch = context_ch
        cin = in_ch + (context_ch if context_ch > 0 else 0)
        self.in_conv = nn.Conv2d(cin, base_ch, 3, padding=1)

        self.down1 = ResBlock(base_ch, base_ch, time_ch)
        self.down2 = ResBlock(base_ch, base_ch*2, time_ch)
        self.pool1 = nn.Conv2d(base_ch*2, base_ch*2, 3, stride=2, padding=1)

        self.mid1 = ResBlock(base_ch*2, base_ch*2, time_ch)
        self.mid_attn = AttnBlock(base_ch*2)
        self.mid2 = ResBlock(base_ch*2, base_ch*2, time_ch)

        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 4, stride=2, padding=1)
        self.up_block = ResBlock(base_ch, base_ch, time_ch)
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t: torch.Tensor, context: Optional[torch.Tensor] = None):
        # x: (B, C, H, W)
        # t: (B,) timesteps
        if context is not None and self.context_ch > 0:
            # context: (B, Cc)
            c = context.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(-2), x.size(-1))
            x = torch.cat([x, c], dim=1)
        t_emb = self.time_mlp(t)

        h = self.in_conv(x)
        h = self.down1(h, t_emb)
        h = self.down2(h, t_emb)
        h = self.pool1(h)
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)
        h = self.up1(h)
        h = self.up_block(h, t_emb)
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


# ------------------------------
# Memory-Guided Temporal Module (MGTM)
# ------------------------------
class MemoryBank(nn.Module):
    """
    Key-Value memory updated over time with simple decay and capacity control.
    Stores a small set of global frame descriptors to guide the denoiser.
    """
    def __init__(self, feat_ch: int, key_ch: int = 128, val_ch: int = 128, capacity: int = 8, decay: float = 0.95):
        super().__init__()
        self.key_proj = nn.Conv2d(feat_ch, key_ch, 1)
        self.val_proj = nn.Conv2d(feat_ch, val_ch, 1)
        self.capacity = capacity
        self.decay = decay
        self.key_ch = key_ch
        self.val_ch = val_ch
        self.register_buffer('keys', torch.zeros(1, 0, key_ch))   # (B, N, Ck)
        self.register_buffer('vals', torch.zeros(1, 0, val_ch))   # (B, N, Cv)

    def reset(self, batch_size: int, device):
        self.keys = torch.zeros(batch_size, 0, self.key_ch, device=device)
        self.vals = torch.zeros(batch_size, 0, self.val_ch, device=device)

    def summarize(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: (B, C, H, W) -> global descriptors
        k = self.key_proj(feat).mean(dim=[2,3])  # (B, Ck)
        v = self.val_proj(feat).mean(dim=[2,3])  # (B, Cv)
        return k, v

    def attend(self, q: torch.Tensor) -> torch.Tensor:
        # q: (B, Ck)
        if self.keys.size(1) == 0:
            return torch.zeros(q.size(0), self.val_ch, device=q.device)
        # cosine attention
        qn = F.normalize(q, dim=-1)
        kn = F.normalize(self.keys, dim=-1)
        attn = (qn.unsqueeze(1) * kn).sum(dim=-1)  # (B, N)
        attn = F.softmax(attn, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), self.vals).squeeze(1)  # (B, Cv)
        return ctx

    def update(self, k: torch.Tensor, v: torch.Tensor):
        # decay old memory
        if self.keys.size(1) > 0:
            self.vals = self.decay * self.vals
        # append and crop to capacity
        self.keys = torch.cat([self.keys, k.unsqueeze(1)], dim=1)
        # self.keys = self.keys.view(0, -1)
        self.vals = torch.cat([self.vals, v.unsqueeze(1)], dim=1)
        if self.keys.size(1) > self.capacity:
            self.keys = self.keys[:, -self.capacity:, :]
            self.vals = self.vals[:, -self.capacity:, :]


class MemoryGuidedTemporalModule(nn.Module):
    def __init__(self, feat_ch: int, key_ch: int = 128, val_ch: int = 128, capacity: int = 8, decay: float = 0.95):
        super().__init__()
        self.bank = MemoryBank(feat_ch, key_ch, val_ch, capacity, decay)
        self.query_proj = nn.Conv2d(feat_ch, key_ch, 1)
        self.context_proj = nn.Linear(val_ch, val_ch)
        self.val_ch = val_ch

    def reset(self, batch_size: int, device):
        self.bank.reset(batch_size, device)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, C, H, W)
        q = self.query_proj(feat).mean(dim=[2,3])                # (B, Ck)
        k, v = self.bank.summarize(feat)                         # (B, Ck), (B, Cv)
        ctx = self.bank.attend(q)                                # (B, Cv)
        # update memory AFTER reading (read-before-write policy)
        self.bank.update(k, v)
        return self.context_proj(ctx)                            # (B, Cv)


# ------------------------------
# Diffusion wrapper
# ------------------------------
class DiffusionDenoiser(nn.Module):
    """
    Lightweight DDPM-style denoiser that runs a small number of reverse steps on
    a feature map. Designed to be called inside another network's forward.
    """
    def __init__(self, feat_ch: int, time_ch: int = 128, context_ch: int = 128, steps: int = 8):
        super().__init__()
        self.steps = steps
        # cosine-like small beta schedule
        betas = torch.linspace(1e-4, 0.02, steps)
        alphas = 1.0 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cum)
        self.unet = SimpleUNet(in_ch=feat_ch, base_ch=min(128, feat_ch), time_ch=time_ch, context_ch=context_ch)

    def p_sample(self, x, t_idx: int, context: Optional[torch.Tensor]):
        b = x.size(0)
        t = torch.full((b,), t_idx, device=x.device, dtype=torch.long)
        eps_theta = self.unet(x, t, context=context)
        beta_t = self.betas[t_idx]
        alpha_t = self.alphas[t_idx]
        alpha_bar_t = self.alphas_cumprod[t_idx]
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_theta)
        if t_idx > 0:
            noise = torch.randn_like(x)
            return mean + torch.sqrt(beta_t) * noise
        else:
            return mean

    def forward(self, feats: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        # Initialize x_T as noisy features
        x = feats
        # Start from moderately noisy state to save compute
        noise = torch.randn_like(x)
        t_start = self.steps - 1
        alpha_bar = self.alphas_cumprod[t_start]
        x = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise
        for t in reversed(range(self.steps)):
            x = self.p_sample(x, t, context)
        return x


# ------------------------------
# Placeholder attention (replace with your own if available)
# ------------------------------
# class Network_3d(nn.Module):
#     def __init__(self, batch_size):
#         super().__init__()
#     def forward(self, x):
#         return x


# ------------------------------
# Updated Generator with Diffusion + MGTM
# ------------------------------
class Generator(nn.Module):
    def __init__(self ,opt, device = "cpu"):
        super(Generator, self).__init__()
        self.upsample0 = nn.ConvTranspose2d(128, 512, kernel_size=4, stride=1, padding=0, bias=False)
        self.upsample1 = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
        self.upsample4 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
        self.conv0 = nn.Sequential(nn.Conv2d(512, 384, kernel_size=3, stride=1, padding=1 ,bias=False),
                                   nn.BatchNorm2d(384),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1 ,bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(256, 192, kernel_size=3, stride=1, padding=1 ,bias=False),
                                   nn.BatchNorm2d(192),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1 ,bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1 ,bias=False),
                                   nn.BatchNorm2d(96),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1 ,bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1 ,bias=False),
                                   nn.BatchNorm2d(48),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(48, 32, kernel_size=3, stride=1, padding=1 ,bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(16),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1 ,bias=False))
        self.sigmoid = nn.Tanh()
        self.device = device
        self.opt = opt

        # --- Memory-Guided Temporal Module + Diffusion ---
        # We place the denoiser at the 64-channel feature scale (after conv2)
        self.mgtm = MemoryGuidedTemporalModule(feat_ch=64, key_ch=128, val_ch=128, capacity=8, decay=0.95)
        self.diffusion = DiffusionDenoiser(feat_ch=64, time_ch=128, context_ch=128, steps=8)

        self.to(self.device)

    def _upsample(self, layer, x):
        return layer(x, output_size=(x.size(0), x.size(1), x.size(2)*2, x.size(3)*2))

    def forward_single(self, X):
        # X: (B, 128, 1, 1) or latent features per frame
        X = self.upsample0(X)
        X = self.conv0(X)
        X = self.upsample1(X, output_size=(X.size(0), X.size(1), X.size(2)*2, X.size(3)*2))
        X = self.conv1(X)
        if self.opt == 1:
            att = Network_3d(X.shape[0])
            X = att(X)
        X = self.upsample2(X, output_size=(X.size(0), X.size(1), X.size(2)*2, X.size(3)*2))
        X = self.conv2(X)  # -> (B, 64, H, W)

        # Memory-guided diffusion refinement at this scale
        context = self.mgtm(X)  # (B, 128)
        X = self.diffusion(X, context)

        X = self.upsample3(X, output_size=(X.size(0), X.size(1), X.size(2)*2, X.size(3)*2))
        X = self.conv3(X)
        X = self.upsample4(X, output_size=(X.size(0), X.size(1), X.size(2)*2, X.size(3)*2))
        X = self.conv4(X)
        return self.sigmoid(X)

    def forward(self, X):
        """
        Accepts either 4D or 5D input:
          - 4D: (B, C, H, W) -> processes a single frame per batch (memory resets)
          - 5D: (B, T, C, H, W) -> iterates over time, updating memory each step
        """
        if X.dim() == 4:
            # reset memory for new sequence
            self.mgtm.reset(batch_size=X.size(0), device=X.device)
            return self.forward_single(X)
        elif X.dim() == 5:
            B, T, C, H, W = X.shape
            outs = []
            self.mgtm.reset(batch_size=B, device=X.device)
            for t in range(T):
                out_t = self.forward_single(X[:, t])
                outs.append(out_t)
            return torch.stack(outs, dim=1)  # (B, T, 3, H_out, W_out)
        else:
            raise ValueError(f"Unexpected input shape: {X.shape}")



class Discriminator(nn.Module):
    def __init__(self,device="cpu"):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(64),nn.LeakyReLU(0.2, inplace=True),
                            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(128),nn.LeakyReLU(0.2, inplace=True),
                            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(256),nn.LeakyReLU(0.2, inplace=True),
                            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(512),nn.LeakyReLU(0.2, inplace=True),
                            nn.Flatten(),
                            nn.Linear(8192,1),
                            nn.Sigmoid())
        self.device = device
        self.to(self.device)

    def forward(self,X):
        return self.discriminator(X)


class Trainer:
    def __init__(self, generator, discriminator, optimizer_generator, optimizer_discriminator, latent_size,
                 device=DEVICE, load_pretrained=True):
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_g = optimizer_generator
        self.optimizer_d = optimizer_discriminator
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_g, 4, gamma=0.9, verbose=True)
        self.latent_size = latent_size
        self.device = device
        if load_pretrained:
            self.load()

    def fit(self, train_loader, epoch_num, save=True):
        train_loader = tqdm(train_loader)
        train_loader = enumerate(train_loader)
        self.generator.train()
        self.discriminator.train()
        for i, real_images in train_loader:
            loss_d, real_score, fake_score = self.train_discriminator(real_images)
            loss_g = self.train_generator(real_images.size(0))

        cprint("Epoch-[{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(epoch_num,
                                                                                                          loss_g,
                                                                                                          loss_d,
                                                                                                          real_score,
                                                                                                          fake_score),
               color='grey',
               on_color='on_cyan')
        self.scheduler.step()
        if save:
            self.save(generator=False)
            self.save()

    def save(self, generator=True):
        if generator:
            torch.save(self.generator.state_dict(), "./generator.pth")
            print("Generator saved!!")
        else:
            torch.save(self.discriminator.state_dict(), "./discriminator.pth")
            print("Discriminator saved!!")

    def load(self):
        print("loading weights...")
        self.discriminator.load_state_dict(torch.load("../input/pretrained-gan/discriminator.pth"))
        self.generator.load_state_dict(torch.load("../input/pretrained-gan/generator.pth"))
        print("weights loaded..")

    def train_discriminator(self, real_images):
        self.optimizer_d.zero_grad()
        real_preds = self.discriminator(real_images)
        real_targets = torch.ones(real_images.size(0), 1, device=self.device)
        real_loss = F.binary_cross_entropy(real_preds, real_targets)
        real_score = torch.mean(real_preds).item()

        latent = torch.randn(real_images.size(0), self.latent_size, 1, 1, device=self.device)
        fake_images = self.generator(latent)
        fake_targets = torch.zeros(fake_images.size(0), 1, device=self.device)
        fake_preds = self.discriminator(fake_images)
        fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
        fake_score = torch.mean(fake_preds).item()

        loss = real_loss + fake_loss
        loss.backward()
        self.optimizer_d.step()
        return loss.item(), real_score, fake_score

    def train_generator(self, batch_size):
        self.optimizer_g.zero_grad()

        latent = torch.randn(batch_size, self.latent_size, 1, 1, device=self.device)
        fake_images = self.generator(latent)

        preds = self.discriminator(fake_images)
        targets = torch.ones(batch_size, 1, device=self.device)
        loss1 = F.binary_cross_entropy(preds, targets)
        loss2 = F.binary_cross_entropy_with_logits(preds, targets)
        loss = (loss1 + loss2) /2

        loss.backward()
        self.optimizer_g.step()

        return loss.item()

    def save_samples(self, epoch_num, latent_tensors, show=True):
        self.generator.eval()
        fake_images = self.generator(latent_tensors)
        fake_fname = './generated-images-{0:0=4d}.png'.format(epoch_num)
        save_image(fake_images, fake_fname, nrow=8)
        print('Saving', fake_fname)
        self.generator.train()
        if show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([]);
            ax.set_yticks([])
            img = make_grid(self.denorm(fake_images.cpu().detach()), nrow=8).permute(1, 2, 0).numpy()
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()

    def denorm(self, img):
        return img * 0.5 + 0.5

    def Anime_generation(self, test_loader):
        test_loader = tqdm(test_loader)
        test_loader = enumerate(test_loader)
        self.generator.eval()
        out_put = []

        for i, real_images in test_loader:
            data = {}
            org_img = real_images.permute(0, 2, 3, 1)

            Generated_images = []
            for i in range(10):

                latent_tensors = torch.randn((1, 128, 1, 1), device=DEVICE)

                # Generate an image using the trained augmentation model
                generated_image = self.generator(latent_tensors)

                gen_image = generated_image.permute(0, 2, 3, 1)

                gen_image = gen_image.detach().numpy()[0]

                #
                # # Normalize the generated image
                # generated_image = 0.5 * generated_image + 0.5
                # generated_image = generated_image[0, :, :, :]
                #
                # # Save the generated image as a JPEG file
                # generated_image = np.asarray(generated_image)
                #
                # im = Image.fromarray((generated_image * 255).astype(np.uint8))
                Generated_images.append(gen_image)

            data['original_image'] = org_img.numpy()[0]
            data['generated_image'] = Generated_images
            out_put.append(data)

        return out_put
