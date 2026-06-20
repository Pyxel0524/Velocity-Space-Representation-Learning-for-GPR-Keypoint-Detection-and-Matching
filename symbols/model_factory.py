
import torch
import torch.nn as nn
from .unet_parts import *
# from vit_pytorch import SimpleViT
import matplotlib.pyplot as plt
from einops import rearrange
import math
from typing import Optional


# class InverseGain(nn.Module):
#     """
#     Param-free attention-like amplitude calibration (depth-wise gain).
#     输入:  img [H, W] (torch.float32/float64 都可)
#     输出:  img_corr [H, W]
#     最近一次的增益曲线保存在 self.last_gain [H]
#     """
#     def __init__(self,
#                  gamma: float = 3,
#                  gmin: float = 0.3,
#                  gmax: float = 5.0,
#                  tau_factor: float = 2.0,
#                  mean_center: bool = True,
#                  k_smooth: int = 5,
#                  eps: float = 1e-6):
#         super().__init__()
#         self.gamma = gamma
#         self.gmin = gmin
#         self.gmax = gmax
#         self.tau_factor = tau_factor
#         self.mean_center = mean_center
#         self.k_smooth = k_smooth
#         self.eps = eps
#         self.last_gain = None  # [B,H]
#
#     @staticmethod
#     def _depth_energy(x: torch.Tensor) -> torch.Tensor:
#         # x: [B,C,H,W] -> u: [B,H] = mean_{c,w} |x|
#         return x.abs().mean(dim=(1, 3))
#
#     def _smooth_gain(self, G: torch.Tensor) -> torch.Tensor:
#         # G: [B,H] -> 平滑后仍为 [B,H]
#         if self.k_smooth is None or self.k_smooth <= 1:
#             return G
#         k = self.k_smooth if self.k_smooth % 2 == 1 else self.k_smooth + 1  # 保证奇数
#         pad = (k - 1) // 2
#         kernel = torch.ones(1, 1, k, dtype=G.dtype, device=G.device) / k
#         Gb = G.unsqueeze(1)                    # [B,1,H]
#         Gs = F.conv1d(F.pad(Gb, (pad, pad), mode='replicate'), kernel)
#         return Gs.squeeze(1)                   # [B,H]
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         assert x.dim() == 4, f"Expected [B,C,H,W], got {tuple(x.shape)}"
#         B, C, H, W = x.shape
#
#         # 1) 深度能量 u[b,h]
#         u = self._depth_energy(x)                            # [B,H]
#
#         # 2) 注意力：A[b,h,j] ~ exp(-(u_h-u_j)^2 / (2 τ^2))
#         tau = self.tau_factor * (u.std(dim=1, keepdim=True) + self.eps)  # [B,1]
#         tau2 = (tau ** 2).unsqueeze(-1)                      # [B,1,1]  <<< 修正广播
#         diff2 = (u.unsqueeze(-1) - u.unsqueeze(-2)) ** 2     # [B,H,H]
#         A = torch.softmax(-diff2 / (2.0 * tau2), dim=-1)     # [B,H,H]
#
#         # 3) 上下文参考能量与目标均值
#         u_ctx = torch.matmul(A, u.unsqueeze(-1)).squeeze(-1) # [B,H]
#         target = u_ctx.mean(dim=1, keepdim=True)             # [B,1]
#
#         # 4) 逆增益（往均值拉平）+ 平滑 + 护栏 + 均值归一
#         G = (target / (u_ctx + self.eps)) ** self.gamma      # [B,H]
#         G = self._smooth_gain(G)
#         G = torch.clamp(G, min=self.gmin, max=self.gmax)
#         if self.mean_center:
#             G = G / (G.mean(dim=1, keepdim=True) + self.eps) # 每样本 mean≈1
#
#         self.last_gain = G.detach()
#
#         # 5) 应用增益（逐深度广播）
#         y = x * G.view(B, 1, H, 1)
#         y_min = y.amin(dim=(1, 2, 3), keepdim=True)  # 每张图最小值
#         y_max = y.amax(dim=(1, 2, 3), keepdim=True)  # 每张图最大值
#         y = 2.0 * (y - y_min) / (y_max - y_min + self.eps) - 1.0
#
#         # plt.figure()
#         # plt.imshow(x[0,0].detach().cpu().numpy())
#         # plt.figure()
#         # plt.imshow(y[0,0].detach().cpu().numpy())
#         # plt.figure()
#         # plt.plot(G[0].detach().cpu().numpy())
#         # plt.show()
#         return y


class InverseGain(nn.Module):
    """
    极简版：无论输入如何，都拉成“浅强、深弱”的分布。
    仅 3 个核心参数：gamma(拉平强度), tilt(浅强力度), shape_p(形状)
    """
    def __init__(self,
                 gamma: float = 0.7,   # 轻度拉平：深弱补、浅强收
                 tilt: float  = 1.0,   # 浅强深弱的整体力度
                 shape_p: float = 1.0, # 模板形状 (1-z)^p, p↑ → 深部更弱
                 gmin: float = 0.1, gmax: float = 3.0,
                 k_smooth: int = 9, mean_center: bool = True, eps: float = 1e-6):
        super().__init__()
        self.gamma = gamma
        self.tilt = tilt
        self.shape_p = shape_p
        self.gmin, self.gmax = gmin, gmax
        self.k_smooth = k_smooth
        self.mean_center = mean_center
        self.eps = eps
        self.last_gain = None

    @staticmethod
    def _depth_energy(x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] -> [B,H]
        return x.abs().mean(dim=(1, 3))

    def _smooth1d(self, G: torch.Tensor) -> torch.Tensor:
        if self.k_smooth is None or self.k_smooth <= 1:
            return G
        k = self.k_smooth if self.k_smooth % 2 == 1 else self.k_smooth + 1
        pad = (k - 1) // 2
        kernel = torch.ones(1, 1, k, dtype=G.dtype, device=G.device) / k
        Gb = G.unsqueeze(1)                                  # [B,1,H]
        Gs = F.conv1d(F.pad(Gb, (pad, pad), mode='replicate'), kernel)
        return Gs.squeeze(1)                                 # [B,H]

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4, f"Expected [B,C,H,W], got {tuple(x.shape)}"
        B, C, H, W = x.shape

        # 1) 深度能量
        u = self._depth_energy(x)                     # [B,H]
        u_mean = u.mean(dim=1, keepdim=True)          # [B,1]

        # 2) 构造“浅强深弱”模板 T(h) = (1 - z)^p，归一化到 mean=1
        z = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype)[None, :]  # [1,H]
        T = (1.0 - z).clamp(min=0) ** self.shape_p                             # [1,H]
        T = T / (T.mean(dim=1, keepdim=True) + self.eps)                       # mean=1

        # 3) 轻度拉平 + 强制浅强
        #    拉平项：(u_mean / u)^gamma   → 深弱补一点、浅强收一点
        #    倾斜项：T^tilt               → 无论输入如何，都推向浅强/深弱
        G = (u_mean / (u + self.eps)) ** self.gamma        # [B,H]
        G = G * (T ** self.tilt)                           # [B,H]

        # 4) 平滑、限幅、均值归一（保证总能量不漂）
        G = self._smooth1d(G)
        G = torch.clamp(G, min=self.gmin, max=self.gmax)
        if self.mean_center:
            G = G / (G.mean(dim=1, keepdim=True) + self.eps)

        self.last_gain = G.detach()                        # [B,H]
        y = x * G.view(B, 1, H, 1)                         # [B,1,H,W]
        y_min = y.amin(dim=(1, 2, 3), keepdim=True)  # 每张图最小值
        y_max = y.amax(dim=(1, 2, 3), keepdim=True)  # 每张图最大值

        y = 2.0 * (y - y_min) / (y_max - y_min + self.eps) - 1.0
        # plt.figure()
        # plt.imshow(x[0,0].detach().cpu().numpy())
        # plt.figure()
        # plt.imshow(y[0,0].detach().cpu().numpy())
        # plt.figure()
        # plt.plot(G[0].detach().cpu().numpy())
        # plt.show()
        return y




class UnsuperShortcut(nn.Module):
    def __init__(self,  **kwargs):
        super(UnsuperShortcut, self).__init__(**kwargs)

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))

        self.pool = nn.MaxPool2d(2, 2)

        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))

        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        layer1 = self.stage1(x)  # 32 channels
        layer2 = self.stage2(self.pool(layer1))  # 64 channels
        layer3 = self.stage3(self.pool(layer2))

        h_new, w_new = layer3.shape[-2:]
        layer1_down = nn.functional.interpolate(layer1, size=[h_new, w_new])
        layer2_down = nn.functional.interpolate(layer2, size=[h_new, w_new])
        out = torch.cat([layer1_down, layer2_down, layer3], axis=1)

        return out

class UnsuperVgg(nn.Module):
    def __init__(self,  **kwargs):
        super(UnsuperVgg, self).__init__(**kwargs)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.cnn(x)
        return out

class UnsuperVggTiny(nn.Module):
    def __init__(self,  **kwargs):
        super(UnsuperVggTiny, self).__init__(**kwargs)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.cnn(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inchannel = 16 # 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.inchannel, kernel_size=3, stride=2, padding=1), # 1
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU(inplace=True),
        )
        self.ResidualBlock = ResidualBlock

        # block, channels, num_blocks, stride
        self.layer1 = self.make_layer(self.ResidualBlock, self.inchannel,  2, stride=2)
        self.layer2 = self.make_layer(self.ResidualBlock, 32, 1, stride=1)
        self.layer3 = self.make_layer(self.ResidualBlock, 64, 1, stride=1)
        self.layer4 = self.make_layer(self.ResidualBlock, 128, 1, stride=1)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ViT(nn.Module):
    def __init__(self, patch_size=4, dim=256, depth=6, heads=8, mlp_dim=512):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim

        self.patch_embedding = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)

        # 相对位置编码 or 不使用绝对位置编码
        self.use_pos_embedding = False

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        B, C, H, W = x.shape
        H_p, W_p = H // self.patch_size, W // self.patch_size

        # Patch embedding
        x = self.patch_embedding(x)  # [B, dim, H/ps, W/ps]
        x = rearrange(x, 'b c h w -> b (h w) c')  # [B, N, C]

        # Position embedding（可选）
        if self.use_pos_embedding:
            pos_embedding = torch.randn(1, x.shape[1], self.dim, device=x.device)
            x = x + pos_embedding

        # Transformer
        x = self.transformer(x)  # [B, N, C]

        # Reshape back to feature map
        x = rearrange(x, 'b (h w) c -> b c h w', h=H_p, w=W_p)  # [B, C, H/ps, W/ps]
        return x

class WaveEncoder(nn.Module):
    def __init__(self, in_channels):
        super(WaveEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 10, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.encoder(x)

class WaveDecoder(nn.Module):
    def __init__(self, out_channels):
        super(WaveDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(10, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)
        )
    def forward(self, x):
        x = self.decoder(x)
        return x


class WaveSenseNet(nn.Module):
    def __init__(self, in_channels):
        super(WaveSenseNet, self).__init__()
        self.encoder = WaveEncoder(in_channels)
        self.decoder = WaveDecoder(in_channels)

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon



if __name__ == '__main__':
    v = ViT(
        patch_size=4,
        dim=128,
        depth=4,
        heads=2,
        mlp_dim=256
    )

    img = torch.randn(1, 3, 240, 320)

    preds = v(img)  # (1, 1000)
    print(preds.shape)
