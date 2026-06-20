#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Learnable depth-wise gain with QKV attention for BCHW inputs (GPR-style).

- 输入:  img [B, C, H, W]
- 输出:  img_corr [B, C, H, W], G_broadcast:
          * [B, 1, H, 1]  当 share_across_channels=True（推荐）
          * [B, C, H, 1]  当 share_across_channels=False（每通道各一条增益）

核心流程（可学习版，保留 QKV 注意力）：
  1) 计算深度维能量 u(z)：对宽度（和可选的通道）做平均，得到 [B,H] 或 [B,C,H]
  2) 将 u(z) 线性嵌入，做 Multi-Head 自注意力（沿 H 维）
  3) 经过小型前馈层得到标量 s(z)，tanh 压缩后放大，指数映射为正增益 G(z)
  4) 可选平滑、裁剪与均值归一；将 G(z) 广播乘到原图像

无任何监督标签也可训练：可用 “深度能量方差 + TV + 均值约束” 的损失微调本层参数。
"""

from typing import Optional, Tuple, Literal
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------- 工具函数 -------------------------

def _make_kernel(k: int, kind: str, device, dtype) -> torch.Tensor:
    if k <= 1:
        return torch.ones(1, 1, 1, device=device, dtype=dtype)
    if kind == "hann":
        n = torch.arange(k, device=device, dtype=dtype)
        win = 0.5 - 0.5 * torch.cos(2 * torch.pi * n / (k - 1))
        win = (win / win.sum()).view(1, 1, k)
        return win
    # box
    return torch.ones(1, 1, k, device=device, dtype=dtype) / k


def _smooth_1d_batched(x: torch.Tensor, k: int, kind: str = "hann") -> torch.Tensor:
    """对最后一维 (H) 做 1D 平滑，x: [..., H]"""
    if k is None or k <= 1:
        return x
    pad = (k - 1) // 2
    w = _make_kernel(k, kind, x.device, x.dtype)  # [1,1,k]
    x2 = x.reshape(-1, 1, x.shape[-1])
    y = F.conv1d(x2, w, padding=pad)
    return y.reshape(x.shape)


# ------------------------- 模块主体 -------------------------

class DepthQKVGain(nn.Module):
    """
    Learnable depth-wise gain via QKV attention（可学习版）

    Args:
        d_model:    嵌入维度
        n_heads:    注意力头数（d_model 必须能整除 n_heads）
        alpha:      tanh 放大系数（控制增益动态范围）
        gmin,gmax:  增益裁剪上下限
        mean_center:是否按深度均值归一到约 1
        smooth_k:   深度维平滑窗口（<=1 代表不平滑）
        smooth_type:'hann' 或 'box'
        share_across_channels: True=样本共享一条 G(z)，False=每通道一条 G_c(z)
        eps:        数值地板
        use_posenc: 是否为深度位置添加正弦位置编码（增强序列建模）
    """
    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        alpha: float = 0.9,
        gmin: float = 0.5,
        gmax: float = 3.0,
        mean_center: bool = True,
        smooth_k: int = 5,
        smooth_type: Literal["hann", "box"] = "hann",
        share_across_channels: bool = True,
        eps: float = 1e-6,
        use_posenc: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.alpha = alpha
        self.gmin, self.gmax = gmin, gmax
        self.mean_center = mean_center
        self.smooth_k = int(smooth_k)
        self.smooth_type = smooth_type
        self.share_across_channels = share_across_channels
        self.eps = eps
        self.use_posenc = use_posenc

        # 输入为 u(z) 的标量，先线性嵌入到 d_model
        self.embed = nn.Linear(1, d_model)
        # QKV
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        # FFN 输出标量
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # 可学习缩放与偏置（稳定训练）
        self.out_scale = nn.Parameter(torch.tensor(1.0))
        self.out_bias  = nn.Parameter(torch.tensor(0.0))

        self.last_gain: Optional[torch.Tensor] = None  # [B,H] 或 [B,C,H]

    # ---------- 基础统计 ----------
    @staticmethod
    def _depth_energy_bchw(img: torch.Tensor, share: bool) -> torch.Tensor:
        """
        share=True  -> u: [B,H]   （对通道与宽度均值）
        share=False -> u: [B,C,H] （对宽度均值）
        """
        if share:
            return img.abs().mean(dim=(1, 3))  # [B,H]
        else:
            return img.abs().mean(dim=3)       # [B,C,H]

    # ---------- 位置编码 ----------
    @staticmethod
    def _posenc_1d(H: int, d_model: int, device, dtype) -> torch.Tensor:
        """
        标准 Transformer 正弦位置编码，返回 [H, d_model]
        """
        pe = torch.zeros(H, d_model, device=device, dtype=dtype)
        position = torch.arange(0, H, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [H, d_model]

    # ---------- 核心注意力 ----------
    def _attend(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z: [N, H, d_model]  （N 可是 B 或 B*C）
        返回: Y: [N, H, d_model]
        """
        N, H, D = Z.shape
        Q = self.to_q(Z).view(N, H, self.n_heads, self.d_head).transpose(1, 2)  # [N, heads, H, d_head]
        K = self.to_k(Z).view(N, H, self.n_heads, self.d_head).transpose(1, 2)  # [N, heads, H, d_head]
        V = self.to_v(Z).view(N, H, self.n_heads, self.d_head).transpose(1, 2)  # [N, heads, H, d_head]

        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head), dim=-1)  # [N, heads, H, H]
        Y = torch.matmul(attn, V)  # [N, heads, H, d_head]
        Y = Y.transpose(1, 2).contiguous().view(N, H, D)  # [N, H, d_model]
        return Y

    # ---------- 计算增益 ----------
    def _compute_gain_shared(self, img: torch.Tensor) -> torch.Tensor:
        """
        共享通道：返回 G: [B, H]
        """
        B, C, H, W = img.shape
        u = self._depth_energy_bchw(img, share=True)  # [B,H]
        # 标准化（每样本）
        u = (u - u.mean(dim=1, keepdim=True)) / (u.std(dim=1, keepdim=True) + self.eps)  # [B,H]
        u_in = u.unsqueeze(-1)  # [B,H,1]
        Z = self.embed(u_in)    # [B,H,d_model]

        # 位置编码
        if self.use_posenc:
            pe = self._posenc_1d(H, self.d_model, Z.device, Z.dtype)  # [H,d_model]
            Z = Z + pe.unsqueeze(0)

        Y = self._attend(Z)                # [B,H,d_model]
        s = self.proj(Y).squeeze(-1)       # [B,H]
        s = s * self.out_scale + self.out_bias

        # tanh 限幅 + 指数为正
        g = torch.tanh(s) * self.alpha     # [-alpha, alpha]
        G = torch.exp(g)                   # (0, e^alpha]
        return G                           # [B,H]

    def _compute_gain_per_channel(self, img: torch.Tensor) -> torch.Tensor:
        """
        每通道一条增益：返回 G: [B, C, H]
        """
        B, C, H, W = img.shape
        u = self._depth_energy_bchw(img, share=False)  # [B,C,H]
        uf = u.reshape(B * C, H)
        uf = (uf - uf.mean(dim=1, keepdim=True)) / (uf.std(dim=1, keepdim=True) + self.eps)  # [BC,H]
        uf_in = uf.unsqueeze(-1)  # [BC,H,1]
        Z = self.embed(uf_in)     # [BC,H,d_model]

        if self.use_posenc:
            pe = self._posenc_1d(H, self.d_model, Z.device, Z.dtype)
            Z = Z + pe.unsqueeze(0)

        Y = self._attend(Z)                # [BC,H,d_model]
        s = self.proj(Y).squeeze(-1)       # [BC,H]
        s = s * self.out_scale + self.out_bias

        g = torch.tanh(s) * self.alpha
        G = torch.exp(g).view(B, C, H)
        return G  # [B,C,H]

    def _normalize_and_clip(self, G: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.mean_center:
            G = G * (G.shape[dim] / (G.sum(dim=dim, keepdim=True) + self.eps))
        G = torch.clamp(G, min=self.gmin, max=self.gmax)
        return G

    # ---------- 前向 ----------
    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        img: [B,C,H,W] -> (img_corr, G_broadcast)
        """
        assert img.dim() == 4, f"Expected BCHW, got {tuple(img.shape)}"
        B, C, H, W = img.shape

        if self.share_across_channels:
            G = self._compute_gain_shared(img)        # [B,H]
            if self.smooth_k > 1:
                G = _smooth_1d_batched(G, self.smooth_k, self.smooth_type)
            G = self._normalize_and_clip(G, dim=-1)   # [B,H]
            G_b = G[:, None, :, None]                 # [B,1,H,1]
        else:
            G = self._compute_gain_per_channel(img)   # [B,C,H]
            if self.smooth_k > 1:
                G = _smooth_1d_batched(G.view(B*C, -1), self.smooth_k, self.smooth_type).view(B, C, H)
            G = self._normalize_and_clip(G, dim=-1)   # [B,C,H]
            G_b = G[:, :, :, None]                    # [B,C,H,1]

        self.last_gain = G.detach()
        img_corr = img * G_b.to(img.dtype)
        return img_corr, G_b


# ------------------------- 简单演示 / 自测 -------------------------
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # 合成一批 GPR 风格数据
    def make_synth(B=2, C=3, H=120, W=240, seed=0):
        rng = torch.Generator().manual_seed(seed)
        y = torch.linspace(0, 1, H)
        x = torch.linspace(0, 1, W)
        X, Y = torch.meshgrid(x, y, indexing="xy")
        fx1, fx2 = 6.0, 18.0
        phase1 = (2.0 * math.pi) * (0.20 * Y)
        phase2 = (2.0 * math.pi) * (0.35 * Y)
        signal = torch.sin(2 * math.pi * fx1 * X + phase1) + 0.5 * torch.sin(2 * math.pi * fx2 * X + phase2)
        envelope = torch.exp(-1.9 * Y)
        noise = 0.06 * torch.randn((H, W), generator=rng)
        refl = torch.zeros((H, W))
        for _ in range(60):
            ry = int(torch.randint(low=int(0.55*H), high=H, size=(1,), generator=rng))
            rx = int(torch.randint(low=0, high=W, size=(1,), generator=rng))
            refl[ry, max(0, rx-1):min(W, rx+2)] = 1.0
        img = (envelope * signal + 0.3 * refl + noise).clamp(-1, 1).float()
        img = img.unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)  # [B,C,H,W]
        return img

    B, C, H, W = 2, 3, 120, 240
    x = make_synth(B, C, H, W)

    # 初始化可学习增益层
    gain = DepthQKVGain(
        d_model=32,
        n_heads=4,
        alpha=0.9,
        gmin=0.6,
        gmax=3.0,
        mean_center=True,
        smooth_k=5,
        smooth_type="hann",
        share_across_channels=True,  # 推荐
        use_posenc=True,
    )

    # 无监督“平能量”微调（示例，可按需删除）
    def depth_energy_shared(t: torch.Tensor) -> torch.Tensor:
        return t.abs().mean(dim=(1, 3))  # [B,H]

    opt = torch.optim.Adam(gain.parameters(), lr=2e-3)
    for _ in range(50):
        opt.zero_grad()
        y, Gb = gain(x)
        u = depth_energy_shared(y)                  # [B,H]
        L_flat = u.var(dim=1).mean()                # 深度能量更平
        tv = (Gb.squeeze(1)[:, 1:, 0] - Gb.squeeze(1)[:, :-1, 0]).abs().mean()
        L_center = (Gb.mean() - 1.0).abs()
        loss = L_flat + 0.05 * tv + 0.01 * L_center
        loss.backward()
        opt.step()

    with torch.no_grad():
        y, Gb = gain(x)

    # 可视化一个样本的前后能量
    u0_before = depth_energy_shared(x)[0].cpu().numpy()
    u0_after  = depth_energy_shared(y)[0].cpu().numpy()
    z = np.arange(H)

    plt.figure(figsize=(5.2, 3.2)); plt.title("Mean |signal| vs depth (sample 0)")
    plt.plot(z, u0_before, label="before"); plt.plot(z, u0_after, label="after", linestyle="--")
    plt.legend(); plt.tight_layout(); plt.show()

    # 显示增益
    plt.figure(figsize=(5.0, 2.6)); plt.title("Learned gain G(z)")
    plt.plot(z, gain.last_gain[0].cpu().numpy())
    plt.tight_layout(); plt.show()

    print("Done. y shape:", y.shape, " G_broadcast shape:", Gb.shape)
