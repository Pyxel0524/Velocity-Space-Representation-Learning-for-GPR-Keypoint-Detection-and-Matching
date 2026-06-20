import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --------------------
# 图像尺寸改成 120x240
# --------------------
H, W = 120, 240
y = torch.linspace(0, 1, H)
x = torch.linspace(0, 1, W)
X, Y = torch.meshgrid(x, y, indexing="xy")

# 合成一张从上到下衰减的“GPR风格”图
fx1, fx2 = 6.0, 18.0
phase1 = (2.0 * np.pi) * (0.20 * Y)
phase2 = (2.0 * np.pi) * (0.35 * Y)
signal = torch.sin(2 * np.pi * fx1 * X + phase1) + 0.5 * torch.sin(2 * np.pi * fx2 * X + phase2)
envelope = torch.exp(-1.9 * Y)

rng = torch.Generator().manual_seed(0)
noise = 0.06 * torch.randn((H, W), generator=rng)
reflectors = torch.zeros((H, W))
for _ in range(60):  # 轻一点
    ry = int(torch.randint(low=int(0.55*H), high=H, size=(1,), generator=rng))
    rx = int(torch.randint(low=0, high=W, size=(1,), generator=rng))
    reflectors[ry, max(0, rx-1):min(W, rx+2)] = 1.0

img = (envelope * signal + 0.3 * reflectors + noise).clamp(-1.0, 1.0)
img = torch.tensor(np.load('F:\data\Registration\CMU\Test\Map\\1613059289_135772_X_-7.7745_Y_-25.0303_T_yaw_180_odom_6.4347_dir_1.0_0.npy'))
# img = torch.tensor(np.load('F:\data\Registration\CMU\Test\Query\\1613059490_852721_X_-7.4863_Y_-25.1467_T_yaw_180_odom_6.0052_dir_1.0_0.npy'))

# 深度注意力增益层（可训练版）
class DepthAttentionGain(nn.Module):
    def __init__(self, d_model=24, alpha=0.9, gmin=0.6, gmax=3.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(1, d_model), nn.GELU())
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.out  = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.alpha = alpha; self.gmin, self.gmax = gmin, gmax

    def forward(self, img_2d):
        u = img_2d.abs().mean(dim=1, keepdim=True)           # [H,1]
        u = (u - u.mean()) / (u.std() + 1e-6)
        Z = self.embed(u.to(torch.float32))
        Q, K, V = self.to_q(Z), self.to_k(Z), self.to_v(Z)
        attn = torch.softmax(Q @ K.T / (Z.shape[1] ** 0.5), dim=-1)   # [H,H]
        s = (attn @ V)
        s = self.out(s).squeeze(1)
        g = torch.tanh(s) * self.alpha
        G = torch.exp(g).clamp(min=self.gmin, max=self.gmax)          # [H]
        G = G / (G.mean() + 1e-6)
        img_corr = (img.T * G).T
        return img_corr

def depth_energy(x2d): return x2d.abs().mean(dim=1)

# A) 无监督训练版（让深度能量更平）
def train_unsupervised(img, steps=1, lr=2e-2, tv_w=0.05, center_w=0.01):
    layer = DepthAttentionGain()
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        G = layer(img); Gn = G / (G.mean() + 1e-6)
        u_after = depth_energy((img.T * Gn).T)
        L_flat = torch.var(u_after)
        L_tv = torch.mean(torch.abs(Gn[1:] - Gn[:-1]))
        L_center = torch.mean((Gn - 1.0)**2)
        loss = L_flat + tv_w*L_tv + center_w*L_center
        loss.backward(); opt.step()
    with torch.no_grad():
        G = layer(img); G = G / (G.mean() + 1e-6)
        img_corr = (img.T * G).T
    return G, img_corr

# B) 无训练“注意力样式”增益（更快）
def param_free_attention_gain(img, gamma=0.7, gmin=0.7, gmax=3.0):
    u = depth_energy(img)                                      # [H]
    tau = 0.15 * u.std().item() + 1e-6
    diff2 = (u[:, None] - u[None, :])**2                       # [H,H]
    A = torch.softmax(-diff2 / (2*(tau**2)), dim=1)            # 幅度相似度注意力
    u_ctx = A @ u
    target = u_ctx.mean()
    G = (target / (u_ctx + 1e-6)) ** gamma
    G = torch.clamp(G, min=gmin, max=gmax)
    G = G * (G.numel() / G.sum())                              # mean≈1
    img_corr = (img.T * G).T
    return G, img_corr

# 选择：训练 or 无训练
USE_TRAINING = True  # True 则用训练版
G, img_corr = (train_unsupervised(img) if USE_TRAINING else param_free_attention_gain(img))

# 可视化
plt.figure(figsize=(5.0, 3.6)); plt.title("Original (120x240)")
plt.imshow(img.numpy(), aspect='auto'); plt.tight_layout(); plt.show()

plt.figure(figsize=(5.0, 3.6)); plt.title("After attention-modulated gain")
plt.imshow(img_corr.numpy(), aspect='auto'); plt.tight_layout(); plt.show()

u_before = depth_energy(img).numpy()
u_after  = depth_energy(img_corr).numpy()
z = np.arange(H)

plt.figure(figsize=(5.0, 3.0)); plt.title("Mean |signal| vs depth")
plt.plot(z, u_before, label="before"); plt.plot(z, u_after, label="after", linestyle='--')
plt.legend(loc='lower right'); plt.tight_layout(); plt.show()

plt.figure(figsize=(5.0, 2.8)); plt.title("Predicted gain G(z)"); plt.plot(z, G.numpy())
plt.tight_layout(); plt.show()
