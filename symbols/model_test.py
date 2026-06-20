import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np


class ViT_Backbone(nn.Module):
    def __init__(self, image_size=256, patch_size=16, dim=768, depth=6, heads=8, mlp_dim=1024):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.patch_dim = 3 * patch_size * patch_size

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),  # [B, dim, H/ps, W/ps]
        )

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # Optional

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embedding(x)  # [B, dim, H/ps, W/ps]
        H_p, W_p = x.shape[2], x.shape[3]

        x = rearrange(x, 'b c h w -> b (h w) c')  # [B, N, C]
        x += self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)  # [B, N, C]

        x = rearrange(x, 'b (h w) c -> b c h w', h=H_p, w=W_p)  # [B, C, H/ps, W/ps]

        # Upsample to original size
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x  # [B, C, H, W]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    model = ViT_Backbone(image_size=256, patch_size=16, dim=64)
    inp = torch.randn(1, 3, 256, 256)
    out = model(inp)
    print(out.shape)
    plt.figure()
    plt.imshow(inp[0,0].detach().numpy())
    plt.figure()
    plt.imshow(out[0,0].detach().numpy())
    plt.show()