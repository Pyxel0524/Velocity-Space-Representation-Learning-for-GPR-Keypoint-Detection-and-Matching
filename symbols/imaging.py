import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Imaging(nn.Module):
    def __init__(self, init_velocity_mean=0.1, velocity_std=0.02, interp_points=8, device="cuda"):
        """
        PyTorch FK 偏移层，支持从高斯分布采样波速，并强制 v1 > 均值, v2 < 均值
        采用可微分的重参数化技巧
        :param init_velocity_mean: 初始波速均值 (默认 0.1 m/ns)
        :param velocity_std: 波速标准差 (手动设定，默认 0.02 m/ns)
        :param interp_points: Stolt 插值点数 (默认 8)
        :param device: 运行设备 (默认 "cuda")
        """
        super(Imaging, self).__init__()
        self.device = device
        self.interp_points = interp_points
        # 让波速均值成为可学习参数
        self.velocity_mean = nn.Parameter(torch.tensor(init_velocity_mean, dtype=torch.float64, requires_grad=True))

        # 波速标准差为固定值
        self.velocity_std = velocity_std

    def sample_velocity(self, type):
        """
        生成两个不同的波速：v1 高于均值, v2 低于均值
        采用重参数化技巧，确保可微分
        :return: (B, 1) 形状的 v1, v2
        """
        epsilon = torch.randn(1, dtype=torch.float64, device=self.device)  # 采样标准正态分布
        delta = self.velocity_std * epsilon  # 计算偏移量 (可微分)
        delta = torch.abs(delta)  # 取绝对值，确保 v1 > 均值, v2 < 均值

        if type == 'h':
            v = self.velocity_mean + delta  # 高于均值
        elif type == 'l':
            v = self.velocity_mean - delta  # 低于均值
        else:
            v = self.velocity_mean

        return v

    def next_pow2(self, n):
        """ 计算比 n 大的最小 2 的幂次方 """
        return 1 << (int(torch.ceil(torch.log2(torch.tensor(n, dtype=torch.float64)))))

    def forward(self, section_tx, dt, dx, type):
        """
        FK 偏移成像前向传播
        :param section_tx: 输入 GPR 数据 (B, C, nt, nx)
        :param dt: 采样时间间隔
        :param dx: 采样空间间隔
        :return: (B, C, nt, nx) 形状的 FK 偏移后的成像
        """
        # 采样两个不同的波速 v1 和 v2（按批次生成）
        v = self.sample_velocity(type)
        # 计算 FK 偏移
        migrated = self.migration(section_tx, dt, dx, v)

        return migrated

    def migration(self, section_tx, dt, dx, v, interp_points=1):
        """
        PyTorch 版本的 F-K 偏移，支持 GPU 和批处理（batch processing）

        参数：
            section_tx : torch.Tensor (B, C, H, W)  批量雷达数据
            dt : float  时间采样间隔 (s)
            dx : float  空间采样间隔 (m)
            v : torch.Tensor (B, 1, 1, 1)  介质速度 (m/s) (支持 batch 变速)
            interp_points : int  插值点数

        返回：
            migrated : torch.Tensor (B, C, H, W)  偏移结果
        """
        device = section_tx.device  # 适配 GPU 计算

        B, Ch, H, W = section_tx.shape  # 读取 batch 维度信息
        new_H = self.next_pow2(2 * H)
        new_W = self.next_pow2(2 * W)

        # **补零并转换到频域**
        padded = torch.zeros((B, Ch, new_H, new_W), dtype=torch.complex64, device=device)#, device=device
        padded[:, :, :H, :W] = section_tx.to(dtype=torch.complex64)

        spectrum_t = torch.fft.fft(padded, dim=2)  # 时间轴 FFT
        spectrum_kx = torch.fft.fft(spectrum_t, dim=3)  # 空间轴 FFT

        # **计算波数频率参数**
        df = 2 * torch.pi / (new_H * dt)
        dkx = 2 * torch.pi / (new_W * dx)

        nf_pos = new_H // 2 + 1
        f_pos = torch.arange(nf_pos, dtype=torch.float32, device=device) * df#, device=device
        kz = f_pos.view(1, 1, -1, 1) / v  # 计算 kz

        nkx_pos = new_W // 2 + 1
        kx_pos = torch.arange(nkx_pos, dtype=torch.float32, device=device) * dkx# device=device

        # **初始化新频谱和插值核**
        spectrum_new = torch.zeros((B, Ch, nf_pos, new_W), dtype=torch.complex64, device=device)
        C = torch.arange(1 - torch.ceil(torch.tensor(interp_points / 2)),
                         torch.ceil(torch.tensor(interp_points / 2)) + 1, device=device).long().view(1, -1)

        # **Stolt 插值**
        for ikx in range(1, nkx_pos):
            kx = kx_pos[ikx]
            f_mapped = (v * torch.sqrt(kz ** 2 + kx ** 2)).squeeze(3)
            n = f_mapped / df

            valid_mask = f_mapped != 0
            factor = torch.where(valid_mask, (f_pos.squeeze(-1) * v) / f_mapped, torch.tensor(0.0, device=device))

            n_floor = torch.floor(n).long()
            delta = n - n_floor

            indices = n_floor.unsqueeze(-1) + C  # 计算索引
            valid = (indices >= 0) & (indices < nf_pos)
            indices_clipped = torch.clamp(indices, 0, nf_pos - 1).squeeze(0).squeeze(0)   # 形状 (B, C, nf_pos, interp_points)

            # **提取正负波数数据**
            ip = spectrum_kx[:, :, indices_clipped, ikx]  # 形状 (B, C, nf_pos, interp_points)
            in_ = spectrum_kx[:, :, indices_clipped, new_W - ikx]

            # **计算插值核**
            distance = delta.unsqueeze(-1) - C
            kernel = torch.sinc(distance) * torch.exp(-1j * torch.pi * distance)
            kernel[~valid] = 0  # 处理无效点

            # **计算插值结果**
            temp_p = factor * torch.sum(ip * kernel, dim=3, keepdim=True).squeeze(3)  # 直接匹配维度
            temp_n = factor * torch.sum(in_ * kernel, dim=3, keepdim=True).squeeze(3)  # 直接匹配维度

            spectrum_new[:, :, :, ikx] = temp_p
            spectrum_new[:, :, :, new_W - ikx] = temp_n
        # **处理零波数**
        spectrum_new[:, :, :, 0] = v * spectrum_kx[:, :, :nf_pos, 0]

        # **重构完整频谱并逆变换**
        spectrum_full = torch.cat([
            spectrum_new,
            torch.conj(torch.flip(spectrum_new[:, :, 1:new_H - nf_pos + 1, :], dims=[2]))  # 处理负频率
        ], dim=2)

        migrated_t = torch.fft.ifft(spectrum_full, dim=2)
        migrated = torch.fft.ifft(migrated_t, dim=3).real

        return migrated[:, :, :H, :W]  # 还原到原始大小


# 测试代码
if __name__ == "__main__":
    import cv2
    import numpy as np
    image_path = "D:\Study\Code\image\gpr.png"  # 你的雷达B扫描图像
    gpr_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gpr_data = gpr_data.astype(np.float32) / 255.0  # 归一化到[0,1]
    gpr_data = cv2.resize(gpr_data, (240, 320), interpolation=cv2.INTER_LINEAR)
    gpr_data = torch.from_numpy(gpr_data).float().to("cuda")
    imaging = Imaging(init_velocity_mean=3, velocity_std=0.5, device="cuda")
    # imaging = FastFKMigrationLayer(init_velocity_mean=4.5, velocity_std=0.5, device="cuda")

    migrated_1, migrated_2, v1, v2 = imaging(gpr_data, 0.01,0.15)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(migrated_1.detach().cpu().numpy())
    plt.show()

    print("采样的波速 v1:", v1.item())
    print("采样的波速 v2:", v2.item())
