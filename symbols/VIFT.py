import torch
import torch.nn as nn
import numpy as np

from symbols.model_factory import ResNet, UnsuperVggTiny, UNet, ViT, InverseGain
from symbols.model_base import ModelTemplate
import torch.nn.functional as F


class UnSuperPoint(ModelTemplate):
    def __init__(self, base_model, model_config, IMAGE_SHAPE):
        super(UnSuperPoint, self).__init__()

        self.downsample = model_config['downsample']
        self.image_shape = IMAGE_SHAPE
        self.feature_hw = [self.image_shape[0] // self.downsample, self.image_shape[1] // self.downsample]

        # export threshold
        self.correspond = model_config['correspond']
        self.position_weight = model_config['position_weight']
        self.score_weight = model_config['score_weight']
        self.rep_weight = model_config['rep_weight']

        # LOSS
        self.usp = model_config['LOSS']['usp']
        self.uni_xy = model_config['LOSS']['uni_xy']
        self.desc = model_config['LOSS']['desc']
        self.decorr = model_config['LOSS']['decorr']
        self.struct = model_config['LOSS']['struct']

        self.d = model_config['d']
        self.t = model_config['mask_th']
        self.m_p = model_config['m_p']
        self.m_n = model_config['m_n']

        self.eps = 1e-12

        # create mesh grid
        x = torch.arange(self.image_shape[1] // self.downsample, requires_grad=False,
                         device='cuda' if torch.cuda.is_available() else 'cpu')
        y = torch.arange(self.image_shape[0] // self.downsample, requires_grad=False,
                         device='cuda' if torch.cuda.is_available() else 'cpu')
        y, x = torch.meshgrid([y, x], indexing='ij')
        self.cell = torch.stack([x, y], dim=0)

        self.base_model = base_model
        self.input_ch = 128
        self.des_ch = 128

        self.score = nn.Sequential(
            nn.Conv2d(self.input_ch, self.input_ch, 3, 1, padding=1),
            nn.BatchNorm2d(self.input_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.input_ch, 1, 1, 1, padding=0),
            nn.Sigmoid()
        )
        self.position = nn.Sequential(
            nn.Conv2d(self.input_ch, self.input_ch, 3, 1, padding=1),
            nn.BatchNorm2d(self.input_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.input_ch, 2, 1, 1, padding=0),
            nn.Sigmoid()
        )
        self.descriptor = nn.Sequential(
            nn.Conv2d(self.input_ch, self.input_ch, 3, 1, padding=1),
            nn.BatchNorm2d(self.input_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.input_ch, self.des_ch, 1, 1, padding=0)
        )


    def forward(self, x):
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(x[0,0,:,:].detach().cpu().numpy());plt.colorbar()
        # plt.show()
        feature = self.base_model(x)

        s = self.score(feature)
        p = self.position(feature)
        d = self.descriptor(feature)
        # desc = self.interpolate(p, d, self.feature_hw[0], self.feature_hw[1])  # (B, C, H, W)
        d = torch.nn.functional.normalize(input=d, p=2, dim=1, eps=self.eps)
        return s, p, d

    def mask(self, echoes,bins = 256,margin = 0.045):
        """
        input :
                radar_echoes
        output:
            background_remove_mask
        """

        # 将像素值量化成整数bin
        bin_indices = (echoes * (bins - 1)).long().clamp(0, bins - 1)  # (H, W)
        # 创建 mask 空间
        B, C, D, A = echoes.shape
        BG_mask = torch.ones_like(echoes[0,0])

        for col in range(A):
            # 当前列所有像素的 bin 值
            col_vals = echoes[B-1,C-1,:,col]
            # 统计频次
            hist = torch.histc(echoes[B-1,C-1,:,col], bins=bins, min=0.0, max=1.0)
            mode_bin = torch.argmax(hist)
            # 将主模态对应的像素值反算回强度值
            mode_val = mode_bin.float() / (bins - 1)
            # 构造 mask：保留不在 mode_val ± margin 范围内的值
            BG_mask[:, col] = (col_vals > (mode_val + margin)).float()
        # import matplotlib.pyplot as plt
        # plt.figure(); plt.imshow(BG_mask.detach().cpu().numpy())
        # plt.figure(); plt.imshow(echoes[0,0].detach().cpu().numpy());
        # plt.show()

        return BG_mask.squeeze()

    def entropy_map(self, echoes, kernel_size=3, num_bins=16, eps=1e-8):
        # Convert image to shape (1, 1, H, W) and normalize
        if isinstance(echoes, np.ndarray):
            echoes = torch.tensor(echoes, dtype=torch.float32)
        if echoes.dim() == 2:
            echoes = echoes.unsqueeze(0).unsqueeze(0)
        echoes = (echoes - echoes.min()) / (echoes.max() - echoes.min() + eps)

        B, _, H, W = echoes.shape
        unfold = F.unfold(echoes, kernel_size=kernel_size, padding=kernel_size // 2)
        patches = unfold.view(B, kernel_size * kernel_size, H, W)

        bin_edges = torch.linspace(0, 1, steps=num_bins + 1, device=echoes.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        hist = torch.zeros((B, num_bins, H, W), device=echoes.device)
        for i, c in enumerate(bin_centers):
            hist[:, i] = ((patches - c).abs() < (0.5 / num_bins)).sum(dim=1)

        prob = hist / (kernel_size * kernel_size + eps)
        entropy = -torch.sum(prob * torch.log(prob + eps), dim=1, keepdim=True)

        entropy_min = entropy.amin(dim=(2, 3), keepdim=True)
        entropy_max = entropy.amax(dim=(2, 3), keepdim=True)
        entropy_norm = (entropy - entropy_min) / (entropy_max - entropy_min + eps)

        return entropy_norm.squeeze()


    def loss(self,a, a_s, a_p, a_d,b, b_s, b_p, b_d):

        loss = 0
        loss_batch_array = np.zeros((5,))  # 根据loss数量来定
        batch = a_s.shape[0]

        for i in np.arange(0, batch):
            loss_batch, loss_item = self.VIFTLoss(a[i], a_s[i], a_p[i], a_d[i],
                                                  b[i], b_s[i], b_p[i], b_d[i])
            loss += loss_batch
            loss_batch_array += loss_item

        return loss / batch, loss_batch_array / batch

    def UnsuperPointLoss(self, a_s, a_p, a_d, b_s, b_p, b_d):
        position_a = self.get_position(a_p, self.cell, self.downsample, flag='B')  # c h w, where c==2
        position_b = self.get_position(b_p, self.cell, self.downsample, flag='B')

        key_dist = self.get_dis(position_a, position_b)  # c h w -> c p p

        batch_loss = 0
        loss_item = []

        if self.usp > 0:
            usp_loss = self.usp * self.usp_loss(a_s, b_s, key_dist)
            batch_loss += usp_loss
            loss_item.append(usp_loss.item())
        else:
            loss_item.append(0.)

        if self.uni_xy > 0:
            uni_xy_loss = self.uni_xy * self.uni_xy_loss(a_p, b_p)
            batch_loss += uni_xy_loss
            loss_item.append(uni_xy_loss.item())
        else:
            loss_item.append(0.)

        if self.desc > 0:
            desc_loss = self.desc * self.desc_loss(a_d, b_d, key_dist)
            batch_loss += desc_loss
            loss_item.append(desc_loss.item())
        else:
            loss_item.append(0.)

        if self.decorr > 0:
            decorr_loss = self.decorr * self.decorr_loss(a_d, b_d)
            batch_loss += decorr_loss
            loss_item.append(decorr_loss.item())
        else:
            loss_item.append(0.)

        return batch_loss, np.array(loss_item)

    def VIFTLoss(self, a, a_s, a_p, a_d, b, b_s, b_p, b_d):
        position_a = self.get_position(a_p, self.cell, self.downsample, flag='B')  # c h w, where c==2
        position_b = self.get_position(b_p, self.cell, self.downsample, flag='B')

        key_dist = self.get_dis(position_a, position_b)  # c h w -> c p p
        # import matplotlib.pyplot as plt;
        # plt.figure(); plt.imshow(a[0].detach().cpu().numpy()); plt.figure(); plt.imshow(a_s[0].detach().cpu().numpy(),cmap = 'jet');
        # plt.figure(); plt.imshow(b[0].detach().cpu().numpy()); plt.figure(); plt.imshow(b_s[0].detach().cpu().numpy(),cmap = 'jet');plt.show()
        batch_loss = 0
        loss_item = []

        B, C, H, W = a.unsqueeze(0).shape

        #-----------通过熵构建mask----------#
        # 计算归一化熵图
        entropy_a = self.entropy_map(a.unsqueeze(0)).view(B, C, H, W)
        # entropy_a = self.mask(a.unsqueeze(0)).view(B, C, H, W)
        entropy_a = F.interpolate(entropy_a, size=(int(H / self.downsample), int(W / self.downsample)),
                                    mode='bilinear', align_corners=False)
        ind_a = (entropy_a.reshape(-1) > self.t)

        entropy_b = self.entropy_map(b.unsqueeze(0),kernel_size=5).view(B, C, H, W)
        # entropy_b = self.mask(b.unsqueeze(0)).view(B, C, H, W)
        entropy_b = F.interpolate(entropy_b, size=(int(H / self.downsample), int(W / self.downsample)),
                                    mode='bilinear', align_corners=False)
        ind_b = (entropy_b.reshape(-1) > self.t)
        # import matplotlib.pyplot as plt;
        # plt.figure();plt.imshow((entropy_a[0,0]).detach().cpu().numpy());
        # plt.figure();plt.imshow((entropy_b[0,0]).detach().cpu().numpy());
        # plt.show()

        #-----------通过熵构建mask----------#

        if self.usp >= 0:
            usp_loss = self.usp * self.usp_loss(a_s, b_s, key_dist)
            batch_loss += usp_loss
            loss_item.append(usp_loss.item())
        else:
            loss_item.append(0.)

        if self.uni_xy > 0:
            uni_xy_loss = self.uni_xy * self.uni_xy_loss(a_p, b_p)
            batch_loss += uni_xy_loss
            loss_item.append(uni_xy_loss.item())
        else:
            loss_item.append(0.)

        if self.desc > 0:
            desc_loss = self.desc * self.desc_loss(a_d, ind_a, b_d, ind_b, key_dist)
            batch_loss += desc_loss
            loss_item.append(desc_loss.item())
        else:
            loss_item.append(0.)

        if self.decorr >= 0:
            decorr_loss = self.decorr * self.decorr_loss(a_d,ind_a,b_d,ind_b)
            batch_loss += decorr_loss
            loss_item.append(decorr_loss.item())
        else:
            loss_item.append(0.)

        if self.struct >= 0:
            struct_loss = self.struct * self.struct_loss(entropy_b,b_s)
            batch_loss += torch.sum(struct_loss)
            loss_item.append(struct_loss.item())
        else:
            loss_item.append(0.)

        print('usp_loss: %f,uni_xy_loss: %f, desc_loss: %f, decorr_loss: %f, struct_loss: %f' %
                  (usp_loss, uni_xy_loss, decorr_loss, decorr_loss, struct_loss))
        print('usp_loss: %f,uni_xy_loss: %f, desc_loss: %f, decorr_loss: %f' %
                  (usp_loss, uni_xy_loss, decorr_loss, decorr_loss))
        return batch_loss, np.array(loss_item)


    def get_position(self, p_map, cell, downsample, flag=None, mat=None):
        res = (cell + p_map) * downsample

        if flag == 'A':
            r = torch.zeros_like(res)  # r用来存储特征点变换后的坐标
            denominator = res[0, :, :] * mat[2, 0] + res[1, :, :] * mat[2, 1] + mat[2, 2]

            r[0, :, :] = (res[0, :, :] * mat[0, 0] + res[1, :, :] * mat[0, 1] + mat[0, 2]) / denominator
            r[1, :, :] = (res[0, :, :] * mat[1, 0] + res[1, :, :] * mat[1, 1] + mat[1, 2]) / denominator

            return r
        else:
            return res

    def get_dis(self, p_a, p_b):
        c = p_a.shape[0]
        reshape_pa = p_a.reshape((c, -1)).permute(1, 0)  # c h w -> c p
        reshape_pb = p_b.reshape((c, -1)).permute(1, 0)

        x = torch.unsqueeze(reshape_pa[:, 0], 1) - torch.unsqueeze(reshape_pb[:, 0], 0)  # c p -> c p 1 - c 1 p -> c p p
        y = torch.unsqueeze(reshape_pa[:, 1], 1) - torch.unsqueeze(reshape_pb[:, 1], 0)
        dis = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + self.eps)
        return dis

    def usp_loss(self, a_s, b_s, dis):
        reshape_as_k, reshape_bs_k, d_k = self.get_point_pair(a_s, b_s, dis)  # p -> k
        position_k_loss = torch.mean(d_k)  # 最小化距离函数，监督offset
        score_k_loss = torch.mean(torch.pow(reshape_as_k - reshape_bs_k, 2))  # 监督分数一致性
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(a_s[0,:,:].detach().cpu().numpy())
        # plt.show()

        sk_ = (reshape_as_k + reshape_bs_k) / 2
        d_ = torch.mean(d_k)
        usp_k_loss = torch.mean(sk_ * (d_k - d_))  # 可重复性监督，分数高的地方->距离就小。分数低的地方->距离就大

        position_k_loss = position_k_loss * self.position_weight
        score_k_loss = score_k_loss * self.score_weight
        usp_k_loss = usp_k_loss * self.rep_weight

        total_usp = position_k_loss + score_k_loss + usp_k_loss
        return total_usp

    def get_point_pair(self, a_s, b_s, dis):
        a2b_min_id = torch.argmin(dis, dim=1)
        len_p = len(a2b_min_id)
        ch = dis[list(range(len_p)), a2b_min_id] < self.correspond
        reshape_as = a_s.reshape(-1)
        reshape_bs = b_s.reshape(-1)

        a_s = reshape_as[ch]
        b_s = reshape_bs[a2b_min_id[ch]]
        d_k = dis[ch, a2b_min_id[ch]]

        return a_s, b_s, d_k

    def get_topk_coords_2d(self, score_map, k=10):
        """
        score_map: torch.Tensor, shape [H, W]
        return: list of (y, x) coordinates for top-k response points
        """
        H, W = score_map.shape
        flat = score_map.view(-1)  # 展平成向量
        topk_vals, topk_indices = torch.topk(flat, k=k)

        coords_y = topk_indices // W
        coords_x = topk_indices % W
        coords = list(zip(coords_y.tolist(), coords_x.tolist()))
        return coords

    def uni_xy_loss(self, a_p, b_p):
        c = a_p.shape[0]
        reshape_pa = a_p.reshape((c, -1)).permute(1, 0)  # c h w -> c p -> p c where c=2
        reshape_pb = b_p.reshape((c, -1)).permute(1, 0)
        loss = (self.get_uni_xy(reshape_pa[:, 0]) + self.get_uni_xy(reshape_pa[:, 1]))
        loss += (self.get_uni_xy(reshape_pb[:, 0]) + self.get_uni_xy(reshape_pb[:, 1]))
        return loss

    def get_uni_xy(self, position):
        idx = torch.argsort(position)
        idx = idx.float()
        p = position.shape[0]
        uni_l2 = torch.mean(torch.pow(position - (idx / p), 2))
        return uni_l2

    def desc_loss(self, d_a, ind_a, d_b, ind_b, dis):
        c = d_a.shape[0]
        reshape_da = d_a.reshape((c, -1)).permute(1, 0)  # c h w -> c p -> p c
        reshape_db = d_b.reshape((c, -1))  # c h w -> c p
        mask = ind_a[:, None] & ind_b[None, :]

        pos = (dis <= 8) & mask
        neg = (dis > 8) & mask
        ab = torch.mm(reshape_da, reshape_db)  # p c * c p -> p p
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(pos.detach().cpu().numpy())
        # plt.show()
        # 监督图a和图b的相同位置生成相似的描述子
        # margin loss
        # pos = min(ab[pos]) neg = max(ab[neg])
        # loss = max(0, m + (neg - pos))
        margin_loss = (self.m_p - self.m_n) + torch.max(ab[neg]) - torch.min(ab[pos])
        margin_loss = torch.clamp(margin_loss, min=0.0)

        ab[pos] = self.d * (self.m_p - ab[pos])
        ab[neg] = ab[neg] - self.m_n
        ab = torch.clamp(ab, min=0.0)

        # loss = torch.mean(ab) + margin_loss * 0.5
        loss = torch.sum(ab[pos])/len(ab[pos]) + torch.sum(ab[neg])/len(ab[neg]) + margin_loss * 0.5

        return loss

    def decorr_loss(self, d_a, ind_a, d_b, ind_b):
        c, h, w = d_a.shape
        reshape_da = d_a.reshape((c, -1))  # .permute(1, 0)  # c h w -> c p
        reshape_db = d_b.reshape((c, -1))  # .permute(1, 0)

        mask_da = reshape_da[:,torch.where(ind_a == True)[0]]
        mask_db = reshape_db[:,torch.where(ind_b == True)[0]]

        loss = self.get_r_b(mask_da)
        loss += self.get_r_b(mask_db)
        return loss

    def struct_loss(self, entropy_map, score_map, high_thr=0.25, low_thr=0.1): # MIT: 0.4, 0.2
        # import matplotlib.pyplot as plt;plt.figure();plt.imshow(high_mask[0,0].detach().cpu().numpy())
        # plt.figure();plt.imshow(entropy_map[0,0].detach().cpu().numpy(),cmap = 'jet')
        # plt.figure();plt.imshow(score_map[0,0].detach().cpu().numpy(),cmap = 'jet')
        # plt.show()
        score_map = score_map.unsqueeze(0)

        # 结构掩膜（高熵区域）
        high_mask = (entropy_map > high_thr).float()
        # 背景掩膜（低熵区域）
        low_mask = (entropy_map < low_thr).float()
        # 结构区域 loss：score 应该和熵一致（MSE/L1）
        structure_loss = F.mse_loss(score_map * high_mask, entropy_map * high_mask)
        # 背景区域 loss：score 应该趋近于 0
        suppress_loss = F.l1_loss(score_map * low_mask, torch.zeros_like(score_map))
        # 总损失 = 结构对齐 + 背景抑制
        total_loss = 2* structure_loss + suppress_loss
        return total_loss

    def get_r_b(self, reshape_d):
        f, p = reshape_d.shape

        # 监督不同位置描数子整体相关性, -1~1 -> 0~2,
        rs = torch.mm(reshape_d.transpose(1, 0), reshape_d) + 1
        ys = rs - 2 * torch.eye(p, device=reshape_d.device)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(rs.detach().cpu().numpy())
        # plt.show()
        loss = torch.mean(ys)

        return loss

    def predict(self, img):
        s1, p1, d1 = self.forward(img)
        batch_size = s1.shape[0]
        # position1 = self.get_batch_position(p1)
        position1 = self.get_position(p1, self.cell, self.downsample)
        position1 = position1.reshape((batch_size, 2, -1)).permute(0, 2, 1)  # B * (HW) * 2
        s1 = s1.reshape((batch_size, -1))
        c = d1.shape[1]
        d1 = d1.reshape((batch_size, c, -1)).permute(0, 2, 1)  # B * (HW) * c

        output_dict = {}
        for i in range(batch_size):
            s1_ = s1[i, ...].cpu().numpy()
            p1_ = position1[i, ...].cpu().numpy()
            d1_ = d1[i, ...].cpu().numpy()
            output_dict[i] = {'s1': s1_, 'p1': p1_, 'd1': d1_}
        return output_dict

    def norm(self, data):
        return 2 * (data - data.min())/(data.max() - data.min()) - 1





def get_sym(model_config, image_shape):
    # base_model = UnsuperVggTiny()# UnsuperShortcut()
    # base_model = ViT(patch_size=4,
    #         dim=128,
    #         depth=4,
    #         heads=1,
    # #         mlp_dim=64)
    # base_model = nn.Sequential(InverseGain()
    #                 ,ResNet())
    base_model = ResNet()
    # base_model = WaveSenseNet(in_channels=1)
    # base_model = UNet(n_channels=1, n_classes=128)
    model = UnSuperPoint(base_model=base_model, model_config=model_config, IMAGE_SHAPE=image_shape)
    return model