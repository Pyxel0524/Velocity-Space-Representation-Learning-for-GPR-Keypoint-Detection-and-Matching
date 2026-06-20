import os

import numpy as np
import torch
import glob
import tqdm
import argparse
from pathlib import Path
from Unsuper.dataset import build_dataloader
from Unsuper.utils import common_utils, utils
from Unsuper.configs.config import (
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    log_config_to_file,
)
from symbols.VIFT import get_sym
import cv2
from registration_eval.registration import *
import torch.nn.functional as F
import matplotlib.cm as cm
import matplotlib.pyplot as plt


class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="../Unsuper/configs/Unsuper.yaml",
        help="specify the config for training",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="SIFT",
        required=False,
        help="Method for Extract Feature and Descriptor",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        required=False,
        help="batch size for training",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="number of workers for dataloader"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="checkpoint to start from"
    )
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="set extra config keys if needed",
    )
    parser.add_argument("--start_epoch", type=int, default=0, help="")
    parser.add_argument(
        "--eval_all",
        action="store_true",
        default=True,
        help="whether to evaluate all checkpoints",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="specify a ckpt directory to be evaluated if needed",
    )
    parser.add_argument("--save_to_file", action="store_true", default=False, help="")

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    # cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(
        args.cfg_file.split("/")[1:-1]
    )  # remove 'cfgs' and 'xxxx.yaml'
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def norm(data):
    norm_data = 2*((data - data.min()) / (data.max() - data.min())) -1
    return norm_data

def heatmap_viz(echo, score_map):
    # translation degree
    echo_ = np.expand_dims(echo, axis=2)
    echo__ = echo_.repeat(3, axis = 2)
    alpha = 0.2
    heatmap = cm.jet(score_map)
    heatmap_ = heatmap[..., :3]
    overlay = ((1 - alpha) * 0.4* (echo__+1)) + alpha * heatmap_
    overlay = np.clip(overlay, 0, 1)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(overlay, cmap = 'jet');plt.colorbar()
    plt.show()


def get_topk_coords(score_map, k=10):
    """
    从二维分数图中获取前 k 个响应点的坐标。
    参数:
    score_map: np.ndarray, 形状为 [H, W] 的二维数组。
    k: int, 要获取的最大响应点的数量。
    返回:
    coords: 包含前 k 个响应点坐标的列表，每个坐标为 (y, x)。
    """
    H, W = score_map.shape
    flat_indices = np.argpartition(score_map.ravel(), -k)[-k:]  # 获取前 k 个最大值的扁平索引
    return flat_indices


def extract_kp_des(cfg, kpmodel, descmodel, raw_echo, point_utm, channel_num, keypoint_num, kp_method='model', desc_method='model',
                   nms=False, downsample =4, depth_scale = 50):
    print("--------------------------- Extract Keypoint and Descriptor ---------------------------")

    positions = []
    descriptors = []
    if len(raw_echo.shape) == 2:
        channel_num = 1
        depth, length = raw_echo.shape
        raw_echo = raw_echo.reshape(1,depth,length)
    else:
        channel, depth, length = raw_echo.shape


    for i in range(channel_num):
        # resize
        radar_echo = torch.tensor(np.expand_dims(cv2.resize(raw_echo[i, :, :].astype(np.float32),
                                                            (cfg['IMAGE_SHAPE'][1], cfg['IMAGE_SHAPE'][0])),
                                                 axis=-1), dtype=torch.float32) / 255.0
        # 这里除255归一化，对应dataloader
        radar_echo = torch.unsqueeze(radar_echo, 0)
        radar_echo = radar_echo.permute(0, 3, 1, 2)
        radar_echo = radar_echo.cuda()

        if (kp_method == 'VIFT') or (kp_method == 'Silk'):
            # keypoint and descriptor combo
            kp_pred = kpmodel.predict(norm(radar_echo))[0]  # norm(radar_echo)
            # keypoint and descriptor combo
            score = kp_pred["s1"]
            # nms extract keypoint resize 240 * 320
            # score_map = F.interpolate(torch.tensor(score.reshape(1, 1, 60, 80)), size=(240, 320), mode='bilinear', align_corners=False)
            # position, topk_ind = utils.nms(score_map, threshold=0.01, dist=1, top_k=keypoint_num, edge_width=1, downsample = downsample)

            # nms extract keypoint 自己预测特征点
            position, topk_ind = utils.nms(torch.tensor(score.reshape(1, 1, 60, 80)), threshold=0.1, dist=1, top_k=keypoint_num, edge_width=1, downsample = downsample)
            position = kp_pred["p1"][topk_ind]# 用vift相当于nms=1，threshold = 0
            t = 0

            # plt.figure()
            # plt.imshow(score_map[0,0],cmap = 'jet');plt.colorbar();plt.show()
            ## 透明 colormap
            # plt.figure()
            # cmap = plt.cm.jet
            # rgba = cmap(score_map[0,0])  # shape: (H, W, 4)
            ## 将 alpha 通道替换为原始值（或其变换）——数值越小越透明
            # rgba[..., 3] = score_map[0,0]
            # plt.imshow(rgba); plt.show()

            # score_map = F.interpolate(torch.tensor(score.reshape(1, 1, 60, 80)), size=(240, 320), mode='bilinear', align_corners=False)
            # plt.figure()
            # plt.imshow(norm(radar_echo)[0,0,:,:].detach().cpu().numpy());plt.colorbar()
            # plt.figure()
            # plt.imshow(score_map[0,0],cmap = 'jet');plt.colorbar()
            # plt.plot(position[:,0],position[:,1],'r+')
            # plt.show()




        elif kp_method == 'MigrationNet':
            radar_echo_ = norm(radar_echo).repeat(1, 256, 1, 1)
            sample_echo = F.interpolate(radar_echo_, size=(62, 87), mode='bilinear', align_corners=False)
            score_map = kpmodel(sample_echo); score_map = -1 * score_map
            # with handicraft
            # heat_map = F.interpolate(score_map, size=(240, 320), mode='bilinear', align_corners=False)
            # position, topk_ind = utils.nms(heat_map, threshold=0.1, dist=1, top_k=keypoint_num, edge_width=10)
            # # with handicraft

            # with learning
            heat_map = F.interpolate(score_map, size=(60, 80), mode='bilinear', align_corners=False)# with learning
            position, topk_ind = utils.nms(heat_map, threshold=0.1, dist=1, top_k=keypoint_num, edge_width=1)
            position = position*4
            # with learning

            t = 0
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(heat_map[0,0].detach().cpu().numpy(),cmap = 'jet');plt.colorbar()
            # plt.figure()
            # plt.imshow(radar_echo[0,0].detach().cpu().numpy());plt.colorbar()
            # plt.plot(position[:,0],position[:,1],'r+')
            # plt.show()

        elif kp_method == 'ParNet':
            radar_echo_ = radar_echo.repeat(1, 3, 1, 1)
            score_map = torch.sum(kpmodel.predict(norm(radar_echo_)), dim=1)

            # with handicraft
            # position, topk_ind = utils.nms(score_map, threshold=0.1, dist=1, top_k=keypoint_num, edge_width=10)
            # with handicraft

            # with learning
            heat_map = F.interpolate(score_map.unsqueeze(0), size=(60, 80), mode='bilinear', align_corners=False)# with learning
            position, topk_ind = utils.nms(heat_map, threshold=0.1, dist=1, top_k=keypoint_num, edge_width=1)
            position = position*4
            # import matplotlib.pyplot as plt
            # plt.figure(); plt.imshow(radar_echo[0,0].detach().cpu().numpy()); plt.plot(position[:,0],position[:,1],'r+')
            # plt.figure(); plt.imshow(score_map[0].detach().cpu().numpy(),cmap = 'jet')
            # plt.show()
            t = 0
        else:
            radar_echo_ = cv2.resize(raw_echo[i, :, :].astype(np.float32),
                                    (cfg['IMAGE_SHAPE'][1], cfg['IMAGE_SHAPE'][0]), interpolation=cv2.INTER_LINEAR)
            position = extract_kps(cv2.resize(radar_echo_,
                                              (cfg['IMAGE_SHAPE'][1], cfg['IMAGE_SHAPE'][0])), kp_method,
                                   keypoint_num)
            if len(position) == 0:
                print('---------------------Without Keypoint--------------------')
                position = np.zeros((1, 2))
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(radar_echo_)
            # plt.plot(position[:,0],position[:,1],'r+')
            # plt.show()
        if (desc_method == 'VIFT') or (desc_method == 'Silk'):
            desc_pred = descmodel.predict(norm(radar_echo))[0]
            if 'topk_ind' in locals():
                desc = desc_pred["d1"][topk_ind]
            else:
                position_scale = (position/4).astype('int16'); shape = (80,60)
                topk_ind = np.ravel_multi_index(position_scale.T, dims = shape)
                desc = desc_pred["d1"][topk_ind]
                # import matplotlib.pyplot as plt
                # score_pred = desc_pred["s1"].reshape(60,80)
                # plt.figure()
                # plt.imshow(score_pred, cmap = 'jet')
                # plt.plot(position_scale[:,0],position_scale[:,1],'r+')
                # plt.show()
        else:
            radar_echo = cv2.resize(raw_echo[i, :, :].astype(np.float32),
                                    (cfg['IMAGE_SHAPE'][1], cfg['IMAGE_SHAPE'][0]), interpolation=cv2.INTER_LINEAR)
            desc = extract_desc(cv2.resize(radar_echo,
                                           (cfg['IMAGE_SHAPE'][1], cfg['IMAGE_SHAPE'][0])), position, desc_method)
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(radar_echo)
            # plt.plot(position[:100,0],position[:100,1],'r+')
            # plt.show()
        # rescale回原索引
        position[:, 1] = (position[:, 1] * (depth / cfg['IMAGE_SHAPE'][0]) - 1).astype('int32')  # 深度维的缩放
        position[:, 0] = (position[:, 0] * (length / cfg['IMAGE_SHAPE'][1]) - 1).astype('int32')
        # rescale回原索引

        # 按照像素索引找到真实索引
        if channel_num > 1:
            position_utm = point_utm[:, i, :][np.array(position[:, 0], dtype='int32')]
            scale = depth_scale
            position_utm = np.hstack((position_utm, np.expand_dims(position[:, 1] / scale, 1)))  # 深度维缩放
        else:
            position_utm = position
        positions.append(position_utm)
        if desc is None:
            desc = np.zeros((1, 32))
        descriptors.append(desc)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(raw_echo[0,:,:])
    # plt.show()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection = '3d')
    # # 绘制散点图
    # ax.scatter(np.array(positions)[:,:,0],np.array(positions)[:,:,1],np.array(positions)[:,:,2])
    # ax.set_xlabel('X')
    # ax.set_xlabel('Y')
    # ax.set_xlabel('Z')
    # plt.title('pcl')
    # plt.show()

    return np.vstack(positions), np.vstack(descriptors)


def extract_kps(radar_echo, method="SIFT", key_number=50, nms = 4):
    if method == "SIFT":
        sift = cv2.SIFT_create()
        radar_echo = 255 * (radar_echo - radar_echo.min()) / (radar_echo.max() - radar_echo.min())
        kp_sift = sift.detect(radar_echo.astype('uint8'))
        kps = sorted(kp_sift, key=lambda x: x.response, reverse=True)

    elif method == "ORB":
        orb = cv2.ORB_create()
        radar_echo = 255 * (radar_echo - radar_echo.min()) / (radar_echo.max() - radar_echo.min())
        kp_orb = orb.detect(radar_echo.astype('uint8'))
        kps = sorted(kp_orb, key=lambda x: x.response, reverse=True)
        # import matplotlib.pyplot as plt
        # plt.figure();plt.imshow(radar_echo)
        # plt.plot(pts[:,0],pts[:,1],'r+')
        # plt.show()
    else:
        raise ValueError("unknown extract keypoint method")

    # pts = np.array([kp.pt for kp in kps])  # (x, y)
    selected = []
    for kp in kps:
        if all(np.linalg.norm(np.array(kp.pt) - np.array(sel)) >= nms for sel in selected):
            selected.append(np.array(kp.pt))
    return np.array(selected[:key_number])


def extract_desc(radar_echo, keypoints, method="SIFT"):
    if method == "SIFT":
        cv2_keypoints = [cv2.KeyPoint(x, y, 1) for x, y in keypoints]
        detector = cv2.SIFT_create()
    elif method == "SURF":
        cv2_keypoints = [cv2.KeyPoint(x, y, 1) for x, y in keypoints]
        detector = cv2.SURF_create(400)
    elif method == "ORB":
        board = 30
        cv2_keypoints = [cv2.KeyPoint(x + board, y + board, 1) for x, y in keypoints]
        detector = cv2.ORB_create()
        radar_echo = cv2.copyMakeBorder(radar_echo, board, board, board, board,
                                        cv2.BORDER_REFLECT)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(radar_echo)
        # plt.show()
    else:
        raise ValueError("unknown extract descriptor method")

    norm_echo = (255 * (radar_echo - radar_echo.min()) / (np.max(radar_echo) - np.min(radar_echo))).astype('uint8')
    keypoints_, descriptors = detector.compute(norm_echo, cv2_keypoints)

    return descriptors


def save_kp_des(cfg, model, dataloader, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)
    logger.info("*************** Export Keypoint and Descriptor *****************")
    iter_step = len(dataloader)
    dataloader_iter = iter(dataloader)

    progress_bar = tqdm.tqdm(
        total=len(dataloader), leave=True, desc="eval", dynamic_ncols=True
    )
    for i in range(iter_step):
        try:
            src_img, dst_img, mat, img_idx = next(dataloader_iter)
        except StopIteration:
            break

        h, w = src_img[0].shape[1:]

        src_img = src_img.cuda()
        dst_img = dst_img.cuda()

        pred_dict_src = model.predict(src_img)
        pred_dict_dst = model.predict(dst_img)

        for j in pred_dict_src.keys():
            suffix = img_idx[j][img_idx[j].rfind("."):]
            img_path = Path(img_idx[j])
            folder = img_path.parent.stem
            img_name = img_path.stem
            data_dir = result_dir / cfg["data"]["export_name"] / folder
            data_dir.mkdir(parents=True, exist_ok=True)

            # for eval
            # cv2.imwrite(os.path.join(str(data_dir), img_name+suffix), img0[j])

            # no nms
            # s1_src = pred_dict_src[j]['s1']
            # loc = np.where(s1_src > 0.5)
            # p1_src = pred_dict_src[j]['p1'][loc]
            # d1_src = pred_dict_src[j]['d1'][loc]
            # s1_src = s1_src[loc]
            #
            # s1_dst = pred_dict_dst[j]['s1']
            # loc = np.where(s1_dst > 0.5)
            # p1_dst = pred_dict_dst[j]['p1'][loc]
            # d1_dst = pred_dict_dst[j]['d1'][loc]
            # s1_dst = s1_dst[loc]

            # with nms
            s1_src = pred_dict_src[j]["s1"]
            s1_src = s1_src.reshape(-1, 1)
            p1_src = pred_dict_src[j]["p1"]
            d1_src = pred_dict_src[j]["d1"]
            input = np.concatenate((s1_src, p1_src, d1_src), axis=1)
            keep = utils.key_nms(input, 8)
            s1_src = input[keep, 0]
            p1_src = input[keep, 1:3]
            d1_src = input[keep, 3:]
            loc_src = np.where(s1_src > 0.5)
            s1_src = s1_src[loc_src]
            p1_src = p1_src[loc_src]
            d1_src = d1_src[loc_src]

            s1_dst = pred_dict_dst[j]["s1"]
            s1_dst = s1_dst.reshape(-1, 1)
            p1_dst = pred_dict_dst[j]["p1"]
            d1_dst = pred_dict_dst[j]["d1"]
            input = np.concatenate((s1_dst, p1_dst, d1_dst), axis=1)
            keep = utils.key_nms(input, 8)
            s1_dst = input[keep, 0]
            p1_dst = input[keep, 1:3]
            d1_dst = input[keep, 3:]
            loc_dst = np.where(s1_dst > 0.5)
            s1_dst = s1_dst[loc_dst]
            p1_dst = p1_dst[loc_dst]
            d1_dst = d1_dst[loc_dst]

            np.savez(
                open(os.path.join(str(data_dir), img_name + ".ppm.usp"), "wb"),
                src_score=s1_src,
                src_point=p1_src,
                src_des=d1_src,
                dst_score=s1_dst,
                dst_point=p1_dst,
                dst_des=d1_dst,
                mat=mat,
                img_wh=(w, h),
            )

        progress_bar.update()

    logger.info("Result is save to %s" % result_dir)
    logger.info("****************Evaluation done.*****************")


if __name__ == "__main__":
    dist_test = False
    args, cfg = parse_config()

    ckpt_dir = '../output/ckpt/checkpoint_epoch_7.pth'

    model = get_sym(model_config=cfg['MODEL'], image_shape=cfg['MIT_data']['IMAGE_SHAPE'], is_training=False)

    eval_out_dir = './test_image_log/'
    if os.path.exists(eval_out_dir):
        print('dir exists')
    else:
        os.mkdir(eval_out_dir, 777)

    logger = common_utils.create_logger(eval_out_dir + 'eval_image.txt')
    config = cfg['Evaluate']

    radar_1 = np.load(os.path.join(config.data_path, config.map_name, 'run.npy'))
    point_utm1 = np.load(os.path.join(config.data_path, config.map_name, 'run_UTM_point.npy'))
    radar_2 = np.load(os.path.join(config.data_path, config.query_name, 'run.npy'))
    point_utm2 = np.load(os.path.join(config.data_path, config.query_name, 'run_UTM_point.npy'))

    total_len = len(radar_1)
    start_ind = config.seq_len;
    end_ind = total_len - config.seq_len
    offset = start_ind - np.argmin(np.linalg.norm(point_utm1[start_ind, 5] - point_utm2[:, 5], axis=1))

    with torch.no_grad():
        model.load_params_from_file(filename=ckpt_dir, logger=logger, to_cpu=dist_test)
        model.cuda()
        model.eval()
        for i in range(start_ind, end_ind, config.seq_len):
            radar_echo1 = radar_1[i:i + config.seq_len, :, ].T
            radar_point_utm1 = point_utm1[i:i + config.seq_len, :]
            radar_echo2 = radar_2[i + offset:i + offset + config.seq_len, :, ].T
            radar_point_utm2 = point_utm2[i + offset:i + offset + config.seq_len, :]

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(radar_echo1[0,:,:])
            # plt.show()
            points1, descriptors1 = extract_kp_des(config, model, radar_echo1, radar_point_utm1, channel_num=11,
                                                   keypoint_num=10, kp_method='ORB', desc_method='ORB', nms=False)
            points2, descriptors2 = extract_kp_des(config, model, radar_echo2, radar_point_utm2, channel_num=11,
                                                   keypoint_num=10, kp_method='ORB', desc_method='ORB', nms=False)

            # feature matching
            matcher = Matcher(config.regist_thre)  # 匹配器
            result = matcher.pcr(points1, points2, descriptors1, descriptors2, mufilter=False)

            # viz
            matcher.visualize_matches(points1, points2, result.correspondence_set, result.transformation,
                                      distance_threshold=config.regist_thre)
