import os
import torch
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.spatial import distance
from registration import *
from Unsuper.utils import common_utils
from Unsuper.configs.config import cfg, cfg_from_list, cfg_from_yaml_file
from symbols.get_model import get_sym

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='../Unsuper/configs/Unsuper.yaml',
                        help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--map', trpe=str, default='Route_1/run_0038/run.npy', nargs=argparse.REMAINDER,
                        help='map data used for test')
    parser.add_argument('--query', trpe=str, default='Route_1/run_0076/run.npy', nargs=argparse.REMAINDER,
                        help='query data used for test')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg

def load_radar_image(data_path, idx_start, idx_end, shape):
    """ 读取雷达数据并调整大小 """
    img = np.load(data_path)[idx_start:idx_end, :, 0].T  # 读取雷达数据
    new_h, new_w = shape
    img_resized = np.expand_dims(cv2.resize(img.astype(np.float32), (new_w, new_h)), axis=-1)
    return torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()

def extract_features(model, img, channel_gap = 1.5):
    """ 通过模型提取特征点、描述子和置信度 """
    pred_dict = model.predict(img)
    keypoints = pred_dict[0]['p1']  # 关键点坐标 (N, 2)
    descriptors = pred_dict[0]['d1']  # 描述子 (N, 128)
    scores = pred_dict[0]['s1']  # 置信度
    return keypoints, descriptors, scores

def match_features(descriptors1, descriptors2, keypoints1, keypoints2, dist_threshold=0.8):
    """ 使用最近邻匹配特征点 """
    matches = []
    for i, desc1 in enumerate(descriptors1):
        dists = np.linalg.norm(descriptors2 - desc1, axis=1)  # 计算 L2 距离
        min_idx = np.argmin(dists)
        if dists[min_idx] < dist_threshold:
            matches.append((i, min_idx, dists[min_idx]))  # (第一张图的索引, 第二张图的索引, 距离)
    return matches

def calculate_utm_error(matches, keypoints1, keypoints2, utm1, utm2):
    """ 计算匹配误差（根据 UTM 坐标） """
    errors = []
    for idx1, idx2, _ in matches:
        pos1 = keypoints1[idx1]
        pos2 = keypoints2[idx2]
        utm_diff = np.linalg.norm(utm1 - utm2)  # UTM 坐标的误差
        pixel_diff = np.linalg.norm(pos1 - pos2)  # 像素坐标的误差
        errors.append((utm_diff, pixel_diff))
    return errors


def compute_inlier_ratio_2d(matches, keypoints1, keypoints2, transformation_matrix, threshold=3.0):
    """
    计算匹配点对的内点率（Inlier Ratio）。

    参数:
        matches: 匹配点对的列表。
        keypoints1: 第一张图的关键点数组。
        keypoints2: 第二张图的关键点数组。
        transformation_matrix: 用于评估匹配质量的变换矩阵（如单应性矩阵或基础矩阵）。
        threshold: 误差阈值，低于此值的匹配视为内点。

    返回:
        inlier_ratio: 内点率（内点数 / 总匹配数）。
        inliers: 内点的索引列表。
    """
    if transformation_matrix is None or len(matches) == 0:
        return 0, []

    src_pts = np.float32([keypoints1[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx] for m in matches]).reshape(-1, 1, 2)

    # 计算变换误差
    projected_pts = cv2.perspectiveTransform(src_pts, transformation_matrix)  # 透视变换
    errors = np.linalg.norm(projected_pts - dst_pts, axis=2).flatten()

    # 计算内点
    inliers = np.where(errors < threshold)[0]
    inlier_ratio = len(inliers) / len(matches)

    return inlier_ratio, inliers


def compute_inlier_ratio_3d(result, source, target, distance_threshold=0.5):
    """
    计算点云匹配的内点率（Inlier Ratio）。

    参数:
        result: Open3D RANSAC 结果 (o3d.pipelines.registration.RegistrationResult)
        source: 源点云 (o3d.geometry.PointCloud)
        target: 目标点云 (o3d.geometry.PointCloud)
        distance_threshold: 内点判断阈值（单位：米）

    返回:
        inlier_ratio: 内点率（内点数 / 总匹配点数）
        inlier_count: 内点数量
    """
    # 获取 RANSAC 匹配的点对
    correspondence_set = np.array(result.correspondence_set)

    if len(correspondence_set) == 0:
        return 0, 0

    # 变换源点云的匹配点
    source_points = np.asarray(source.points)[correspondence_set[:, 0]]
    target_points = np.asarray(target.points)[correspondence_set[:, 1]]

    # 计算匹配误差（欧式距离）
    transformed_source_points = np.dot(result.transformation[:3, :3], source_points.T).T + result.transformation[:3, 3]
    errors = np.linalg.norm(transformed_source_points - target_points, axis=1)

    # 计算内点数量
    inliers = np.where(errors < distance_threshold)[0]
    inlier_ratio = len(inliers) / len(correspondence_set)

    return inlier_ratio, len(inliers)


def thresold_estimate(keypoints1, keypoints2, des1, des2, score1, score2, num, inlier = 3):
    '''测试不同数量的特征点数情况，正确匹配的点数'''
    sorted_ind1 = np.argsort(score1)[::-1]
    sorted_ind2 = np.argsort(score2)[::-1]

    loc1 = sorted_ind1[:num]
    loc2 = sorted_ind2[:num]

    top_kp1 = keypoints1[loc1]; top_des1 = des1[loc1]
    top_kp2 = keypoints2[loc2]; top_des2 = des2[loc2]
    '''match'''
    matcher = Matcher()
    matches = matcher.pc(top_kp1,top_kp2,top_des1,top_des2)

    '''compute inlier ratio'''
    inlier_ratio, inliers = compute_inlier_ratio_3d(matches, top_kp1, top_kp2, matches.transformation)

    return matches, inlier_ratio, inliers





def main():
    args, cfg = parse_config()
    ckpt_path = '../../output/ckpt/checkpoint_epoch_7.pth'

    # 加载模型
    model = get_sym(model_config=cfg['MODEL'], image_shape=cfg['data']['IMAGE_SHAPE'], is_training=False)
    model.load_params_from_file(filename=ckpt_path, logger=None, to_cpu=False)
    model.cuda()
    model.eval()

    # 选择两段雷达数据（可以更改索引）
    map_path = os.path.join(cfg['data']['data_path'], args.map)
    query_path = os.path.join(cfg['data']['data_path'], args.query)
    cfg.seqlen


    img1 = load_radar_image(map_path, 8600, 8900, cfg['data']['IMAGE_SHAPE'])
    img2 = load_radar_image(query_path, 9600, 9900, cfg['data']['IMAGE_SHAPE'])

    # 提取特征点
    keypoints1, descriptors1, scores1 = extract_features(model, img1)
    keypoints2, descriptors2, scores2 = extract_features(model, img2)

    # 过滤低置信度的特征点
    conf_thresh = 0.7
    valid1 = scores1 > conf_thresh
    valid2 = scores2 > conf_thresh
    keypoints1, descriptors1 = keypoints1[valid1], descriptors1[valid1]
    keypoints2, descriptors2 = keypoints2[valid2], descriptors2[valid2]

    # 进行特征匹配
    matches = match_features(descriptors1, descriptors2, keypoints1, keypoints2)

    # 这里 UTM 坐标需要你提供
    utm1 = np.array([123456.7, 234567.8])  # 示例 UTM 坐标
    utm2 = np.array([123458.2, 234569.5])

    # 计算 UTM 误差
    errors = calculate_utm_error(matches, keypoints1, keypoints2, utm1, utm2)

    # 打印匹配结果
    print(f"总共找到 {len(matches)} 对匹配点")
    for i, (utm_err, pixel_err) in enumerate(errors[:10]):  # 仅显示前10个
        print(f"Match {i+1}: UTM 误差 = {utm_err:.3f} m, 像素误差 = {pixel_err:.3f} px")

    # 可视化匹配结果
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img1.squeeze().cpu().numpy(), cmap='gray')
    ax[0].set_title("Image 1")
    ax[1].imshow(img2.squeeze().cpu().numpy(), cmap='gray')
    ax[1].set_title("Image 2")

    # 画出匹配点
    for idx1, idx2, _ in matches:
        ax[0].scatter(*keypoints1[idx1], color='r', marker='x')
        ax[1].scatter(*keypoints2[idx2], color='b', marker='o')

    plt.show()

if __name__ == '__main__':
    main()
