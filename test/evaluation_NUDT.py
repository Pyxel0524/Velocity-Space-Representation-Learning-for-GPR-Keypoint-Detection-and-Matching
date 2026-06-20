import os.path

import numpy as np
import tqdm
from export_kp_des import *
from symbols.migrationnet_model import MigrationNet
from symbols.PARNet import HourglassNet
import matplotlib.cm as cm  # 伪彩色映射
import matplotlib.pyplot as plt
import open3d as pcd
import time

def volume_viz_pc(vol, coords, down_sample = 1):
    vol = vol[:, ::down_sample, ::down_sample]
    coords = coords[::down_sample, ::down_sample, :]
    points = []
    intensities = []
    for c in range(vol.shape[0]):
        for x in range(vol.shape[1]):
            for y in range(vol.shape[2]):
                intensity = vol[c, x, y]
                points.append(np.array([coords[y,x,c][0], coords[y,x,c][1], coords[y,x,c][2]]))
                # points.append(np.array([c, x/5, y]))
                intensities.append(intensity)

    pcd = o3d.geometry.PointCloud()
    intensities = np.array(intensities).astype('int16')
    intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-6)
    colors = cm.viridis(intensities)[:, :3]
    # colors = cm.jet(intensities)[:, :3]
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    view_ctl = vis.get_view_control()
    view_ctl.set_front([0.5, 0.5, -1.0])
    view_ctl.set_up([0, 1, 0])
    # view_ctl.set_lookat([230, 50, 50])
    view_ctl.set_zoom(0.5)

    vis.run()
    vis.destroy_window()





def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument( "--cfg_file", type=str, default="../Unsuper/configs/Unsuper.yaml", help="specify the config for training")
    parser.add_argument( "--kps", type=str, default="Silk", required=False, help="VIFT,MigrationNet,ParNet,SIFT,ORB")
    parser.add_argument( "--des", type=str, default="VIFT", required=False, help="Method for Extract Descriptor")
    parser.add_argument( "--depth_scale", type=int, default=50, required=False, help="Used for evaluate repeatablility and registration")
    parser.add_argument("--batch_size", type=int, default=1, required=False, help="batch size for training")
    parser.add_argument("--workers", type=int, default=4, help="number of workers for dataloader")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint to start from")
    parser.add_argument("--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER, help="set extra config keys if needed")
    parser.add_argument("--start_epoch", type=int, default=0, help="")
    parser.add_argument("--eval_all", action="store_true", default=True, help="whether to evaluate all checkpoints")
    parser.add_argument("--ckpt_dir", type=str,default=None, help="specify a ckpt directory to be evaluated if needed")
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

if __name__ == "__main__":

    dist_test = False
    args, cfg = parse_config()
    eval_out_dir = './test_image_log/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.exists(eval_out_dir):
        print('dir exists')
    else:
        os.mkdir(eval_out_dir, 777)
    logger = common_utils.create_logger(eval_out_dir + 'eval_image.txt')
    config = cfg['Evaluate_NUDT']

    if args.kps == 'MigrationNet':
        ckpt_dir = '../output/ckpt/Migration_epoch381.pth'
        model = MigrationNet(n_channels=256, n_classes=1)
        model.load_state_dict(torch.load(ckpt_dir, map_location=device))
        model.cuda()
        model.eval()
    elif args.kps == 'ParNet':
        ckpt_dir = '../output/ckpt/HourglassNet_epoch101.pth'
        model = HourglassNet(n_channels=3, n_classes=1)
        model.load_state_dict(torch.load(ckpt_dir, map_location=device))
        model.cuda()
        model.eval()
    elif args.kps == 'VIFT':
        ckpt_dir = '../output/ckpt/VIFT_NUDT.pth'
        model = get_sym(model_config=cfg['MODEL'], image_shape=cfg['MIT_data']['IMAGE_SHAPE'])
        model.load_state_dict(torch.load(ckpt_dir, map_location=device)['model_state'])
        # model.load_params_from_file(filename=ckpt_dir, logger=logger, to_cpu=dist_test)
        model.cuda()
        model.eval()
    elif args.kps == 'Silk':
        ckpt_dir = '../output/ckpt/silk.pth'
        model = get_sym(model_config=cfg['MODEL'], image_shape=cfg['CMU_data']['IMAGE_SHAPE'])
        model.load_state_dict(torch.load(ckpt_dir, map_location=device)['model_state'])
        model.cuda()
        model.eval()
    else:
        ckpt_dir = '../output/ckpt/VIFT_NUDT.pth' # ablation: VIFT_MIT_withoutusp.pth, VIFT_MIT_withoutRegularDecorr.pth, silk (without decorr)
        model = get_sym(model_config=cfg['MODEL'], image_shape=cfg['MIT_data']['IMAGE_SHAPE'])
        model.load_state_dict(torch.load(ckpt_dir, map_location=device)['model_state'])
        # model.load_params_from_file(filename=ckpt_dir, logger=logger, to_cpu=dist_test)
        model.cuda()
        model.eval()

    # 用来组合深度学习和描述子模型
    if args.des == 'Silk':
        ckpt_dir = '../output/ckpt/silk.pth'# silk
    else:
        ckpt_dir = '../output/ckpt/VIFT_NUDT.pth'  # VIFT_MIT_withoutRegularDecorr.pth, MIT_checkpoint_epoch_46.pth, VIFT_MIT_withoutusp.pth
    descmodel = get_sym(model_config=cfg['MODEL'], image_shape=cfg['MIT_data']['IMAGE_SHAPE'])
    descmodel.load_state_dict(torch.load(ckpt_dir, map_location=device)['model_state'])
    # model.load_params_from_file(filename=ckpt_dir, logger=logger, to_cpu=dist_test)
    descmodel.cuda()
    descmodel.eval()
    # 用来组合深度学习和描述子模型


    radar_1 = np.load(os.path.join(config.data_path, config.map_name, 'run.npy')) # map
    point_utm1 = np.load(os.path.join(config.data_path, config.map_name, 'run_UTM_point.npy'))
    radar_2 = np.load(os.path.join(config.data_path, config.query_name, 'run.npy')) # query
    point_utm2 = np.load(os.path.join(config.data_path, config.query_name, 'run_UTM_point.npy'))

    total_len = len(radar_2)
    start_ind = config.seq_len; end_ind = total_len - config.seq_len
    offset = start_ind - np.argmin(np.linalg.norm(point_utm1[start_ind,5] - point_utm2[:,5],axis=1))


    inliers_save = []; repeatability_save = []; translation_err_save = []; rotation_err_save = [];
    with torch.no_grad():
        matcher = Matcher(config.regist_thre)  # 匹配器
        for i in range(start_ind, end_ind, config.stride):# start_ind, end_ind, config.stride
            print('=============================index_number:', i)
            radar_echo1 = radar_1[i + offset:i + offset + config.seq_len,:,].T
            radar_point_utm1 = point_utm1[i + offset:i + offset + config.seq_len,:]
            radar_echo2 = radar_2[i :i + config.seq_len,:,].T
            radar_point_utm2 = point_utm2[i :i + config.seq_len,:]

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(radar_echo1[0,:,:])
            # plt.show()

            points1, descriptors1 = extract_kp_des(config, model, descmodel, radar_echo1, radar_point_utm1,
                                                   channel_num = 15, keypoint_num=config.keypoint_num, kp_method=args.kps, desc_method=args.des, nms=False, depth_scale = args.depth_scale ,downsample=cfg['MODEL']['downsample'])
            points2, descriptors2 = extract_kp_des(config, model, descmodel, radar_echo2, radar_point_utm2,
                                                   channel_num = 15, keypoint_num=config.keypoint_num, kp_method=args.kps, desc_method=args.des, nms=False, depth_scale = args.depth_scale ,downsample=cfg['MODEL']['downsample'])


            # feature matching
            result = matcher.pcr(points1, points2, descriptors1, descriptors2, viz = False, mufilter=False)
            print('correspondence_number: ', len(result.correspondence_set))

            # calculate repeatablity
            repeatability, repeats = matcher.compute_repeatablity(points1, points2, threshold = config.repeat_thre)
            repeatability_save.append(repeatability)
            print('repeatability: %.4f' % (repeatability))

            # calculate localization error
            translation_err, rotation_err = matcher.Relative_Error(radar_point_utm1, radar_point_utm2, result.transformation)
            translation_err_save.append(translation_err)
            rotation_err_save.append(rotation_err)
            print('translation_error: %.4f, rotation_error: %.4f' % (translation_err, rotation_err))

            # calculate inlier ratio
            inlier_ratio = matcher.compute_inlier_ratio(points1, points2, result.correspondence_set,
                                                        radar_point_utm1, radar_point_utm2,
                                                        distance_threshold=config.inlier_thre, viz=True, ind = i)

            # local to map animation
            # plt.figure()
            # plt.plot(point_utm1[:,5,0],point_utm1[:,5,1], 'b-')
            # plt.scatter(radar_point_utm2[150,5,0],radar_point_utm2[150,5,1], c='hotpink', marker='*', s=100)
            # # 添加图例
            # plt.axis('off')  # 关闭坐标轴（包括刻度和边框）
            # plt.savefig(f"../Animation/Traj/GT_traj_{i}.png", dpi=300, bbox_inches='tight')
            # plt.close()


            # raw scan
            # fig, axes = plt.subplots(2, 1)  # 两行一列，figsize控制总大小
            # axes[0].imshow(cv2.resize(radar_echo1[0,:,:].astype(np.float32), (320, 240), interpolation=cv2.INTER_LINEAR))
            # axes[0].axis('off')
            # axes[1].imshow(cv2.resize(radar_echo2[0,:,:].astype(np.float32), (320, 240), interpolation=cv2.INTER_LINEAR))
            # axes[1].axis('off')
            #
            # # 保存图像
            # plt.tight_layout()
            # plt.savefig(f"../Animation/raw_scan/{i}.png", dpi=300, bbox_inches='tight')
            # plt.close()
            # local to map animation


            ############################ volume visualization
            # vol_coord = np.repeat(radar_point_utm2[:, np.newaxis, :, :], 369, axis=1)
            # depth = np.linspace(0, 369/50, 369)     # depth_dimension_scale
            # z_coords = np.tile(depth[:, np.newaxis], (1, 11))  # (369, 11)
            # z_coords = np.tile(z_coords[np.newaxis, :, :, np.newaxis], (300, 1, 1, 1))  # (300, 369, 11, 1)
            # # expand to 3d
            # vol_coords = np.concatenate([vol_coord, z_coords], axis=-1)
            # volume_viz_pc(radar_echo2, vol_coords)
            ############################ volume visualization


            inliers_save.append(inlier_ratio)
            print('inlier_ratio: %.4f' % (inlier_ratio))

            # visualization radar echo


        inlier_name = f"inlier_kp-{args.kps}_desc-{args.des}_Num-{config.keypoint_num}_M-{config.map_name}_Q-{config.query_name}_regist-{config.regist_thre}_inlier-{config.inlier_thre}.npy"
        np.save(os.path.join('../output/eval/inlier/', inlier_name), np.array(inliers_save))

        repeatability_name = f"Repeat_kp-{args.kps}_desc-{args.des}_Num-{config.keypoint_num}_M-{config.map_name}_Q-{config.query_name}_regist-{config.regist_thre}_repeat-{config.repeat_thre}.npy"
        np.save(os.path.join('../output/eval/repeat/', repeatability_name), np.array(repeatability_save))

        trans_name = f"TransErr_kp-{args.kps}_desc-{args.des}_Num-{config.keypoint_num}_M-{config.map_name}_Q-{config.query_name}_regist-{config.regist_thre}.npy"
        np.save(os.path.join('../output/eval/localization/', trans_name), np.array(translation_err_save))

        rotation_name = f"RotationErr_kp-{args.kps}_desc-{args.des}_Num-{config.keypoint_num}_M-{config.map_name}_Q-{config.query_name}_regist-{config.regist_thre}.npy"
        np.save(os.path.join('../output/eval/localization/', rotation_name), np.array(rotation_err_save))

