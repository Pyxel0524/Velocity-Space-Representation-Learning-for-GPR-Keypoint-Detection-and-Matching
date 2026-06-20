import os.path
import matplotlib.pyplot as plt
from export_kp_des import *
from symbols.migrationnet_model import MigrationNet
from symbols.PARNet import HourglassNet
import re
import faiss
from matplotlib import cm
import time

def get_coords(file_path):
    pattern = re.compile(r"X_([-+]?[0-9]*\.?[0-9]+)_Y_([-+]?[0-9]*\.?[0-9]+)")
    coords = []
    for filename in os.listdir(file_path):
        if not filename.lower().endswith('.npy'):
            continue
        match = pattern.search(filename)
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            coords.append([x, y])
    return coords

def split_files_by_suffix(folder, suffixes=['png', 'npy']):
    files = os.listdir(folder)
    result = {s: [] for s in suffixes}
    for f in files:
        for s in suffixes:
            if f.lower().endswith(s):
                result[s].append(os.path.join(folder, f))
    return result


def filter_matches(query, query_pts, target, target_pts, inliers, distance_thresh, viz = True):
    h, w = query.shape
    gap = 20

    query = cm.viridis(query / 255.0)[..., :3]
    target = cm.viridis(target / 255.0)[..., :3]

    gap_band = np.ones((gap, w, 3), dtype=np.float32)
    concat_viz = np.vstack((query, gap_band, target))
    offset_y_with_gap = h + gap

    filtered = []
    viz_match = []
    for (x1, y1), (x2, y2), inlier in zip(query_pts, target_pts, inliers):
        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        is_close = (dist <= distance_thresh)
        viz_match.append(((x1, y1), (x2, y2), inlier and is_close))
        if inlier and is_close:
            filtered.append(((x1, y1), (x2, y2), inlier and is_close))

    if viz:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(concat_viz)

        # 绘制匹配点和线
        for (x1, y1), (x2, y2), inlier in viz_match:
            if not inlier:
                ax.plot(x1, y1, 'r*')
                ax.plot(x2, y2 + offset_y_with_gap, 'r*')
                # ax.plot([x1, x2], [y1, y2 + offset_y_with_gap], 'r-', linewidth=1)
            else:
                continue

        for (x1, y1), (x2, y2), inlier in viz_match:
            if inlier:
                ax.plot(x1, y1, 'b*')
                ax.plot(x2, y2 + offset_y_with_gap, 'b*')
                ax.plot([x1, x2], [y1, y2 + offset_y_with_gap], color='#00ff00', linewidth=1)
            else:
                continue

        plt.tight_layout()
        plt.show()
        plt.close()  # 自动关闭
        return filtered
    else:
        return filtered

def norm(data):
    norm_data = ((data - data.min()) / (data.max() - data.min()))
    return 255*norm_data

def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument( "--cfg_file", type=str, default="../Unsuper/configs/Unsuper.yaml", help="specify the config for training")
    parser.add_argument( "--kps", type=str, default="VIFT", required=False, help="VIFT,MigrationNet,ParNet,SIFT,ORB")
    parser.add_argument( "--des", type=str, default="VIFT", required=False, help="Method for Extract Descriptor")
    parser.add_argument( "--type", type=str, default="npy", required=False, help="dataset type")
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
    config = cfg['Evaluate_CMU']

    if args.kps == 'MigrationNet':
        ckpt_dir = '../output/ckpt/Migration_CP_epoch361.pth'
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
    elif args.kps == 'VIFT':#记住vift要改data type
        ckpt_dir = '../output/ckpt/CMU_best.pth'#'../output/ckpt/CMU_best.pth'
        model = get_sym(model_config=cfg['MODEL'], image_shape=cfg['CMU_data']['IMAGE_SHAPE'])
        model.load_state_dict(torch.load(ckpt_dir, map_location=device)['model_state'])
        model.cuda()
        model.eval()
    elif args.kps == 'Silk':#记住vift要改data type
        ckpt_dir = '../output/ckpt/silk.pth'#silk
        model = get_sym(model_config=cfg['MODEL'], image_shape=cfg['CMU_data']['IMAGE_SHAPE'])
        model.load_state_dict(torch.load(ckpt_dir, map_location=device)['model_state'])
        model.cuda()
        model.eval()
    else:
        model = get_sym(model_config=cfg['MODEL'], image_shape=cfg['CMU_data']['IMAGE_SHAPE'])


    # 用来组合深度学习和描述子模型
    if args.des == 'Silk':
        ckpt_dir = '../output/ckpt/silk.pth'# silk
    else:
        ckpt_dir = '../output/ckpt/CMU_best.pth'  # '../output/ckpt/CMU_best.pth'

    descmodel = get_sym(model_config=cfg['MODEL'], image_shape=cfg['CMU_data']['IMAGE_SHAPE'])
    descmodel.load_state_dict(torch.load(ckpt_dir, map_location=device)['model_state'])
    # model.load_params_from_file(filename=ckpt_dir, logger=logger, to_cpu=dist_test)
    descmodel.cuda()
    descmodel.eval()
    # 用来组合深度学习和描述子模型


    inliers_save = []; repeatability_save = []; translation_err_save = []; rotation_err_save = [];
    with torch.no_grad():
        matcher = Matcher(config.regist_thre)  # 匹配器
        coords_q = get_coords(os.path.join(config.data_path, config.query_name))
        radar_q = split_files_by_suffix(os.path.join(config.data_path, config.query_name))[args.type]

        coords_m = get_coords(os.path.join(config.data_path, config.map_name))
        radar_m = split_files_by_suffix(os.path.join(config.data_path, config.map_name))[args.type]

        index = faiss.IndexFlatL2(np.array(coords_m).shape[1])  # 这里是二维空间
        index.add(np.array(coords_m).astype('float32'))

        for i in range(len(radar_q)):
            print('=============================index_number:', i)
            radar_path_q = radar_q[i]
            radar_point_utm_q = coords_q[i]

            if args.type == 'npy':
                radar_echo_q = np.load(os.path.join(config.data_path, config.query_name,radar_path_q))
            else:
                radar_echo_q = cv2.imread(os.path.join(config.data_path, config.query_name,radar_path_q), cv2.IMREAD_GRAYSCALE)

            dist, map_ind = index.search(np.array(radar_point_utm_q).astype('float32').reshape(1,-1),1)

            radar_path_m = radar_m[map_ind[0][0]]
            radar_point_utm_m = coords_m[map_ind[0][0]]

            if args.type == 'npy':
                radar_echo_m = np.load(os.path.join(config.data_path, config.map_name,radar_path_m))
            else:
                radar_echo_m = cv2.imread(os.path.join(config.data_path, config.map_name,radar_path_m), cv2.IMREAD_GRAYSCALE)


            points1, descriptors1 = extract_kp_des(config, model, descmodel, radar_echo_q, radar_point_utm_q,
                                                   channel_num = 1, keypoint_num=config.keypoint_num, kp_method=args.kps, desc_method=args.des, nms=False, downsample = cfg['MODEL']['downsample'])
            points2, descriptors2 = extract_kp_des(config, model, descmodel, radar_echo_m, radar_point_utm_m,
                                                   channel_num = 1, keypoint_num=config.keypoint_num, kp_method=args.kps, desc_method=args.des, nms=False, downsample = cfg['MODEL']['downsample'])
            if len(points1) <= 3:
                continue
            # import matplotlib.pyplot as plt
            # plt.figure
            # plt.imshow(radar_echo_q)
            # plt.plot(points1[:,0], points1[:,1],'r+')
            # plt.figure()
            # plt.imshow(radar_echo_m)
            # plt.plot(points2[:,0], points2[:,1],'r+')
            # plt.show()

            # feature matching
            M_est, inliers, src_matched, dst_matched = matcher.imr(points1, points2, descriptors1, descriptors2, inlier_thre=config.inlier_thre)
            print('correspondence_number: ', len(src_matched))

            # filter result
            inliers = filter_matches(norm(radar_echo_q), src_matched, norm(radar_echo_m), dst_matched, inliers, distance_thresh = config.inlier_thre, viz=False)

            # calculate repeatablity
            repeatability, repeats = matcher.compute_repeatablity(points1, points2, threshold = config.repeat_thre)
            repeatability_save.append(repeatability)
            print('repeatability: %.4f' % (repeatability))

            # calculate localization error
            translation_err = matcher.Single_Channel_Error(radar_point_utm_q, radar_point_utm_m, M_est, resolution = 10/cfg['CMU_data']['IMAGE_SHAPE'][1])
            translation_err_save.append(translation_err)
            print('translation_error: %.4f' % (translation_err))

            # calculate inlier ratio
            inlier_ratio = len(inliers)/len(src_matched)

            inliers_save.append(inlier_ratio)
            print('inlier_ratio: %.4f' % (inlier_ratio))


        inlier_name = f"inlier_kp-{args.kps}_desc-{args.des}_Num-{config.keypoint_num}_M-{config.map_name}_Q-{config.query_name}_regist-{config.regist_thre}_inlier-{config.inlier_thre}.npy"
        np.save(os.path.join('../output/eval/inlier/', inlier_name), np.array(inliers_save))

        repeatability_name = f"Repeat_kp-{args.kps}_desc-{args.des}_Num-{config.keypoint_num}_M-{config.map_name}_Q-{config.query_name}_regist-{config.regist_thre}_repeat-{config.repeat_thre}.npy"
        np.save(os.path.join('../output/eval/repeat/', repeatability_name), np.array(repeatability_save))

        trans_name = f"TransErr_kp-{args.kps}_desc-{args.des}_Num-{config.keypoint_num}_M-{config.map_name}_Q-{config.query_name}_regist-{config.regist_thre}.npy"
        np.save(os.path.join('../output/eval/localization/', trans_name), np.array(translation_err_save))


