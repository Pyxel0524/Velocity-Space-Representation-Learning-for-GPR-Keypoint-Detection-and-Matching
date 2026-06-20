import numpy as np
import open3d as o3d
import faiss
from sklearn.neighbors import NearestNeighbors
from skimage.measure import ransac
from skimage.transform import AffineTransform
from scipy.spatial import KDTree
import cv2

class Matcher:
    """
    input: point cloud: x, y, z // image_feature keypoint: x, y
           desc: N X D

    output: correspondence
    """

    def __init__(self, distance_threshold):
        self.dist_thre = distance_threshold

    def pcr(self, source_np, target_np, source_desc_np, target_desc_np, viz, mufilter = True):
        print('target_shape:',target_np.shape)
        centroid = np.mean(target_np.reshape(-1,3), axis=0)  # 计算质心

        source_np = self.center_point_cloud(source_np, centroid)
        target_np = self.center_point_cloud(target_np, centroid)

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_np)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_np)

        source_desc = o3d.pipelines.registration.Feature()
        target_desc = o3d.pipelines.registration.Feature()
        source_desc.data = source_desc_np.T
        target_desc.data = target_desc_np.T

        print('--------------------------keypoint registration-----------------------------')


        if mufilter == False:
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(source, target, source_desc,
                    target_desc, True, self.dist_thre,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
                    [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.dist_thre)],
                    o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 5000))# Converge condition

        else:
            filtered_correspondences = self.mutual_nearest_neighbor(source_desc_np.reshape(-1,source_desc_np.shape[2]),
                                                                    target_desc_np.reshape(-1,target_desc_np.shape[2]))
            correspondences = o3d.utility.Vector2iVector(filtered_correspondences)

            # RANSAC
            ransac_n = 8
            estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
            checkers = [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.dist_thre)
            ]
            criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 5000)

            # 使用 correspondence 进行 RANSAC
            result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                source, target, correspondences, self.dist_thre, estimation, ransac_n, checkers, criteria
            )

        if viz:
            # visualize registration result
            # 给点云上色
            source.paint_uniform_color([0, 0, 1])
            target.paint_uniform_color([1, 0, 1])


            # 显示
            # Convert points to homogeneous coordinates (Nx4)
            ones = np.ones((np.array(source.points).shape[0], 1))
            homogeneous_points = np.hstack((np.array(source.points), ones))


            # Apply transformation
            transformed_points = (result.transformation @ homogeneous_points.T).T
            trans_point = transformed_points[:, :3]
            trans_points = o3d.geometry.PointCloud()
            trans_points.points = o3d.utility.Vector3dVector(trans_point)

            # Convert back to 3D coordinates
            trans_points.paint_uniform_color([0, 0, 1])

            # before registration
            o3d.visualization.draw_geometries([source, target])
            # after registration
            o3d.visualization.draw_geometries([trans_points, target])

        return result

    def center_point_cloud(self, points, centroid):
        """
        Calculate point cloud center
        """
        centered_points = points - centroid  # 平移点云到中心
        return centered_points

    def imr(self, source, target, source_desc, target_desc, inlier_thre = 1.0):
        # Neighborhood Nearest
        d = source_desc.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(target_desc.astype('float32'))
        distances, indices = index.search(source_desc.astype('float32'), k=2)

        # Ratio Test
        good_matches = []
        for i in range(len(source_desc)):
            d1, d2 = distances[i]
            j1, j2 = indices[i]
            if d1 <= 1 * d2:  # 比值过滤
                good_matches.append((i, j1))

        # Matching Pair
        if len(good_matches) <= 3:
            raise ValueError("有效匹配点不足，无法估计变换")

        matched_query = np.array([source[i] for i, _ in good_matches])
        matched_target = np.array([target[j] for _, j in good_matches])

        # Only Translation
        class TranslationOnlyTransform(AffineTransform):
            def __init__(self):
                super().__init__()

            def estimate(self, src, dst):
                t = np.mean(dst - src, axis=0)
                self.params = np.array([
                    [1, 0, t[0]],
                    [0, 1, t[1]],
                    [0, 0, 1]
                ])
                return True
        model, inliers = ransac(
            (matched_query, matched_target),
            TranslationOnlyTransform,
            min_samples=3,
            residual_threshold=inlier_thre,
            max_trials=5000
        )

        return model, inliers, matched_query, matched_target


    def mutual_nearest_neighbor(self, source_desc, target_desc, confidence_threshold = 0.8):
        """
        进行双向最近邻（MNN）匹配，去除一对多错误匹配。

        Parameters:
        - source_desc: (N, F) 源点云特征
        - target_desc: (M, F) 目标点云特征

        Returns:
        - 过滤后的一对一匹配对 (Nx2 数组)
        """
        # 1 计算 Source → Target 最近邻匹配
        nn1 = KDTree(target_desc)
        distances1, indices1 = nn1.query(source_desc)

        # 2 计算 Target → Source 最近邻匹配
        nn2 = KDTree(source_desc)
        distances2, indices2 = nn2.query(target_desc)

        # 3 仅保留 Mutual Nearest Neighbors
        mutual_matches = []
        match_distances = []  # 记录匹配距离（用于筛选置信度）

        for i, idx1 in enumerate(indices1.flatten()):
            if indices2[idx1] == i:  # 确保是互相最近邻
                mutual_matches.append([i, idx1])
                match_distances.append(distances1[i][0])  # 记录匹配的距离

        mutual_matches = np.array(mutual_matches)
        # match_distances = np.array(match_distances)

        return np.array(mutual_matches)


    def compute_repeatablity(self, source_points_np, target_points_np, threshold=0.5, scale = 50):
        """
        计算特征点的重复率（Repeatability）。

        :param source_features: 源点云的特征点 (Nx3) NumPy 数组
        :param target_features: 目标点云的特征点 (Mx3) NumPy 数组
        :param threshold: 判断是否为重复特征点的欧式距离阈值
        :return: 重复率 (float), 重复的点对索引 (list)
        """

        # 构建 KD-Tree 进行最近邻搜索
        tree = KDTree(target_points_np)
        # 计算每个源特征点到目标特征点的最近邻
        distances, indices = tree.query(source_points_np, k=1)
        # 找到符合阈值的匹配点
        inliers = np.where(distances.flatten() < threshold)[0]
        # 计算重复率
        repeatability = len(inliers) / len(source_points_np)
        return repeatability, inliers.tolist()



    def compute_heading(self, start, end):
        """
        Compute the heading (azimuth angle) between two UTM points.

        Parameters:
        start (numpy.ndarray): [x, y] coordinates of the start point.
        end (numpy.ndarray): [x, y] coordinates of the end point.

        Returns:
        float: Heading angle in degrees.
        """
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        heading = np.degrees(np.arctan2(delta_y, delta_x))  # Convert radians to degrees
        return heading


    def Relative_Error(self, source_points_np, target_points_np, transformation):
        """
        Compute the pose error between ground truth and estimated transformation matrices.

        Parameters:
        T_gt (numpy.ndarray): 4x4 ground truth transformation matrix.
        T_est (numpy.ndarray): 4x4 estimated transformation matrix.

        Returns:
        tuple: (translation_error, rotation_error_in_degrees)
        """
        central = int(len(source_points_np)/2)
        rotation_target = self.compute_heading(target_points_np[central - 5,5], target_points_np[central + 5,5])
        rotation_source = self.compute_heading(source_points_np[central - 5,5], source_points_np[central + 5,5])
        rotation_gt = rotation_target - rotation_source

        trans_gt = target_points_np[central,5] - source_points_np[central,5]

        theta = np.radians(rotation_gt)  # Convert degrees to radians
        T_gt = np.array([
            [np.cos(theta), -np.sin(theta), 0, trans_gt[0]],
            [np.sin(theta), np.cos(theta), 0, trans_gt[1]],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        T_est = transformation

        # Extract translation vectors
        t_gt = T_gt[:3, 3]
        t_est = T_est[:3, 3]

        # Compute translation error (Euclidean distance)
        translation_error = np.linalg.norm(t_gt - t_est)

        # Extract rotation matrices
        R_gt = T_gt[:3, :3]
        R_est = T_est[:3, :3]

        # Compute rotation error (angle in radians)
        rotation_matrix_diff = R_gt.T @ R_est
        cos_theta = (np.trace(rotation_matrix_diff) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Avoid numerical issues
        rotation_error = np.arccos(cos_theta)  # In radians


        rotation_error_degrees = np.degrees(rotation_error)

        return translation_error, rotation_error_degrees

    def Single_Channel_Error(self, source_points_np, target_points_np, transformation, resolution):

        trans_gt = np.array(target_points_np) - np.array(source_points_np)
        trans_gt = np.linalg.norm(trans_gt)/resolution if (source_points_np[0] < target_points_np[0]) else -np.linalg.norm(trans_gt)
        trans_est = np.array(transformation.params[0,2])

        translation_error = np.linalg.norm(trans_gt - trans_est)

        return translation_error


    def compute_inlier_ratio(self, source_np, target_np, correspondences, source_points_np, target_points_np, distance_threshold, ind, viz = True):
        centroid = np.mean(target_np.reshape(-1,3), axis = 0)  # 计算质心

        source_np = self.center_point_cloud(source_np, centroid)
        target_np = self.center_point_cloud(target_np, centroid)

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_np)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_np)

        # 给点云上色
        source.paint_uniform_color([1, 0, 1])
        target.paint_uniform_color([0, 0, 1])


        # calculate transformation_gt
        central = int(len(source_points_np)/2)
        rotation_target = self.compute_heading(target_points_np[central - 5,5], target_points_np[central + 5,5])
        rotation_source = self.compute_heading(source_points_np[central - 5,5], source_points_np[central + 5,5])
        rotation_gt = rotation_target - rotation_source

        trans_gt = target_points_np[central,5] - source_points_np[central,5]

        theta = np.radians(rotation_gt)  # Convert degrees to radians
        transformation = np.array([
            [np.cos(theta), -np.sin(theta), 0, trans_gt[0]],
            [np.sin(theta), np.cos(theta), 0, trans_gt[1]],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])


        points = []
        lines = []
        colors = []
        corr = 0
        for match in correspondences:
            source_idx, target_idx = match
            source_point = np.asarray(source.points)[source_idx]
            target_point = np.asarray(target.points)[target_idx]

            lines.append([source_point, target_point])

            # 判断是否为正确匹配
            source_point_stack = np.hstack((np.array(source_point), 1))

            # Apply transformation
            transformed_point = (transformation @ source_point_stack)[:3]

            is_correct = np.linalg.norm(transformed_point - target_point) < distance_threshold

            if is_correct:
                colors.append([0, 1, 0])  # Green for correct matches
                corr += 1
            else:
                colors.append([1, 0, 0])  # Red for incorrect matches

        inlier_ratio = corr / len(correspondences) if len(correspondences) > 0 else 0
        if viz:
            # source点云偏移
            # 沿 Z 轴移 10
            translation = np.array([0.0, 0.0, -10.0])  # Z 方向平移 10
            source.translate(translation)


            line_set = o3d.geometry.LineSet()
            print(np.array(lines).shape)
            line_set.points = o3d.utility.Vector3dVector(np.vstack(lines))
            line_set.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(0, len(lines) * 2, 2)])
            line_set.colors = o3d.utility.Vector3dVector(colors)

            # 使用Visualizer进行可视化
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)  # 显式设置窗口可见

            # 添加几何体
            vis.add_geometry(source)
            vis.add_geometry(target)
            vis.add_geometry(line_set)

            # 获取视图控制器并设置视角
            view_ctl = vis.get_view_control()
            view_ctl.set_front([1, 0, -0.5])  # X, Y, Z
            view_ctl.set_lookat((np.array(source.get_center()) + np.array(target.get_center())) / 2)  # 观察点居中
            view_ctl.set_up([0, 0, -1])  # 上方向为Z轴, 要反过来看才对
            view_ctl.set_zoom(0.9)  # 根据需要调整缩放

            # 设置渲染选项
            vis.get_render_option().background_color = [1, 1, 1]  # 白色背景

            # 强制刷新并截图
            vis.update_renderer()
            vis.poll_events()
            vis.capture_screen_image(f"../Animation/{ind}.png")
            # 运行可视化
            # vis.run()
            vis.destroy_window()
            return inlier_ratio
        else:
            return inlier_ratio