import open3d as o3d
import numpy as np
import copy

def preprocess_point_cloud(pcd, voxel_size):
    """
    预处理：降采样、估计法线、计算FPFH特征
    """
    # 1. 降采样
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # 2. 估计法线 (ICP点到面 和 FPFH特征计算都需要)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # 3. 计算FPFH特征 (用于粗配准)
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    粗配准：使用基于RANSAC的全局配准 (替代4PCS)
    """
    distance_threshold = voxel_size * 1.5
    print("   -> 正在进行粗配准 (Global RANSAC)...")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, result_ransac, voxel_size):
    """
    精配准：使用ICP (点到面)
    """
    distance_threshold = voxel_size * 0.4
    print("   -> 正在进行精配准 (ICP)...")
    
    # 必须确保目标点云有法线信息，否则只能用PointToPoint
    if not target.has_normals():
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def main():
    # ------------------- 参数设置 -------------------
    voxel_size = 4  # 体素大小，根据点云尺度调整（单位通常是米）
    # 假设您的文件名为 cloud_0.pcd 到 cloud_5.pcd
    file_paths = [f"D:\\vs_ws\\vision_detect\\point_cloud_ws\\point_clound_data\\Brake_band_point_cloud\\cloud_{i}.pcd" for i in range(6)] 
    
    # ------------------- 加载点云 -------------------
    pcds = []
    for path in file_paths:
        try:
            pcd = o3d.io.read_point_cloud(path)
            if pcd.is_empty():
                raise IOError(f"文件为空: {path}")
            pcds.append(pcd)
            print(f"成功加载: {path}, 点数: {len(pcd.points)}")
        except Exception as e:
            print(f"加载失败 {path}: {e}")
            return

    # ------------------- 顺序拼接 -------------------
    # 策略：将 cloud_i 配准到 cloud_{i-1}，然后累乘变换矩阵
    
    # 用于显示的完整点云
    full_cloud = o3d.geometry.PointCloud()
    full_cloud += pcds[0]  # 第0段作为基准，不需要移动
    
    # 保存每一段相对于全局（第0段）的变换矩阵
    global_transforms = [np.identity(4)] 
    
    print("\n开始拼接过程...")
    
    for i in range(1, len(pcds)):
        source_original = pcds[i]      # 当前要拼进去的点云 (待变换)
        target_original = pcds[i-1]    # 前一段点云 (参考目标)
        
        print(f"\n正在拼接第 {i} 段 (Source) 到第 {i-1} 段 (Target)...")
        
        # 1. 预处理
        source_down, source_fpfh = preprocess_point_cloud(source_original, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target_original, voxel_size)
        
        # 2. 粗配准 (Coarse)
        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        print(f"   粗配准 Fitness: {result_ransac.fitness:.4f}")
        
        # 3. 精配准 (ICP)
        result_icp = refine_registration(source_original, target_original, result_ransac, voxel_size)
        print(f"   精配准 Fitness: {result_icp.fitness:.4f}, RMSE: {result_icp.inlier_rmse:.4f}")
        
        # 4. 计算全局变换
        # result_icp.transformation 是将 Source 变到 Target (即 i -> i-1)
        # 我们需要 i -> 0 (Global)，公式：T_{i->0} = T_{i-1->0} @ T_{i->i-1}
        trans_i_to_i_minus_1 = result_icp.transformation
        trans_i_minus_1_to_0 = global_transforms[i-1]
        
        trans_i_to_0 = np.dot(trans_i_minus_1_to_0, trans_i_to_i_minus_1)
        global_transforms.append(trans_i_to_0)
        
        # 5. 变换并合并
        source_transformed = copy.deepcopy(source_original)
        source_transformed.transform(trans_i_to_0)
        full_cloud += source_transformed

    # ------------------- 结果可视化 -------------------
    print("\n拼接完成，正在降采样并可视化结果...")
    
    # 统一降采样以便显示（可选）
    full_cloud_down = full_cloud.voxel_down_sample(voxel_size)
    
    # 重新计算法线以便渲染更美观
    full_cloud_down.estimate_normals()
    
    # 保存结果
    o3d.io.write_point_cloud("merged_result.pcd", full_cloud)
    print("已保存拼接结果为 merged_result.pcd")
    
    # 可视化
    o3d.visualization.draw_geometries([full_cloud_down], 
                                      window_name="Stitched Point Cloud",
                                      width=1024, height=768)

if __name__ == "__main__":
    main()