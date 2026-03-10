#显示所有能拟合的面



import open3d as o3d
import numpy as np
import os

def fit_plane_from_pcd(file_path):
    # --- 1. 读取 PCD 文件 ---
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    print(f"正在读取文件: {file_path} ...")
    pcd = o3d.io.read_point_cloud(file_path)
    
    # 检查点云是否为空
    if pcd.is_empty():
        print("错误: 读取的点云为空，请检查文件格式或路径。")
        return

    print(f"原始点云点数: {len(pcd.points)}")

    # (可选) 如果点云非常大（例如超过百万点），为了速度可以先降采样
    # pcd = pcd.voxel_down_sample(voxel_size=0.005) 

    # --- 2. 执行 RANSAC 平面拟合 ---
    # distance_threshold: 设为 0.01 表示 1mm (假设单位是米) 或 0.01mm (取决于你的点云单位)
    # 请根据阀体密封面的加工精度调整此参数
    threshold = 0.01 
    print(f"正在进行 RANSAC 拟合 (距离阈值={threshold})...")
    
    plane_model, inliers = pcd.segment_plane(distance_threshold=threshold,
                                             ransac_n=3,
                                             num_iterations=1000)
    
    [a, b, c, d] = plane_model
    print(f"拟合结果 -> 平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(f"平面法向量 (Normal): [{a:.4f}, {b:.4f}, {c:.4f}]")
    
    # --- 3. 提取结果 ---
    # 局内点 (Inliers): 属于密封面的点
    inlier_cloud = pcd.select_by_index(inliers)
    # 离群点 (Outliers): 阀孔、背景或其他噪点
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    
    print(f"平面点数: {len(inlier_cloud.points)}")
    print(f"噪声/离群点数: {len(outlier_cloud.points)}")

    # --- 4. 可视化 ---
    # 拟合出的平面涂成红色
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # 其余部分保持灰色 (或者涂成其他颜色区分)
    # outlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    
    # 创建坐标轴辅助观看 (原点位置，红=X, 绿=Y, 蓝=Z)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, mesh_frame], 
                                      window_name="PCD RANSAC Result",
                                      width=1024, height=768,
                                      left=50, top=50)



# === 使用说明 ===
if __name__ == "__main__":
    # 使用你提供的路径
    input_file = r"D:\vs_ws\vision_detect\point_cloud_ws\point_clound_data\2026-01-12-16-14-22_RTPointCloud.pcd" 
    
    fit_plane_from_pcd(input_file)