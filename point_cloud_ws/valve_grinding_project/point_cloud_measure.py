#点云数据跨度范围
import open3d as o3d
import numpy as np

# 读取你的第一个点云
pcd = o3d.io.read_point_cloud(r"D:\vs_ws\vision_detect\point_cloud_ws\point_clound_data\Brake_band_point_cloud\cloud_0.pcd") # 替换为实际路径
points = np.asarray(pcd.points)

print(f"点数: {len(points)}")
print(f"X轴范围: {points[:,0].min()} ~ {points[:,0].max()}")
print(f"Y轴范围: {points[:,1].min()} ~ {points[:,1].max()}")
print(f"Z轴范围: {points[:,2].min()} ~ {points[:,2].max()}")

# 简单的尺度估算
max_range = np.max(points.max(0) - points.min(0))
print(f"最大跨度: {max_range}")